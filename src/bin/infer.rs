//! BitNet b1.58-2B GGUF 推理（标量参考实现）。

use std::path::Path;

use bitnet_rs::gguf::{GgufFile, GgufValue, type_block_size};
use bitnet_rs::tokenizer::BpeTokenizer;
use bitnet_rs::{I2Tensor, QK_I2_S, linear_row_q, i2_matvec_parallel, quantize_i8_blocks};

const I2_S_TYPE: u32 = 36; // GGUF 里 BitNet 三值权重类型

struct Model {
    n_embd: usize,
    n_layer: usize,
    n_head: usize,
    n_head_kv: usize,
    head_dim: usize,
    n_ff: usize,
    vocab_size: usize,
    rope_theta: f32,
    rope_freq_scale: f32,
    rms_eps: f32,
    n_ctx: usize,

    token_embd: TensorData,
    final_norm: Vec<f32>,
    layers: Vec<Layer>,
}

struct Layer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    attn_sub_norm: Vec<f32>,
    ffn_sub_norm: Vec<f32>,
    q: TensorData,
    k: TensorData,
    v: TensorData,
    o: TensorData,
    gate: TensorData,
    up: TensorData,
    down: TensorData,
}

enum TensorData {
    F32(Vec<f32>),
    I2(I2Tensor),
}

fn load_tensor(g: &mut GgufFile, name: &str) -> TensorData {
    let t = g.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("tensor {name} not found")).clone();
    let ne = t.dims.iter().product::<u64>();
    match t.ggml_type {
        0 => {
            let mut buf = vec![0u8; (ne * 4) as usize];
            g.read_tensor_data(&t, &mut buf).unwrap();
            let v: Vec<f32> = buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            TensorData::F32(v)
        }
        2 => {
            let mut buf = vec![0u8; (ne * 2) as usize];
            g.read_tensor_data(&t, &mut buf).unwrap();
            let v: Vec<f32> = buf.chunks_exact(2).map(|c| {
                let u = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((u as u32) << 16)
            }).collect();
            TensorData::F32(v)
        }
        1 => {
            let mut buf = vec![0u8; (ne * 2) as usize];
            g.read_tensor_data(&t, &mut buf).unwrap();
            let v: Vec<f32> = buf.chunks_exact(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
            TensorData::F32(v)
        }
        tt if tt == I2_S_TYPE => {
            // 官方 GGUF I2_S 布局：每 128 权重 = 32 字节（组 32，位段 6-2*(j/32)），
            // 张量尾 32 字节对齐，前 4 字节 = 张量级 f32 scale。
            let n_elem = ne as usize;
            let packed_size = n_elem / 4;
            let mut buf = vec![0u8; packed_size + 32];
            g.read_tensor_data(&t, &mut buf).unwrap();
            let scale = f32::from_le_bytes(buf[packed_size..packed_size + 4].try_into().unwrap());

            // 转成 NEON 布局（64 权重/16 字节，组 16）：packed[b*16+p] 位段 (6-2g) ← 权重 64b+16g+p
            let src = &buf[..packed_size];
            let n_blocks_64 = n_elem / QK_I2_S;
            let mut packed = vec![0u8; n_blocks_64 * 16];
            for idx in 0..n_elem {
                let code = (src[(idx / 128) * 32 + (idx % 128) % 32] >> (6 - 2 * ((idx % 128) / 32))) & 0b11;
                let b = idx / QK_I2_S;
                let j = idx % QK_I2_S;
                let g = j / 16;
                let p = j % 16;
                packed[b * 16 + p] |= code << (6 - 2 * g);
            }
            let scales = vec![scale; n_blocks_64];
            TensorData::I2(I2Tensor::from_blocks(packed, scales))
        }
        other => panic!("unsupported tensor type {other} for {name}"),
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut e = 127i32 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

fn i2_matvec(w: &TensorData, x: &[f32]) -> Vec<f32> {
    match w {
        TensorData::I2(t) => {
            if std::env::var("F32_MODE").is_ok() {
                // 实验：全精度（dequantize 权重 × f32 激活）
                let dq = t.dequantize();
                let k = x.len();
                let n_rows = t.n / k;
                dq.chunks_exact(k).map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
            } else {
                i2_matvec_parallel(t, x.len(), x)
            }
        }
        TensorData::F32(m) => {
            let k = x.len();
            let n_rows = m.len() / k;
            m.chunks_exact(k).map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect()
        }
    }
}

/// q/k/v 与 gate/up 共享激活量化：量化一次，多矩阵复用
fn i2_matvec_q(w: &TensorData, y: &[i8], act_scales: &[f32], k: usize) -> Vec<f32> {
    match w {
        TensorData::I2(t) => {
            let n_rows = t.n / k;
            (0..n_rows).map(|r| linear_row_q(t, r, k, y, act_scales)).collect()
        }
        TensorData::F32(m) => {
            let n_rows = m.len() / k;
            m.chunks_exact(k).map(|row| {
                let x: Vec<f32> = y.iter().enumerate().map(|(i, &v)| v as f32 * act_scales[i / QK_I2_S]).collect();
                row.iter().zip(&x).map(|(a, b)| a * b).sum()
            }).collect()
        }
    }
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * inv * b).collect()
}

/// BitNet 的 relu2 = ReLUSquaredActivation：max(x, 0)²
fn relu2(v: f32) -> f32 {
    let r = if v > 0.0 { v } else { 0.0 };
    r * r
}

fn rope(q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, n_head: usize, n_head_kv: usize, theta: f32) {
    let apply = |vec: &mut [f32], heads: usize| {
        for h in 0..heads {
            for d in (0..head_dim).step_by(2) {
                // 标准 llama rope：freq = theta^(-d/head_dim)（d 为偶数索引）
                let freq = 1.0 / theta.powf(d as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let (s, c) = angle.sin_cos();
                let idx = h * head_dim + d;
                let a = vec[idx];
                let b = vec[idx + 1];
                vec[idx] = a * c - b * s;
                vec[idx + 1] = a * s + b * c;
            }
        }
    };
    apply(q, n_head);
    apply(k, n_head_kv);
}

fn softmax(v: &[f32]) -> Vec<f32> {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = v.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn load_model(path: &Path) -> (Model, BpeTokenizer) {
    let mut g = GgufFile::open(path).unwrap();
    let s = |k: &str| g.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let u = |k: &str| g.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let f = |k: &str| g.get(k).and_then(|v| v.as_f32()).unwrap_or(1e-5);

    let arch = s("general.architecture");
    let p = "bitnet-b1.58.";
    let n_layer = u(&format!("{p}block_count"));
    let n_embd = u(&format!("{p}embedding_length"));
    let n_head = u(&format!("{p}attention.head_count"));
    let n_head_kv = u(&format!("{p}attention.head_count_kv"));
    let head_dim = u(&format!("{p}rope.dimension_count"));
    let head_dim = if head_dim == 0 { n_embd / n_head } else { head_dim };
    let n_ff = u(&format!("{p}feed_forward_length"));
    let vocab_size = u(&format!("{p}vocab_size"));
    let rope_theta = f(&format!("{p}rope.freq_base"));
    let rope_freq_scale = f(&format!("{p}rope.frequency_scale"));
    let rms_eps = f(&format!("{p}attention.layer_norm_rms_epsilon"));

    println!("arch={arch} layers={n_layer} embd={n_embd} head={n_head} kv={n_head_kv} hd={head_dim} ff={n_ff} vocab={vocab_size} theta={rope_theta} freq_scale={rope_freq_scale} eps={rms_eps}");

    let token_embd = load_tensor(&mut g, "token_embd.weight");
    let final_norm = match load_tensor(&mut g, "output_norm.weight") { TensorData::F32(v) => v, _ => panic!("output_norm not f32") };
    // 输出层与 embedding 共享（tied weights）

    let mut layers = Vec::new();
    for i in 0..n_layer {
        let p = format!("blk.{i}.");
        layers.push(Layer {
            attn_norm: match load_tensor(&mut g, &format!("{p}attn_norm.weight")) { TensorData::F32(v) => v, _ => panic!("norm not f32") },
            ffn_norm: match load_tensor(&mut g, &format!("{p}ffn_norm.weight")) { TensorData::F32(v) => v, _ => panic!("norm not f32") },
            attn_sub_norm: match load_tensor(&mut g, &format!("{p}attn_sub_norm.weight")) { TensorData::F32(v) => v, _ => panic!("attn_sub_norm not f32") },
            ffn_sub_norm: match load_tensor(&mut g, &format!("{p}ffn_sub_norm.weight")) { TensorData::F32(v) => v, _ => panic!("ffn_sub_norm not f32") },
            q: load_tensor(&mut g, &format!("{p}attn_q.weight")),
            k: load_tensor(&mut g, &format!("{p}attn_k.weight")),
            v: load_tensor(&mut g, &format!("{p}attn_v.weight")),
            o: load_tensor(&mut g, &format!("{p}attn_output.weight")),
            gate: load_tensor(&mut g, &format!("{p}ffn_gate.weight")),
            up: load_tensor(&mut g, &format!("{p}ffn_up.weight")),
            down: load_tensor(&mut g, &format!("{p}ffn_down.weight")),
        });
        if i % 4 == 0 {
            println!("  layer {i} loaded");
        }
    }
    println!("{} layers loaded", layers.len());

    let tokens: Vec<(String, u32)> = g.get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().enumerate().filter_map(|(i, v)| v.as_str().map(|s| (s.to_string(), i as u32))).collect())
        .unwrap_or_default();
    let merges: Vec<(String, String)> = g.get("tokenizer.ggml.merges")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).filter_map(|s| s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string()))).collect())
        .unwrap_or_default();
    let special: Vec<(String, u32)> = g.get("tokenizer.ggml.special_tokens")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().enumerate().filter_map(|(i, v)| v.as_str().map(|s| (s.to_string(), i as u32))).collect())
        .unwrap_or_default();
    let bos = g.get("tokenizer.ggml.bos_token_id").and_then(|v| v.as_u32()).unwrap_or(1);
    let eos = g.get("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u32()).unwrap_or(2);
    println!("tokenizer: {} tokens, {} merges, {} special, bos={bos} eos={eos}", tokens.len(), merges.len(), special.len());
    let tok = BpeTokenizer::new(tokens, merges, special);

    let model = Model {
        n_embd, n_layer, n_head, n_head_kv, head_dim, n_ff, vocab_size,
        rope_theta, rope_freq_scale, rms_eps, n_ctx: 4096,
        token_embd, final_norm, layers,
    };
    (model, tok)
}

struct KVCache {
    k: Vec<f32>, // [layer][head_kv][ctx][head_dim]
    v: Vec<f32>,
    pos: usize,
}

impl KVCache {
    fn new(model: &Model) -> KVCache {
        let per = model.n_head_kv * model.n_ctx * model.head_dim;
        KVCache {
            k: vec![0.0; model.n_layer * per],
            v: vec![0.0; model.n_layer * per],
            pos: 0,
        }
    }
}

/// 单 token 前向，返回 logits
fn forward_token(model: &Model, kv: &mut KVCache, token: u32) -> Vec<f32> {
    let n = model.n_embd;
    let mut x: Vec<f32> = match &model.token_embd {
        TensorData::F32(e) => e[(token as usize) * n..(token as usize + 1) * n].to_vec(),
        TensorData::I2(_) => panic!("token_embd I2 unsupported"),
    };

    for (li, layer) in model.layers.iter().enumerate() {
        if std::env::var("DBG").is_ok() && li % 5 == 0 {
            let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!("L{li} |x|={norm:.3}");
        }
        // --- attention ---
        let xn = rms_norm(&x, &layer.attn_norm, model.rms_eps);
        let (xn_q, xn_scales) = quantize_i8_blocks(&xn);
        let q = i2_matvec_q(&layer.q, &xn_q, &xn_scales, model.n_embd);
        let mut k = i2_matvec_q(&layer.k, &xn_q, &xn_scales, model.n_embd);
        let v = i2_matvec_q(&layer.v, &xn_q, &xn_scales, model.n_embd);

        let mut q = q;

        rope(&mut q, &mut k, kv.pos, model.head_dim, model.n_head, model.n_head_kv, model.rope_theta);

        // 写 KV cache
        let per = model.n_head_kv * model.n_ctx * model.head_dim;
        let k_off = li * per;
        for h in 0..model.n_head_kv {
            for d in 0..model.head_dim {
                let idx = k_off + (h * model.n_ctx + kv.pos) * model.head_dim + d;
                kv.k[idx] = k[h * model.head_dim + d];
                kv.v[idx] = v[h * model.head_dim + d];
            }
        }

        // attention: 每 query head 对全部 kv heads（n_head_kv == n_head 时 1:1）
        let n_groups = model.n_head / model.n_head_kv;
        let ctx_len = kv.pos + 1;
        let mut attn_out = vec![0f32; n];
        for h in 0..model.n_head {
            let kvh = h / n_groups;
            let q_base = h * model.head_dim;
            let kv_base = kvh * model.n_ctx * model.head_dim;
            let mut scores = Vec::with_capacity(ctx_len);
            for t in 0..ctx_len {
                let mut acc = 0f32;
                for d in 0..model.head_dim {
                    acc += q[q_base + d] * kv.k[k_off + kv_base + t * model.head_dim + d];
                }
                scores.push(acc / (model.head_dim as f32).sqrt());
            }
            let probs = softmax(&scores);
            for d in 0..model.head_dim {
                let mut acc = 0f32;
                for t in 0..ctx_len {
                    acc += probs[t] * kv.v[k_off + kv_base + t * model.head_dim + d];
                }
                attn_out[q_base + d] = acc;
            }
        }

        // SubLN: attn_sub_norm 在 o_proj 之前（inner_attn_ln）
        let attn_ln = rms_norm(&attn_out, &layer.attn_sub_norm, model.rms_eps);
        let o = i2_matvec(&layer.o, &attn_ln);
        for i in 0..n {
            x[i] += o[i];
        }

        // --- MLP ---
        let xn = rms_norm(&x, &layer.ffn_norm, model.rms_eps);
        let (xn_q, xn_scales) = quantize_i8_blocks(&xn);
        let g = i2_matvec_q(&layer.gate, &xn_q, &xn_scales, model.n_embd);
        let up = i2_matvec_q(&layer.up, &xn_q, &xn_scales, model.n_embd);
        let mut gated = vec![0f32; model.n_ff];
        for i in 0..model.n_ff {
            gated[i] = relu2(g[i]) * up[i];
        }
        // SubLN: ffn_sub_norm 在 down_proj 之前（ffn_layernorm）
        let gated_ln = rms_norm(&gated, &layer.ffn_sub_norm, model.rms_eps);
        let down = i2_matvec(&layer.down, &gated_ln);
        for i in 0..n {
            x[i] += down[i];
        }
    }

    let xn = rms_norm(&x, &model.final_norm, model.rms_eps);
    // tied: logits = token_embd (F16→F32) × xn
    let logits = match &model.token_embd {
        TensorData::F32(e) => {
            let k = model.n_embd;
            let n_rows = e.len() / k;
            let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(n_rows);
            let chunk = n_rows.div_ceil(n_threads);
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(n_threads);
                for t_id in 0..n_threads {
                    let start = t_id * chunk;
                    let end = (start + chunk).min(n_rows);
                    if start >= end {
                        continue;
                    }
                    let e = &e[..];
                    let xn = &xn[..];
                    handles.push(s.spawn(move || {
                        (start..end).map(|r| e[r * k..(r + 1) * k].iter().zip(xn).map(|(a, b)| a * b).sum()).collect::<Vec<f32>>()
                    }));
                }
                let mut out = vec![0f32; n_rows];
                for (t_id, h) in handles.into_iter().enumerate() {
                    let part = h.join().unwrap();
                    out[t_id * chunk..t_id * chunk + part.len()].copy_from_slice(&part);
                }
                out
            })
        }
        TensorData::I2(_) => panic!("token_embd I2 unsupported"),
    };
    kv.pos += 1;
    logits
}

fn sample(logits: &[f32], temp: f32) -> u32 {
    if temp <= 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i as u32).unwrap();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|l| ((l - max) / temp).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    let mut r: u32 = unsafe { std::mem::transmute::<[u8; 4], u32>(rand_bytes()) };
    r = r % 1_000_000_000;
    let mut acc = 0.0;
    let target = r as f32 / 1_000_000_000.0;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if target <= acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

fn rand_bytes() -> [u8; 4] {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    [(t >> 0) as u8, (t >> 8) as u8, (t >> 16) as u8, (t >> 24) as u8]
}

/// 交互聊天：持续会话（KV cache 跨轮保留，支持长上下文）
/// chat 模板: Human: {input}\n\nBITNETAssistant: {eos}
fn chat_loop(model: &Model, tok: &BpeTokenizer) {
    use std::io::{BufRead, Write};
    let bos = 128000u32;
    let eos = 128001u32;
    let mut kv = KVCache::new(model);
    let mut history: Vec<u32> = Vec::new();
    let stdin = std::io::stdin();
    println!("=== BitNet b1.58-2B Chat (Ctrl+D 退出) ===");
    loop {
        print!("\nYou: ");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            println!();
            break;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        // 组装本轮输入（含历史，但只需增量 token）
        let prompt = format!("Human: {}\n\nBITNETAssistant: ", input);
        let ids = tok.encode(&prompt, false, bos);
        let t0 = std::time::Instant::now();
        for &t in &ids {
            forward_token(model, &mut kv, t);
        }
        history.extend_from_slice(&ids);

        // 生成直到 eos 或上限
        let mut out_tokens = Vec::new();
        let max_gen = std::env::var("MAX_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
        let mut last = *history.last().unwrap();
        let mut done = false;
        for _ in 0..max_gen {
            if kv.pos >= model.n_ctx {
                println!("\n[context full: {} tokens]", kv.pos);
                done = true;
                break;
            }
            let logits = forward_token(model, &mut kv, last);
            let next = sample(&logits, 0.6);
            if next == eos {
                done = true;
                break;
            }
            out_tokens.push(next);
            last = next;
        }
        let _ = done;
        let dt = t0.elapsed().as_secs_f32();
        let reply = tok.decode(&out_tokens, &std::collections::HashMap::new());
        println!("\nBitnet: {reply}  ({dt:.1}s, {} tokens, ctx {})", out_tokens.len(), kv.pos);
        history.extend_from_slice(&out_tokens);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: infer <model.gguf> [prompt] [-t temp] [-n tokens]");
        std::process::exit(1);
    }
    let path = &args[1];
    let mut prompt = "The meaning of life is".to_string();
    let mut temp = 0.0;
    let mut n_tokens = 32;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "-t" => { temp = args[i + 1].parse().unwrap(); i += 2; }
            "-n" => { n_tokens = args[i + 1].parse().unwrap(); i += 2; }
            s => { prompt = s.to_string(); i += 1; }
        }
    }

    let start = std::time::Instant::now();
    let (model, tok) = load_model(Path::new(path));
    println!("model loaded in {:.1}s", start.elapsed().as_secs_f32());

    if std::env::var("CHAT").is_ok() || args.iter().any(|a| a == "--chat") {
        chat_loop(&model, &tok);
        return;
    }

    let bos = 128000u32;
    let ids = tok.encode(&prompt, true, bos);
    println!("prompt: {} tokens", ids.len());

    let mut kv = KVCache::new(&model);
    let mut all: Vec<u32> = ids.clone();
    let t0 = std::time::Instant::now();
    for &t in &ids {
        forward_token(&model, &mut kv, t);
    }
    let pp = t0.elapsed().as_secs_f32() / ids.len() as f32;
    println!("prompt eval: {pp:.2}s/token");

    if std::env::var("DBG_LOGITS").is_ok() {
        let logits = forward_token(&model, &mut kv, *ids.last().unwrap());
        let mut top: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("== logits top-10 ==");
        for (i, (id, v)) in top.iter().take(10).enumerate() {
            let dec = tok.decode(&[*id as u32], &std::collections::HashMap::new());
            eprintln!("  {i}. {id}: {v:.4} '{dec}'");
        }
        eprintln!("logits: min={:.2} max={:.2} nan={}",
            logits.iter().cloned().fold(f32::INFINITY, f32::min),
            logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            logits.iter().filter(|x| x.is_nan()).count());
        return;
    }
    let mut out = prompt.clone();
    for i in 0..n_tokens {
        let t0 = std::time::Instant::now();
        let logits = forward_token(&model, &mut kv, *all.last().unwrap());
        let next = sample(&logits, temp);
        let dt = t0.elapsed().as_secs_f32();
        all.push(next);
        out.push_str(" ");
        out.push_str(&tok.decode(&[next], &std::collections::HashMap::new()));
        print!("token {i}: {dt:.1}s  {}", tok.decode(&[next], &std::collections::HashMap::new()));
        println!();
        if next == 2 {
            break;
        }
    }
    println!("\n=== OUTPUT ===\n{}", out);
}
