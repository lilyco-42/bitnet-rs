//! BitNet b1.58-2B GGUF 模型加载与生成（从 `bin/infer.rs` 提升进 lib 的最小补丁）
//!
//! 与原 bin 实现的差别：
//! - 加载失败不再 panic，改返回 [`ModelError`]（lib 不能炸宿主进程）
//! - 去掉加载期 `println!` 与 `F32_MODE` / `DBG` 环境变量分支
//! - [`Model::generate`] 采用标准自回归循环：prompt 最后一遍 forward 的 logits
//!   直接采样（原 bin 单轮路径会把最后一个 prompt token 重复 forward 一次）
//! - 温度采样的随机源由 `transmute` 改为 xorshift（语义等价，去掉 unsafe）
//!
//! 逐位计算路径（rms_norm / rope / relu2 / I2 点积）与原实现完全一致。

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::gguf::GgufFile;
use crate::tokenizer::BpeTokenizer;
use crate::{I2Tensor, QK_I2_S, linear_row_q, quantize_i8_blocks};

/// GGUF 里 BitNet 三值权重的 ggml_type 编号（官方 I2_S 布局）
pub const GGML_TYPE_I2_S: u32 = 36;

/// 模型加载 / 推理错误
#[derive(Debug)]
pub enum ModelError {
    /// 文件 IO（含"文件不存在"——上游可据此给下载指引）
    Io(std::io::Error),
    /// GGUF 结构 / 张量布局不符合预期
    Format(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Io(e) => write!(f, "IO 错误: {e}"),
            ModelError::Format(m) => write!(f, "模型格式错误: {m}"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        ModelError::Io(e)
    }
}

/// 单个权重的存储形态：F32/BF16/F16 反量化为 f32，I2_S 保持打包
pub enum TensorData {
    F32(Vec<f32>),
    I2(I2Tensor),
}

fn load_tensor(g: &mut GgufFile, name: &str) -> Result<TensorData, ModelError> {
    let t = g
        .tensors
        .iter()
        .find(|t| t.name == name)
        .cloned()
        .ok_or_else(|| ModelError::Format(format!("张量 {name} 不存在")))?;
    let ne = t.dims.iter().product::<u64>();
    match t.ggml_type {
        0 => {
            let mut buf = vec![0u8; (ne * 4) as usize];
            g.read_tensor_data(&t, &mut buf)?;
            let v: Vec<f32> = buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(TensorData::F32(v))
        }
        2 => {
            let mut buf = vec![0u8; (ne * 2) as usize];
            g.read_tensor_data(&t, &mut buf)?;
            let v: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let u = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((u as u32) << 16)
                })
                .collect();
            Ok(TensorData::F32(v))
        }
        1 => {
            let mut buf = vec![0u8; (ne * 2) as usize];
            g.read_tensor_data(&t, &mut buf)?;
            let v: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect();
            Ok(TensorData::F32(v))
        }
        tt if tt == GGML_TYPE_I2_S => {
            // 官方 GGUF I2_S 布局：每 128 权重 = 32 字节（组 32，位段 6-2*(j/32)），
            // 张量尾 32 字节对齐，前 4 字节 = 张量级 f32 scale。
            let n_elem = ne as usize;
            let packed_size = n_elem / 4;
            let mut buf = vec![0u8; packed_size + 32];
            g.read_tensor_data(&t, &mut buf)?;
            let scale = f32::from_le_bytes(buf[packed_size..packed_size + 4].try_into().unwrap());

            // 转成 NEON 布局（64 权重/16 字节，组 16）：packed[b*16+p] 位段 (6-2g) ← 权重 64b+16g+p
            let src = &buf[..packed_size];
            let n_blocks_64 = n_elem / QK_I2_S;
            let mut packed = vec![0u8; n_blocks_64 * 16];
            for idx in 0..n_elem {
                let code = (src[(idx / 128) * 32 + (idx % 128) % 32]
                    >> (6 - 2 * ((idx % 128) / 32)))
                    & 0b11;
                let b = idx / QK_I2_S;
                let j = idx % QK_I2_S;
                let g16 = j / 16;
                let p = j % 16;
                packed[b * 16 + p] |= code << (6 - 2 * g16);
            }
            let scales = vec![scale; n_blocks_64];
            Ok(TensorData::I2(I2Tensor::from_blocks(packed, scales)))
        }
        other => Err(ModelError::Format(format!(
            "不支持的张量类型 {other}（{name}）"
        ))),
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
        TensorData::I2(t) => crate::i2_matvec_parallel(t, x.len(), x),
        TensorData::F32(m) => {
            let k = x.len();
            m.chunks_exact(k)
                .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
                .collect()
        }
    }
}

/// q/k/v 与 gate/up 共享激活量化：量化一次，多矩阵复用
fn i2_matvec_q(w: &TensorData, y: &[i8], act_scales: &[f32], k: usize) -> Vec<f32> {
    match w {
        TensorData::I2(t) => {
            let n_rows = t.n / k;
            (0..n_rows)
                .map(|r| linear_row_q(t, r, k, y, act_scales))
                .collect()
        }
        TensorData::F32(m) => {
            let n_rows = m.len() / k;
            m.chunks_exact(k)
                .map(|row| {
                    let x: Vec<f32> = y
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| v as f32 * act_scales[i / QK_I2_S])
                        .collect();
                    row.iter().zip(&x).map(|(a, b)| a * b).sum()
                })
                .collect()
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

fn rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    head_dim: usize,
    n_head: usize,
    n_head_kv: usize,
    theta: f32,
) {
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

/// 已加载的 BitNet b1.58 模型（权重 + 分词器 + 结构参数）
pub struct Model {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rope_freq_scale: f32,
    pub rms_eps: f32,
    pub n_ctx: usize,
    /// BOS token id（GGUF 元数据，缺省 128000）
    pub bos_id: u32,
    /// EOS token id（GGUF 元数据，缺省 128001）
    pub eos_id: u32,

    pub token_embd: TensorData,
    pub final_norm: Vec<f32>,
    pub layers: Vec<Layer>,
    pub tokenizer: BpeTokenizer,
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub attn_sub_norm: Vec<f32>,
    pub ffn_sub_norm: Vec<f32>,
    pub q: TensorData,
    pub k: TensorData,
    pub v: TensorData,
    pub o: TensorData,
    pub gate: TensorData,
    pub up: TensorData,
    pub down: TensorData,
}

impl Model {
    /// 从 GGUF 文件加载模型（阻塞，2B 模型约数秒到数十秒）
    pub fn load(path: &Path) -> Result<Model, ModelError> {
        let mut g = GgufFile::open(path)?;
        let s = |g: &GgufFile, k: &str| g.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let u = |g: &GgufFile, k: &str| g.get(k).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let f = |g: &GgufFile, k: &str| g.get(k).and_then(|v| v.as_f32()).unwrap_or(1e-5);

        let arch = s(&g, "general.architecture");
        if !arch.starts_with("bitnet") && !arch.is_empty() {
            return Err(ModelError::Format(format!(
                "不是 BitNet 架构的 gguf（general.architecture = {arch}）"
            )));
        }
        let p = "bitnet-b1.58.";
        let n_layer = u(&g, &format!("{p}block_count"));
        let n_embd = u(&g, &format!("{p}embedding_length"));
        let n_head = u(&g, &format!("{p}attention.head_count"));
        let n_head_kv = u(&g, &format!("{p}attention.head_count_kv"));
        let head_dim = u(&g, &format!("{p}rope.dimension_count"));
        let head_dim = if head_dim == 0 {
            n_embd / n_head
        } else {
            head_dim
        };
        let n_ff = u(&g, &format!("{p}feed_forward_length"));
        let vocab_size = u(&g, &format!("{p}vocab_size"));
        let rope_theta = f(&g, &format!("{p}rope.freq_base"));
        let rope_freq_scale = f(&g, &format!("{p}rope.frequency_scale"));
        let rms_eps = f(&g, &format!("{p}attention.layer_norm_rms_epsilon"));
        if n_layer == 0 || n_embd == 0 {
            return Err(ModelError::Format(
                "缺少 bitnet-b1.58.* 结构元数据（block_count / embedding_length）".into(),
            ));
        }

        let token_embd = load_tensor(&mut g, "token_embd.weight")?;
        let final_norm = match load_tensor(&mut g, "output_norm.weight")? {
            TensorData::F32(v) => v,
            _ => return Err(ModelError::Format("output_norm 不是 f32".into())),
        };
        // 输出层与 embedding 共享（tied weights）

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("blk.{i}.");
            let mut norm = |name: String| -> Result<Vec<f32>, ModelError> {
                match load_tensor(&mut g, &name)? {
                    TensorData::F32(v) => Ok(v),
                    _ => Err(ModelError::Format(format!("{name} 不是 f32"))),
                }
            };
            layers.push(Layer {
                attn_norm: norm(format!("{p}attn_norm.weight"))?,
                ffn_norm: norm(format!("{p}ffn_norm.weight"))?,
                attn_sub_norm: norm(format!("{p}attn_sub_norm.weight"))?,
                ffn_sub_norm: norm(format!("{p}ffn_sub_norm.weight"))?,
                q: load_tensor(&mut g, &format!("{p}attn_q.weight"))?,
                k: load_tensor(&mut g, &format!("{p}attn_k.weight"))?,
                v: load_tensor(&mut g, &format!("{p}attn_v.weight"))?,
                o: load_tensor(&mut g, &format!("{p}attn_output.weight"))?,
                gate: load_tensor(&mut g, &format!("{p}ffn_gate.weight"))?,
                up: load_tensor(&mut g, &format!("{p}ffn_up.weight"))?,
                down: load_tensor(&mut g, &format!("{p}ffn_down.weight"))?,
            });
        }

        let tokens: Vec<(String, u32)> = g
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .enumerate()
                    .filter_map(|(i, v)| v.as_str().map(|s| (s.to_string(), i as u32)))
                    .collect()
            })
            .unwrap_or_default();
        let merges: Vec<(String, String)> = g
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .filter_map(|s| {
                        s.split_once(' ')
                            .map(|(a, b)| (a.to_string(), b.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let special: Vec<(String, u32)> = g
            .get("tokenizer.ggml.special_tokens")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .enumerate()
                    .filter_map(|(i, v)| v.as_str().map(|s| (s.to_string(), i as u32)))
                    .collect()
            })
            .unwrap_or_default();
        let bos = g
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(128000);
        let eos = g
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.as_u32())
            .unwrap_or(128001);
        let tok = BpeTokenizer::new(tokens, merges, special);

        Ok(Model {
            n_embd,
            n_layer,
            n_head,
            n_head_kv,
            head_dim,
            n_ff,
            vocab_size,
            rope_theta,
            rope_freq_scale,
            rms_eps,
            n_ctx: 4096,
            bos_id: bos,
            eos_id: eos,
            token_embd,
            final_norm,
            layers,
            tokenizer: tok,
        })
    }

    /// 文本 → token ids（自动加 BOS）
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text, true, self.bos_id)
    }

    /// token ids → 文本
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, &HashMap::new())
    }

    /// 自回归生成（阻塞）：返回生成文本与 token 统计
    ///
    /// 每次调用使用全新 KV cache（无状态，适合桥接场景的"一问一答"）。
    /// `temp <= 0` 为贪心解码（确定性，工具/Agent 采样推荐）。
    pub fn generate(
        &self,
        prompt: &str,
        max_tokens: u32,
        temp: f32,
    ) -> Result<Generation, ModelError> {
        let mut kv = KVCache::new(self);
        let ids = self.encode(prompt);
        let mut last_logits = Vec::new();
        for &t in &ids {
            last_logits = forward_token(self, &mut kv, t);
        }
        if last_logits.is_empty() {
            return Err(ModelError::Format("prompt 编码结果为空".into()));
        }

        let mut out: Vec<u32> = Vec::new();
        for _ in 0..max_tokens {
            if kv.pos >= self.n_ctx {
                break; // 上下文写满，提前收束
            }
            let next = sample_token(&last_logits, temp);
            if next == self.eos_id {
                break;
            }
            out.push(next);
            last_logits = forward_token(self, &mut kv, next);
        }
        Ok(Generation {
            text: self.decode(&out),
            prompt_tokens: ids.len(),
            gen_tokens: out.len(),
        })
    }
}

/// 一次生成的结果
pub struct Generation {
    /// 生成文本（不含 prompt，不含 EOS）
    pub text: String,
    /// prompt 编码后的 token 数
    pub prompt_tokens: usize,
    /// 实际生成的 token 数（≤ max_tokens）
    pub gen_tokens: usize,
}

pub struct KVCache {
    /// [layer][head_kv][ctx][head_dim]
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub pos: usize,
}

impl KVCache {
    pub fn new(model: &Model) -> KVCache {
        let per = model.n_head_kv * model.n_ctx * model.head_dim;
        KVCache {
            k: vec![0.0; model.n_layer * per],
            v: vec![0.0; model.n_layer * per],
            pos: 0,
        }
    }
}

/// 单 token 前向，返回 logits（推理速度慢是标量参考实现的预期行为）
pub fn forward_token(model: &Model, kv: &mut KVCache, token: u32) -> Vec<f32> {
    let n = model.n_embd;
    let mut x: Vec<f32> = match &model.token_embd {
        TensorData::F32(e) => e[(token as usize) * n..(token as usize + 1) * n].to_vec(),
        TensorData::I2(_) => panic!("token_embd I2 不支持（embedding 必须是 F32/F16/BF16）"),
    };

    for (li, layer) in model.layers.iter().enumerate() {
        // --- attention ---
        let xn = rms_norm(&x, &layer.attn_norm, model.rms_eps);
        let (xn_q, xn_scales) = quantize_i8_blocks(&xn);
        let q = i2_matvec_q(&layer.q, &xn_q, &xn_scales, model.n_embd);
        let mut k = i2_matvec_q(&layer.k, &xn_q, &xn_scales, model.n_embd);
        let v = i2_matvec_q(&layer.v, &xn_q, &xn_scales, model.n_embd);

        let mut q = q;

        rope(
            &mut q,
            &mut k,
            kv.pos,
            model.head_dim,
            model.n_head,
            model.n_head_kv,
            model.rope_theta,
        );

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
            let n_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .min(n_rows);
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
                        (start..end)
                            .map(|r| {
                                e[r * k..(r + 1) * k]
                                    .iter()
                                    .zip(xn)
                                    .map(|(a, b)| a * b)
                                    .sum()
                            })
                            .collect::<Vec<f32>>()
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
        TensorData::I2(_) => panic!("token_embd I2 不支持"),
    };
    kv.pos += 1;
    logits
}

/// 从 logits 采样下一个 token：`temp <= 0` 贪心（argmax），否则温度采样
///
/// 随机源：纳秒时钟种子的 xorshift（轻量、零依赖；与原 bin 的时钟字节等价）。
pub fn sample_token(logits: &[f32], temp: f32) -> u32 {
    if temp <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|l| ((l - max) / temp).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    let mut seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    let r = (seed % 1_000_000_000) as f32 / 1_000_000_000.0;
    let mut acc = 0.0;
    for (i, p) in probs.iter().enumerate() {
        acc += p;
        if r <= acc {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}
