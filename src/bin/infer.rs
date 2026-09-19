//! BitNet b1.58-2B GGUF 推理（标量参考实现）。
//!
//! 模型加载 / 前向 / 采样已提升进 lib（`bitnet_rs::model`），本 bin 只保留
//! CLI 外壳：单轮补全 / 交互聊天 / DBG_LOGITS 调试输出。

use std::path::Path;

use bitnet_rs::model::{KVCache, Model, forward_token, sample_token};

/// 交互聊天：持续会话（KV cache 跨轮保留，支持长上下文）
/// chat 模板: Human: {input}\n\nBITNETAssistant: {eos}
fn chat_loop(model: &Model) {
    use std::io::{BufRead, Write};
    let bos = model.bos_id;
    let eos = model.eos_id;
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
        let prompt = format!("Human: {input}\n\nBITNETAssistant: ");
        let ids = model.tokenizer.encode(&prompt, false, bos);
        let t0 = std::time::Instant::now();
        for &t in &ids {
            forward_token(model, &mut kv, t);
        }
        history.extend_from_slice(&ids);

        // 生成直到 eos 或上限
        let mut out_tokens = Vec::new();
        let max_gen = std::env::var("MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(120);
        let mut last = *history.last().unwrap();
        let mut done = false;
        for _ in 0..max_gen {
            if kv.pos >= model.n_ctx {
                println!("\n[context full: {} tokens]", kv.pos);
                done = true;
                break;
            }
            let logits = forward_token(model, &mut kv, last);
            let next = sample_token(&logits, 0.6);
            if next == eos {
                done = true;
                break;
            }
            out_tokens.push(next);
            last = next;
        }
        let _ = done;
        let dt = t0.elapsed().as_secs_f32();
        let reply = model.decode(&out_tokens);
        println!(
            "\nBitnet: {reply}  ({dt:.1}s, {} tokens, ctx {})",
            out_tokens.len(),
            kv.pos
        );
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
            "-t" => {
                temp = args[i + 1].parse().unwrap();
                i += 2;
            }
            "-n" => {
                n_tokens = args[i + 1].parse().unwrap();
                i += 2;
            }
            s => {
                prompt = s.to_string();
                i += 1;
            }
        }
    }

    let start = std::time::Instant::now();
    let model = Model::load(Path::new(path)).unwrap_or_else(|e| {
        eprintln!("模型加载失败: {e}");
        std::process::exit(1);
    });
    println!(
        "arch loaded: layers={} embd={} head={} kv={} hd={} ff={} vocab={} theta={} eps={}",
        model.n_layer,
        model.n_embd,
        model.n_head,
        model.n_head_kv,
        model.head_dim,
        model.n_ff,
        model.vocab_size,
        model.rope_theta,
        model.rms_eps
    );
    println!("model loaded in {:.1}s", start.elapsed().as_secs_f32());

    if std::env::var("CHAT").is_ok() || args.iter().any(|a| a == "--chat") {
        chat_loop(&model);
        return;
    }

    let ids = model.encode(&prompt);
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
            let dec = model.decode(&[*id as u32]);
            eprintln!("  {i}. {id}: {v:.4} '{dec}'");
        }
        eprintln!(
            "logits: min={:.2} max={:.2} nan={}",
            logits.iter().cloned().fold(f32::INFINITY, f32::min),
            logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            logits.iter().filter(|x| x.is_nan()).count()
        );
        return;
    }
    let mut out = prompt.clone();
    for i in 0..n_tokens {
        let t0 = std::time::Instant::now();
        let logits = forward_token(&model, &mut kv, *all.last().unwrap());
        let next = sample_token(&logits, temp);
        let dt = t0.elapsed().as_secs_f32();
        all.push(next);
        out.push(' ');
        out.push_str(&model.decode(&[next]));
        print!("token {i}: {dt:.1}s  {}", model.decode(&[next]));
        println!();
        if next == model.eos_id {
            break;
        }
    }
    println!("\n=== OUTPUT ===\n{out}");
}
