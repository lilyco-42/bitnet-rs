//! GGUF 元数据 dump：架构参数 + 张量类型分布。

use std::path::Path;

use bitnet_rs::gguf::{GgufFile, type_block_size};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: dump <model.gguf>");
        std::process::exit(1);
    });
    let mut f = GgufFile::open(Path::new(&path)).unwrap();
    println!("== 元数据 ==");
    let mut keys: Vec<_> = f.kv.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let v = &f.kv[&k];
        let s = match v {
            bitnet_rs::gguf::GgufValue::String(s) => format!("string: {s}"),
            bitnet_rs::gguf::GgufValue::Bool(b) => format!("bool: {b}"),
            bitnet_rs::gguf::GgufValue::U32(x) => format!("u32: {x}"),
            bitnet_rs::gguf::GgufValue::I32(x) => format!("i32: {x}"),
            bitnet_rs::gguf::GgufValue::U64(x) => format!("u64: {x}"),
            bitnet_rs::gguf::GgufValue::I64(x) => format!("i64: {x}"),
            bitnet_rs::gguf::GgufValue::F32(x) => format!("f32: {x}"),
            bitnet_rs::gguf::GgufValue::F64(x) => format!("f64: {x}"),
            bitnet_rs::gguf::GgufValue::Array(a) if !a.is_empty() => format!("array[{}]: {:?}..", a.len(), a[0]),
            other => format!("{:?}", other),
        };
        println!("  {k}: {s}");
    }

    println!("\n== 张量 ({} 个) ==", f.tensors.len());
    let mut type_counts: std::collections::HashMap<u32, (usize, u64)> = std::collections::HashMap::new();
    for t in &f.tensors {
        let e = type_counts.entry(t.ggml_type).or_insert((0, 0));
        e.0 += 1;
        e.1 += f.tensor_data_len(t);
    }
    for (t, (count, bytes)) in type_counts {
        let (blk, per) = type_block_size(t);
        println!("  ggml_type {t}: {count} 个张量, {:.2} GB, blk={blk}B/{per}el",
                 bytes as f64 / 1e9);
    }
    println!("\n张量名样本:");
    for t in f.tensors.iter().take(16) {
        println!("  {:<44} dims={:?} type={}", t.name, t.dims, t.ggml_type);
    }
    let q = f.tensors.iter().find(|t| t.name.contains("attn_q")).unwrap();
    println!("\nattn_q: dims={:?} type={} ne={}", q.dims, q.ggml_type, q.dims.iter().product::<u64>());
    println!("\n前 12 个张量:");
    for t in f.tensors.iter().take(12) {
        println!("  {:<40} dims={:?} type={} off={}", t.name, t.dims, t.ggml_type, t.offset);
    }
    let _ = &mut f;
}
