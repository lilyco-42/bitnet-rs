//! 对拍工具：与 reference/bitnet_ref.cpp 输出逐字节一致。
//!
//! 用法: compare <n_weights> <n_activ> <input.bin> <output.bin>
//! input.bin 布局: n_weights 个 f32 权重 + n_activ 个 f32 激活
//! output.bin 布局: n_weights/4 字节打包 + f32 scale + f32 y_scale + f32 dot

use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::process::exit;

use bitnet_rs::{quantize_i2_s, vec_dot_i2_i8_s_1x1};

fn read_f32s(f: &mut File, n: usize) -> Vec<f32> {
    let mut buf = vec![0u8; n * 4];
    f.read_exact(&mut buf).unwrap();
    buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!("usage: {} <n_weights> <n_activ> <input.bin> <output.bin>", args[0]);
        exit(1);
    }
    let n_weights: usize = args[1].parse().unwrap();
    let n_activ: usize = args[2].parse().unwrap();

    let mut fin = File::open(&args[3]).unwrap();
    let w = read_f32s(&mut fin, n_weights);
    let a = read_f32s(&mut fin, n_activ);

    // 量化（与 C++ 参考一致：round-half-to-even）
    let t = quantize_i2_s(&w);

    // 激活 f32 -> i8（lrintf = round to nearest even）
    let y_scale = a.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let y_scale = if y_scale == 0.0 { 1.0 } else { y_scale };
    let inv = 127.0f32 / y_scale;
    let y: Vec<i8> = a.iter().map(|&v| libm::rintf(v * inv) as i8).collect();

    // 点积（1x1，nrc=1）
    let dots = vec_dot_i2_i8_s_1x1(n_weights, &t.packed, &y, 1);

    // 写输出（同 C++ 格式）
    let mut out = Vec::with_capacity(n_weights / 4 + 12);
    out.extend_from_slice(&t.packed);
    out.extend_from_slice(&t.scales[0].to_le_bytes());
    out.extend_from_slice(&y_scale.to_le_bytes());
    out.extend_from_slice(&dots[0].to_le_bytes());
    File::create(&args[4]).unwrap().write_all(&out).unwrap();
}
