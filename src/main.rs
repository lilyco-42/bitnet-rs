//! demo：随机 MLP 权重 → 1.58-bit 量化 → 整数推理 + 与 C++ 参考实现的语义验证。

use bitnet_rs::{quantize_i2_s, vec_dot_i2_i8_s_1x1, quantize_i8, linear_row, row_bytes, QK_I2_S};

fn main() {
    // 一个 MLP：in=64 → hidden=128 → out=16（64 的倍数，满足 QK_I2_S 对齐）
    let (k, n_hidden, out_size) = (64usize, 128usize, 16usize);

    let mut rng = 42u64;
    let mut rand = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        ((rng as f64) / u64::MAX as f64 - 0.5) as f32 * 2.0
    };

    let w1: Vec<f32> = (0..n_hidden * k).map(|_| rand()).collect();
    let w2: Vec<f32> = (0..out_size * n_hidden).map(|_| rand()).collect();

    let q1 = quantize_i2_s(&w1);
    let q2 = quantize_i2_s(&w2);
    println!(
        "w1: {} 权重 -> {} 字节打包 + scale={:.4} (块大小 {})",
        q1.n, q1.packed.len(), q1.scales[0], QK_I2_S
    );

    let x: Vec<f32> = (0..k).map(|_| rand()).collect();

    // 全精度参考
    let ref1: Vec<f32> = w1
        .chunks_exact(k)
        .map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum())
        .collect();
    let ref2: Vec<f32> = w2
        .chunks_exact(n_hidden)
        .map(|row| row.iter().zip(&ref1).map(|(a, b)| a * b).sum())
        .collect();

    // 1.58-bit 整数推理（1×1 布局）
    let h: Vec<f32> = (0..n_hidden).map(|i| linear_row(&q1, i, k, &x)).collect();
    let y: Vec<f32> = (0..out_size).map(|i| linear_row(&q2, i, n_hidden, &h)).collect();

    // 整数路径 vs 解包权重 fp32 路径（应仅差激活量化误差）
    let dq1 = q1.dequantize();
    let ref_q1: Vec<f32> = dq1
        .chunks_exact(k)
        .map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum())
        .collect();
    let (xq, xscale) = quantize_i8(&x);
    let dot_q = vec_dot_i2_i8_s_1x1(k, row_bytes(&q1, 0, k), &xq, 1)[0];
    println!("整数点积(dot={dot_q}) vs 解包 fp32 点积(ref={:.4})，scale 后: {:.4} vs {:.4}",
             ref_q1[0], dot_q as f32 * q1.scales[0] * xscale, ref_q1[0]);

    println!("\n全精度 vs 1.58-bit 输出：");
    for i in 0..out_size.min(6) {
        let err = ((y[i] - ref2[i]).abs() / (ref2[i].abs() + 1e-6)) * 100.0;
        println!("out[{i}]: bitnet={:+.4}  fp32={:+.4}  相对误差 {err:5.1}%", y[i], ref2[i]);
    }
}
