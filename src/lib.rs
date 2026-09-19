//! bitnet-rs — BitNet b1.58 推理核心（从 microsoft/BitNet bitnet.cpp 移植）
//!
//! 移植自 `src/ggml-bitnet-mad.cpp`，语义与 NEON 路径逐位一致：
//! - `quantize_i2_s`: f32 权重 → 1.58-bit 三值 {-1,0,+1} 打包
//!   （QK_I2_S=64：每 64 权重 = 16 字节，每 16 权重一组 2-bit 位段 + 1 个 f32 scale）
//! - `vec_dot_i2_i8_s_1x1`: 打包权重 × i8 激活的整数点积（i16 中间累加 wrap 语义）
//!
//! 参考实现（标量，可读优先）。no_std(alloc) 兼容。

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(clippy::needless_range_loop)]
#![allow(unused_imports)]

extern crate alloc;

use alloc::vec::Vec;

/// NEON 路径的块大小（QK_I2_S = 64）
pub const QK_I2_S: usize = 64;
/// 每组的权重数（打包位段宽度 2-bit）
pub const GROUP_SIZE: usize = 16;
/// C++ 中判定为 0 的阈值
const ZERO_EPS: f64 = 1e-6;

/// 打包后的三值权重（Q2_I2 格式，NEON 布局）
pub struct I2Tensor {
    /// 每 64 权重打包为 16 字节：权重 j → byte[j % 16] 的位段 (6 - 2*(j/16))
    pub packed: Vec<u8>,
    /// 每块（64 权重）一个 scale
    pub scales: Vec<f32>,
    /// 权重总数
    pub n: usize,
}

impl I2Tensor {
    /// 从 GGUF 块布局构造（每 64 权重 = 16 字节 + 4 字节 f32 scale）
    pub fn from_blocks(packed: Vec<u8>, scales: Vec<f32>) -> I2Tensor {
        let n = packed.len() * 4;
        I2Tensor { packed, scales, n }
    }

    /// 块 scale（权重 idx 所在块）
    #[inline]
    pub fn scale_of(&self, idx: usize) -> f32 {
        self.scales[idx / QK_I2_S]
    }
}

impl I2Tensor {
    /// 第 idx 个权重的三值码 {-1, 0, +1}
    #[inline]
    pub fn get_code(&self, idx: usize) -> i8 {
        debug_assert!(idx < self.n);
        let b = idx / QK_I2_S;
        let j = idx % QK_I2_S;
        let g = j / GROUP_SIZE;
        let p = j % GROUP_SIZE;
        let q = (self.packed[b * (QK_I2_S / 4) + p] >> (6 - 2 * g)) & 0b11;
        (q as i8) - 1
    }

    /// 解包回 f32（乘所在块 scale），用于验证
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = alloc::vec![0f32; self.n];
        for idx in 0..self.n {
            out[idx] = self.get_code(idx) as f32 * self.scale_of(idx);
        }
        out
    }
}

/// 量化：f32 → Q2_I2 三值打包（与 C++ NEON 分支逐位一致）
pub fn quantize_i2_s(src: &[f32]) -> I2Tensor {
    let n = src.len();
    debug_assert!(n % QK_I2_S == 0, "n 必须是 64 的倍数");

    // scale = max |x|
    let mut max = 0f64;
    for &x in src {
        max = max.max(x.abs() as f64);
    }
    let scale = max as f32;

    // f32 → 码 {0,1,2}（语义同 C++：1e-6 阈值）
    let mut q8: Vec<u8> = alloc::vec![0u8; n];
    for (i, &x) in src.iter().enumerate() {
        q8[i] = if (x.abs() as f64) < ZERO_EPS {
            1
        } else if (x as f64) * max > 0.0 {
            2
        } else {
            0
        };
    }

    // 打包：每 64 权重 16 字节，j 的位段 = 6 - 2*(j/16)
    let mut packed = alloc::vec![0u8; n / 4];
    for b in 0..n / QK_I2_S {
        for j in 0..QK_I2_S {
            let g = j / GROUP_SIZE;
            let p = j % GROUP_SIZE;
            packed[b * (QK_I2_S / 4) + p] |= q8[b * QK_I2_S + j] << (6 - 2 * g);
        }
    }

    I2Tensor {
        packed,
        scales: alloc::vec![scale; n / QK_I2_S],
        n,
    }
}

/// 激活量化：f32 → i8（对称，scale = max/127）
pub fn quantize_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let mut max = 0f32;
    for &v in x {
        max = max.max(v.abs());
    }
    if max == 0.0 {
        return (alloc::vec![0i8; x.len()], 1.0);
    }
    let scale = max / 127.0;
    let q = x.iter().map(|&v| round_f32(v / scale) as i8).collect();
    (q, scale)
}

/// 激活量化（per-block，bitnet.cpp Q8_0 语义）：每 QK_I2_S=64 元素一块，
/// 每块独立 scale = max/127，减少长向量上的量化误差。
pub fn quantize_i8_blocks(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n = x.len();
    let n_blocks = n.div_ceil(QK_I2_S);
    let mut q = alloc::vec![0i8; n];
    let mut scales = alloc::vec![1f32; n_blocks];
    for b in 0..n_blocks {
        let start = b * QK_I2_S;
        let end = core::cmp::min(start + QK_I2_S, n);
        let mut max = 0f32;
        for &v in &x[start..end] {
            max = max.max(v.abs());
        }
        if max == 0.0 {
            continue;
        }
        let scale = max / 127.0;
        scales[b] = scale;
        for (i, &v) in x[start..end].iter().enumerate() {
            q[start + i] = round_f32(v / scale) as i8;
        }
    }
    (q, scales)
}

/// no_std 可用的四舍五入
#[inline]
fn round_f32(v: f32) -> f32 {
    let f = libm::floorf(v);
    if v - f >= 0.5 { f + 1.0 } else { f }
}

/// 整数点积：打包权重 × i8 激活（与 C++ `ggml_vec_dot_i2_i8_s_1x1` NEON 分支一致）
///
/// 语义：Σ_j code(j) * y(j)，code ∈ {0,1,2}；中间累加按 NEON vmlal 的
/// i16 wrap 分 lane 模拟，任何输入下输出都与 C++ 逐位一致。
pub fn vec_dot_i2_i8_s_1x1(
    n: usize,
    x_row: &[u8], // 打包权重行（bx 步长语义：行内连续）
    y: &[i8],     // 激活
    nrc: usize,
) -> Vec<f32> {
    let nb = n / QK_I2_S;
    let group32_num = nb / 32;
    let la_num = nb % 32;
    let mut out = alloc::vec![0f32; nrc];

    for row in 0..nrc {
        // 行偏移：每行 bx/4 字节（对拍中 bx = n/4，行连续）
        let x_off = row * (n / 4);
        let mut accu: [i32; 4] = [0; 4];

        // 完整 32 块组（每块 2048 权重）
        for i in 0..group32_num {
            let mut accu32: [i16; 8] = [0; 8];
            for j in 0..32 {
                let xb = &x_row[x_off + i * 512 + j * 16..x_off + i * 512 + j * 16 + 16];
                for g in 0..4 {
                    let y_base = i * 2048 + j * 64 + g * 16;
                    for t in 0..16 {
                        let code = (xb[t] >> (6 - 2 * g)) & 0b11;
                        // vmlal 的 lane 划分：low 半（t<8）→ lane 0..3，high 半 → lane 4..7
                        let lane = t % 8;
                        accu32[lane] =
                            accu32[lane].wrapping_add(code as i16 * y[y_base + t] as i16);
                    }
                }
            }
            for l in 0..4 {
                accu[l] += accu32[l] as i32;
                accu[l] += accu32[4 + l] as i32;
            }
        }

        // 尾部不足 32 块
        if la_num > 0 {
            let mut accula: [i16; 8] = [0; 8];
            for j in 0..la_num {
                let xb = &x_row
                    [x_off + group32_num * 512 + j * 16..x_off + group32_num * 512 + j * 16 + 16];
                for g in 0..4 {
                    let y_base = group32_num * 2048 + j * 64 + g * 16;
                    for t in 0..16 {
                        let code = (xb[t] >> (6 - 2 * g)) & 0b11;
                        let lane = t % 8;
                        accula[lane] =
                            accula[lane].wrapping_add(code as i16 * y[y_base + t] as i16);
                    }
                }
            }
            for l in 0..4 {
                accu[l] += accula[l] as i32;
                accu[l] += accula[4 + l] as i32;
            }
        }

        out[row] = (accu[0] + accu[1] + accu[2] + accu[3]) as f32;
    }
    out
}

/// 单块（64 权重）整数点积，返回 i32（NEON dotprod 或标量）
#[cfg(target_arch = "aarch64")]
#[inline]
fn dot64(p: &[u8], y: &[i8]) -> i32 {
    unsafe {
        use std::arch::aarch64::*;
        let xb = vld1q_u8(p.as_ptr());
        let mask = vdupq_n_u8(3);
        let mut acc16 = vdupq_n_s16(0);
        for g in 0..4 {
            let code = match g {
                0 => vshrq_n_u8(xb, 6),
                1 => vshrq_n_u8(xb, 4),
                2 => vshrq_n_u8(xb, 2),
                _ => xb,
            };
            let code = vandq_u8(code, mask);
            let q = vreinterpretq_s8_u8(code);
            let yv = vld1q_s8(y.as_ptr().add(g * 16));
            acc16 = vmlal_s8(acc16, vget_low_s8(q), vget_low_s8(yv));
            acc16 = vmlal_s8(acc16, vget_high_s8(q), vget_high_s8(yv));
        }
        let lo = vmovl_s16(vget_low_s16(acc16));
        let hi = vmovl_s16(vget_high_s16(acc16));
        vaddvq_s32(vaddq_s32(lo, hi))
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn dot64(p: &[u8], y: &[i8]) -> i32 {
    let mut acc = 0i32;
    for j in 0..64 {
        let g = j / 16;
        let code = (p[j % 16] >> (6 - 2 * g)) & 0b11;
        acc += code as i32 * y[j] as i32;
    }
    acc
}

/// 单块（64 权重）整数点积 × 块 scale
#[inline]
fn dot_block(w: &I2Tensor, start: usize, y: &[i8]) -> f32 {
    let b = start / QK_I2_S;
    let p = &w.packed[b * (QK_I2_S / 4)..(b + 1) * (QK_I2_S / 4)];
    dot64(p, y) as f32 * w.scales[b]
}

/// 线性层（单行权重，k == w.n）：out = Σ_blocks dot_block * x_scale
pub fn linear(w: &I2Tensor, x: &[f32]) -> f32 {
    debug_assert_eq!(w.n, x.len());
    let (y, x_scale) = quantize_i8(x);
    let mut acc = 0f32;
    for start in (0..w.n).step_by(QK_I2_S) {
        acc += dot_block(w, start, &y[start..start + QK_I2_S]);
    }
    acc * x_scale
}

/// 单行权重 × 输入向量（k 维），激活已量化
///
/// 权重码 {0,1,2} 映射真实三值 {−1,0,+1} = 码 − 1，
/// 因此每块点积需补偿 −Σy（C++ 在 mul_mat 集成层做，对拍只覆盖 vec_dot 原始输出）。
#[inline]
pub fn linear_row_q(w: &I2Tensor, row: usize, k: usize, y: &[i8], act_scales: &[f32]) -> f32 {
    debug_assert_eq!(w.n % k, 0);
    let row_start = row * k;
    let mut acc = 0f32;
    for start in (0..k).step_by(QK_I2_S) {
        let abs = row_start + start;
        let b = abs / QK_I2_S;
        let p = &w.packed[b * (QK_I2_S / 4)..(b + 1) * (QK_I2_S / 4)];
        let yb = &y[start..start + QK_I2_S];
        let ysum: i32 = yb.iter().map(|&v| v as i32).sum();
        // 权重码 {0,1,2} → 真实 {-1,0,+1} = 码 − 1，补偿 −Σy
        acc += (dot64(p, yb) - ysum) as f32 * w.scales[b] * act_scales[start / QK_I2_S];
    }
    acc
}

/// 多行权重矩阵的第 row 行 × 输入向量（k 维）
pub fn linear_row(w: &I2Tensor, row: usize, k: usize, x: &[f32]) -> f32 {
    let (y, act_scales) = quantize_i8_blocks(x);
    linear_row_q(w, row, k, &y, &act_scales)
}

/// 并行 matvec：I2 权重矩阵（n_rows×k）× 输入 → n_rows 输出
/// 每行输出独立，按可用核数分块并行（每线程独立 Vec，避免共享写）。
pub fn i2_matvec_parallel(w: &I2Tensor, k: usize, x: &[f32]) -> Vec<f32> {
    let n_rows = w.n / k;
    let (y, act_scales) = quantize_i8_blocks(x);
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
            let (w, y, act_scales) = (w, &y[..], &act_scales[..]);
            handles.push(s.spawn(move || {
                (start..end)
                    .map(|r| linear_row_q(w, r, k, y, act_scales))
                    .collect::<Vec<f32>>()
            }));
        }
        let mut out = alloc::vec![0f32; n_rows];
        for (t_id, h) in handles.into_iter().enumerate() {
            let part = h.join().unwrap();
            out[t_id * chunk..t_id * chunk + part.len()].copy_from_slice(&part);
        }
        out
    })
}

/// 行切片视图：取出第 row 行的打包字节（bx/4 步长语义）
pub fn row_bytes(w: &I2Tensor, row: usize, k: usize) -> &[u8] {
    debug_assert_eq!(w.n % k, 0);
    let row_len = k / 4;
    &w.packed[row * row_len..(row + 1) * row_len]
}
#[cfg(feature = "std")]
pub mod gguf;
#[cfg(feature = "std")]
pub mod model;
#[cfg(feature = "std")]
pub mod tokenizer;
