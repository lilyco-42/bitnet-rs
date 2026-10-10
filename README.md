# bitnet-rs

纯 Rust 实现的 BitNet b1.58 推理引擎 —— 从 [microsoft/BitNet (bitnet.cpp)](https://github.com/microsoft/BitNet) 移植。

在手机上跑 2B 参数模型：**~0.5s/token**（NEON SIMD + 多线程）。

## 特性

- **完整移植** `ggml-bitnet-mad.cpp`：`quantize_i2_s`（Q2_I2 三值打包）+ `vec_dot_i2_i8_s` 整数点积，与 C++ **逐字节对拍一致**（含 i16 wrap 语义）
- **GGUF 解析**：官方 `microsoft/bitnet-b1.58-2B-4T-gguf` 直接加载（I2_S 类型）
- **Tokenizer**：llama3 BPE（128K vocab / 280K merges），**完整中文支持**（llama.cpp 字节表 + ByteLevel）
- **推理**：SubLN 结构（attn_sub_norm / ffn_sub_norm）、relu2 激活、GQA、per-block 激活量化、Σy 补偿、tied embedding
- **性能**：NEON `vmlal` 点积 + 8 线程并行 matvec + 激活量化缓存
- **聊天模式**：持续会话（KV cache 跨轮保留，支持 4096 上下文）

## 快速开始

```bash
# 下载模型（官方 GGUF，1.1GB）
curl -L -o models/ggml-model-i2_s.gguf \
  "https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf"

cargo build --release

# 单次生成
./target/release/infer models/ggml-model-i2_s.gguf "Once upon a time" -n 40 -t 0.7

# 交互聊天（Ctrl+D 退出）
CHAT=1 ./target/release/infer models/ggml-model-i2_s.gguf --chat
```

## 与 C++ 对拍

```bash
g++ -O2 -o reference/bitnet_ref reference/bitnet_ref.cpp   # C++ 参考（原版 NEON 语义）
./target/release/compare <n> <n> input.bin out_rs.bin      # Rust 输出
# 与 reference 输出逐字节 diff == 0
```

覆盖：打包字节、scale、整数点积（常规 + 尾部块 + i16 wrap 极端情况）。

## 作为库使用（lib API）

模型加载 / KV cache / 前向 / 采样已提升进 lib（`bitnet_rs::model`），下游可
直接依赖本 crate 做本地推理，无需起子进程：

```rust,ignore
use bitnet_rs::model::Model;

let model = Model::load(std::path::Path::new("models/ggml-model-i2_s.gguf"))?;
let gen = model.generate("你好，介绍一下你自己", 256, 0.0)?; // temp=0 贪心解码
println!("{}", gen.text);           // 生成文本
println!("{} tok", gen.gen_tokens); // 实际生成 token 数（≤ max_tokens）
```

低层 API（聊天等需要跨轮保留 KV cache 的场景）：

- `Model::encode / decode` — BPE 分词（自动加 BOS）
- `KVCache::new(&model)` + `forward_token(&model, &mut kv, token)` — 单 token 前向
- `sample_token(&logits, temp)` — 贪心 / 温度采样

## 架构

```
src/
├── lib.rs        # quantize_i2_s / vec_dot（NEON+标量）/ 并行 matvec
├── model.rs      # Model 加载 / KVCache / forward_token / generate（lib 推理 API）
├── gguf.rs       # GGUF v3 解析（32 字节数据区对齐）
├── tokenizer.rs  # llama3 BPE（llama.cpp 字节表）
└── bin/
    ├── infer.rs  # 推理 + 聊天（lib API 的 CLI 外壳）
    ├── compare.rs # 对拍工具
    └── dump.rs   # GGUF 元数据查看
```

## 踩过的坑

1. GGUF 数据区 32 字节对齐（header 尾部 padding 18 字节）
2. I2_S 权重 = 纯 2bit 打包（QK=128/组 32），scale 为张量级（张量尾 4 字节）
3. 权重码 {0,1,2} → 真实三值 {−1,0,+1} 需要 **Σy 补偿**
4. 激活量化必须 **per-block**（全局量化导致每层 1.35x 放大）
5. 激活是 **relu2**（平方 ReLU）不是 silu
6. 字节表跳过 0xAD（llama.cpp 版，否则中文编码失败）

## 性能

| 优化前 | 优化后 |
|--------|--------|
| 19s/token（纯标量） | **0.5s/token**（NEON + 8 线程） |

## License

MIT
