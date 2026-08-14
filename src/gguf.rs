//! GGUF 文件解析器（v3，支持读取元数据 + 张量目录 + 数据）。
//! 参考 llama.cpp gguf 格式规范。

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};

pub const MAGIC: u32 = 0x4655_4747; // "GGUF"

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            GgufValue::I32(v) => Some(*v as u32),
            GgufValue::U64(v) => Some(*v as u32),
            GgufValue::I64(v) => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U64(v) => Some(*v),
            GgufValue::I64(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::I32(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

pub struct GgufFile {
    pub file: std::fs::File,
    pub kv: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offset: u64,
}

fn read_string<R: Read>(r: &mut R) -> std::io::Result<String> {
    let len = r.read_u64::<LittleEndian>()?;
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn read_value<R: Read>(r: &mut R, t: u32) -> std::io::Result<GgufValue> {
    Ok(match t {
        0 => GgufValue::U8(r.read_u8()?),
        1 => GgufValue::I8(r.read_i8()?),
        2 => GgufValue::U16(r.read_u16::<LittleEndian>()?),
        3 => GgufValue::I16(r.read_i16::<LittleEndian>()?),
        4 => GgufValue::U32(r.read_u32::<LittleEndian>()?),
        5 => GgufValue::I32(r.read_i32::<LittleEndian>()?),
        6 => GgufValue::F32(r.read_f32::<LittleEndian>()?),
        7 => GgufValue::Bool(r.read_u8()? != 0),
        8 => GgufValue::String(read_string(r)?),
        9 => {
            let elem_t = r.read_u32::<LittleEndian>()?;
            let len = r.read_u64::<LittleEndian>()?;
            let mut items = Vec::with_capacity(len as usize);
            for _ in 0..len {
                items.push(read_value(r, elem_t)?);
            }
            GgufValue::Array(items)
        }
        10 => GgufValue::U64(r.read_u64::<LittleEndian>()?),
        11 => GgufValue::I64(r.read_i64::<LittleEndian>()?),
        12 => GgufValue::F64(r.read_f64::<LittleEndian>()?),
        _ => return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("unknown gguf type {t}"))),
    })
}

impl GgufFile {
    pub fn open(path: &Path) -> std::io::Result<GgufFile> {
        let mut f = std::fs::File::open(path)?;
        let magic = f.read_u32::<LittleEndian>()?;
        if magic != MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not a GGUF file"));
        }
        let _version = f.read_u32::<LittleEndian>()?;
        let n_tensors = f.read_u64::<LittleEndian>()?;
        let n_kv = f.read_u64::<LittleEndian>()?;

        let mut kv = HashMap::new();
        for _ in 0..n_kv {
            let key = read_string(&mut f)?;
            let t = f.read_u32::<LittleEndian>()?;
            let v = read_value(&mut f, t)?;
            kv.insert(key, v);
        }

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_string(&mut f)?;
            let n_dims = f.read_u32::<LittleEndian>()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(f.read_u64::<LittleEndian>()?);
            }
            let ggml_type = f.read_u32::<LittleEndian>()?;
            let offset = f.read_u64::<LittleEndian>()?;
            tensors.push(GgufTensorInfo { name, n_dims, dims, ggml_type, offset });
        }
        let pos = f.stream_position()?;
        // ggml 张量数据区 32 字节对齐（header 尾部有 padding）
        let data_offset = (pos + 31) / 32 * 32;
        Ok(GgufFile { file: f, kv, tensors, data_offset })
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.kv.get(key)
    }

    /// 张量数据读取（I2_S / F32 / BF16 / F16 / Q4_0 等按需）
    pub fn read_tensor_data(&mut self, t: &GgufTensorInfo, buf: &mut [u8]) -> std::io::Result<()> {
        self.file.seek(SeekFrom::Start(self.data_offset + t.offset))?;
        self.file.read_exact(buf)
    }

    pub fn tensor_data_len(&self, t: &GgufTensorInfo) -> u64 {
        let ne: u64 = t.dims.iter().product();
        match t.ggml_type {
            // 与 llama.cpp ggml_type_size 一致
            0 | 1 | 2 => ne * 4,                 // F32 / F16 / BF16（size 4/2/2 分开处理）
            _ => {
                // QK=64 的量化类型按块大小计算
                let (blk, per_blk) = type_block_size(t.ggml_type);
                ne / per_blk * blk
            }
        }
    }

    pub fn n_elements(&self, t: &GgufTensorInfo) -> u64 {
        t.dims.iter().product()
    }
}

/// ggml_type 的块大小：返回 (block 字节数, 每块元素数)
pub fn type_block_size(t: u32) -> (u64, u64) {
    match t {
        0 => (4, 1),                              // F32
        1 => (2, 1),                              // F16
        2 => (2, 1),                              // BF16
        7 => (4 + 1, 4),                          // Q4_0 (block_q4_0)
        8 => (4 + 2, 8),                          // Q4_1
        32 => (4 + 1, 4),                         // IQ4_NL? 忽略
        34 => (4 + 1, 4),                         // Q4_K? 忽略
        40 => (8, 1),                             // I8
        41 => (4, 1),                             // I16
        42 => (8, 1),                             // I32
        43 => (4, 1),                             // I64
        // 自定义: BitNet I2_S —— llama.cpp bitnet 分支: QK=64? 这里按 bitnet.cpp 的 I2_S 布局:
        // 每 64 权重 = 16 字节 + 4 字节 scale
        99 => (20, 64),                           // 自定义 I2_S（bitnet）占位，需按实际映射
        _ => (0, 1),
    }
}
