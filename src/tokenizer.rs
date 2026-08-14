//! GPT-2 风格 byte-level BPE tokenizer（从 GGUF 元数据加载）。
//! 与 llama.cpp 的 gpt2 tokenizer 行为一致。

use std::collections::HashMap;

/// byte → unicode 字符映射（llama.cpp 的 gpt2 字节表）
///
/// 与标准 GPT2 表的关键差异：bs 跳过 0xAD（软连字符），它被映射到
/// 偏移区 U+0143；若用标准表，中文（UTF-8 常含 0xAD 续字节）会编码失败。
pub fn byte_to_unicode() -> Vec<(u8, char)> {
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();
    bs.extend(0xA1u8..=0xAC); // 0xA1-0xAC（跳过 0xAD）
    bs.extend(0xAEu8..=0xFF); // 0xAE-0xFF
    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
    let mut n = 0;
    for b in 0u16..=255 {
        if !bs.contains(&(b as u8)) {
            bs.push(b as u8);
            cs.push(char::from_u32(0x100 + n).unwrap());
            n += 1;
        }
    }
    bs.into_iter().zip(cs).collect()
}

pub struct BpeTokenizer {
    encoder: HashMap<String, u32>,
    bpe_ranks: HashMap<(String, String), u32>,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    special_tokens: HashMap<String, u32>,
    vocab_size: u32,
}

fn bytes_to_unicode_map() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut enc = HashMap::new();
    let mut dec = HashMap::new();
    for (b, c) in byte_to_unicode() {
        enc.insert(b, c);
        dec.insert(c, b);
    }
    (enc, dec)
}

impl BpeTokenizer {
    pub fn new(
        tokens: Vec<(String, u32)>,
        merges: Vec<(String, String)>,
        special: Vec<(String, u32)>,
    ) -> BpeTokenizer {
        let mut encoder = HashMap::new();
        for (t, id) in &tokens {
            encoder.insert(t.clone(), *id);
        }
        let mut bpe_ranks = HashMap::new();
        for (i, (a, b)) in merges.iter().enumerate() {
            bpe_ranks.insert((a.clone(), b.clone()), i as u32);
        }
        let special_tokens: HashMap<String, u32> = special.into_iter().collect();
        let vocab_size = tokens.len() as u32 + special_tokens.len() as u32;
        let (byte_encoder, byte_decoder) = bytes_to_unicode_map();
        BpeTokenizer {
            encoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            special_tokens,
            vocab_size,
        }
    }

    fn get_pairs(word: &[String]) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        for i in 0..word.len() - 1 {
            pairs.push((word[i].clone(), word[i + 1].clone()));
        }
        pairs
    }

    fn bpe(&self, token: &str) -> String {
        let chars: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if chars.len() < 2 {
            return token.to_string();
        }
        let mut word = chars;
        loop {
            let pairs = Self::get_pairs(&word);
            let mut best: Option<((String, String), u32)> = None;
            for p in &pairs {
                if let Some(rank) = self.bpe_ranks.get(p) {
                    if best.as_ref().map(|(_, r)| *rank < *r).unwrap_or(true) {
                        best = Some((p.clone(), *rank));
                    }
                }
            }
            let Some(((a, b), _)) = best else {
                break;
            };
            let mut new_word: Vec<String> = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i < word.len() - 1 && word[i] == a && word[i + 1] == b {
                    new_word.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            if new_word == word {
                break;
            }
            word = new_word;
        }
        word.join(" ")
    }

    /// 文本 → token ids（ByteLevel：整句字节映射 + BPE，空格参与编码）
    pub fn encode(&self, text: &str, add_bos: bool, bos_id: u32) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos {
            out.push(bos_id);
        }
        // 特殊 token 优先（整句替换为特殊 id）
        let bytes: Vec<u8> = text.as_bytes().to_vec();
        let unicode: String = bytes.iter().map(|b| self.byte_encoder[b]).collect();
        let bpe_str = self.bpe(&unicode);
        for piece in bpe_str.split(' ') {
            if let Some(&id) = self.encoder.get(piece) {
                out.push(id);
                continue;
            }
            // 未登录：按单字节 token 兜底（字节映射字符的单字符 token）
            for c in piece.chars() {
                if let Some(&id) = self.encoder.get(&c.to_string()) {
                    out.push(id);
                }
            }
        }
        out
    }

    /// token ids → 文本（UTF-8 字节正确拼接，支持中文等多字节字符）
    pub fn decode(&self, ids: &[u32], special: &HashMap<u32, String>) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(s) = special.get(&id) {
                // 特殊 token 追加原文本
                for b in s.bytes() {
                    bytes.push(b);
                }
                continue;
            }
            let mut tok = String::new();
            for (t, tid) in &self.encoder {
                if *tid == id {
                    tok = t.clone();
                    break;
                }
            }
            if tok.is_empty() {
                continue;
            }
            for c in tok.chars() {
                if let Some(&b) = self.byte_decoder.get(&c) {
                    bytes.push(b);
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
