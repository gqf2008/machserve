//! Safetensors weight loading.
//!
//! Implements the safetensors binary format (8-byte little-endian header
//! length, JSON header, raw tensor data) and maps Llama/Qwen-style tensor
//! names onto the slice [`Weights`] layout. F32/F16/BF16 tensors are loaded
//! and converted to f32.

use crate::{Config, Error, LayerWeights, Weights};
use std::collections::HashMap;
use std::path::Path;

/// A decoded tensor from a safetensors file.
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    /// Data offsets relative to the start of the data section.
    start: usize,
    end: usize,
}

/// Reads a safetensors file and returns `(name -> raw tensor, data_bytes)`.
fn parse_safetensors(path: &Path) -> Result<(HashMap<String, RawTensor>, Vec<u8>), Error> {
    let bytes = std::fs::read(path).map_err(|e| Error::Model(format!("read {path:?}: {e}")))?;
    if bytes.len() < 8 {
        return Err(Error::Model("file too short".into()));
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if 8 + header_len > bytes.len() {
        return Err(Error::Model("header length out of range".into()));
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])
        .map_err(|e| Error::Model(format!("bad JSON header: {e}")))?;
    let data = bytes[8 + header_len..].to_vec();

    let mut tensors = HashMap::new();
    let obj = header
        .as_object()
        .ok_or_else(|| Error::Model("header not an object".into()))?;
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        let o = v
            .as_object()
            .ok_or_else(|| Error::Model(format!("tensor {name}: not an object")))?;
        let dtype = o
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Model(format!("tensor {name}: missing dtype")))?
            .to_string();
        let shape = o
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::Model(format!("tensor {name}: missing shape")))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = o
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::Model(format!("tensor {name}: missing data_offsets")))?;
        let start = off[0].as_u64().unwrap_or(0) as usize;
        let end = off[1].as_u64().unwrap_or(0) as usize;
        tensors.insert(
            name.clone(),
            RawTensor {
                dtype,
                shape,
                start,
                end,
            },
        );
    }
    Ok((tensors, data))
}

/// Loads a tensor and converts it to f32 `[out, in]` row-major.
fn tensor_f32(data: &[u8], t: &RawTensor, expected: usize, name: &str) -> Result<Vec<f32>, Error> {
    let n: usize = t.shape.iter().product();
    if n != expected {
        return Err(Error::Model(format!(
            "{name}: shape {:?} has {n} elems, expected {expected}",
            t.shape
        )));
    }
    let span = t.end - t.start;
    let bytes = data
        .get(t.start..t.end)
        .ok_or_else(|| Error::Model(format!("{name}: data range out of bounds")))?;
    let mut out = Vec::with_capacity(n);
    match t.dtype.as_str() {
        "F32" => {
            if span != n * 4 {
                return Err(Error::Model(format!("{name}: F32 size mismatch")));
            }
            for i in 0..n {
                out.push(f32::from_le_bytes(
                    bytes[i * 4..i * 4 + 4].try_into().unwrap(),
                ));
            }
        }
        "F16" => {
            if span != n * 2 {
                return Err(Error::Model(format!("{name}: F16 size mismatch")));
            }
            for i in 0..n {
                let u = u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
                out.push(f16_to_f32(u));
            }
        }
        "BF16" => {
            if span != n * 2 {
                return Err(Error::Model(format!("{name}: BF16 size mismatch")));
            }
            for i in 0..n {
                let u = u16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap());
                out.push(bf16_to_f32(u));
            }
        }
        other => return Err(Error::Model(format!("{name}: unsupported dtype {other}"))),
    }
    Ok(out)
}

/// Half-precision float to f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// BF16 to f32 (zero-extend).
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Loads a single safetensors checkpoint into [`Weights`].
pub fn load_safetensors(path: &Path, cfg: &Config, tie_embeddings: bool) -> Result<Weights, Error> {
    let (tensors, data) = parse_safetensors(path)?;
    build_weights(&tensors, &data, cfg, tie_embeddings)
}

/// Loads every `*.safetensors` shard in `path` and merges them into one
/// [`Weights`] (Qwen-8B+ checkpoints ship as 5..65 shards; per-file tensor
/// offsets are rebased onto the concatenated data section).
pub fn load_safetensors_dir(
    path: &Path,
    cfg: &Config,
    tie_embeddings: bool,
) -> Result<Weights, Error> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(path)
        .map_err(|e| Error::Model(format!("read dir {path:?}: {e}")))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(Error::Model(format!("no .safetensors files in {path:?}")));
    }
    let mut tensors = HashMap::new();
    let mut data = Vec::new();
    for f in &files {
        let (mut t, d) = parse_safetensors(f)?;
        let base = data.len();
        for rt in t.values_mut() {
            rt.start += base;
            rt.end += base;
        }
        tensors.extend(t);
        data.extend(d);
    }
    build_weights(&tensors, &data, cfg, tie_embeddings)
}

/// Builds [`Weights`] from a merged tensor map (single file or shards).
fn build_weights(
    tensors: &HashMap<String, RawTensor>,
    data: &[u8],
    cfg: &Config,
    tie_embeddings: bool,
) -> Result<Weights, Error> {
    let d = cfg.d_model;
    let nq = cfg.n_heads * cfg.head_dim;
    let nkv = cfg.n_kv_heads * cfg.head_dim;

    // Like `get`, but returns an empty vector when the tensor is absent
    // (optional biases).
    let get_opt = |name: &str, expected: usize| -> Result<Vec<f32>, Error> {
        if !tensors.contains_key(name) {
            return Ok(Vec::new());
        }
        tensor_f32(data, &tensors[name], expected, name)
    };
    let get = |name: &str, expected: usize| -> Result<Vec<f32>, Error> {
        let t = tensors
            .get(name)
            .ok_or_else(|| Error::Model(format!("missing tensor {name}")))?;
        tensor_f32(data, t, expected, name)
    };

    let tok_emb = get("model.embed_tokens.weight", cfg.vocab_size * d)?;
    let rms_final = get("model.norm.weight", d)?;
    let lm_head = match tensors.get("lm_head.weight") {
        Some(_) => get("lm_head.weight", cfg.vocab_size * d)?,
        None if tie_embeddings => tok_emb.clone(),
        None => {
            return Err(Error::Model(
                "lm_head.weight missing and tie_embeddings=false".into(),
            ));
        }
    };

    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = |suffix: &str| format!("model.layers.{i}.{suffix}");
        let (moe_router, moe_wg, moe_wu, moe_wd) = if cfg.num_experts > 0 {
            let ne = cfg.num_experts;
            let router = get(&p("mlp.gate.weight"), ne * d)?;
            let mut wg = Vec::with_capacity(ne * cfg.intermediate_size * d);
            let mut wu = Vec::with_capacity(ne * cfg.intermediate_size * d);
            let mut wd = Vec::with_capacity(ne * d * cfg.intermediate_size);
            for e in 0..ne {
                let ep = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}");
                wg.extend(get(&ep("gate_proj.weight"), cfg.intermediate_size * d)?);
                wu.extend(get(&ep("up_proj.weight"), cfg.intermediate_size * d)?);
                wd.extend(get(&ep("down_proj.weight"), d * cfg.intermediate_size)?);
            }
            (router, wg, wu, wd)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        let lw = LayerWeights {
            wq: get(&p("self_attn.q_proj.weight"), d * nq)?,
            wk: get(&p("self_attn.k_proj.weight"), d * nkv)?,
            wv: get(&p("self_attn.v_proj.weight"), d * nkv)?,
            wo: get(&p("self_attn.o_proj.weight"), nq * d)?,
            rms_attn: get(&p("input_layernorm.weight"), d)?,
            wg: get(&p("mlp.gate_proj.weight"), cfg.intermediate_size * d)?,
            wu: get(&p("mlp.up_proj.weight"), cfg.intermediate_size * d)?,
            wd: get(&p("mlp.down_proj.weight"), d * cfg.intermediate_size)?,
            rms_mlp: get(&p("post_attention_layernorm.weight"), d)?,
            // Qwen2 checkpoints ship q/k/v biases (even with
            // `attention_bias: false`); default to empty (no bias) when absent.
            bq: get_opt(&p("self_attn.q_proj.bias"), nq)?,
            bk: get_opt(&p("self_attn.k_proj.bias"), nkv)?,
            bv: get_opt(&p("self_attn.v_proj.bias"), nkv)?,
            // Qwen3 QK-norm: per-head RMSNorm on q/k after projection.
            q_norm: get_opt(&p("self_attn.q_norm.weight"), cfg.n_heads * cfg.head_dim)?,
            k_norm: get_opt(&p("self_attn.k_norm.weight"), cfg.n_kv_heads * cfg.head_dim)?,
            moe_router,
            moe_wg,
            moe_wu,
            moe_wd,
        };
        layers.push(lw);
    }

    Ok(Weights {
        tok_emb,
        rms_final,
        lm_head,
        layers,
    })
}
