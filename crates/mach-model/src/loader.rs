//! Safetensors weight loading.
//!
//! Implements the safetensors binary format (8-byte little-endian header
//! length, JSON header, raw tensor data) and maps Llama/Qwen-style tensor
//! names onto the slice [`Weights`] layout. F32/F16/BF16 tensors are loaded
//! and converted to f32.

use crate::fp8::Fp8Tensor;
use crate::q4::Q4Tensor;
use crate::weights::{LayerWeightsFp8, LayerWeightsQ4, WeightsFp8, WeightsQ4};
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
    let mut bytes = std::fs::read(path).map_err(|e| Error::Model(format!("read {path:?}: {e}")))?;
    if bytes.len() < 8 {
        return Err(Error::Model("file too short".into()));
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if 8 + header_len > bytes.len() {
        return Err(Error::Model("header length out of range".into()));
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])
        .map_err(|e| Error::Model(format!("bad JSON header: {e}")))?;
    // Move the data segment out of the file buffer instead of copying it:
    // `bytes` keeps only the small header, so the whole file never lives twice
    // during the tensor loop (multi-GB shards would double host RAM).
    let data = bytes.split_off(8 + header_len);

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
        if off.len() < 2 {
            return Err(Error::Model(format!(
                "tensor {name}: data_offsets must have 2 entries, got {}",
                off.len()
            )));
        }
        let start = off[0]
            .as_u64()
            .ok_or_else(|| Error::Model(format!("tensor {name}: data_offsets[0] not a u64")))?
            as usize;
        let end = off[1]
            .as_u64()
            .ok_or_else(|| Error::Model(format!("tensor {name}: data_offsets[1] not a u64")))?
            as usize;
        if start > end || end > data.len() {
            return Err(Error::Model(format!(
                "tensor {name}: data_offsets [{start}, {end}) out of bounds (data len {})",
                data.len()
            )));
        }
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

/// Parallel f32 conversion for large tensors: splits the element range across
/// threads (each converts its byte slice to f32), concatenating in order.
/// Bit-equal to [`tensor_f32`]. Falls back to the scalar path below ~1M
/// elements (thread spawn overhead is not worth it for small tensors).
fn tensor_f32_par(
    data: &[u8],
    t: &RawTensor,
    expected: usize,
    name: &str,
) -> Result<Vec<f32>, Error> {
    let n: usize = t.shape.iter().product();
    if n != expected {
        return Err(Error::Model(format!(
            "{name}: shape {:?} has {n} elems, expected {expected}",
            t.shape
        )));
    }
    if n < (1 << 20) {
        return tensor_f32(data, t, expected, name);
    }
    let elem = match t.dtype.as_str() {
        "F32" => 4usize,
        "F16" | "BF16" => 2usize,
        other => return Err(Error::Model(format!("{name}: unsupported dtype {other}"))),
    };
    let span = t.end - t.start;
    if span != n * elem {
        return Err(Error::Model(format!("{name}: size mismatch")));
    }
    let bytes = data
        .get(t.start..t.end)
        .ok_or_else(|| Error::Model(format!("{name}: data range out of bounds")))?;
    let n_threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(4)
        .min(16);
    let chunk = n.div_ceil(n_threads);
    let dtype = t.dtype.as_str();
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(n_threads);
        for th in 0..n_threads {
            let e0 = th * chunk;
            let e1 = (e0 + chunk).min(n);
            if e0 >= e1 {
                continue;
            }
            handles.push(s.spawn(move || {
                let mut local = Vec::with_capacity(e1 - e0);
                match dtype {
                    "F32" => {
                        for i in e0..e1 {
                            let off = i * 4;
                            local.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
                        }
                    }
                    "F16" => {
                        for i in e0..e1 {
                            let off = i * 2;
                            let u = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
                            local.push(f16_to_f32(u));
                        }
                    }
                    "BF16" => {
                        for i in e0..e1 {
                            let off = i * 2;
                            let u = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap());
                            local.push(bf16_to_f32(u));
                        }
                    }
                    other => {
                        return Err(Error::Model(format!("{name}: unsupported dtype {other}")));
                    }
                }
                Ok(local)
            }));
        }
        let mut merged = Vec::with_capacity(n);
        for h in handles {
            merged.extend_from_slice(&h.join().unwrap()?);
        }
        Ok(merged)
    })
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
    // Accept either a single .safetensors file or a directory of shards.
    let mut files: Vec<std::path::PathBuf> = if path.is_dir() {
        std::fs::read_dir(path)
            .map_err(|e| Error::Model(format!("read dir {path:?}: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect()
    } else {
        vec![path.to_path_buf()]
    };
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

/// Qwen3 QK-norm weight: real checkpoints store ONE `[head_dim]` vector
/// shared across all heads; older test checkpoints may store per-head
/// `[n_heads * head_dim]`. Accept both, tiling the shared form to per-head.
fn load_qk_norm(
    tensors: &HashMap<String, RawTensor>,
    data: &[u8],
    name: &str,
    per_head: usize,
    head_dim: usize,
) -> Result<Vec<f32>, Error> {
    let Some(t) = tensors.get(name) else {
        return Ok(Vec::new());
    };
    let n: usize = t.shape.iter().product();
    if n == per_head {
        return tensor_f32(data, t, per_head, name);
    }
    if n == head_dim {
        let shared = tensor_f32(data, t, head_dim, name)?;
        let mut v = Vec::with_capacity(per_head);
        for _ in 0..(per_head / head_dim) {
            v.extend_from_slice(&shared);
        }
        return Ok(v);
    }
    Err(Error::Model(format!("{name}: unexpected QK-norm size {n}")))
}

/// Qwen3 ships q_norm/k_norm as one SHARED `[head_dim]` vector, while the
/// QK-norm kernel indexes per head (`[n_heads, head_dim]`). The f32 loader
/// broadcasts via `load_qk_norm`; the Q4/FP8 loaders call this after the
/// layer is built so both storage paths match the kernel contract.
fn broadcast_qk_norm(q_norm: &mut Vec<f32>, k_norm: &mut Vec<f32>, cfg: &Config) {
    let hd = cfg.head_dim;
    if q_norm.len() == hd && cfg.n_heads > 1 {
        let shared = q_norm.clone();
        for _ in 1..cfg.n_heads {
            q_norm.extend_from_slice(&shared);
        }
    }
    if k_norm.len() == hd && cfg.n_kv_heads > 1 {
        let shared = k_norm.clone();
        for _ in 1..cfg.n_kv_heads {
            k_norm.extend_from_slice(&shared);
        }
    }
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
        // Per-layer MoE detection: Qwen-MoE checkpoints mix dense layers
        // (`mlp_only_layers`, e.g. Qwen3-MoE) with routed-expert layers. A layer
        // is MoE iff it carries a router tensor (`mlp.gate.weight`).
        let is_moe = cfg.num_experts > 0 && tensors.contains_key(&p("mlp.gate.weight"));
        let einter = cfg.expert_size();
        let (moe_router, moe_wg, moe_wu, moe_wd) = if is_moe {
            let ne = cfg.num_experts;
            let router = get(&p("mlp.gate.weight"), ne * d)?;
            let mut wg = Vec::with_capacity(ne * einter * d);
            let mut wu = Vec::with_capacity(ne * einter * d);
            let mut wd = Vec::with_capacity(ne * d * einter);
            for e in 0..ne {
                let ep = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}");
                wg.extend(get(&ep("gate_proj.weight"), einter * d)?);
                wu.extend(get(&ep("up_proj.weight"), einter * d)?);
                wd.extend(get(&ep("down_proj.weight"), d * einter)?);
            }
            (router, wg, wu, wd)
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        // MLA (DeepSeek-V2 style): low-rank Q + compressed KV replaces the
        // standard q/k/v/o projections when kv_lora_rank > 0.
        let mla = cfg.kv_lora_rank > 0;
        let (mla_q_a, mla_q_a_norm, mla_q_b, mla_q_rope, mla_kv_a, mla_kv_a_norm, mla_kv_b, mla_o) =
            if mla {
                (
                    get(&p("self_attn.q_a_proj.weight"), cfg.q_lora_rank * d)?,
                    get(&p("self_attn.q_a_layernorm.weight"), cfg.q_lora_rank)?,
                    get(
                        &p("self_attn.q_b_proj.weight"),
                        cfg.n_heads * cfg.qk_nope_head_dim * cfg.q_lora_rank,
                    )?,
                    get(
                        &p("self_attn.q_rope_proj.weight"),
                        cfg.n_heads * cfg.qk_rope_head_dim * d,
                    )?,
                    get(
                        &p("self_attn.kv_a_proj_with_mqa.weight"),
                        (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * d,
                    )?,
                    get(&p("self_attn.kv_a_layernorm.weight"), cfg.kv_lora_rank)?,
                    get(
                        &p("self_attn.kv_b_proj.weight"),
                        cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim) * cfg.kv_lora_rank,
                    )?,
                    get(
                        &p("self_attn.o_proj.weight"),
                        d * cfg.n_heads * cfg.v_head_dim,
                    )?,
                )
            } else {
                (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            };
        let lw = LayerWeights {
            wq: if mla {
                Vec::new()
            } else {
                get(&p("self_attn.q_proj.weight"), d * nq)?
            },
            wk: if mla {
                Vec::new()
            } else {
                get(&p("self_attn.k_proj.weight"), d * nkv)?
            },
            wv: if mla {
                Vec::new()
            } else {
                get(&p("self_attn.v_proj.weight"), d * nkv)?
            },
            wo: if mla {
                Vec::new()
            } else {
                get(&p("self_attn.o_proj.weight"), nq * d)?
            },
            rms_attn: get(&p("input_layernorm.weight"), d)?,
            wg: if is_moe {
                Vec::new()
            } else {
                get(&p("mlp.gate_proj.weight"), cfg.intermediate_size * d)?
            },
            wu: if is_moe {
                Vec::new()
            } else {
                get(&p("mlp.up_proj.weight"), cfg.intermediate_size * d)?
            },
            wd: if is_moe {
                Vec::new()
            } else {
                get(&p("mlp.down_proj.weight"), d * cfg.intermediate_size)?
            },
            rms_mlp: get(&p("post_attention_layernorm.weight"), d)?,
            // Qwen2 checkpoints ship q/k/v biases (even with
            // `attention_bias: false`); default to empty (no bias) when absent.
            bq: get_opt(&p("self_attn.q_proj.bias"), nq)?,
            bk: get_opt(&p("self_attn.k_proj.bias"), nkv)?,
            bv: get_opt(&p("self_attn.v_proj.bias"), nkv)?,
            // Qwen3 QK-norm: per-head RMSNorm on q/k after projection.
            q_norm: load_qk_norm(
                tensors,
                data,
                &p("self_attn.q_norm.weight"),
                cfg.n_heads * cfg.head_dim,
                cfg.head_dim,
            )?,
            k_norm: load_qk_norm(
                tensors,
                data,
                &p("self_attn.k_norm.weight"),
                cfg.n_kv_heads * cfg.head_dim,
                cfg.head_dim,
            )?,
            mla_q_a,
            mla_q_a_norm,
            mla_q_b,
            mla_q_rope,
            mla_kv_a,
            mla_kv_a_norm,
            mla_kv_b,
            mla_o,
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

/// Loads a checkpoint into storage-Q4 form, streaming shards one at a time so
/// host memory stays ~= the packed Q4 weights + one shard of raw bytes (8B
/// model: ~5GB instead of ~48GB for the f32 path).
///
/// Every GEMM weight is quantized to int4 as it is read and the raw shard
/// bytes are dropped before the next shard. Norms and biases stay f32.
pub fn load_safetensors_q4(
    path: &Path,
    cfg: &Config,
    tie_embeddings: bool,
) -> Result<WeightsQ4, Error> {
    let d = cfg.d_model;
    let nq = cfg.n_heads * cfg.head_dim;
    let nkv = cfg.n_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let einter = cfg.expert_size();
    let mla = cfg.kv_lora_rank > 0;
    let ne = cfg.num_experts;

    // GEMM tensors are quantized; small tensors (norms/biases) stay f32.
    let mut big: HashMap<String, Q4Tensor> = HashMap::new();
    let mut small: HashMap<String, Vec<f32>> = HashMap::new();

    // Accept either a single .safetensors file or a directory of shards.
    let mut files: Vec<std::path::PathBuf> = if path.is_dir() {
        std::fs::read_dir(path)
            .map_err(|e| Error::Model(format!("read dir {path:?}: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect()
    } else {
        vec![path.to_path_buf()]
    };
    files.sort();
    if files.is_empty() {
        return Err(Error::Model(format!("no .safetensors files in {path:?}")));
    }

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(16);

    for file in &files {
        let (tensors, data) = parse_safetensors(file)?;

        // Classify this shard's tensors: GEMM weights are quantized (parallel),
        // norms/biases stay f32 (inline). Big-vs-small is decided by whether the
        // name maps to a known GEMM weight.
        let mut big_work: Vec<(String, usize)> = Vec::new();
        for (name, t) in &tensors {
            let n: usize = t.shape.iter().product();
            let expected = if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
                Some(cfg.vocab_size * d)
            } else if let Some(e) = expected_q4_size(name, cfg, d, nq, nkv, inter, einter, mla) {
                Some(e)
            } else if let Some(e) = expected_small_size(name, cfg, d, nq, nkv, mla, ne) {
                // q_norm/k_norm: accept both shared [head_dim] and per-head
                // [n_heads*head_dim] / [n_kv_heads*head_dim] forms.
                let e = if (name.contains("q_norm.weight") || name.contains("k_norm.weight"))
                    && n != e
                    && (n == cfg.n_heads * cfg.head_dim || n == cfg.n_kv_heads * cfg.head_dim)
                {
                    n
                } else {
                    e
                };
                small.insert(name.clone(), tensor_f32(&data, t, e, name)?);
                None
            } else {
                // Unknown auxiliary tensors (e.g. shared_expert.*) are skipped,
                // matching the f32 loader's behavior.
                eprintln!("q4 loader: skipping unknown tensor {name}");
                None
            };
            if let Some(e) = expected {
                big_work.push((name.clone(), e));
            }
        }

        // Decode + quantize the GEMM tensors in parallel: per-element CPU work
        // dominates Q4 load, so the shard's tensors are split across threads
        // (peak host RAM is one shard's decoded f32 tensors, then the shard's
        // raw bytes are dropped before the next shard).
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: std::sync::Mutex<Vec<(String, Q4Tensor)>> =
            std::sync::Mutex::new(Vec::with_capacity(big_work.len()));
        let err: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);
        std::thread::scope(|s| {
            for _ in 0..n_threads {
                s.spawn(|| {
                    loop {
                        if err.lock().unwrap().is_some() {
                            break;
                        }
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((name, expected)) = big_work.get(i) else {
                            break;
                        };
                        match tensor_f32_par(&data, &tensors[name], *expected, name) {
                            Ok(f) => {
                                let q = Q4Tensor::quantize_par(&f);
                                results.lock().unwrap().push((name.clone(), q));
                            }
                            Err(e) => {
                                let mut g = err.lock().unwrap();
                                if g.is_none() {
                                    *g = Some(e);
                                }
                                break;
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = err.into_inner().unwrap() {
            return Err(e);
        }
        for (name, q) in results.into_inner().unwrap() {
            big.insert(name, q);
        }
        // Drop this shard's raw bytes before reading the next.
        drop(data);
    }

    // Assemble per-layer Q4 weights.
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = |suffix: &str| format!("model.layers.{i}.{suffix}");
        let is_moe = ne > 0 && small.contains_key(&p("mlp.gate.weight"));
        let mut lw = LayerWeightsQ4 {
            wq: if mla {
                Q4Tensor::default()
            } else {
                big.remove(&p("self_attn.q_proj.weight")).expect("q_proj")
            },
            wk: if mla {
                Q4Tensor::default()
            } else {
                big.remove(&p("self_attn.k_proj.weight")).expect("k_proj")
            },
            wv: if mla {
                Q4Tensor::default()
            } else {
                big.remove(&p("self_attn.v_proj.weight")).expect("v_proj")
            },
            wo: if mla {
                Q4Tensor::default()
            } else {
                big.remove(&p("self_attn.o_proj.weight")).expect("o_proj")
            },
            rms_attn: small
                .remove(&p("input_layernorm.weight"))
                .expect("input_layernorm"),
            wg: if is_moe {
                Q4Tensor::default()
            } else {
                big.remove(&p("mlp.gate_proj.weight")).expect("gate_proj")
            },
            wu: if is_moe {
                Q4Tensor::default()
            } else {
                big.remove(&p("mlp.up_proj.weight")).expect("up_proj")
            },
            wd: if is_moe {
                Q4Tensor::default()
            } else {
                big.remove(&p("mlp.down_proj.weight")).expect("down_proj")
            },
            rms_mlp: small
                .remove(&p("post_attention_layernorm.weight"))
                .expect("post_attention_layernorm"),
            bq: small
                .remove(&p("self_attn.q_proj.bias"))
                .unwrap_or_default(),
            bk: small
                .remove(&p("self_attn.k_proj.bias"))
                .unwrap_or_default(),
            bv: small
                .remove(&p("self_attn.v_proj.bias"))
                .unwrap_or_default(),
            q_norm: small
                .remove(&p("self_attn.q_norm.weight"))
                .unwrap_or_default(),
            k_norm: small
                .remove(&p("self_attn.k_norm.weight"))
                .unwrap_or_default(),
            mla_q_a: if mla {
                big.remove(&p("self_attn.q_a_proj.weight"))
                    .expect("mla q_a")
            } else {
                Q4Tensor::default()
            },
            mla_q_a_norm: if mla {
                small
                    .remove(&p("self_attn.q_a_layernorm.weight"))
                    .expect("mla q_a_norm")
            } else {
                Vec::new()
            },
            mla_q_b: if mla {
                big.remove(&p("self_attn.q_b_proj.weight"))
                    .expect("mla q_b")
            } else {
                Q4Tensor::default()
            },
            mla_q_rope: if mla {
                big.remove(&p("self_attn.q_rope_proj.weight"))
                    .expect("mla q_rope")
            } else {
                Q4Tensor::default()
            },
            mla_kv_a: if mla {
                big.remove(&p("self_attn.kv_a_proj_with_mqa.weight"))
                    .expect("mla kv_a")
            } else {
                Q4Tensor::default()
            },
            mla_kv_a_norm: if mla {
                small
                    .remove(&p("self_attn.kv_a_layernorm.weight"))
                    .expect("mla kv_a_norm")
            } else {
                Vec::new()
            },
            mla_kv_b: if mla {
                big.remove(&p("self_attn.kv_b_proj.weight"))
                    .expect("mla kv_b")
            } else {
                Q4Tensor::default()
            },
            mla_o: if mla {
                big.remove(&p("self_attn.o_proj.weight")).expect("mla o")
            } else {
                Q4Tensor::default()
            },
            moe_router: small.remove(&p("mlp.gate.weight")).unwrap_or_default(),
            moe_wg: Q4Tensor::default(),
            moe_wu: Q4Tensor::default(),
            moe_wd: Q4Tensor::default(),
        };
        if is_moe {
            let ne_i = ne;
            // Single-pass per-tensor concat: the sequential fold re-clones the
            // growing prefix per expert (O(n²) byte traffic — minutes on
            // 256-expert checkpoints); concat_many appends in one O(total)
            // pass (byte-identical for the group-aligned expert tensors).
            let mut concat = |name: &str| -> Q4Tensor {
                let parts: Vec<Q4Tensor> = (0..ne_i)
                    .map(|e| {
                        big.remove(&format!("model.layers.{i}.mlp.experts.{e}.{name}"))
                            .expect("exp tensor")
                    })
                    .collect();
                Q4Tensor::concat_many(&parts)
            };
            lw.moe_wg = concat("gate_proj.weight");
            lw.moe_wu = concat("up_proj.weight");
            lw.moe_wd = concat("down_proj.weight");
        }
        broadcast_qk_norm(&mut lw.q_norm, &mut lw.k_norm, cfg);
        layers.push(lw);
    }

    let tok_emb = big.remove("model.embed_tokens.weight").expect("tok_emb");
    let lm_head = match big.remove("lm_head.weight") {
        Some(t) => t,
        None if tie_embeddings => tok_emb.clone(),
        None => {
            return Err(Error::Model(
                "lm_head.weight missing and tie_embeddings=false".into(),
            ));
        }
    };
    Ok(WeightsQ4 {
        tok_emb,
        rms_final: small.remove("model.norm.weight").expect("norm"),
        lm_head,
        layers,
    })
}

/// Loads a checkpoint into storage-FP8 form, streaming shards one at a time so
/// host memory stays ~= the packed FP8 weights + one shard of raw bytes
/// (8B model: ~8GB instead of ~48GB for the f32 path / ~16GB for f16).
///
/// Every GEMM weight is quantized to E4M3 (one byte/element + one f32 scale
/// per tensor; MoE experts keep per-expert scales and concatenate by appending
/// packed bytes + scales) as it is read, and the raw shard bytes are dropped
/// before the next shard. Norms and biases stay f32.
pub fn load_safetensors_fp8(
    path: &Path,
    cfg: &Config,
    tie_embeddings: bool,
) -> Result<WeightsFp8, Error> {
    let d = cfg.d_model;
    let nq = cfg.n_heads * cfg.head_dim;
    let nkv = cfg.n_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let einter = cfg.expert_size();
    let mla = cfg.kv_lora_rank > 0;
    let ne = cfg.num_experts;

    // GEMM tensors are quantized; small tensors (norms/biases) stay f32.
    let mut big: HashMap<String, Fp8Tensor> = HashMap::new();
    let mut small: HashMap<String, Vec<f32>> = HashMap::new();

    // Accept either a single .safetensors file or a directory of shards.
    let mut files: Vec<std::path::PathBuf> = if path.is_dir() {
        std::fs::read_dir(path)
            .map_err(|e| Error::Model(format!("read dir {path:?}: {e}")))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect()
    } else {
        vec![path.to_path_buf()]
    };
    files.sort();
    if files.is_empty() {
        return Err(Error::Model(format!("no .safetensors files in {path:?}")));
    }

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(16);

    for file in &files {
        let (tensors, data) = parse_safetensors(file)?;

        // Classify this shard's tensors: GEMM weights are quantized (parallel),
        // norms/biases stay f32 (inline). Big-vs-small is decided by whether the
        // name maps to a known GEMM weight (same shape matcher as the Q4
        // loader; only the quantization differs).
        let mut big_work: Vec<(String, usize)> = Vec::new();
        for (name, t) in &tensors {
            let n: usize = t.shape.iter().product();
            let expected = if name == "model.embed_tokens.weight" || name == "lm_head.weight" {
                Some(cfg.vocab_size * d)
            } else if let Some(e) = expected_q4_size(name, cfg, d, nq, nkv, inter, einter, mla) {
                Some(e)
            } else if let Some(e) = expected_small_size(name, cfg, d, nq, nkv, mla, ne) {
                // q_norm/k_norm: accept both shared [head_dim] and per-head
                // [n_heads*head_dim] / [n_kv_heads*head_dim] forms.
                let e = if (name.contains("q_norm.weight") || name.contains("k_norm.weight"))
                    && n != e
                    && (n == cfg.n_heads * cfg.head_dim || n == cfg.n_kv_heads * cfg.head_dim)
                {
                    n
                } else {
                    e
                };
                small.insert(name.clone(), tensor_f32(&data, t, e, name)?);
                None
            } else {
                // Unknown auxiliary tensors (e.g. shared_expert.*) are skipped,
                // matching the f32 loader's behavior.
                eprintln!("fp8 loader: skipping unknown tensor {name}");
                None
            };
            if let Some(e) = expected {
                big_work.push((name.clone(), e));
            }
        }

        // Decode + quantize the GEMM tensors in parallel: per-element CPU work
        // dominates FP8 load, so the shard's tensors are split across threads
        // (peak host RAM is one shard's decoded f32 tensors, then the shard's
        // raw bytes are dropped before the next shard).
        let next = std::sync::atomic::AtomicUsize::new(0);
        let results: std::sync::Mutex<Vec<(String, Fp8Tensor)>> =
            std::sync::Mutex::new(Vec::with_capacity(big_work.len()));
        let err: std::sync::Mutex<Option<Error>> = std::sync::Mutex::new(None);
        std::thread::scope(|s| {
            for _ in 0..n_threads {
                s.spawn(|| {
                    loop {
                        if err.lock().unwrap().is_some() {
                            break;
                        }
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((name, expected)) = big_work.get(i) else {
                            break;
                        };
                        match tensor_f32(&data, &tensors[name], *expected, name) {
                            Ok(f) => {
                                let q = Fp8Tensor::quantize_par(&f);
                                results.lock().unwrap().push((name.clone(), q));
                            }
                            Err(e) => {
                                let mut g = err.lock().unwrap();
                                if g.is_none() {
                                    *g = Some(e);
                                }
                                break;
                            }
                        }
                    }
                });
            }
        });
        if let Some(e) = err.into_inner().unwrap() {
            return Err(e);
        }
        for (name, q) in results.into_inner().unwrap() {
            big.insert(name, q);
        }
        // Drop this shard's raw bytes before reading the next.
        drop(data);
    }

    // Assemble per-layer FP8 weights.
    let mut layers = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let p = |suffix: &str| format!("model.layers.{i}.{suffix}");
        let is_moe = ne > 0 && small.contains_key(&p("mlp.gate.weight"));
        let mut lw = LayerWeightsFp8 {
            wq: if mla {
                Fp8Tensor::default()
            } else {
                big.remove(&p("self_attn.q_proj.weight")).expect("q_proj")
            },
            wk: if mla {
                Fp8Tensor::default()
            } else {
                big.remove(&p("self_attn.k_proj.weight")).expect("k_proj")
            },
            wv: if mla {
                Fp8Tensor::default()
            } else {
                big.remove(&p("self_attn.v_proj.weight")).expect("v_proj")
            },
            wo: if mla {
                Fp8Tensor::default()
            } else {
                big.remove(&p("self_attn.o_proj.weight")).expect("o_proj")
            },
            rms_attn: small
                .remove(&p("input_layernorm.weight"))
                .expect("input_layernorm"),
            wg: if is_moe {
                Fp8Tensor::default()
            } else {
                big.remove(&p("mlp.gate_proj.weight")).expect("gate_proj")
            },
            wu: if is_moe {
                Fp8Tensor::default()
            } else {
                big.remove(&p("mlp.up_proj.weight")).expect("up_proj")
            },
            wd: if is_moe {
                Fp8Tensor::default()
            } else {
                big.remove(&p("mlp.down_proj.weight")).expect("down_proj")
            },
            rms_mlp: small
                .remove(&p("post_attention_layernorm.weight"))
                .expect("post_attention_layernorm"),
            bq: small
                .remove(&p("self_attn.q_proj.bias"))
                .unwrap_or_default(),
            bk: small
                .remove(&p("self_attn.k_proj.bias"))
                .unwrap_or_default(),
            bv: small
                .remove(&p("self_attn.v_proj.bias"))
                .unwrap_or_default(),
            q_norm: small
                .remove(&p("self_attn.q_norm.weight"))
                .unwrap_or_default(),
            k_norm: small
                .remove(&p("self_attn.k_norm.weight"))
                .unwrap_or_default(),
            mla_q_a: if mla {
                big.remove(&p("self_attn.q_a_proj.weight"))
                    .expect("mla q_a")
            } else {
                Fp8Tensor::default()
            },
            mla_q_a_norm: if mla {
                small
                    .remove(&p("self_attn.q_a_layernorm.weight"))
                    .expect("mla q_a_norm")
            } else {
                Vec::new()
            },
            mla_q_b: if mla {
                big.remove(&p("self_attn.q_b_proj.weight"))
                    .expect("mla q_b")
            } else {
                Fp8Tensor::default()
            },
            mla_q_rope: if mla {
                big.remove(&p("self_attn.q_rope_proj.weight"))
                    .expect("mla q_rope")
            } else {
                Fp8Tensor::default()
            },
            mla_kv_a: if mla {
                big.remove(&p("self_attn.kv_a_proj_with_mqa.weight"))
                    .expect("mla kv_a")
            } else {
                Fp8Tensor::default()
            },
            mla_kv_a_norm: if mla {
                small
                    .remove(&p("self_attn.kv_a_layernorm.weight"))
                    .expect("mla kv_a_norm")
            } else {
                Vec::new()
            },
            mla_kv_b: if mla {
                big.remove(&p("self_attn.kv_b_proj.weight"))
                    .expect("mla kv_b")
            } else {
                Fp8Tensor::default()
            },
            mla_o: if mla {
                big.remove(&p("self_attn.o_proj.weight")).expect("mla o")
            } else {
                Fp8Tensor::default()
            },
            moe_router: small.remove(&p("mlp.gate.weight")).unwrap_or_default(),
            moe_wg: Fp8Tensor::default(),
            moe_wu: Fp8Tensor::default(),
            moe_wd: Fp8Tensor::default(),
        };
        if is_moe {
            let ne_i = ne;
            // Single-pass per-tensor concat (the sequential fold re-clones the
            // growing prefix per expert — O(n²) byte traffic).
            let mut concat = |name: &str| -> Fp8Tensor {
                let parts: Vec<Fp8Tensor> = (0..ne_i)
                    .map(|e| {
                        big.remove(&format!("model.layers.{i}.mlp.experts.{e}.{name}"))
                            .expect("exp tensor")
                    })
                    .collect();
                Fp8Tensor::concat_many(&parts)
            };
            lw.moe_wg = concat("gate_proj.weight");
            lw.moe_wu = concat("up_proj.weight");
            lw.moe_wd = concat("down_proj.weight");
        }
        broadcast_qk_norm(&mut lw.q_norm, &mut lw.k_norm, cfg);
        layers.push(lw);
    }

    let tok_emb = big.remove("model.embed_tokens.weight").expect("tok_emb");
    let lm_head = match big.remove("lm_head.weight") {
        Some(t) => t,
        None if tie_embeddings => tok_emb.clone(),
        None => {
            return Err(Error::Model(
                "lm_head.weight missing and tie_embeddings=false".into(),
            ));
        }
    };
    Ok(WeightsFp8 {
        tok_emb,
        rms_final: small.remove("model.norm.weight").expect("norm"),
        lm_head,
        layers,
    })
}

/// Expected element count for a quantized (GEMM) weight tensor by name.
#[allow(clippy::too_many_arguments)]
fn expected_q4_size(
    name: &str,
    cfg: &Config,
    d: usize,
    nq: usize,
    nkv: usize,
    inter: usize,
    einter: usize,
    mla: bool,
) -> Option<usize> {
    // Match against the same names build_weights uses. MLA's `o_proj` is a
    // different shape, so it must be matched before the dense `o_proj`.
    if mla && name.contains("self_attn.o_proj.weight") {
        Some(d * cfg.n_heads * cfg.v_head_dim)
    } else if mla && name.contains("self_attn.q_a_proj.weight") {
        Some(cfg.q_lora_rank * d)
    } else if mla && name.contains("self_attn.q_b_proj.weight") {
        Some(cfg.n_heads * cfg.qk_nope_head_dim * cfg.q_lora_rank)
    } else if mla && name.contains("self_attn.q_rope_proj.weight") {
        Some(cfg.n_heads * cfg.qk_rope_head_dim * d)
    } else if mla && name.contains("self_attn.kv_a_proj_with_mqa.weight") {
        Some((cfg.kv_lora_rank + cfg.qk_rope_head_dim) * d)
    } else if mla && name.contains("self_attn.kv_b_proj.weight") {
        Some(cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim) * cfg.kv_lora_rank)
    } else if name.contains("self_attn.q_proj.weight") {
        Some(d * nq)
    } else if name.contains("self_attn.k_proj.weight") || name.contains("self_attn.v_proj.weight") {
        Some(d * nkv)
    } else if name.contains("self_attn.o_proj.weight") {
        Some(nq * d)
    } else if name.contains("mlp.gate_proj.weight")
        || name.contains("mlp.up_proj.weight")
        || name.contains("mlp.down_proj.weight")
    {
        Some(inter * d)
    } else if name.contains("mlp.experts.")
        && (name.ends_with("gate_proj.weight")
            || name.ends_with("up_proj.weight")
            || name.ends_with("down_proj.weight"))
    {
        Some(einter * d)
    } else {
        None
    }
}

/// Expected element count for a small (f32) tensor by name.
fn expected_small_size(
    name: &str,
    cfg: &Config,
    d: usize,
    nq: usize,
    nkv: usize,
    mla: bool,
    ne: usize,
) -> Option<usize> {
    if name == "model.norm.weight"
        || name.contains("input_layernorm.weight")
        || name.contains("post_attention_layernorm.weight")
    {
        Some(d)
    } else if name.ends_with("self_attn.q_proj.bias") {
        Some(nq)
    } else if name.ends_with("self_attn.k_proj.bias") || name.ends_with("self_attn.v_proj.bias") {
        Some(nkv)
    } else if name.contains("mlp.gate.weight") {
        Some(ne * d)
    } else if name.contains("self_attn.q_norm.weight") || name.contains("self_attn.k_norm.weight") {
        Some(cfg.head_dim)
    } else if mla && name.contains("self_attn.q_a_layernorm.weight") {
        Some(cfg.q_lora_rank)
    } else if mla && name.contains("self_attn.kv_a_layernorm.weight") {
        Some(cfg.kv_lora_rank)
    } else {
        None
    }
}

/// Concatenates two quantized tensors (per-expert MoE tensors). Delegates to
/// `Q4Tensor::concat`: group-aligned tensors append packed int4 bytes + scales
/// directly (O(1) per expert, fixing the old O(ne^2) dequant+requant that made
/// Q4 loading of MoE checkpoints ~30x slower than f32).
#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(dtype: &str, n: usize, seed: u64) -> (RawTensor, Vec<u8>) {
        let elem = match dtype {
            "F32" => 4usize,
            _ => 2usize,
        };
        let mut data = Vec::with_capacity(n * elem);
        let mut r = seed;
        for i in 0..n {
            r = r
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = (((r >> 33) as f64) / ((1u64 << 31) as f64)) as f32 * 3.0 - 1.5;
            match dtype {
                "F32" => data.extend_from_slice(&v.to_le_bytes()),
                "F16" => data.extend_from_slice(&crate::fp16::f32_to_f16(v).to_le_bytes()),
                "BF16" => data.extend_from_slice(&(((v.to_bits() >> 16) as u16).to_le_bytes())),
                _ => unreachable!(),
            }
            let _ = i;
        }
        let t = RawTensor {
            dtype: dtype.to_string(),
            shape: vec![n],
            start: 0,
            end: n * elem,
        };
        (t, data)
    }

    #[test]
    fn tensor_f32_par_matches_scalar_bitwise() {
        for dtype in ["F32", "F16", "BF16"] {
            // Large (parallel path) and small (fallback path).
            for n in [2usize << 20, 1usize << 21, 1000usize] {
                let (t, data) = synthetic(dtype, n, 42 + n as u64);
                let a = tensor_f32(&data, &t, n, "w").unwrap();
                let b = tensor_f32_par(&data, &t, n, "w").unwrap();
                assert_eq!(a, b, "dtype={dtype} n={n}");
            }
        }
    }
}
