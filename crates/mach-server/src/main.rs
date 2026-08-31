//! MachServe OpenAI-compatible server binary.
//!
//! Loads a safetensors Llama/Qwen checkpoint and serves the continuous-batching
//! engine over HTTP:
//!   cargo run -p mach-server --release --features hip
//!
//! Env: MACH_MODELS (default ".models"), MACH_MODEL (default
//! "qwen-0.5b.safetensors"), MACH_CONFIG (default "qwen-config.json"),
//! MACH_CAPACITY (default 64), MACH_PREFILL_ROWS (default 512),
//! MACH_ADDR (default "127.0.0.1:8080"), MACH_Q4 / MACH_FP8 (storage-quantized
//! host weights: int4 or E4M3, dequantized to f16 on the device),
//! MACH_Q4_DEVICE=1 (with MACH_Q4: keep the MoE expert pool in raw Q4 on the
//! device, dequantized in-kernel — the memory path for 30B-class checkpoints
//! whose f16 experts would not fit in VRAM),
//! MACH_PAGED=1 (paged-KV engine with cross-request prefix reuse) with
//! MACH_TPP (KV page size in tokens, default 64; only read by the modes that
//! engage paged KV — plain, Q4 and FP8 non-MLA). Limitations: paged KV serves
//! MLA models in F32 only (quantized MLA warns and falls back to continuous),
//! and MACH_SPEC / MoE-offload modes ignore MACH_PAGED (warned).
//! MACH_MOE_GROUPED=0 (default on; disable the batched-MoE decode grouped
//! GEMV device path — A/B switch and ops lever). NOTE: Q4-on-device models
//! (MACH_Q4_DEVICE=1) have no f16/f32 expert copy, so they ALWAYS run the
//! grouped path and this knob is a no-op for them.
//! MACH_STEP_PROFILE=1 (diagnostic): per-layer attention/MoE HIP event
//! bracketing printed after each decode step.
#[cfg(feature = "hip")]
use mach_kernel_sys::hip;
#[cfg(feature = "hip")]
use mach_model::batched::BatchedModel;
#[cfg(feature = "hip")]
use mach_model::config::ModelDType;
#[cfg(feature = "hip")]
use mach_model::loader::{load_safetensors, load_safetensors_fp8, load_safetensors_q4};
#[cfg(feature = "hip")]
use mach_model::tokenizer::Tokenizer;
#[cfg(feature = "hip")]
use mach_model::{Config, Weights, WeightsFp8, WeightsQ4};
#[cfg(feature = "hip")]
use mach_server::{AppState, ServerEngine, router};
#[cfg(feature = "hip")]
use std::path::PathBuf;

#[cfg(feature = "hip")]
fn config_from_json(path: &std::path::Path) -> Config {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read config"))
            .expect("parse config");
    let hidden = v["hidden_size"].as_u64().unwrap_or(896) as usize;
    let layers = v["num_hidden_layers"].as_u64().unwrap_or(24) as usize;
    let heads = v["num_attention_heads"].as_u64().unwrap_or(14) as usize;
    let kv = v["num_key_value_heads"].as_u64().unwrap_or(heads as u64) as usize;
    let vocab = v["vocab_size"].as_u64().unwrap_or(151936) as usize;
    let inter = v["intermediate_size"].as_u64().unwrap_or(4 * hidden as u64) as usize;
    let max_seq = v["max_position_embeddings"]
        .as_u64()
        .unwrap_or(2048)
        .min(8192) as usize;
    let eps = v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    let theta = v["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let mut cfg = Config::llama(hidden, layers, heads, kv, vocab, max_seq);
    // Some configs (e.g. Qwen3-30B-A3B) ship an explicit `head_dim` that
    // differs from hidden/n_heads (q/o width = n_heads*head_dim is wider
    // than hidden). Honor it, or the loader under-sizes q/o projections.
    if let Some(hd) = v["head_dim"].as_u64() {
        cfg.head_dim = hd as usize;
    }
    cfg.intermediate_size = inter;
    cfg.rms_eps = eps;
    cfg.rope_theta = theta;
    // MLA (DeepSeek-V2 style): compressed KV + low-rank Q replace q/k/v/o.
    cfg.q_lora_rank = v["q_lora_rank"].as_u64().unwrap_or(0) as usize;
    cfg.kv_lora_rank = v["kv_lora_rank"].as_u64().unwrap_or(0) as usize;
    cfg.qk_nope_head_dim = v["qk_nope_head_dim"].as_u64().unwrap_or(0) as usize;
    cfg.qk_rope_head_dim = v["qk_rope_head_dim"].as_u64().unwrap_or(0) as usize;
    cfg.v_head_dim = v["v_head_dim"].as_u64().unwrap_or(0) as usize;
    if cfg.kv_lora_rank > 0 {
        // MLA: per-head q is (nope + rope); the expanded KV cache is per-head
        // f32. head_dim from hidden/heads would be too small and under-size the
        // q scratch (mla_assemble_q_batched writes nope+rope per head).
        if cfg.qk_nope_head_dim == 0 || cfg.qk_rope_head_dim == 0 || cfg.v_head_dim == 0 {
            panic!(
                "MLA config (kv_lora_rank={}) requires qk_nope_head_dim, qk_rope_head_dim and v_head_dim to be > 0",
                cfg.kv_lora_rank
            );
        }
        cfg.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        cfg.n_kv_heads = cfg.n_heads;
    }
    // MoE (Qwen2.5-MoE style): num_experts / num_experts_per_tok.
    cfg.num_experts = v["num_experts"].as_u64().unwrap_or(0) as usize;
    cfg.num_experts_per_tok = v["num_experts_per_tok"].as_u64().unwrap_or(0) as usize;
    // Qwen-MoE expert FFN width (moe_intermediate_size); 0 = use intermediate_size.
    cfg.moe_intermediate_size = v["moe_intermediate_size"].as_u64().unwrap_or(0) as usize;
    // Qwen3 QK-norm: HF configs express it via `model_type` ("qwen3" /
    // "qwen3_moe") rather than an explicit flag, so default to ON for those
    // and honor an explicit `use_qk_norm` / `qk_norm` key when present.
    cfg.qk_norm = v["model_type"]
        .as_str()
        .is_some_and(|t| t.starts_with("qwen3"));
    if let Some(qk) = v["use_qk_norm"]
        .as_bool()
        .or_else(|| v["qk_norm"].as_bool())
    {
        cfg.qk_norm = qk;
    }
    // MACH_MOE_GROUPED=0 (default on): batched-MoE decode falls back to the
    // hipBLAS host loop. Parsed HERE, in the server's env-knob area, not in
    // the library (the library reads no MoE env; the field lives on Config).
    cfg.moe_grouped = std::env::var("MACH_MOE_GROUPED")
        .map(|x| x != "0")
        .unwrap_or(true);
    // MACH_STEP_PROFILE=1 (diagnostic): per-layer attention/MoE HIP event
    // bracketing, reported after each decode step.
    cfg.step_profile = std::env::var("MACH_STEP_PROFILE").is_ok_and(|v| v != "0");
    cfg
}

/// Total weight-payload size for the VRAM preflight: a single checkpoint file,
/// or the sum of every `*.safetensors` shard when `path` is a directory of
/// shards (Qwen-8B+ checkpoints ship as 5..65 files). A bare directory's
/// `metadata().len()` is ~0 on Windows, so using it directly would let the
/// preflight pass while the actual upload OOMs. Best-effort: unreadable shard
/// entries are skipped, and a directory with no shards sums to 0.
#[cfg(feature = "hip")]
fn model_file_bytes(path: &std::path::Path) -> u64 {
    if let Ok(entries) = std::fs::read_dir(path) {
        // Directory of shards: sum every `*.safetensors`; non-shard files and
        // unreadable entries are skipped (best-effort preflight estimate).
        entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum()
    } else {
        // Single checkpoint file (read_dir on a file path fails).
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    }
}

/// Rough device-memory estimate for the preflight: weight file + KV cache +
/// 256MiB scratch margin (+ draft model in spec mode). MLA uses the expanded
/// per-head KV cache (always f32); dense uses the GQA formula with the dtype's
/// element size. Sharded weight files are counted via [`model_file_bytes`];
/// hipBLAS workspace and compiled kernels are not counted; the margin covers
/// today's scenarios.
#[cfg(feature = "hip")]
fn estimate_vram(
    cfg: &Config,
    capacity: usize,
    file_bytes: u64,
    draft: Option<(&Config, u64)>,
    q4: bool,
    fp8: bool,
    q4_device: bool,
) -> u64 {
    let kv_elem = if cfg.dtype == ModelDType::F16 { 2 } else { 4 };
    let kv = if cfg.kv_lora_rank > 0 {
        capacity
            * cfg.max_seq_len
            * cfg.n_heads
            * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim)
            * 4
    } else {
        capacity * cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim * kv_elem * 2
    };
    // MACH_Q4 (host-side storage): the server loads standard BF16/F16
    // safetensors and quantizes at load; the device holds dequantized f16 =
    // the same bytes as the file. (The x4 device multiplier would only apply
    // to checkpoints already stored as packed int4, which the server never
    // reads.) MACH_Q4_DEVICE keeps the expert pool packed int4 on the device
    // (~0.28x the BF16 file bytes incl. scales) while non-expert weights
    // dequantize to f16 (~1.0x); MoE checkpoints are expert-dominated, so
    // x0.3 is a safe estimate (the 30B measured ~16.5GB device for a 61GB
    // file). FP8 stores packed E4M3 (1 byte/weight) but the device holds
    // dequantized f16 (2 bytes/weight); x2 is the exact device multiplier.
    let weight = if q4_device {
        file_bytes * 3 / 10
    } else if fp8 {
        file_bytes * 2
    } else {
        file_bytes
    };
    let mut est = weight + (kv * cfg.n_layers) as u64 + (256 << 20);
    if let Some((dcfg, dfb)) = draft {
        let dkv_elem = if dcfg.dtype == ModelDType::F16 { 2 } else { 4 };
        let dkv = if dcfg.kv_lora_rank > 0 {
            capacity
                * dcfg.max_seq_len
                * dcfg.n_heads
                * (dcfg.qk_nope_head_dim + dcfg.qk_rope_head_dim + dcfg.v_head_dim)
                * 4
        } else {
            capacity * dcfg.max_seq_len * dcfg.n_kv_heads * dcfg.head_dim * dkv_elem * 2
        };
        est += dfb + (dkv * dcfg.n_layers) as u64;
    }
    est
}

/// Pure validation of a raw `MACH_TPP` value against `cfg` — testable
/// without touching process env (see `parse_paged_tpp` for the fatal
/// wrapper). Missing/non-numeric values fall back to the default 64
/// (non-numeric warns); `0` and non-divisors of `max_seq_len` are fatal.
// Only the hip path calls this; kept un-gated so the CPU test below covers it.
#[cfg_attr(not(feature = "hip"), allow(dead_code))]
fn validate_paged_tpp(
    cfg: &mach_model::config::Config,
    raw: Option<&str>,
) -> Result<usize, String> {
    let tpp: usize = match raw {
        None => 64,
        Some(v) => match v.parse() {
            Ok(t) => t,
            Err(_) => {
                eprintln!("warning: MACH_TPP={v} is not a number; using default 64");
                64
            }
        },
    };
    if tpp == 0 || !cfg.max_seq_len.is_multiple_of(tpp) {
        return Err(format!(
            "MACH_TPP={tpp} is invalid: must be a non-zero divisor of max_seq_len {}",
            cfg.max_seq_len
        ));
    }
    Ok(tpp)
}

/// Parses and validates `MACH_TPP` for a branch that actually engages paged
/// KV (`MACH_PAGED` already checked by the caller). Runs BEFORE any weight
/// load: a bad value fails fast instead of aborting after the multi-minute
/// load. The MACH_SPEC / MoE-offload branches ignore MACH_PAGED (warned)
/// and never call this, so a stale value must not abort them. Returns
/// `None` (after reporting) for a fatal configuration — the caller degrades.
#[cfg(feature = "hip")]
fn parse_paged_tpp(cfg: &Config) -> Option<usize> {
    match validate_paged_tpp(cfg, std::env::var("MACH_TPP").ok().as_deref()) {
        Ok(tpp) => Some(tpp),
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(1);
        }
    }
}

/// One-shot diagnostic report (`mach-server doctor`): OS/host, HIP/GPU/VRAM,
/// MACH_* env, model files and a VRAM estimate. Exits 0 even when HIP is
/// missing (the report explains what to fix). Reuses the preflight queries.
#[cfg(feature = "hip")]
fn run_doctor() {
    let rev = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    println!("mach-server {}", env!("CARGO_PKG_VERSION"));
    println!("git rev: {}", rev.as_deref().unwrap_or("n/a"));
    println!("os: {} {}", std::env::consts::OS, std::env::consts::ARCH);
    println!(
        "host cpus: {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0)
    );
    println!(
        "MACH_HIP_PATH: {:?}",
        std::env::var("MACH_HIP_PATH").unwrap_or_default()
    );
    println!("MACH_* env:");
    for (k, v) in std::env::vars().filter(|(k, _)| k.starts_with("MACH_")) {
        println!("  {k}={v}");
    }

    match hip::hip() {
        Err(e) => {
            println!("HIP runtime: UNAVAILABLE ({e})");
            println!("  -> install ROCm and/or set MACH_HIP_PATH to the ROCm bin dir");
        }
        Ok(h) => match hip::device_count() {
            Err(e) => println!("device_count: ERROR ({e})"),
            Ok(n) if n <= 0 => println!("device_count: 0 (no HIP device)"),
            Ok(n) => {
                println!("device_count: {n}");
                for d in 0..n {
                    let name = hip::device_name(d).unwrap_or_else(|e| format!("<{e}>"));
                    println!("  gpu[{d}]: {name}");
                }
                match hip::mem_info() {
                    Ok((free, total)) => {
                        let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
                        println!(
                            "vram: {:.2} GiB free / {:.2} GiB total",
                            gib(free as u64),
                            gib(total as u64)
                        );
                        let _ = (h, gib);
                    }
                    Err(e) => println!("vram: mem_info error ({e})"),
                }
            }
        },
    }

    let root = std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into());
    let model_name = std::env::var("MACH_MODEL").unwrap_or_else(|_| "qwen-0.5b.safetensors".into());
    let config_name = std::env::var("MACH_CONFIG").unwrap_or_else(|_| "qwen-config.json".into());
    println!("models dir: {root}");
    let root = std::path::PathBuf::from(root);
    for (label, f) in [("model", &model_name), ("config", &config_name)] {
        let path = root.join(f);
        match std::fs::metadata(&path) {
            Ok(m) => println!("  {label} {f}: {} bytes", m.len()),
            Err(e) => println!("  {label} {f}: MISSING ({e})"),
        }
    }
    // Best-effort VRAM estimate; the server preflight is authoritative.
    let cfg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        config_from_json(&root.join(&config_name))
    }))
    .ok();
    if let Some(cfg) = cfg {
        let fb = model_file_bytes(&root.join(&model_name));
        let cap = std::env::var("MACH_CAPACITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let q4 = std::env::var("MACH_Q4").is_ok_and(|v| v != "0");
        let fp8 = std::env::var("MACH_FP8").is_ok_and(|v| v != "0");
        let q4_device = std::env::var("MACH_Q4_DEVICE").is_ok_and(|v| v != "0");
        let need = estimate_vram(&cfg, cap, fb, None, q4, fp8, q4_device);
        let gib = need as f64 / (1024.0 * 1024.0 * 1024.0);
        println!(
            "estimate: d_model={} layers={} experts={} need ~{:.2} GiB (capacity {cap})",
            cfg.d_model, cfg.n_layers, cfg.num_experts, gib
        );
    }
}

#[cfg(feature = "hip")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `mach-server doctor` / `--version`: one-shot diagnostics, no model load.
    match std::env::args().nth(1).as_deref() {
        Some("--version") | Some("-V") => {
            println!("mach-server {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        Some("doctor") | Some("--doctor") => {
            run_doctor();
            return Ok(());
        }
        _ => {}
    }
    let root = std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into());
    let model_name = std::env::var("MACH_MODEL").unwrap_or_else(|_| "qwen-0.5b.safetensors".into());
    let config_name = std::env::var("MACH_CONFIG").unwrap_or_else(|_| "qwen-config.json".into());
    let tokenizer_name =
        std::env::var("MACH_TOKENIZER").unwrap_or_else(|_| "tokenizer.json".into());
    let capacity = std::env::var("MACH_CAPACITY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let prefill_rows = std::env::var("MACH_PREFILL_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let moe_slots = std::env::var("MACH_MOE_SLOTS")
        .ok()
        .and_then(|s| s.parse().ok());
    // Storage-Q4 mode: weights stay packed int4 on the host (dequantized to
    // f16 per tensor on the device), cutting host RAM ~4x vs f32.
    let q4 = std::env::var("MACH_Q4").is_ok_and(|v| v != "0");
    // Q4-on-device: the MoE expert pool stays raw Q4 on the device (in-kernel
    // dequant) instead of the dequantized f16 pool — the memory path for
    // 30B-class checkpoints whose f16 experts would not fit in VRAM.
    let q4_device = std::env::var("MACH_Q4_DEVICE").is_ok_and(|v| v != "0");
    // Storage-FP8 mode: weights stay E4M3 on the host (dequantized to f16
    // per tensor on the device), cutting host RAM ~2x vs f16 / ~4x vs f32.
    let fp8 = std::env::var("MACH_FP8").is_ok_and(|v| v != "0");
    let addr = std::env::var("MACH_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
    // Compute dtype: default fp16 (2x+ GEMM, verified vs fp32), MACH_DTYPE=f32
    // opts out. bf16 is not wired yet.
    let dtype = std::env::var("MACH_DTYPE").unwrap_or_else(|_| "f16".into());
    let spec = std::env::var("MACH_SPEC").is_ok();
    let spec_k = std::env::var("MACH_SPEC_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let draft_name = std::env::var("MACH_DRAFT").unwrap_or_else(|_| "qwen-0.5b.safetensors".into());
    let draft_config =
        std::env::var("MACH_DRAFT_CONFIG").unwrap_or_else(|_| "qwen-config.json".into());

    let root = PathBuf::from(root);
    let mut cfg = config_from_json(&root.join(&config_name));
    match dtype.as_str() {
        "f32" => cfg.dtype = ModelDType::F32,
        "f16" => cfg.dtype = ModelDType::F16,
        other => {
            eprintln!("MACH_DTYPE must be f32 or f16, got {other:?}");
            std::process::exit(1);
        }
    }
    if q4 {
        if cfg.dtype != ModelDType::F16 {
            eprintln!(
                "MACH_Q4=1 requires dtype f16 (Q4 dequantizes to f16 on device); set MACH_DTYPE=f16 or drop MACH_Q4"
            );
            std::process::exit(1);
        }
        if spec {
            eprintln!(
                "MACH_Q4=1 and MACH_SPEC are mutually exclusive (spec mode loads a second model)"
            );
            std::process::exit(1);
        }
        if moe_slots.is_some() {
            eprintln!(
                "MACH_Q4=1 and MACH_MOE_SLOTS are mutually exclusive (cpu-backend offload needs f32 Weights)"
            );
            std::process::exit(1);
        }
    }
    if q4_device && !q4 {
        eprintln!(
            "warning: MACH_Q4_DEVICE=1 requires MACH_Q4=1 (it selects the Q4 expert-pool layout); ignoring"
        );
    }

    if fp8 {
        if q4 {
            eprintln!("MACH_FP8=1 and MACH_Q4 are mutually exclusive; choose one storage format");
            std::process::exit(1);
        }
        if cfg.dtype != ModelDType::F16 {
            eprintln!(
                "MACH_FP8=1 requires dtype f16 (FP8 dequantizes to f16 on device); set MACH_DTYPE=f16 or drop MACH_FP8"
            );
            std::process::exit(1);
        }
        if spec {
            eprintln!(
                "MACH_FP8=1 and MACH_SPEC are mutually exclusive (spec mode loads a second model)"
            );
            std::process::exit(1);
        }
        if moe_slots.is_some() {
            eprintln!(
                "MACH_FP8=1 and MACH_MOE_SLOTS are mutually exclusive (cpu-backend offload needs f32 Weights)"
            );
            std::process::exit(1);
        }
    }

    // Preflight (before any heavy loading): HIP runtime + device + VRAM. A
    // missing/busy device or grossly insufficient memory should fail fast with
    // a readable error, not hang the host during the ~36 serial hiprtc kernel
    // compiles that follow.
    let hip = match hip::hip() {
        Ok(h) => h,
        Err(e) => {
            eprintln!(
                "HIP runtime unavailable: {e}\n  set MACH_HIP_PATH to the ROCm bin dir if needed"
            );
            std::process::exit(1);
        }
    };
    let devices = match hip::device_count() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("hipGetDeviceCount failed: {e}");
            std::process::exit(1);
        }
    };
    if devices <= 0 {
        eprintln!("no HIP device found (device_count={devices}); refusing to load a model");
        std::process::exit(1);
    }
    let (free, total) = match hip::mem_info() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("hipMemGetInfo failed: {e}");
            std::process::exit(1);
        }
    };
    // NOTE: hipBLAS workspace and compiled kernels are not counted; the 256MiB
    // margin covers today's tiny/1.5B scenarios.
    let file_bytes = model_file_bytes(&root.join(&model_name));
    let draft_est = if spec {
        let dfb = model_file_bytes(&root.join(&draft_name));
        let mut dcfg = config_from_json(&root.join(&draft_config));
        match dtype.as_str() {
            "f32" => dcfg.dtype = ModelDType::F32,
            _ => dcfg.dtype = ModelDType::F16,
        }
        Some((dcfg, dfb))
    } else {
        None
    };
    // In Q4 mode the device holds dequantized f16 weights (~4x the packed int4
    // file size), so the preflight weight term must account for that or it can
    // pass while the upload OOMs.
    let estimate = estimate_vram(
        &cfg,
        capacity,
        file_bytes,
        draft_est.as_ref().map(|(c, b)| (c, *b)),
        q4,
        fp8,
        q4_device,
    );
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "GPU preflight: device_count={devices}, VRAM free {:.2}GiB / {:.2}GiB, estimated need {:.2}GiB",
        gib(free as u64),
        gib(total as u64),
        gib(estimate)
    );
    if q4 {
        println!(
            "storage Q4: host weights stay packed int4 (~4x smaller than f32); device still holds dequantized f16 weights"
        );
    }
    if estimate > free as u64 {
        eprintln!(
            "insufficient VRAM: need ~{:.2}GiB but only {:.2}GiB free; lower MACH_CAPACITY / MACH_PREFILL_ROWS or use a smaller model",
            gib(estimate),
            gib(free as u64)
        );
        std::process::exit(1);
    }

    // Bind early: a port conflict fails here (before the multi-minute model
    // load + kernel compile) instead of after.
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("bound http://{addr} (preflight passed)");

    // Load the real tokenizer when available; fall back to naive bytes.
    let tok_path = root.join(&tokenizer_name);
    let tok = if tok_path.exists() {
        let t = Tokenizer::from_path(&tok_path).expect("load tokenizer");
        println!(
            "tokenizer: {} (vocab {})",
            tok_path.display(),
            t.vocab_size()
        );
        Some(std::sync::Arc::new(t))
    } else {
        println!("tokenizer {tok_path:?} not found; using naive byte mapping");
        None
    };

    // Q4 mode loads packed int4 weights directly (host RAM stays small) and
    // spawns the Q4 engine; f16/f32 and spec modes keep the f32 host load.
    // Paged mode is wired for the plain-Weights and storage-quantized paths
    // (device f16 served by the f16 paged kernels); MACH_SPEC remains
    // contiguous-only (warned).
    let paged_requested = std::env::var("MACH_PAGED").is_ok_and(|v| v != "0");
    // Resolve paged engagement BEFORE any weight load: a stale/invalid
    // MACH_TPP or a paged-incompatible checkpoint must fail fast (or degrade
    // with a warning) up front, not abort after the multi-minute load.
    // Paged KV serves MLA in F32 only; the quantized branches always serve
    // KV in device f16, so quantized MLA never qualifies. MACH_SPEC and
    // MoE-offload ignore MACH_PAGED (warned in-branch) and skip this. The
    // model-side paged_guards stay the authoritative last-resort checks.
    let paged_tpp = if paged_requested && !spec && moe_slots.is_none() {
        if cfg.kv_lora_rank > 0 && (q4 || fp8) {
            // Quantized builds force device dtype F16 (build_q4/build_fp8),
            // so a quantized MLA checkpoint can never qualify for paged KV
            // (MLA is F32-only). Warn and degrade before the load.
            eprintln!(
                "warning: MACH_PAGED is unsupported with this model/mode combination (paged KV serves MLA in F32 only); serving continuous"
            );
            None
        } else {
            match parse_paged_tpp(&cfg) {
                Some(tpp) => match BatchedModel::check_paged_support(&cfg, tpp) {
                    Ok(()) => Some(tpp),
                    // The authoritative model-side checks (page geometry,
                    // attention smem, dtype coverage) degrade up front —
                    // never abort after the multi-minute weight load.
                    Err(e) => {
                        eprintln!("warning: MACH_PAGED unsupported: {e}; serving continuous");
                        None
                    }
                },
                None => None, // invalid MACH_TPP: already reported and exited
            }
        }
    } else {
        None
    };
    let (engine, engine_handle) = if q4 {
        let wq4: WeightsQ4 =
            load_safetensors_q4(&root.join(&model_name), &cfg, true).expect("load q4 weights");
        println!(
            "model {model_name}: d_model={} layers={} heads={} kv={} vocab={} dtype={:?} (storage Q4; host weights stay packed int4, device dequantizes to f16)",
            cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.dtype
        );
        let eng = match paged_tpp {
            Some(tpp) => ServerEngine::with_paged(capacity, prefill_rows, tpp),
            None => ServerEngine::with_prefill_rows(capacity, prefill_rows),
        };
        let handle = if q4_device {
            eng.clone().spawn_q4_device(hip, cfg, wq4)?
        } else {
            eng.clone().spawn_q4(hip, cfg, wq4)?
        };
        (eng, handle)
    } else if fp8 {
        let wfp8: WeightsFp8 =
            load_safetensors_fp8(&root.join(&model_name), &cfg, true).expect("load fp8 weights");
        println!(
            "model {model_name}: d_model={} layers={} heads={} kv={} vocab={} dtype={:?} (storage FP8; host weights stay packed E4M3, device dequantizes to f16)",
            cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.dtype
        );
        let eng = match paged_tpp {
            Some(tpp) => ServerEngine::with_paged(capacity, prefill_rows, tpp),
            None => ServerEngine::with_prefill_rows(capacity, prefill_rows),
        };
        let handle = eng.clone().spawn_fp8(hip, cfg, wfp8)?;
        (eng, handle)
    } else if spec {
        if paged_requested {
            eprintln!(
                "warning: MACH_PAGED is ignored in MACH_SPEC mode (paged spec wiring is a follow-up)"
            );
        }
        let w: Weights =
            load_safetensors(&root.join(&model_name), &cfg, true).expect("load target weights");
        println!(
            "model {model_name}: d_model={} layers={} heads={} kv={} vocab={} dtype={:?}",
            cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.dtype
        );
        // Speculative-decoding mode: MACH_SPEC=1 serves greedy requests through
        // a draft + target engine (greedy-only; other params rejected).
        let mut dcfg = config_from_json(&root.join(&draft_config));
        // Two [prof] streams (draft + target) would interleave without any
        // role tag — disable the profiler on the draft.
        dcfg.step_profile = false;
        match dtype.as_str() {
            "f32" => dcfg.dtype = ModelDType::F32,
            "f16" => dcfg.dtype = ModelDType::F16,
            _ => {}
        }
        let dw: Weights =
            load_safetensors(&root.join(&draft_name), &dcfg, true).expect("load draft weights");
        println!(
            "draft {draft_name}: d_model={} layers={} dtype={:?} K={spec_k}",
            dcfg.d_model, dcfg.n_layers, dcfg.dtype
        );
        let eng = ServerEngine::with_spec(capacity, spec_k);
        let handle = eng.clone().spawn_spec(hip, cfg, w, dcfg, dw)?;
        (eng, handle)
    } else {
        let w: Weights =
            load_safetensors(&root.join(&model_name), &cfg, true).expect("load weights");
        println!(
            "model {model_name}: d_model={} layers={} heads={} kv={} vocab={} dtype={:?}",
            cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.dtype
        );
        let eng = if let Some(slots) = moe_slots {
            if paged_requested {
                eprintln!(
                    "warning: MACH_PAGED is ignored in MoE-offload mode (paged offload wiring is a follow-up)"
                );
            }
            ServerEngine::with_offload(capacity, prefill_rows, slots)
        } else if let Some(tpp) = paged_tpp {
            ServerEngine::with_paged(capacity, prefill_rows, tpp)
        } else {
            ServerEngine::with_prefill_rows(capacity, prefill_rows)
        };
        let handle = eng.clone().spawn(hip, cfg, w)?;
        (eng, handle)
    };
    let state = AppState {
        engine: engine.clone(),
        model: model_name,
        tok,
    };
    let app = router(state);
    println!(
        "mach-server listening on http://{addr} (capacity {capacity}, prefill rows {prefill_rows}{}{}{})",
        if q4 {
            ", storage Q4"
        } else if fp8 {
            ", storage FP8"
        } else {
            ""
        },
        if spec { ", spec-decode" } else { "" },
        if let Some(slots) = moe_slots {
            format!(", moe-offload slots={slots}")
        } else {
            String::new()
        }
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            println!("ctrl-c received; draining in-flight requests...");
            engine.shutdown();
        })
        .await?;

    // The engine thread drains queued + active sequences, then exits.
    engine_handle
        .join()
        .map_err(|_| "engine thread panicked during shutdown".to_string())?;
    println!("engine drained; exiting");
    Ok(())
}
#[cfg(not(feature = "hip"))]
fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("--version") | Some("-V") => {
            println!("mach-server {}", env!("CARGO_PKG_VERSION"));
        }
        Some("doctor") | Some("--doctor") => {
            println!("mach-server {}", env!("CARGO_PKG_VERSION"));
            println!("os: {} {}", std::env::consts::OS, std::env::consts::ARCH);
            println!("HIP: NOT COMPILED (build with --features hip)");
            for (k, v) in std::env::vars().filter(|(k, _)| k.starts_with("MACH_")) {
                println!("{k}={v}");
            }
        }
        _ => eprintln!(
            "mach-server requires the `hip` feature: cargo run -p mach-server --features hip"
        ),
    }
}

#[cfg(all(test, feature = "hip"))]
mod tests {
    use super::*;

    fn dense_cfg() -> Config {
        Config::llama(128, 2, 4, 2, 1024, 64)
    }

    fn mla_cfg() -> Config {
        Config::mla(128, 2, 4, 1024, 64, 32, 16, 16, 8, 16)
    }

    #[test]
    fn dense_estimate_includes_weights_kv_and_margin() {
        let cfg = dense_cfg();
        let est = estimate_vram(&cfg, 8, 1_000_000, None, false, false, false);
        // KV (f32) = capacity*max_seq*kv_heads*head_dim*4 per layer.
        let kv =
            (8 * cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim * 4 * 2 * cfg.n_layers) as u64;
        assert_eq!(est, 1_000_000 + kv + (256 << 20));
    }

    #[test]
    fn f16_dense_uses_two_byte_kv() {
        let mut cfg = dense_cfg();
        cfg.dtype = ModelDType::F16;
        let f16 = estimate_vram(&cfg, 8, 0, None, false, false, false);
        cfg.dtype = ModelDType::F32;
        let f32 = estimate_vram(&cfg, 8, 0, None, false, false, false);
        // KV diff = layers * capacity*max_seq*kv_heads*head_dim*(4-2).
        let kv_diff =
            (cfg.n_layers * 8 * cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim * 2 * 2) as u64;
        assert_eq!(
            f32 - f16,
            kv_diff,
            "f32 KV must exceed f16 KV by the elem diff"
        );
    }

    #[test]
    fn mla_estimate_uses_expanded_per_head_kv() {
        let cfg = mla_cfg();
        // MLA KV/layer is f32: capacity*max_seq*heads*(nope+rope+v_hd)*4.
        let kv = (8
            * cfg.max_seq_len
            * cfg.n_heads
            * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim)
            * 4
            * cfg.n_layers) as u64;
        assert_eq!(
            estimate_vram(&cfg, 8, 0, None, false, false, false),
            kv + (256 << 20)
        );
    }

    #[test]
    fn spec_adds_draft_weights_and_kv() {
        let tcfg = dense_cfg();
        let dcfg = dense_cfg();
        let base = estimate_vram(&tcfg, 8, 1_000, None, false, false, false);
        let spec = estimate_vram(&tcfg, 8, 1_000, Some((&dcfg, 500)), false, false, false);
        let dkv =
            (dcfg.n_layers * 8 * dcfg.max_seq_len * dcfg.n_kv_heads * dcfg.head_dim * 4 * 2) as u64;
        assert_eq!(spec - base, 500 + dkv);
    }

    /// Removes the temp file on drop (also on test panic).
    struct TempFile(std::path::PathBuf);
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn parse_json(json: &str) -> Config {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "machserve_cfg_test_{}_{}.json",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, json).unwrap();
        let _guard = TempFile(path.clone());
        config_from_json(&path)
    }

    #[test]
    fn config_parses_dense_defaults() {
        let cfg = parse_json(
            r#"{"hidden_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":1024,"intermediate_size":512,"max_position_embeddings":64}"#,
        );
        assert_eq!(cfg.kv_lora_rank, 0);
        assert_eq!(cfg.q_lora_rank, 0);
        assert_eq!(cfg.num_experts, 0);
        assert_eq!(cfg.head_dim, 32, "dense head_dim = hidden/heads");
        assert_eq!(cfg.n_kv_heads, 2);
    }

    #[test]
    fn config_parses_mla_hyperparams() {
        let cfg = parse_json(
            r#"{"hidden_size":5120,"num_hidden_layers":2,"num_attention_heads":128,"vocab_size":102400,"max_position_embeddings":4096,"q_lora_rank":1536,"kv_lora_rank":512,"qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128}"#,
        );
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.q_lora_rank, 1536);
        assert_eq!(
            cfg.head_dim,
            128 + 64,
            "MLA head_dim must be qk_nope_head_dim + qk_rope_head_dim"
        );
        assert_eq!(cfg.n_kv_heads, 128, "MLA n_kv_heads == n_heads");
        assert_eq!(cfg.num_experts, 0);
    }

    #[test]
    fn config_parses_explicit_head_dim() {
        // Qwen3-30B-A3B-style: n_heads*head_dim (32*128=4096) is wider than
        // hidden (2048); the explicit head_dim must win over hidden/n_heads.
        let cfg = parse_json(
            r#"{"hidden_size":2048,"num_hidden_layers":48,"num_attention_heads":32,"num_key_value_heads":4,"vocab_size":151936,"intermediate_size":6144,"moe_intermediate_size":768,"num_experts":128,"num_experts_per_tok":8,"head_dim":128,"max_position_embeddings":40960}"#,
        );
        assert_eq!(cfg.head_dim, 128, "explicit head_dim wins");
        assert_eq!(cfg.n_heads * cfg.head_dim, 4096);
        assert_eq!(cfg.n_kv_heads * cfg.head_dim, 512);
        assert_eq!(cfg.num_experts, 128);
        assert_eq!(cfg.expert_size(), 768);
    }

    #[test]
    fn config_parses_moe_hyperparams() {
        let cfg = parse_json(
            r#"{"hidden_size":1024,"num_hidden_layers":2,"num_attention_heads":8,"num_key_value_heads":2,"vocab_size":151936,"intermediate_size":512,"max_position_embeddings":2048,"num_experts":64,"num_experts_per_tok":8,"moe_intermediate_size":256}"#,
        );
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.moe_intermediate_size, 256);
        assert_eq!(
            cfg.expert_size(),
            256,
            "Qwen-MoE experts use moe_intermediate_size"
        );
        assert_eq!(cfg.kv_lora_rank, 0);
    }

    #[test]
    #[should_panic(expected = "requires qk_nope_head_dim")]
    fn config_rejects_mla_missing_dims() {
        parse_json(
            r#"{"hidden_size":5120,"num_hidden_layers":2,"num_attention_heads":128,"vocab_size":102400,"max_position_embeddings":4096,"kv_lora_rank":512}"#,
        );
    }

    #[test]
    fn model_file_bytes_sums_shard_dir() {
        let dir =
            std::env::temp_dir().join(format!("mach_preflight_shard_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // A single checkpoint file: metadata length, not a dir sum.
        let single = dir.join("model.safetensors");
        std::fs::write(&single, vec![0u8; 1234]).unwrap();
        assert_eq!(model_file_bytes(&single), 1234);
        // A directory of shards: every *.safetensors counted, non-shards not.
        std::fs::write(
            dir.join("model-00001-of-00002.safetensors"),
            vec![0u8; 1000],
        )
        .unwrap();
        std::fs::write(
            dir.join("model-00002-of-00002.safetensors"),
            vec![0u8; 2000],
        )
        .unwrap();
        std::fs::write(dir.join("config.json"), vec![0u8; 999]).unwrap();
        assert_eq!(model_file_bytes(&dir), 4234); // 1234 single + 1000 + 2000 shards
        // A directory with no shards sums to 0, not the dir's own size.
        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert_eq!(model_file_bytes(&empty), 0);
        std::fs::write(empty.join("config.json"), vec![0u8; 64]).unwrap();
        assert_eq!(model_file_bytes(&empty), 0);
        // Missing path falls back to 0.
        assert_eq!(model_file_bytes(&dir.join("nope.safetensors")), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn q4_scales_weight_term_by_four() {
        let cfg = dense_cfg();
        let base = estimate_vram(&cfg, 8, 1_000_000, None, false, false, false);
        // MACH_Q4 loads standard BF16/F16 files and quantizes at load: the
        // device holds f16 = the same bytes as the file, so the weight term
        // is unchanged vs dense.
        let q4 = estimate_vram(&cfg, 8, 1_000_000, None, true, false, false);
        assert_eq!(q4, base);
        // MACH_Q4_DEVICE keeps the expert pool packed on the device: the
        // weight term is 0.3x the file bytes.
        let q4d = estimate_vram(&cfg, 8, 1_000_000, None, true, false, true);
        assert_eq!(q4d, base - 700_000);
    }

    #[test]
    fn fp8_scales_weight_term_by_two() {
        let cfg = dense_cfg();
        let base = estimate_vram(&cfg, 8, 1_000_000, None, false, false, false);
        let fp8 = estimate_vram(&cfg, 8, 1_000_000, None, false, true, false);
        // FP8 stores E4M3 (1 byte/weight) but the device holds dequantized f16
        // (2 bytes/weight): the weight term must be x2, or the preflight can
        // pass while the upload OOMs (regression).
        assert_eq!(fp8 - base, 1_000_000);
    }
}

/// Ungated: `validate_paged_tpp` is pure cfg logic, so CPU CI covers it.
#[cfg(test)]
mod paged_tpp_tests {
    use super::*;

    #[test]
    fn validate_paged_tpp_defaults_warnings_and_fatals() {
        let cfg = mach_model::config::Config::tiny(); // max_seq_len 256
        // Missing env: default 64 (divides 256).
        assert_eq!(validate_paged_tpp(&cfg, None).unwrap(), 64);
        // Non-numeric: warned fallback to 64 (not fatal).
        assert_eq!(validate_paged_tpp(&cfg, Some("64x")).unwrap(), 64);
        // Zero: fatal.
        assert!(validate_paged_tpp(&cfg, Some("0")).is_err());
        // Non-divisor: fatal.
        assert!(validate_paged_tpp(&cfg, Some("48")).is_err());
        // Valid custom value.
        assert_eq!(validate_paged_tpp(&cfg, Some("128")).unwrap(), 128);
    }
}
