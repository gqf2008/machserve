//! MachServe OpenAI-compatible server binary.
//!
//! Loads a safetensors Llama/Qwen checkpoint and serves the continuous-batching
//! engine over HTTP:
//!   cargo run -p mach-server --release --features hip
//!
//! Env: MACH_MODELS (default ".models"), MACH_MODEL (default
//! "qwen-0.5b.safetensors"; an HF shard filename resolves to its directory),
//! MACH_CONFIG (optional; defaults to `config.json` beside the checkpoint,
//! then the legacy `qwen-config.json`),
//! MACH_CAPACITY (default 64), MACH_PREFILL_ROWS (default 512),
//! MACH_ADDR (default "127.0.0.1:8080"), MACH_Q4 / MACH_FP8 (storage-quantized
//! host weights: int4 or E4M3, dequantized to f16 on the device),
//! MACH_Q4_DEVICE=1 (with MACH_Q4: keep the MoE expert pool in raw Q4 on the
//! device, dequantized in-kernel — the memory path for 30B-class checkpoints
//! whose f16 experts would not fit in VRAM) or MACH_Q4_DEVICE=2 (=1 PLUS every
//! big dense tensor — q/k/v/o projections, dense MLP, GDN projections,
//! embedding, lm_head — raw Q4 on device, `gemv_q4` in-kernel dequant; the
//! memory path for DENSE 27B-class checkpoints whose dequantized f16 weights
//! ~54GB would not fit in VRAM; all-Q4 ~13.5GB does),
//! MACH_PAGED=1 (paged-KV engine with cross-request prefix reuse) with
//! MACH_TPP (KV page size in tokens, default 64; only read by the modes that
//! engage paged KV — plain, Q4 and FP8 non-MLA). The paged-path safety cap is
//! MACH_PREFILL_ROWS<=64 / MACH_CAPACITY<=64 until controlled 512-row GPU
//! validation of the tiled attention path lands.
//! Limitations: paged KV serves
//! MLA models in F32 only (quantized MLA warns and falls back to continuous),
//! and MACH_SPEC / MoE-offload modes ignore MACH_PAGED (warned).
//! MACH_MOE_GROUPED=0 (default on; disable the batched-MoE decode grouped
//! GEMV device path — A/B switch and ops lever). NOTE: Q4-on-device models
//! (MACH_Q4_DEVICE>=1) have no f16/f32 expert copy, so they ALWAYS run the
//! grouped path and this knob is a no-op for them.
//! MACH_STEP_PROFILE=1 (diagnostic): per-layer attention/MoE HIP event
//! bracketing printed after each decode step.
//! MACH_GRAPH=1 (experimental, #103): capture the decode step as a HIP graph
//! and replay it (per active-row-count bucket). Raw decode wins ~7% at short
//! context; the service chain shows no end-to-end gain (host bookkeeping
//! dominates), and on ROCm 6.2 / Windows long-lived replays degenerate after
//! ~1-4k launches (silent GPU-side no-op — see docs/roadmap.md and the
//! qwen3_30b_graph_churn example). Off by default.
#[cfg(feature = "hip")]
use mach_kernel_sys::hip;
#[cfg(feature = "hip")]
use mach_model::batched::BatchedModel;
#[cfg(feature = "hip")]
use mach_model::config::ModelDType;
#[cfg(feature = "hip")]
use mach_model::image_processor::ImageProcessorConfig;
#[cfg(feature = "hip")]
use mach_model::loader::{
    load_safetensors, load_safetensors_fp8, load_safetensors_q4, load_vision_weights,
    validate_checkpoint,
};
#[cfg(feature = "hip")]
use mach_model::tokenizer::Tokenizer;
#[cfg(feature = "hip")]
use mach_model::vision::VisionConfig;
#[cfg(feature = "hip")]
use mach_model::{Config, Weights, WeightsFp8, WeightsQ4};
#[cfg(feature = "hip")]
use mach_server::startup::{
    PAGED_PREFILL_ROWS_MAX, cap_paged_prefill_rows, config_from_json, estimate_vram,
    model_file_bytes, resolve_preprocessor_path,
};
#[cfg(feature = "hip")]
use mach_server::{AppState, ChatFormat, ImageRuntimeConfig, ServerEngine, VisionSetup, router};
#[cfg(any(feature = "hip", test))]
use std::ffi::OsStr;
#[cfg(any(feature = "hip", test))]
use std::path::{Path, PathBuf};

/// Picks the chat template for a checkpoint's `model_type`.
///
/// The template is a property of the checkpoint, not of the request: rendering
/// ChatML markers into a DeepSeek prompt makes the model treat them as prose.
/// Unknown model types keep the Qwen/ChatML default, which is what every model
/// served before DeepSeek used.
#[cfg(feature = "hip")]
fn chat_format_from_json(path: &std::path::Path) -> ChatFormat {
    let Ok(text) = std::fs::read_to_string(path) else {
        return ChatFormat::default();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return ChatFormat::default();
    };
    match v["model_type"].as_str() {
        Some(t) if t.starts_with("deepseek") => ChatFormat::DeepSeek,
        _ => ChatFormat::default(),
    }
}

/// True when `name` follows the Hugging Face multi-shard convention
/// (`model-00001-of-00018.safetensors`, also used with a `pytorch_model`
/// prefix). Such a file is only one piece of a checkpoint; the loader must be
/// given its containing directory or it will report unrelated tensors missing.
#[cfg(any(feature = "hip", test))]
fn is_hf_shard_file(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|x| x.to_str()) else {
        return false;
    };
    let Some(stem) = name.strip_suffix(".safetensors") else {
        return false;
    };
    let Some((prefix, total)) = stem.rsplit_once("-of-") else {
        return false;
    };
    let Some((stem, index)) = prefix.rsplit_once('-') else {
        return false;
    };
    !stem.is_empty()
        && !index.is_empty()
        && !total.is_empty()
        && index.chars().all(|c| c.is_ascii_digit())
        && total.chars().all(|c| c.is_ascii_digit())
}

/// Resolve a checkpoint input to the path the loaders actually need.
///
/// A directory stays a directory. A shard file is promoted to its parent so
/// `MACH_MODEL=model-00001-of-00018.safetensors` follows the documented HF
/// workflow instead of loading one shard in isolation. Standalone files stay
/// files, which keeps `.models/qwen-0.5b.safetensors` from absorbing every
/// unrelated checkpoint in the models root.
#[cfg(any(feature = "hip", test))]
fn resolve_checkpoint_path(root: &Path, model: &str) -> PathBuf {
    let path = root.join(model);
    if path.is_file() && is_hf_shard_file(&path) {
        return path.parent().unwrap_or(root).to_path_buf();
    }
    path
}

/// Resolve the checkpoint config. An explicit `MACH_CONFIG` always wins.
/// Otherwise use `config.json` next to the model (directory or shard parent),
/// falling back to the legacy models-root `qwen-config.json`.
#[cfg(any(feature = "hip", test))]
fn resolve_config_path(root: &Path, model: &str, explicit: Option<&OsStr>) -> PathBuf {
    if let Some(name) = explicit {
        return root.join(name);
    }
    let checkpoint = resolve_checkpoint_path(root, model);
    let dir = if checkpoint.is_dir() {
        checkpoint.as_path()
    } else {
        checkpoint.parent().unwrap_or(root)
    };
    let local = dir.join("config.json");
    if local.is_file() {
        local
    } else {
        root.join("qwen-config.json")
    }
}

/// Resolve the tokenizer beside a checkpoint directory/shard parent, with an
/// explicit `MACH_TOKENIZER` override and the legacy models-root fallback.
#[cfg(any(feature = "hip", test))]
fn resolve_tokenizer_path(root: &Path, model: &str, explicit: Option<&OsStr>) -> PathBuf {
    if let Some(name) = explicit {
        return root.join(name);
    }
    let checkpoint = resolve_checkpoint_path(root, model);
    let dir = if checkpoint.is_dir() {
        checkpoint.as_path()
    } else {
        checkpoint.parent().unwrap_or(root)
    };
    let local = dir.join("tokenizer.json");
    if local.is_file() {
        local
    } else {
        root.join("tokenizer.json")
    }
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
    let explicit_config = std::env::var_os("MACH_CONFIG");
    let explicit_tokenizer = std::env::var_os("MACH_TOKENIZER");
    println!("models dir: {root}");
    let root = std::path::PathBuf::from(root);
    let checkpoint_path = resolve_checkpoint_path(&root, &model_name);
    let config_path = resolve_config_path(&root, &model_name, explicit_config.as_deref());
    let tokenizer_path = resolve_tokenizer_path(&root, &model_name, explicit_tokenizer.as_deref());
    for (label, path) in [
        ("model", &checkpoint_path),
        ("config", &config_path),
        ("tokenizer", &tokenizer_path),
    ] {
        let size = if label == "model" {
            model_file_bytes(path)
        } else {
            std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
        };
        match std::fs::metadata(path) {
            Ok(_) => println!("  {label} {}: {size} bytes", path.display()),
            Err(e) => println!("  {label} {}: MISSING ({e})", path.display()),
        }
    }
    match validate_checkpoint(&checkpoint_path) {
        Ok(layout) => println!(
            "  checkpoint layout: {} shards, {} tensors, {} payload bytes",
            layout.shards, layout.tensors, layout.payload_bytes
        ),
        Err(e) => println!("  checkpoint layout: ERROR ({e})"),
    }
    // Best-effort VRAM estimate; the server preflight is authoritative.
    let cfg = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        config_from_json(&config_path)
    }))
    .ok();
    if let Some(cfg) = cfg {
        let fb = model_file_bytes(&checkpoint_path);
        let cap = std::env::var("MACH_CAPACITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let fp8 = std::env::var("MACH_FP8").is_ok_and(|v| v != "0");
        let q4_device = std::env::var("MACH_Q4_DEVICE").is_ok_and(|v| v != "0");
        // Doctor is diagnostic-only: mirror the runtime's reachable INT8-KV
        // constraints, but fall back to the conservative f16 estimate for an
        // invalid combination instead of aborting the whole doctor run.
        let kv_int8 = std::env::var("MACH_KV").is_ok_and(|v| v == "int8")
            && std::env::var("MACH_Q4").is_ok_and(|v| v != "0")
            && std::env::var("MACH_Q4_DEVICE").is_ok_and(|v| v == "2")
            && !std::env::var("MACH_PAGED").is_ok_and(|v| v != "0")
            && cfg.kv_lora_rank == 0;
        let need = estimate_vram(&cfg, cap, fb, None, fp8, q4_device, kv_int8);
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
    let explicit_config = std::env::var_os("MACH_CONFIG");
    let explicit_tokenizer = std::env::var_os("MACH_TOKENIZER");
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
    // Q4-on-device level: 1 keeps the MoE expert pool raw Q4 on the device
    // (in-kernel dequant); 2 additionally keeps EVERY big dense tensor raw
    // Q4 (`gemv_q4`) — the memory path for DENSE 27B-class checkpoints whose
    // dequantized f16 weights would not fit in VRAM.
    let q4_device = match std::env::var("MACH_Q4_DEVICE") {
        Ok(v) if v == "2" => 2u8,
        Ok(v) if v != "0" && !v.is_empty() => 1u8,
        _ => 0u8,
    };
    // Storage-FP8 mode: weights stay E4M3 on the host (dequantized to f16
    // per tensor on the device), cutting host RAM ~2x vs f16 / ~4x vs f32.
    let fp8 = std::env::var("MACH_FP8").is_ok_and(|v| v != "0");
    let paged_requested = std::env::var("MACH_PAGED").is_ok_and(|v| v != "0");
    let kv_int8 = match std::env::var("MACH_KV").as_deref() {
        Err(_) | Ok("") | Ok("0") | Ok("f16") => false,
        Ok("int8") => true,
        Ok(other) => {
            eprintln!("MACH_KV must be f16 or int8, got {other:?}");
            std::process::exit(1);
        }
    };
    if kv_int8 && (!q4 || q4_device != 2) {
        eprintln!("MACH_KV=int8 currently requires MACH_Q4=1 MACH_Q4_DEVICE=2");
        std::process::exit(1);
    }
    if kv_int8 && paged_requested {
        eprintln!("MACH_KV=int8 is not wired for MACH_PAGED yet; unset MACH_PAGED");
        std::process::exit(1);
    }
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
    let checkpoint_path = resolve_checkpoint_path(&root, &model_name);
    let config_path = resolve_config_path(&root, &model_name, explicit_config.as_deref());
    let tokenizer_path = resolve_tokenizer_path(&root, &model_name, explicit_tokenizer.as_deref());
    let mut cfg = config_from_json(&config_path);
    match dtype.as_str() {
        "f32" => cfg.dtype = ModelDType::F32,
        "f16" => cfg.dtype = ModelDType::F16,
        other => {
            eprintln!("MACH_DTYPE must be f32 or f16, got {other:?}");
            std::process::exit(1);
        }
    }
    if kv_int8 && cfg.kv_lora_rank > 0 {
        eprintln!("MACH_KV=int8 does not support MLA checkpoints yet (kv_lora_rank > 0)");
        std::process::exit(1);
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
    if q4_device != 0 && !q4 {
        eprintln!(
            "warning: MACH_Q4_DEVICE={} requires MACH_Q4=1 (it selects a Q4 device layout); ignoring",
            q4_device
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

    // Header/index validation is a few MB of I/O and happens before HIP
    // preflight or any multi-GB weight allocation, so a missing/truncated
    // 18-shard Qwen3.8 checkpoint fails fast instead of after a full load.
    let layout = validate_checkpoint(&checkpoint_path).unwrap_or_else(|e| {
        eprintln!("invalid checkpoint {}: {e}", checkpoint_path.display());
        std::process::exit(1);
    });
    println!(
        "checkpoint: {} shards, {} tensors, {:.2} GiB payload",
        layout.shards,
        layout.tensors,
        layout.payload_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
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
    let file_bytes = model_file_bytes(&checkpoint_path);
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
        fp8,
        q4_device != 0,
        kv_int8,
    );
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "GPU preflight: device_count={devices}, VRAM free {:.2}GiB / {:.2}GiB, estimated need {:.2}GiB",
        gib(free as u64),
        gib(total as u64),
        gib(estimate)
    );
    if q4 {
        match q4_device {
            2 => println!(
                "storage Q4 + dense Q4-on-device: host AND device weights stay packed int4 (in-kernel dequant)"
            ),
            1 => println!(
                "storage Q4 + Q4-on-device experts: the expert pool stays packed int4 on device; other weights dequantize to f16"
            ),
            _ => println!(
                "storage Q4: host weights stay packed int4 (~4x smaller than f32); device still holds dequantized f16 weights"
            ),
        }
    }
    if kv_int8 {
        println!(
            "INT8 KV: contiguous full-attention payload i8 + per-token/head f32 scales (exact preflight estimate)"
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
    // A directory checkpoint (HF layout) ships its own `tokenizer.json`, so
    // look there first: pointing MACH_MODEL at a directory must not silently
    // pick up some other model's vocabulary from the models root (a
    // DeepSeek checkpoint would then be served with Qwen's 151k vocab).
    let tok_path = tokenizer_path;
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
    // Safety cap: paged prefill batches above the empirically safe limit have
    // caused corrupted outputs and whole-machine resets on the project's 7900
    // XTX. Apply this before any weight load and make an overridden non-default
    // value visible in the server log.
    let prefill_rows = match cap_paged_prefill_rows(prefill_rows, capacity, paged_tpp.is_some()) {
        Ok(capped) => {
            if paged_tpp.is_some() && capped != prefill_rows {
                eprintln!(
                    "warning: paged prefill rows capped from {prefill_rows} to {capped} \
                     ({PAGED_PREFILL_ROWS_MAX} is the current conservative policy cap; \
                     128+ observed unsafe)"
                );
            }
            capped
        }
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(1);
        }
    };

    // Optional multimodal (Qwen3.5 vision) support: MACH_VISION=1 loads the
    // vision tower weights and image preprocessor config, and serves
    // image_url requests through the OpenAI chat endpoint.
    let mut vision_setup = None;
    let mut image_runtime = None;
    if std::env::var("MACH_VISION").is_ok_and(|v| v != "0") && !spec {
        let raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&config_path).expect("read config for vision"),
        )
        .expect("parse config for vision");
        let vision_cfg = VisionConfig::from_hf_json(&raw).expect("vision_config");
        let pre_path = resolve_preprocessor_path(&checkpoint_path)
            .expect("MACH_VISION=1 requires preprocessor_config.json");
        let pre_raw: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&pre_path).expect("read preprocessor_config.json"),
        )
        .expect("parse preprocessor_config.json");
        let processor =
            ImageProcessorConfig::from_hf_json(&pre_raw).expect("image processor config");
        let max_tokens = match std::env::var("MACH_VISION_MAX_TOKENS") {
            Ok(v) => v
                .parse()
                .unwrap_or_else(|_| panic!("invalid MACH_VISION_MAX_TOKENS: {v}")),
            Err(_) => 8192,
        };
        let weights =
            load_vision_weights(&checkpoint_path, &vision_cfg).expect("load vision weights");
        println!(
            "vision: depth={} hidden={} merge={} image_token={} max_patches={max_tokens}",
            vision_cfg.depth,
            vision_cfg.hidden_size,
            vision_cfg.spatial_merge_size,
            vision_cfg.image_token_id
        );
        image_runtime = Some(ImageRuntimeConfig {
            processor,
            image_token_id: vision_cfg.image_token_id,
            spatial_merge_size: vision_cfg.spatial_merge_size,
            max_patches: max_tokens,
            downscale_oversized: std::env::var("MACH_VISION_DOWNSCALE").is_ok_and(|v| v != "0"),
        });
        vision_setup = Some(VisionSetup {
            cfg: vision_cfg,
            weights,
            max_tokens,
        });
    } else if std::env::var("MACH_VISION").is_ok_and(|v| v != "0") {
        eprintln!("warning: MACH_VISION is not supported with MACH_SPEC; disabling vision");
    }
    if vision_setup.is_some() && paged_tpp.is_some() {
        eprintln!("warning: MACH_VISION is not supported with paged KV; disabling vision");
        vision_setup = None;
        image_runtime = None;
    }

    let (engine, engine_handle) = if q4 {
        let wq4: WeightsQ4 =
            load_safetensors_q4(&checkpoint_path, &cfg, true).expect("load q4 weights");
        println!(
            "model {model_name}: d_model={} layers={} heads={} kv={} vocab={} dtype={:?} (storage Q4; host weights stay packed int4, device dequantizes to f16)",
            cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.dtype
        );
        let eng = match paged_tpp {
            Some(tpp) => ServerEngine::with_paged(capacity, prefill_rows, tpp),
            None => ServerEngine::with_prefill_rows(capacity, prefill_rows),
        };
        if let Some(setup) = vision_setup.take() {
            eng.set_vision(hip.clone(), setup);
        }
        let handle = match q4_device {
            2 => eng.clone().spawn_q4_all(hip, cfg, wq4, kv_int8)?,
            1 => eng.clone().spawn_q4_device(hip, cfg, wq4)?,
            _ => eng.clone().spawn_q4(hip, cfg, wq4)?,
        };
        (eng, handle)
    } else if fp8 {
        let wfp8: WeightsFp8 =
            load_safetensors_fp8(&checkpoint_path, &cfg, true).expect("load fp8 weights");
        println!(
            "model {model_name}: d_model={} layers={} heads={} kv={} vocab={} dtype={:?} (storage FP8; host weights stay packed E4M3, device dequantizes to f16)",
            cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.dtype
        );
        let eng = match paged_tpp {
            Some(tpp) => ServerEngine::with_paged(capacity, prefill_rows, tpp),
            None => ServerEngine::with_prefill_rows(capacity, prefill_rows),
        };
        if let Some(setup) = vision_setup.take() {
            eng.set_vision(hip.clone(), setup);
        }
        let handle = eng.clone().spawn_fp8(hip, cfg, wfp8)?;
        (eng, handle)
    } else if spec {
        if paged_requested {
            eprintln!(
                "warning: MACH_PAGED is ignored in MACH_SPEC mode (paged spec wiring is a follow-up)"
            );
        }
        let w: Weights =
            load_safetensors(&checkpoint_path, &cfg, true).expect("load target weights");
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
        let w: Weights = load_safetensors(&checkpoint_path, &cfg, true).expect("load weights");
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
        if let Some(setup) = vision_setup.take() {
            eng.set_vision(hip.clone(), setup);
        }
        let handle = eng.clone().spawn(hip, cfg, w)?;
        (eng, handle)
    };
    if let Some(image) = image_runtime {
        engine.set_image_runtime(image);
    }
    let state = AppState {
        engine: engine.clone(),
        model: model_name,
        tok,
        chat_format: chat_format_from_json(&config_path),
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

#[cfg(test)]
mod checkpoint_path_tests {
    use super::*;

    #[test]
    fn hf_shard_and_local_metadata_resolution() {
        let root =
            std::env::temp_dir().join(format!("mach_checkpoint_paths_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let repo = root.join("qwen3.8-27b");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::write(repo.join("config.json"), b"{}").unwrap();
        std::fs::write(repo.join("tokenizer.json"), b"{}").unwrap();
        std::fs::write(repo.join("model-00001-of-00002.safetensors"), b"one").unwrap();
        std::fs::write(repo.join("model-00002-of-00002.safetensors"), b"two").unwrap();

        assert!(is_hf_shard_file(
            &repo.join("model-00001-of-00002.safetensors")
        ));
        assert!(!is_hf_shard_file(&repo.join("model.safetensors")));
        assert_eq!(
            resolve_checkpoint_path(&root, "qwen3.8-27b"),
            repo,
            "directory input stays a directory"
        );
        assert_eq!(
            resolve_checkpoint_path(&root, "qwen3.8-27b/model-00001-of-00002.safetensors"),
            repo,
            "any HF shard resolves to the shard directory"
        );
        assert_eq!(
            resolve_config_path(&root, "qwen3.8-27b/model-00001-of-00002.safetensors", None),
            repo.join("config.json"),
            "model-local config is discovered from a shard path"
        );
        assert_eq!(
            resolve_tokenizer_path(&root, "qwen3.8-27b/model-00001-of-00002.safetensors", None),
            repo.join("tokenizer.json"),
            "model-local tokenizer is discovered from a shard path"
        );
        assert_eq!(
            resolve_config_path(
                &root,
                "qwen3.8-27b/model-00001-of-00002.safetensors",
                Some(OsStr::new("custom.json"))
            ),
            root.join("custom.json"),
            "explicit MACH_CONFIG wins"
        );
        assert_eq!(
            resolve_tokenizer_path(
                &root,
                "qwen3.8-27b/model-00001-of-00002.safetensors",
                Some(OsStr::new("custom-tokenizer.json"))
            ),
            root.join("custom-tokenizer.json"),
            "explicit MACH_TOKENIZER wins"
        );

        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt;
            // WTF-16 permits an unpaired surrogate; this is the Windows form of
            // a non-Unicode OsString and must survive path resolution intact.
            let non_unicode = std::ffi::OsString::from_wide(&[0xD800, b'x' as u16]);
            assert_eq!(
                resolve_config_path(&root, "qwen3.8-27b", Some(non_unicode.as_os_str())),
                root.join(&non_unicode),
                "non-Unicode explicit config path must not be converted to UTF-8"
            );
        }

        let standalone = root.join("qwen-0.5b.safetensors");
        std::fs::write(&standalone, b"single").unwrap();
        assert_eq!(
            resolve_checkpoint_path(&root, "qwen-0.5b.safetensors"),
            standalone,
            "standalone checkpoints do not absorb their parent directory"
        );
        assert_eq!(
            resolve_config_path(&root, "qwen-0.5b.safetensors", None),
            root.join("qwen-config.json"),
            "standalone files keep the legacy root fallback"
        );

        let bare = root.join("bare");
        std::fs::create_dir_all(&bare).unwrap();
        assert_eq!(
            resolve_config_path(&root, "bare", None),
            root.join("qwen-config.json"),
            "a directory without config.json keeps the legacy fallback"
        );
        let _ = std::fs::remove_dir_all(&root);
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
