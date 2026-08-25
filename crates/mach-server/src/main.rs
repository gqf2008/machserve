//! MachServe OpenAI-compatible server binary.
//!
//! Loads a safetensors Llama/Qwen checkpoint and serves the continuous-batching
//! engine over HTTP:
//!   cargo run -p mach-server --release --features hip
//!
//! Env: MACH_MODELS (default ".models"), MACH_MODEL (default
//! "qwen-0.5b.safetensors"), MACH_CONFIG (default "qwen-config.json"),
//! MACH_CAPACITY (default 64), MACH_PREFILL_ROWS (default 512),
//! MACH_ADDR (default "127.0.0.1:8080"), MACH_Q4 (storage-int4 weights:
//! host RAM stays packed int4, dequantized to f16 on the device).

#[cfg(feature = "hip")]
use mach_kernel_sys::hip;
#[cfg(feature = "hip")]
use mach_model::config::ModelDType;
#[cfg(feature = "hip")]
use mach_model::loader::{load_safetensors, load_safetensors_q4};
#[cfg(feature = "hip")]
use mach_model::tokenizer::Tokenizer;
#[cfg(feature = "hip")]
use mach_model::{Config, Weights, WeightsQ4};
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
    // Qwen3 QK-norm.
    if let Some(qk) = v["qk_norm"].as_bool() {
        cfg.qk_norm = qk;
    }
    cfg
}

/// Rough device-memory estimate for the preflight: weight file + KV cache +
/// 256MiB scratch margin (+ draft model in spec mode). MLA uses the expanded
/// per-head KV cache (always f32); dense uses the GQA formula with the dtype's
/// element size. Sharded weight files, hipBLAS workspace and compiled kernels
/// are not counted; the margin covers today's scenarios.
#[cfg(feature = "hip")]
fn estimate_vram(
    cfg: &Config,
    capacity: usize,
    model_file_bytes: u64,
    draft: Option<(&Config, u64)>,
    q4: bool,
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
    // Q4 stores packed int4 on the host but the device holds dequantized f16
    // weights (~3.2x the packed size incl. scales); x4 is a conservative
    // over-estimate so the preflight cannot pass while the upload OOMs.
    let weight = if q4 {
        model_file_bytes * 4
    } else {
        model_file_bytes
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
        let fb = std::fs::metadata(root.join(&model_name))
            .map(|m| m.len())
            .unwrap_or(0);
        let cap = std::env::var("MACH_CAPACITY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let need = estimate_vram(&cfg, cap, fb, None, false);
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
        other => panic!("MACH_DTYPE must be f32 or f16, got {other:?}"),
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
                "MACH_Q4=1 and MACH_SPEC are mutually exclusive (spec mode loads a second f32 model)"
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
    // NOTE: sharded weight files, hipBLAS workspace and compiled kernels are
    // not counted; the 256MiB margin covers today's tiny/1.5B scenarios.
    let file_bytes = std::fs::metadata(root.join(&model_name))
        .map(|m| m.len())
        .unwrap_or(0);
    let draft_est = if spec {
        let dfb = std::fs::metadata(root.join(&draft_name))
            .map(|m| m.len())
            .unwrap_or(0);
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
    let (engine, engine_handle) = if q4 {
        let wq4: WeightsQ4 =
            load_safetensors_q4(&root.join(&model_name), &cfg, true).expect("load q4 weights");
        println!(
            "model {model_name}: d_model={} layers={} heads={} kv={} vocab={} dtype={:?} (storage Q4; host weights stay packed int4, device dequantizes to f16)",
            cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.dtype
        );
        let eng = ServerEngine::with_prefill_rows(capacity, prefill_rows);
        let handle = eng.clone().spawn_q4(hip, cfg, wq4)?;
        (eng, handle)
    } else if spec {
        let w: Weights =
            load_safetensors(&root.join(&model_name), &cfg, true).expect("load target weights");
        println!(
            "model {model_name}: d_model={} layers={} heads={} kv={} vocab={} dtype={:?}",
            cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.dtype
        );
        // Speculative-decoding mode: MACH_SPEC=1 serves greedy requests through
        // a draft + target engine (greedy-only; other params rejected).
        let mut dcfg = config_from_json(&root.join(&draft_config));
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
            ServerEngine::with_offload(capacity, prefill_rows, slots)
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
        if q4 { ", storage Q4" } else { "" },
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
        let est = estimate_vram(&cfg, 8, 1_000_000, None, false);
        // KV (f32) = capacity*max_seq*kv_heads*head_dim*4 per layer.
        let kv =
            (8 * cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim * 4 * 2 * cfg.n_layers) as u64;
        assert_eq!(est, 1_000_000 + kv + (256 << 20));
    }

    #[test]
    fn f16_dense_uses_two_byte_kv() {
        let mut cfg = dense_cfg();
        cfg.dtype = ModelDType::F16;
        let f16 = estimate_vram(&cfg, 8, 0, None, false);
        cfg.dtype = ModelDType::F32;
        let f32 = estimate_vram(&cfg, 8, 0, None, false);
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
        assert_eq!(estimate_vram(&cfg, 8, 0, None, false), kv + (256 << 20));
    }

    #[test]
    fn spec_adds_draft_weights_and_kv() {
        let tcfg = dense_cfg();
        let dcfg = dense_cfg();
        let base = estimate_vram(&tcfg, 8, 1_000, None, false);
        let spec = estimate_vram(&tcfg, 8, 1_000, Some((&dcfg, 500)), false);
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
    fn q4_scales_weight_term_by_four() {
        let cfg = dense_cfg();
        let base = estimate_vram(&cfg, 8, 1_000_000, None, false);
        let q4 = estimate_vram(&cfg, 8, 1_000_000, None, true);
        // Q4 adds 3x file_bytes (weight term x4 vs x1).
        assert_eq!(q4 - base, 3_000_000);
    }
}
