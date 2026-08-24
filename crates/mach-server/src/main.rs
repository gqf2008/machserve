//! MachServe OpenAI-compatible server binary.
//!
//! Loads a safetensors Llama/Qwen checkpoint and serves the continuous-batching
//! engine over HTTP:
//!   cargo run -p mach-server --release --features hip
//!
//! Env: MACH_MODELS (default ".models"), MACH_MODEL (default
//! "qwen-0.5b.safetensors"), MACH_CONFIG (default "qwen-config.json"),
//! MACH_CAPACITY (default 64), MACH_PREFILL_ROWS (default 512),
//! MACH_ADDR (default "127.0.0.1:8080").

#[cfg(feature = "hip")]
use mach_kernel_sys::hip;
#[cfg(feature = "hip")]
use mach_model::config::ModelDType;
#[cfg(feature = "hip")]
use mach_model::loader::load_safetensors;
#[cfg(feature = "hip")]
use mach_model::tokenizer::Tokenizer;
#[cfg(feature = "hip")]
use mach_model::{Config, Weights};
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
    cfg
}

#[cfg(feature = "hip")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    // NOTE: the estimate uses the dense-GQA KV formula; MLA's expanded
    // per-head KV cache (heads * (nope+rope+v_hd), f32) is larger and is not
    // accounted for yet — revisit before serving real MLA checkpoints. Sharded
    // weight files, hipBLAS workspace and compiled modules are also omitted;
    // the 256MiB margin covers today's dense tiny/1.5B scenarios.
    let kv_elem = if cfg.dtype == ModelDType::F16 { 2 } else { 4 };
    let file_bytes = std::fs::metadata(root.join(&model_name))
        .map(|m| m.len())
        .unwrap_or(0);
    let kv_bytes =
        capacity * cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim * kv_elem * cfg.n_layers;
    let mut estimate = file_bytes + kv_bytes as u64 + (256 << 20); // +256MiB scratch margin
    if spec {
        let dfb = std::fs::metadata(root.join(&draft_name))
            .map(|m| m.len())
            .unwrap_or(0);
        let mut dcfg = config_from_json(&root.join(&draft_config));
        match dtype.as_str() {
            "f32" => dcfg.dtype = ModelDType::F32,
            _ => dcfg.dtype = ModelDType::F16,
        }
        let dkv =
            capacity * dcfg.max_seq_len * dcfg.n_kv_heads * dcfg.head_dim * kv_elem * dcfg.n_layers;
        estimate += dfb + dkv as u64;
    }
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    println!(
        "GPU preflight: device_count={devices}, VRAM free {:.2}GiB / {:.2}GiB, estimated need {:.2}GiB",
        gib(free as u64),
        gib(total as u64),
        gib(estimate)
    );
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

    let w: Weights = load_safetensors(&root.join(&model_name), &cfg, true).expect("load weights");
    println!(
        "model {model_name}: d_model={} layers={} heads={} kv={} vocab={} dtype={:?}",
        cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.dtype
    );

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

    // Speculative-decoding mode: MACH_SPEC=1 serves greedy requests through a
    // draft + target engine (greedy-only; other params rejected).
    let (engine, engine_handle) = if spec {
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
        let eng = ServerEngine::with_prefill_rows(capacity, prefill_rows);
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
        "mach-server listening on http://{addr} (capacity {capacity}, prefill rows {prefill_rows}{})",
        if spec { ", spec-decode" } else { "" }
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
    eprintln!("mach-server requires the `hip` feature: cargo run -p mach-server --features hip");
}
