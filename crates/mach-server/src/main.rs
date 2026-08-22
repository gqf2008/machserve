//! MachServe OpenAI-compatible server binary.
//!
//! Loads a safetensors Llama/Qwen checkpoint and serves the continuous-batching
//! engine over HTTP:
//!   cargo run -p mach-server --release --features hip
//!
//! Env: MACH_MODELS (default ".models"), MACH_MODEL (default
//! "qwen-0.5b.safetensors"), MACH_CONFIG (default "qwen-config.json"),
//! MACH_CAPACITY (default 64), MACH_ADDR (default "127.0.0.1:8080").

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
    let addr = std::env::var("MACH_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".into());
    // Compute dtype: default fp16 (2x+ GEMM, verified vs fp32), MACH_DTYPE=f32
    // opts out. bf16 is not wired yet.
    let dtype = std::env::var("MACH_DTYPE").unwrap_or_else(|_| "f16".into());

    let root = PathBuf::from(root);
    let mut cfg = config_from_json(&root.join(&config_name));
    match dtype.as_str() {
        "f32" => cfg.dtype = ModelDType::F32,
        "f16" => cfg.dtype = ModelDType::F16,
        other => panic!("MACH_DTYPE must be f32 or f16, got {other:?}"),
    }
    let w: Weights = load_safetensors(&root.join(&model_name), &cfg, true).expect("load weights");
    println!(
        "model {model_name}: d_model={} layers={} heads={} kv={} vocab={} dtype={:?}",
        cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, cfg.dtype
    );

    let hip = hip::hip().expect("HIP runtime");
    assert!(hip::device_count().expect("devices") > 0, "no HIP device");

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

    let engine = ServerEngine::new(capacity);
    engine.clone().spawn(hip, cfg, w)?;
    let state = AppState {
        engine,
        model: model_name,
        tok,
    };
    let app = router(state);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("mach-server listening on http://{addr} (capacity {capacity})");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!("mach-server requires the `hip` feature: cargo run -p mach-server --features hip");
}
