//! Long-context f16 decode benchmark over the continuous-batching path
//! (`BatchedModel`): fills `ctx` KV positions per sequence, then times N
//! decode steps and reports ms/step and per-sequence token throughput.
//!
//!   cargo run -p mach-model --release --features hip --example lctx_bench
//!
//! Env: MACH_MODEL (default qwen-0.5b.safetensors), MACH_CONFIG (default
//! qwen-config.json), MACH_DTYPE (f16/f32, default f16), MACH_BATCH (default
//! 64), MACH_CTX (default 2048).
#[cfg(feature = "hip")]
fn main() {
    use mach_kernel_sys::hip;
    use mach_model::batched::BatchedModel;
    use mach_model::config::ModelDType;
    use mach_model::loader::load_safetensors;
    use mach_model::{Config, Weights};
    use std::path::PathBuf;
    use std::time::Instant;

    let root = PathBuf::from(std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into()));
    let model_name = std::env::var("MACH_MODEL").unwrap_or_else(|_| "qwen-0.5b.safetensors".into());
    let config_name = std::env::var("MACH_CONFIG").unwrap_or_else(|_| "qwen-config.json".into());
    let batch: usize = std::env::var("MACH_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let ctx: usize = std::env::var("MACH_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);
    let dtype = std::env::var("MACH_DTYPE").unwrap_or_else(|_| "f16".into());

    let v: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join(&config_name)).expect("read config"),
    )
    .expect("parse config");
    let hidden = v["hidden_size"].as_u64().unwrap() as usize;
    let layers = v["num_hidden_layers"].as_u64().unwrap() as usize;
    let heads = v["num_attention_heads"].as_u64().unwrap() as usize;
    let kv = v["num_key_value_heads"].as_u64().unwrap() as usize;
    let vocab = v["vocab_size"].as_u64().unwrap() as usize;
    let inter = v["intermediate_size"].as_u64().unwrap() as usize;
    let max_seq = v["max_position_embeddings"]
        .as_u64()
        .unwrap_or(2048)
        .min(8192) as usize;
    let mut cfg = Config::llama(hidden, layers, heads, kv, vocab, max_seq);
    cfg.intermediate_size = inter;
    cfg.rms_eps = v["rms_norm_eps"].as_f64().unwrap() as f32;
    cfg.rope_theta = v["rope_theta"].as_f64().unwrap() as f32;
    cfg.dtype = match dtype.as_str() {
        "f16" => ModelDType::F16,
        _ => ModelDType::F32,
    };
    let w: Weights = load_safetensors(&root.join(&model_name), &cfg, true).expect("weights");
    println!(
        "model {model_name}: d_model={} layers={} heads={} kv={} head_dim={} dtype={:?} batch={batch} ctx={ctx}",
        cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.dtype
    );

    let hip = hip::hip().expect("hip");
    let mut bm = BatchedModel::new(hip, cfg, &w, batch).expect("model");
    let mut tokens: Vec<u32> = (0..batch).map(|i| (i % 977) as u32).collect();
    // Fill KV to `ctx` positions (each decode_step advances one position).
    for _ in 0..ctx {
        tokens = bm.decode_step(&tokens).expect("prefill");
    }
    bm.reset_state().expect("reset");
    for _ in 0..10 {
        tokens = bm.decode_step(&tokens).expect("warmup");
    }
    let n = 50usize;
    let t0 = Instant::now();
    for _ in 0..n {
        tokens = bm.decode_step(&tokens).expect("step");
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
    let per_tok_us = ms / batch as f64 * 1000.0;
    println!(
        "long-context decode: {ms:.3} ms/step | {per_tok_us:.1} us/seq-tok | {:.0} tok/s/seq",
        1e6 / per_tok_us
    );
}
#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!("lctx_bench requires the hip feature");
}
