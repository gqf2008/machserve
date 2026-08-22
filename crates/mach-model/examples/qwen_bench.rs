//! Loads a real Qwen2.5-0.5B-Instruct checkpoint (BF16 safetensors) and
//! benchmarks decode TPOT on the GPU, eager vs HIP graph.
//!
//! Also dumps the first tokens' logits to `.models/qwen_rust_logits.json` for
//! numeric validation against `tools/ref_llama.py` (fp64 reference).
//!
//! Run with:
//!   cargo run -p mach-model --release --features hip --example qwen_bench
//!
//! Download first:
//!   curl -L -o .models/qwen-0.5b.safetensors \
//!     https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors
//!   curl -L -o .models/qwen-config.json \
//!     https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json

#[cfg(feature = "hip")]
use mach_model::Config;

#[cfg(feature = "hip")]
fn main() {
    use mach_kernel_sys::hip;
    use mach_model::Weights;
    use mach_model::loader::load_safetensors;
    use mach_model::model::GpuModel;
    use std::path::PathBuf;
    use std::time::Instant;

    let root = std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into());
    let root = PathBuf::from(&root);
    let model_path = root.join("qwen-0.5b.safetensors");
    let cfg_path = root.join("qwen-config.json");
    assert!(
        model_path.exists(),
        "missing {model_path:?} (see doc comment)"
    );
    assert!(cfg_path.exists(), "missing {cfg_path:?}");

    let cfg = config_from_json(&cfg_path);
    println!(
        "model: Qwen2.5-0.5B-Instruct (d_model={} layers={} heads={} kv={} head_dim={} inter={} vocab={})",
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.intermediate_size,
        cfg.vocab_size
    );

    let w: Weights = load_safetensors(&model_path, &cfg, true).expect("load weights");
    println!("weights: {:.1} MB fp32", w.byte_size() as f64 / 1e6);

    let hip = hip::hip().expect("HIP runtime");
    assert!(hip::device_count().expect("devices") > 0, "no HIP device");
    let mut model = GpuModel::new(hip, cfg, &w).expect("build model");

    // Numeric validation: tokens from args (default 1 2 3), dump logits.
    let check_tokens: Vec<u32> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let check_tokens = if check_tokens.is_empty() {
        vec![1, 2, 3]
    } else {
        check_tokens
    };
    let logits = model.forward(&check_tokens).expect("decode");
    assert!(logits.iter().all(|v| v.is_finite()));
    std::fs::write(
        root.join("qwen_rust_logits.json"),
        serde_json::to_string(&logits).unwrap(),
    )
    .expect("write");
    println!("validated tokens {check_tokens:?} -> logits finite, dumped for ref_llama.py");

    let n_tokens = 100usize;
    let seq: Vec<u32> = (0..n_tokens).map(|i| (i % 977) as u32).collect();

    // --- full step (input copy + kernels + logits readback): the single-seq TPOT path ---
    model.reset_state().expect("reset");
    for _ in 0..5 {
        model.decode_step(1).expect("warmup");
    }
    model.reset_state().expect("reset");
    let t0 = Instant::now();
    for &t in &seq {
        model.decode_step(t).expect("eager");
    }
    let eager_full_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_tokens as f64;

    let graph = model.capture_decode().expect("capture");
    let t1 = Instant::now();
    for &t in &seq {
        model.decode_step_graph(&*graph, t).expect("graph");
    }
    let graph_full_ms = t1.elapsed().as_secs_f64() * 1000.0 / n_tokens as f64;

    // --- launch-only path ---
    model.reset_state().expect("reset");
    let t2 = Instant::now();
    for &t in &seq {
        model.step_eager(t).expect("eager launch");
    }
    let eager_launch_ms = t2.elapsed().as_secs_f64() * 1000.0 / n_tokens as f64;

    let graph2 = model.capture_decode().expect("capture2");
    let t3 = Instant::now();
    for &t in &seq {
        model.step_graph(&*graph2, t).expect("graph launch");
    }
    let graph_launch_ms = t3.elapsed().as_secs_f64() * 1000.0 / n_tokens as f64;

    println!("\n=== decode TPOT (Qwen2.5-0.5B, 7900 XTX) ===");
    println!(
        "full step (incl. logits readback): eager {eager_full_ms:.2} ms/tok | graph {graph_full_ms:.2} ms/tok | {:.2}x",
        eager_full_ms / graph_full_ms
    );
    println!(
        "launch-only: eager {eager_launch_ms:.2} ms/tok | graph {graph_launch_ms:.2} ms/tok | {:.2}x",
        eager_launch_ms / graph_launch_ms
    );

    // --- real generation demo: 8 tokens, argmax sampling ---
    model.reset_state().expect("reset");
    let mut token = 151643u32; // <|im_start|>
    let mut out = Vec::new();
    let t4 = Instant::now();
    for _ in 0..8 {
        let logits = model.decode_step(token).expect("gen");
        token = argmax(&logits) as u32;
        out.push(token);
    }
    let gen_ms = t4.elapsed().as_secs_f64() * 1000.0 / 8.0;
    println!("\ngeneration (argmax): {gen_ms:.2} ms/token, tokens {out:?}");
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!(
        "qwen_bench requires the `hip` feature: cargo run -p mach-model --features hip --example qwen_bench"
    );
}

#[cfg(feature = "hip")]
fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

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
        .min(2048) as usize;
    let eps = v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    let theta = v["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let mut cfg = Config::llama(hidden, layers, heads, kv, vocab, max_seq);
    cfg.intermediate_size = inter;
    cfg.rms_eps = eps;
    cfg.rope_theta = theta;
    cfg
}
