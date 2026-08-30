//! TEMP→real: Qwen3-30B-A3B Q4-on-device 真机验证 (issue #85 P2).
//!
//!   MACH_MODELS=<dir with model-*.safetensors + config.json> \
//!     cargo run -p mach-model --release --features hip --example qwen3_30b_q4_check
//!
//! Verifies: load (Q4 quantize, ~16GB host), build (Q4-on-device, ~24GB
//! VRAM), decode steps — logits finite, greedy token stability, timing.
#[cfg(feature = "hip")]
use mach_model::batched::BatchedModel;
#[cfg(feature = "hip")]
use mach_model::loader::load_safetensors_q4;
#[cfg(feature = "hip")]
use mach_model::{Config, WeightsQ4};
#[cfg(feature = "hip")]
use std::path::PathBuf;

#[cfg(feature = "hip")]
fn config_from_json(path: &std::path::Path) -> Config {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read config"))
            .expect("parse config");
    let hidden = v["hidden_size"].as_u64().unwrap_or(2048) as usize;
    let layers = v["num_hidden_layers"].as_u64().unwrap_or(48) as usize;
    let heads = v["num_attention_heads"].as_u64().unwrap_or(32) as usize;
    let kv = v["num_key_value_heads"].as_u64().unwrap_or(heads as u64) as usize;
    let vocab = v["vocab_size"].as_u64().unwrap_or(151936) as usize;
    let inter = v["intermediate_size"].as_u64().unwrap_or(6144) as usize;
    let max_seq = v["max_position_embeddings"]
        .as_u64()
        .unwrap_or(32768)
        .min(8192) as usize;
    let eps = v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    let theta = v["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let ne = v["num_experts"]
        .as_u64()
        .or_else(|| v["n_routed_experts"].as_u64())
        .unwrap_or(0) as usize;
    let topk = v["num_experts_per_tok"]
        .as_u64()
        .or_else(|| v["top_k"].as_u64())
        .unwrap_or(0) as usize;
    let mut cfg = Config::llama(hidden, layers, heads, kv, vocab, max_seq);
    if let Some(hd) = v["head_dim"].as_u64() {
        cfg.head_dim = hd as usize;
    }
    cfg.intermediate_size = inter;
    cfg.rms_eps = eps;
    cfg.rope_theta = theta;
    cfg.num_experts = ne;
    cfg.num_experts_per_tok = topk;
    cfg.moe_intermediate_size = v["moe_intermediate_size"].as_u64().unwrap_or(0) as usize;
    cfg.qk_norm = v["use_qk_norm"].as_bool().unwrap_or(false);
    cfg.dtype = mach_model::config::ModelDType::F16;
    cfg
}

#[cfg(feature = "hip")]
fn main() {
    let root = PathBuf::from(std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into()));
    let model_dir = root.join("qwen3-30b-a3b");
    let cfg_path = model_dir.join("config.json");
    assert!(cfg_path.exists(), "missing {cfg_path:?}");
    let cfg = config_from_json(&cfg_path);
    println!(
        "config: d={} layers={} experts={} topk={} moe_inter={} vocab={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.expert_size(),
        cfg.vocab_size
    );

    let t0 = std::time::Instant::now();
    let w: WeightsQ4 = load_safetensors_q4(&model_dir, &cfg, true).expect("load q4");
    println!(
        "load: {:.1}s (host Q4 ~{} MiB)",
        t0.elapsed().as_secs_f64(),
        0
    );
    let q4_mib = w
        .layers
        .iter()
        .map(|l| l.moe_wg.q_bytes().len() + l.moe_wu.q_bytes().len() + l.moe_wd.q_bytes().len())
        .sum::<usize>()
        / (1024 * 1024);
    println!("expert Q4 bytes: ~{q4_mib} MiB (3 tensors)");

    let hip = mach_kernel_sys::hip::hip().expect("HIP runtime");
    assert!(
        mach_kernel_sys::hip::device_count().expect("devices") > 0,
        "no device"
    );

    let t1 = std::time::Instant::now();
    let mut model = BatchedModel::with_rows_q4_device(hip, cfg, &w, 1, 8).expect("build");
    println!("build: {:.1}s", t1.elapsed().as_secs_f64());
    drop(w);

    // Warmup, then two full passes over the same sequence: logits finite on
    // every step, greedy tokens run-to-run stable (deterministic Q4 path),
    // and per-step timing.
    let n_steps = 16usize;
    let run = |model: &mut BatchedModel| -> (Vec<u32>, std::time::Duration) {
        model.reset_state().unwrap();
        let mut got = Vec::with_capacity(n_steps);
        let t = std::time::Instant::now();
        for i in 0..n_steps {
            let tok = ((i as u32) * 37 + 5) % cfg.vocab_size as u32;
            model.decode_step(&[tok]).unwrap();
            let logits = model.read_logits().unwrap();
            assert_eq!(logits.len(), cfg.vocab_size);
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "step {i}: non-finite logits"
            );
            let greedy = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            got.push(greedy);
        }
        (got, t.elapsed())
    };
    let (a, el) = run(&mut model);
    let (b, _) = run(&mut model);
    assert_eq!(
        a, b,
        "greedy tokens must be run-to-run stable (deterministic)"
    );
    for (i, &g) in a.iter().enumerate() {
        println!("step {i}: greedy token {g}");
    }
    println!(
        "decode: {:.2} ms/step ({:.0} tok/s, {} steps)",
        el.as_secs_f64() * 1000.0 / n_steps as f64,
        n_steps as f64 / el.as_secs_f64(),
        n_steps
    );
    println!("OK: Q4-on-device 30B decode verified (finite logits, stable greedy)");
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!("qwen3_30b_q4_check requires the `hip` feature");
}
