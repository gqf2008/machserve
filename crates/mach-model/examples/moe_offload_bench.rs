//! Reproducible benchmark for the MoE offload engine: compares full-resident,
//! bounded-slot, and bandwidth-adaptive (q*) placement on the SAME model + input.
//!
//! Run on a GPU box (needs a MoE checkpoint; opt-in, no model is loaded here):
//!   MACH_MODEL=<model.safetensors> MACH_CONFIG=<config.json> MACH_MOE_SLOTS=2 \
//!     cargo run -p mach-model --release --features hip --example moe_offload_bench
//!
//! Metrics: TTFT (first decoded token, ms) and TPOT (avg per generated token over
//! MACH_BENCH_TOKENS, default 32).

#[cfg(feature = "hip")]
use mach_model::{Config, Weights};
#[cfg(feature = "hip")]
use std::sync::Arc;
#[cfg(feature = "hip")]
use std::time::Instant;

#[cfg(feature = "hip")]
fn main() {
    use mach_kernel_sys::hip;
    use mach_model::loader::load_safetensors;
    use std::path::PathBuf;

    let root = PathBuf::from(std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into()));
    let model_name = std::env::var("MACH_MODEL").unwrap_or_else(|_| "qwen-0.5b.safetensors".into());
    let config_name = std::env::var("MACH_CONFIG").unwrap_or_else(|_| "qwen-config.json".into());
    let model_path = root.join(&model_name);
    let cfg_path = root.join(&config_name);
    assert!(
        model_path.exists(),
        "missing {model_path:?} (set MACH_MODEL)"
    );
    assert!(cfg_path.exists(), "missing {cfg_path:?}");

    let cfg = config_from_json(&cfg_path);
    assert!(
        cfg.num_experts > 0,
        "benchmark requires a MoE checkpoint (num_experts > 0)"
    );
    let n: u32 = std::env::var("MACH_BENCH_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let slots: usize = std::env::var("MACH_MOE_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);

    let w: Weights = load_safetensors(&model_path, &cfg, true).expect("load weights");
    let hip = hip::hip().expect("HIP runtime");
    assert!(hip::device_count().expect("devices") > 0, "no HIP device");

    let seq: Vec<u32> = (0..n).map(|i| i % 977).collect();

    let (full, full_logits) = run_mode(&hip, cfg, &w, Mode::Full, &seq);
    let (slot, slot_logits) = run_mode(&hip, cfg, &w, Mode::Slots(slots), &seq);
    let (adapt, adapt_logits) = run_mode(&hip, cfg, &w, Mode::Adaptive(slots), &seq);

    // Placement-invariance on the real checkpoint: offloaded modes must match
    // the full-resident logits exactly (scheduling is a numeric no-op).
    let argmax = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    let max_diff = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    println!();
    println!("=== MoE offload benchmark ===");
    println!(
        "model: {model_name} | d_model={} layers={} experts={} topk={} moe_inter={} | tokens={n}",
        cfg.d_model,
        cfg.n_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.expert_size()
    );
    println!("mode           | TTFT(ms) | TPOT(ms/tok) | tok/s");
    println!(
        "full          | {:8.2} | {:10.2} | {:8.1}",
        full.ttft_ms,
        full.tpot_ms,
        1000.0 / full.tpot_ms
    );
    println!(
        "slots={slots:<4}    | {:8.2} | {:10.2} | {:8.1}",
        slot.ttft_ms,
        slot.tpot_ms,
        1000.0 / slot.tpot_ms
    );
    println!(
        "adaptive      | {:8.2} | {:10.2} | {:8.1}",
        adapt.ttft_ms,
        adapt.tpot_ms,
        1000.0 / adapt.tpot_ms
    );
    println!("note: TTFT/TPOT include the offload path syncs/D2H; placement is");
    println!("      invariance-agnostic, so any diff vs full is scheduling, not accuracy.");
    println!();
    println!("placement invariance vs full | max|logit diff| | argmax match");
    println!(
        "slots={slots:<4}                   {:>14.6} | {}",
        max_diff(&full_logits, &slot_logits),
        argmax(&full_logits) == argmax(&slot_logits)
    );
    println!(
        "adaptive                         {:>14.6} | {}",
        max_diff(&full_logits, &adapt_logits),
        argmax(&full_logits) == argmax(&adapt_logits)
    );
}

#[cfg(feature = "hip")]
enum Mode {
    Full,
    Slots(usize),
    Adaptive(usize),
}

#[cfg(feature = "hip")]
struct Meas {
    ttft_ms: f64,
    tpot_ms: f64,
}

#[cfg(feature = "hip")]
fn run_mode(
    hip: &Arc<mach_kernel_sys::hip::Hip>,
    cfg: Config,
    w: &Weights,
    mode: Mode,
    seq: &[u32],
) -> (Meas, Vec<f32>) {
    use mach_model::model::GpuModel;
    let mut model = match mode {
        Mode::Full => GpuModel::new(Arc::clone(hip), cfg, w).unwrap(),
        Mode::Slots(s) => GpuModel::with_expert_slots(Arc::clone(hip), cfg, w, s).unwrap(),
        Mode::Adaptive(s) => GpuModel::with_adaptive(Arc::clone(hip), cfg, w, s).unwrap(),
    };
    model.reset_state().unwrap();
    for _ in 0..3 {
        model.decode_step(1).unwrap();
    }
    model.reset_state().unwrap();

    let t0 = Instant::now();
    let mut last = model.decode_step(seq[0]).unwrap();
    let ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    for &t in &seq[1..] {
        last = model.decode_step(t).unwrap();
    }
    let tpot_ms = t1.elapsed().as_secs_f64() * 1000.0 / (seq.len() - 1).max(1) as f64;
    (Meas { ttft_ms, tpot_ms }, last)
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
    // Some configs (e.g. Qwen3-30B-A3B) ship an explicit `head_dim` that
    // differs from hidden/n_heads (q/o width = n_heads*head_dim is wider
    // than hidden). Honor it, or the loader under-sizes q/o projections.
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
    cfg
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!("moe_offload_bench requires the `hip` feature");
}
