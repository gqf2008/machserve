//! Reproducible multi-turn TTFT A/B for agentic state reuse.
//!
//! Same context + a new round: turn 2 either re-prefills the whole context
//! (baseline) or restores the turn-1 anchor and prefills only the delta
//! (state reuse). TTFT = time from the turn-2 request to its first generated
//! token, exactly what a user perceives.
//!
//! Run on a GPU box (needs a MoE checkpoint; opt-in, no model is loaded here):
//!   MACH_MODELS=.models MACH_MODEL=model.safetensors MACH_CONFIG=config.json \
//!     cargo run -p mach-model --release --features hip --example state_reuse_bench
//!
//! Env knobs: MACH_PREFIX_TOKENS (turn-1 prompt, default 128),
//! MACH_RESP_TOKENS (turn-1 response, default 32),
//! MACH_DELTA_TOKENS (turn-2 new user message, default 16),
//! MACH_TURN2_GEN (turn-2 generated tokens, default 8).

#[cfg(feature = "hip")]
use mach_kernel_sys::hip;
#[cfg(feature = "hip")]
use mach_model::continuous::ContinuousModel;
#[cfg(feature = "hip")]
use mach_model::loader::load_safetensors;
#[cfg(feature = "hip")]
use mach_model::sampling::SamplingParams;
#[cfg(feature = "hip")]
use mach_model::state_reuse::{ReuseStats, StateReuse};
#[cfg(feature = "hip")]
use mach_model::{Config, Weights};
#[cfg(feature = "hip")]
use std::path::PathBuf;
#[cfg(feature = "hip")]
use std::sync::Arc;
#[cfg(feature = "hip")]
use std::time::Instant;

#[cfg(feature = "hip")]
fn main() {
    let root = PathBuf::from(std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into()));
    let model_name = std::env::var("MACH_MODEL").unwrap_or_else(|_| "model.safetensors".into());
    let config_name = std::env::var("MACH_CONFIG").unwrap_or_else(|_| "config.json".into());
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
    let prefix_len: usize = env_usize("MACH_PREFIX_TOKENS", 128);
    let resp_len: usize = env_usize("MACH_RESP_TOKENS", 32);
    let delta_len: usize = env_usize("MACH_DELTA_TOKENS", 16);
    let turn2_gen: usize = env_usize("MACH_TURN2_GEN", 8);

    let w: Weights = load_safetensors(&model_path, &cfg, false).expect("load weights");
    let hip = hip::hip().expect("HIP runtime");
    assert!(hip::device_count().expect("devices") > 0, "no HIP device");
    let hip = Arc::new(hip);

    // Warm up hiprtc compilation + weight upload on a throwaway engine so the
    // measured turns start from a warm runtime.
    let mut warm = ContinuousModel::new(Arc::clone(&hip), cfg, &w, 2).unwrap();
    let id = warm
        .add(
            &[1, 2, 3],
            2,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !warm.is_done(id) {
        warm.step().unwrap();
    }
    drop(warm);

    let prompt1: Vec<u32> = (0..prefix_len).map(|i| (i % 977) as u32 + 1).collect();
    let user2: Vec<u32> = (0..delta_len).map(|i| (i % 977) as u32 + 500).collect();

    let spec = TurnSpec {
        prompt1: &prompt1,
        resp_len,
        user2: &user2,
        turn2_gen,
    };
    let (base, _base_stats, base_out) = run_turns(false, Arc::clone(&hip), cfg, &w, spec);
    let (reuse, reuse_stats, reuse_out) = run_turns(true, Arc::clone(&hip), cfg, &w, spec);

    let red = (base.ttft_ms - reuse.ttft_ms) / base.ttft_ms * 100.0;
    let skipped = reuse_stats.map_or(0, |s: ReuseStats| s.tokens_reused as usize);
    let total = prefix_len + resp_len + delta_len;
    let bound = skipped as f64 / total as f64 * 100.0;

    println!();
    println!("=== State-reuse multi-turn TTFT A/B ===");
    println!(
        "model: {model_name} | d_model={} layers={} experts={} topk={} moe_inter={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.expert_size()
    );
    println!(
        "tokens: turn1_prompt={prefix_len} turn1_resp={resp_len} turn2_delta={delta_len} turn2_gen={turn2_gen}"
    );
    println!(
        "turn-2 prefill: no-reuse={total} tokens | reuse skips {skipped} (max theoretical {bound:.1}%)"
    );
    println!("mode          | TTFT(ms) | tokens_reused | output matches baseline");
    println!("no-reuse      | {:8.2} | {:>12} | -", base.ttft_ms, 0);
    println!(
        "reuse         | {:8.2} | {:>12} | {}",
        reuse.ttft_ms,
        reuse_stats.map_or(0, |s| s.tokens_reused),
        reuse_out == base_out
    );
    println!("TTFT reduction: {red:.2}% (reuse vs no-reuse, same turn-2 request)");
    println!(
        "reuse stats: hits={} lookups={}",
        reuse_stats.map_or(0, |s| s.hits),
        reuse_stats.map_or(0, |s| s.lookups)
    );
    println!(
        "note: reuse output must equal baseline (greedy, deterministic). FreeToken reports 65-80%;"
    );
    println!(
        "      the gap here is fixed engine overhead (step/decode+sample) that TTFT includes."
    );
}

#[cfg(feature = "hip")]
struct Meas {
    ttft_ms: f64,
}

#[cfg(feature = "hip")]
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Per-turn token layout shared by both benchmark arms.
#[derive(Clone, Copy)]
#[cfg(feature = "hip")]
struct TurnSpec<'a> {
    prompt1: &'a [u32],
    resp_len: usize,
    user2: &'a [u32],
    turn2_gen: usize,
}

#[cfg(feature = "hip")]
fn run_turns(
    reuse: bool,
    hip: Arc<mach_kernel_sys::hip::Hip>,
    cfg: Config,
    w: &Weights,
    spec: TurnSpec<'_>,
) -> (Meas, Option<ReuseStats>, Vec<u32>) {
    let mut eng = if reuse {
        ContinuousModel::with_state_reuse(hip, cfg, w, 2, StateReuse::new(16)).unwrap()
    } else {
        ContinuousModel::new(hip, cfg, w, 2).unwrap()
    };
    let params = SamplingParams::default(); // greedy -> deterministic

    // Turn 1: prefix + a short assistant response (anchor saved on finish).
    let id1 = eng
        .add(
            spec.prompt1,
            spec.resp_len,
            None,
            Vec::new(),
            Vec::new(),
            params,
        )
        .unwrap();
    while !eng.is_done(id1) {
        eng.step().unwrap();
    }
    let resp1 = eng.generated(id1);
    assert_eq!(resp1.len(), spec.resp_len, "turn-1 response length");

    // Turn 2: same context + a new user message; TTFT = first generated token.
    let mut prompt2 = spec.prompt1.to_vec();
    prompt2.extend_from_slice(&resp1);
    prompt2.extend_from_slice(spec.user2);
    let t0 = Instant::now();
    let id2 = eng
        .add(
            &prompt2,
            spec.turn2_gen,
            None,
            Vec::new(),
            Vec::new(),
            params,
        )
        .unwrap();
    let mut first: Option<f64> = None;
    let mut out = Vec::new();
    while !eng.is_done(id2) {
        let outs = eng.step().unwrap();
        if first.is_none() && !outs.is_empty() {
            first = Some(t0.elapsed().as_secs_f64() * 1000.0);
        }
        for (_, t) in outs {
            out.push(t);
        }
    }
    (
        Meas {
            ttft_ms: first.expect("turn 2 must generate"),
        },
        eng.reuse_stats(),
        out,
    )
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
    eprintln!("state_reuse_bench requires --features hip");
}
