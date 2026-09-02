//! Qwen3-30B-A3B request-churn repro for the #103 decode-graph corruption.
//!
//!   MACH_MODELS=<dir> MACH_GRAPH=1 \
//!     cargo run -p mach-model --release --features hip --example qwen3_30b_graph_churn
//!
//! Drives the continuous engine exactly like the mach-server single-stream
//! loop: sequential requests, each an eager hipBLAS prefill followed by
//! ~512 greedy decode steps (graph-captured when MACH_GRAPH=1). Greedy decode
//! is deterministic, so every request must produce the IDENTICAL token
//! sequence; the first divergence (or a suspiciously fast request, the
//! no-op-replay signature) pinpoints where the graph path breaks.
#[cfg(feature = "hip")]
use mach_model::continuous::ContinuousModel;
#[cfg(feature = "hip")]
use mach_model::loader::load_safetensors_q4;
#[cfg(feature = "hip")]
use mach_model::sampling::SamplingParams;
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
fn run_engine(eng: &mut ContinuousModel, prompt: &[u32], n_reqs: usize, n_new: usize) -> bool {
    let mut reference: Option<Vec<u32>> = None;
    let mut failed = false;
    for r in 0..n_reqs {
        let t = std::time::Instant::now();
        let id = eng
            .add(
                prompt,
                n_new,
                None,
                Vec::new(),
                Vec::new(),
                SamplingParams::default(),
            )
            .expect("add");
        while !eng.is_done(id) {
            eng.step().expect("step");
        }
        let got = eng.generated(id);
        let el = t.elapsed().as_secs_f64();
        let status = match &reference {
            None => {
                reference = Some(got.clone());
                "ref".to_string()
            }
            Some(want) if *want == got => "match".to_string(),
            Some(want) => {
                failed = true;
                let div = want
                    .iter()
                    .zip(got.iter())
                    .position(|(a, b)| a != b)
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "len".to_string());
                format!("DIVERGE@{div}")
            }
        };
        println!(
            "req {r}: {} tokens in {el:.2}s ({:.1} ms/tok) [{status}]",
            got.len(),
            el * 1000.0 / got.len().max(1) as f64
        );
    }
    failed
}

/// Raw BatchedModel drive, no engine: each "request" is an optional eager
/// prefill chunk followed by `n_new` explicit decode steps — isolates whether
/// the corruption needs the engine's bookkeeping or just the
/// prefill↔decode-graph alternation (or nothing but explicit-path replays).
#[cfg(feature = "hip")]
fn run_raw(
    model: &mut mach_model::batched::BatchedModel,
    prompt: &[u32],
    n_reqs: usize,
    n_new: usize,
    prefill: bool,
) -> bool {
    let mut reference: Option<Vec<u32>> = None;
    let mut failed = false;
    for r in 0..n_reqs {
        let t = std::time::Instant::now();
        let mut pos = 0u32;
        let mut tok;
        if prefill {
            let lens: Vec<u32> = (0..prompt.len() as u32).collect();
            let slots = vec![0u32; prompt.len()];
            let mut params = vec![SamplingParams::default(); prompt.len()];
            let counts = vec![Vec::new(); prompt.len()];
            let bias = vec![Vec::new(); prompt.len()];
            let out = model
                .decode_step_explicit(prompt, &lens, &slots, &mut params, &counts, &bias, false)
                .expect("prefill");
            tok = out.0[prompt.len() - 1];
            pos += prompt.len() as u32;
        } else {
            model.reset_state().expect("reset");
            tok = prompt[0];
        }
        let mut got = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let lens = [pos];
            let slots = [0u32];
            let mut params = [SamplingParams::default()];
            let counts: [Vec<(u32, u32)>; 1] = [Vec::new()];
            let bias: [Vec<(u32, f32)>; 1] = [Vec::new()];
            let out = model
                .decode_step_explicit(&[tok], &lens, &slots, &mut params, &counts, &bias, true)
                .expect("decode");
            tok = out.0[0];
            got.push(tok);
            pos += 1;
        }
        let el = t.elapsed().as_secs_f64();
        let status = match &reference {
            None => {
                reference = Some(got.clone());
                "ref".to_string()
            }
            Some(want) if *want == got => "match".to_string(),
            Some(want) => {
                failed = true;
                let div = want
                    .iter()
                    .zip(got.iter())
                    .position(|(a, b)| a != b)
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "len".to_string());
                format!("DIVERGE@{div}")
            }
        };
        println!(
            "req {r}: {} tokens in {el:.2}s ({:.1} ms/tok) [{status}]",
            got.len(),
            el * 1000.0 / got.len().max(1) as f64
        );
    }
    failed
}

#[cfg(feature = "hip")]
fn main() {
    let root = PathBuf::from(std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into()));
    let model_dir = root.join("qwen3-30b-a3b");
    let cfg = config_from_json(&model_dir.join("config.json"));
    let n_reqs = std::env::var("MACH_CHURN_REQS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8usize);
    let n_new = std::env::var("MACH_CHURN_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512usize);
    // Prompt length is a knob: >8 rows routes the prefill through eager
    // hipBLAS (the server shape); <=8 rows stays on the custom GEMV path.
    let prompt_len = std::env::var("MACH_CHURN_PROMPT_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16usize);
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|i| i * 997 + 11).collect();
    // Mode: "engine" (ContinuousModel, the server shape), "raw" (direct
    // BatchedModel prefill+decode), "raw_noprefill" (decode only, reset_state
    // between requests).
    let mode = std::env::var("MACH_CHURN_MODE").unwrap_or_else(|_| "engine".into());

    let w: WeightsQ4 = load_safetensors_q4(&model_dir, &cfg, true).expect("load q4");
    let hip = mach_kernel_sys::hip::hip().expect("HIP runtime");
    let failed = match mode.as_str() {
        "engine" => {
            let mut eng =
                ContinuousModel::with_prefill_rows_q4_device(hip, cfg, &w, 4, 512).expect("engine");
            run_engine(&mut eng, &prompt, n_reqs, n_new)
        }
        "raw" | "raw_noprefill" => {
            let mut model =
                mach_model::batched::BatchedModel::with_rows_q4_device(hip, cfg, &w, 4, 512)
                    .expect("model");
            run_raw(&mut model, &prompt, n_reqs, n_new, mode == "raw")
        }
        other => panic!("unknown MACH_CHURN_MODE {other}"),
    };
    if failed {
        eprintln!("FAIL: greedy output diverged across identical requests");
        std::process::exit(1);
    }
    println!("OK: {n_reqs} requests token-identical");
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!("qwen3_30b_graph_churn requires the `hip` feature");
}
