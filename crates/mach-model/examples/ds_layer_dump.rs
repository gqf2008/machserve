//! DeepSeek-V2-Lite Q4-on-device 逐层 hidden-state dump (issue #107).
//!
//!   MACH_PROMPT_IDS="1,2,3..." MACH_MODELS=<root> \
//!     cargo run -p mach-model --release --features hip --example ds_layer_dump
//!
//! Loads deepseek-v2-lite-chat through the storage-Q4 loader, builds the
//! Q4-on-device batched model (expert pool stays int4 in VRAM), feeds the
//! PROMPT_IDS one token at a time, and snapshots the last row's residual
//! after EVERY layer of the final step to `.scratch/dump_ds/hs_L{li:02}.npy`
//! (via `BatchedModel::set_layer_dump`). Those files line up 1:1 with what
//! `.scratch/np_ref.py` records at `REC_POS` (`FAKE_Q4=1 DUMP=1`), so the
//! first layer where the device forward diverges from the host reference is
//! read straight off a diff. Ids MUST come from the same tokenizer run as
//! the reference (np_ref prints them; PROMPT_IDS feeds them back in) — the
//! Rust tokenizer is deliberately not used here so it can never smuggle a
//! tokenization difference into the comparison.
#[cfg(feature = "hip")]
use mach_model::batched::BatchedModel;
#[cfg(feature = "hip")]
use mach_model::loader::load_safetensors_q4;
#[cfg(feature = "hip")]
use mach_model::{Config, WeightsQ4};
#[cfg(feature = "hip")]
use std::path::PathBuf;

/// V2-Lite-relevant subset of the server's config_from_json (same mapping,
/// same conventions; the family allowlist collapses to a model_type assert
/// because this example only accepts the V2-Lite checkpoint).
#[cfg(feature = "hip")]
fn config_from_json(path: &std::path::Path) -> Config {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read config"))
            .expect("parse config");
    let model_type = v["model_type"].as_str().unwrap_or_default().to_string();
    let family: String = model_type
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    assert!(
        ["deepseekv2", "deepseekv3", "deepseekvlv2"]
            .iter()
            .any(|p| family.starts_with(p)),
        "ds_layer_dump expects the DeepSeek MLA lineage, got model_type {model_type:?}"
    );
    let hidden = v["hidden_size"].as_u64().unwrap_or(2048) as usize;
    let layers = v["num_hidden_layers"].as_u64().unwrap_or(27) as usize;
    let heads = v["num_attention_heads"].as_u64().unwrap_or(16) as usize;
    let kv = v["num_key_value_heads"].as_u64().unwrap_or(heads as u64) as usize;
    let vocab = v["vocab_size"].as_u64().unwrap_or(102400) as usize;
    let inter = v["intermediate_size"].as_u64().unwrap_or(4 * hidden as u64) as usize;
    // The dump only walks a short prompt; a small max_seq keeps the
    // preallocated MLA KV cache tiny. YaRN numerics key off
    // original_max_position_embeddings, not this, so it stays identical.
    let max_seq = 512usize;
    let eps = v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    let theta = v["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let mut cfg = Config::llama(hidden, layers, heads, kv, vocab, max_seq);
    cfg.intermediate_size = inter;
    cfg.rms_eps = eps;
    cfg.rope_theta = theta;
    // MLA lineage rotates ADJACENT coordinates (#104): allowlist, same as the
    // server (V1/DeepSeekMoE's `model_type: "deepseek"` is excluded above).
    cfg.rope_interleave = true;
    cfg.q_lora_rank = v["q_lora_rank"].as_u64().unwrap_or(0) as usize;
    cfg.kv_lora_rank = v["kv_lora_rank"].as_u64().unwrap_or(0) as usize;
    cfg.qk_nope_head_dim = v["qk_nope_head_dim"].as_u64().unwrap_or(0) as usize;
    cfg.qk_rope_head_dim = v["qk_rope_head_dim"].as_u64().unwrap_or(0) as usize;
    cfg.v_head_dim = v["v_head_dim"].as_u64().unwrap_or(0) as usize;
    assert!(
        cfg.kv_lora_rank > 0
            && cfg.qk_nope_head_dim > 0
            && cfg.qk_rope_head_dim > 0
            && cfg.v_head_dim > 0,
        "DeepSeek MLA config must carry kv_lora_rank + head dims"
    );
    cfg.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
    cfg.n_kv_heads = cfg.n_heads;
    cfg.num_experts = v["num_experts"]
        .as_u64()
        .or_else(|| v["n_routed_experts"].as_u64())
        .unwrap_or(0) as usize;
    cfg.num_experts_per_tok = v["num_experts_per_tok"].as_u64().unwrap_or(0) as usize;
    cfg.moe_intermediate_size = v["moe_intermediate_size"].as_u64().unwrap_or(0) as usize;
    cfg.n_shared_experts = v["n_shared_experts"].as_u64().unwrap_or(0) as usize;
    cfg.moe_norm_topk = v["norm_topk_prob"].as_bool().unwrap_or(true);
    cfg.moe_routed_scale = v["routed_scaling_factor"].as_f64().unwrap_or(1.0) as f32;
    let scoring = v["scoring_func"].as_str().unwrap_or("softmax");
    let topk_method = v["topk_method"].as_str().unwrap_or("greedy");
    assert!(
        scoring == "softmax" && topk_method == "greedy",
        "unsupported router scoring/topk ({scoring:?}/{topk_method:?})"
    );
    // YaRN (DeepSeek-V2): same parameter plumbing as the server mapping.
    if let Some(rs) = v["rope_scaling"].as_object() {
        match rs.get("type").and_then(|t| t.as_str()) {
            Some("yarn") => {
                cfg.rope_yarn_factor = rs["factor"].as_f64().unwrap_or(1.0) as f32;
                cfg.rope_yarn_orig_len =
                    rs["original_max_position_embeddings"].as_u64().unwrap_or(0) as usize;
                cfg.rope_yarn_beta_fast = rs["beta_fast"].as_f64().unwrap_or(32.0) as f32;
                cfg.rope_yarn_beta_slow = rs["beta_slow"].as_f64().unwrap_or(1.0) as f32;
                cfg.rope_yarn_mscale = rs["mscale"].as_f64().unwrap_or(1.0) as f32;
                cfg.rope_yarn_mscale_all_dim = rs["mscale_all_dim"].as_f64().unwrap_or(0.0) as f32;
                assert!(cfg.yarn(), "yarn config incomplete");
            }
            Some(other) => panic!("unsupported rope_scaling type {other:?}"),
            None => {}
        }
    }
    cfg.dtype = mach_model::config::ModelDType::F16;
    cfg
}

#[cfg(feature = "hip")]
fn main() {
    let root = PathBuf::from(std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into()));
    let model_dir = root.join("deepseek-v2-lite-chat");
    let cfg_path = model_dir.join("config.json");
    assert!(cfg_path.exists(), "missing {cfg_path:?}");
    let cfg = config_from_json(&cfg_path);
    println!(
        "config: d={} layers={} experts={} topk={} moe_inter={} shared={} vocab={} rope_interleave={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.expert_size(),
        cfg.shared_size(),
        cfg.vocab_size,
        cfg.rope_interleave
    );

    let ids: Vec<u32> = std::env::var("MACH_PROMPT_IDS")
        .expect(
            "MACH_PROMPT_IDS is required (comma/space separated; take them from np_ref's stderr)",
        )
        .replace(',', " ")
        .split_whitespace()
        .map(|s| s.parse().expect("token id"))
        .collect();
    assert!(!ids.is_empty(), "empty prompt");
    // max_seq is 512 and the generation loop below walks 24 more steps; keep
    // the prompt short enough that greedy decode cannot run past the cache.
    assert!(ids.len() + 24 < 512, "prompt longer than max_seq headroom");
    println!("prompt ids ({}): {ids:?}", ids.len());

    let t0 = std::time::Instant::now();
    let w: WeightsQ4 = load_safetensors_q4(&model_dir, &cfg, true).expect("load q4");
    println!("load: {:.1}s", t0.elapsed().as_secs_f64());

    let hip = mach_kernel_sys::hip::hip().expect("HIP runtime");
    assert!(
        mach_kernel_sys::hip::device_count().expect("devices") > 0,
        "no device"
    );

    let t1 = std::time::Instant::now();
    let mut model = BatchedModel::with_rows_q4_device(hip, cfg, &w, 1, 1).expect("build");
    println!("build: {:.1}s", t1.elapsed().as_secs_f64());
    drop(w);

    let dump_dir = PathBuf::from(".scratch/dump_ds");
    model.set_layer_dump(&dump_dir).expect("layer dump dir");

    // One token per decode_step: every step rewrites the per-layer files, so
    // after the loop they hold the LAST position's trace (REC_POS = len-1).
    for &t in &ids {
        model.decode_step(&[t]).expect("decode step");
    }
    println!(
        "dumped {} layer files to {} (pos {})",
        cfg.n_layers,
        dump_dir.display(),
        ids.len() - 1
    );

    // The #107 question, asked directly: is THIS model's own text coherent?
    // The layer-diff above only shows the server sits on a different (but
    // equally valid, tie-rounded) Q4 realization than the numpy sim —
    // coherence of each realization must be judged separately. Greedy-
    // generate and print for eyeball comparison against the numpy reference
    // ("The capital of France is Paris.").
    let tok = mach_model::tokenizer::Tokenizer::from_path(&model_dir.join("tokenizer.json"))
        .expect("load tokenizer");
    model.clear_layer_dump();
    let mut cur = *ids.last().expect("non-empty");
    let mut out_ids = Vec::new();
    for step in 0..24 {
        let logits = model.read_logits().expect("logits");
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        println!(
            "out_ids step {step}: argmax={next} ({:.2}) top2 margin {:.2}",
            logits[next as usize],
            {
                let mut s: Vec<f32> = logits.to_vec();
                s.select_nth_unstable_by(cfg.vocab_size - 2, |a, b| a.partial_cmp(b).unwrap());
                logits[next as usize] - s[cfg.vocab_size - 2]
            }
        );
        out_ids.push(next);
        if next == 100001 {
            break; // <|end▁of▁sentence|>
        }
        cur = next;
        model.decode_step(&[cur]).expect("decode step");
    }
    println!("GENERATED: {}", tok.decode(&out_ids));
    let _ = cur;
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!("ds_layer_dump requires the `hip` feature");
}
