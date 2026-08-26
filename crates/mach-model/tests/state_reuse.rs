//! Agentic state reuse: token-boundary anchors + incremental prefill.
//!
//! Correctness contract: a prefix restored from an anchor and then extended
//! with the delta must produce **exactly** the logits of a full recompute
//! (CPU reference pair, independent implementation). GPU parity tests are
//! `#[ignore]`d (opt-in, serial: `-- --ignored --test-threads=1`).

use mach_model::config::ModelDType;
use mach_model::ref_model::RefModel;
use mach_model::state_reuse::{AnchorStore, StateReuse};
use mach_model::{Config, Weights};

fn cfg_dense() -> Config {
    Config::tiny()
}

fn cfg_moe() -> Config {
    let mut cfg = Config::tiny();
    cfg.intermediate_size = 64;
    cfg.moe_intermediate_size = 48;
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg
}

fn cfg_mla() -> Config {
    Config::mla(128, 2, 4, 1024, 256, 8, 16, 64, 64, 64)
}

/// Per-position logits of a full recompute (independent reference).
fn full_recompute_logits(cfg: Config, w: &Weights, tokens: &[u32]) -> Vec<Vec<f32>> {
    let mut m = RefModel::new(cfg, w.clone());
    tokens.iter().map(|&t| m.decode_step(t)).collect::<Vec<_>>()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_reuse_vs_full(cfg: Config, seed: u64, tokens: &[u32], split: usize) {
    let w = Weights::random(&cfg, seed).unwrap();
    assert!(
        split > 0 && split < tokens.len(),
        "split must leave a delta"
    );
    let (prefix, delta) = tokens.split_at(split);

    // Independent reference: full recompute, per-position logits captured.
    let full = full_recompute_logits(cfg, &w, tokens);
    let full_final = full.last().unwrap().clone();

    // Reuse path: process the prefix, save an anchor at its end, restore into
    // a fresh model, then prefill only the delta.
    let mut first = RefModel::new(cfg, w.clone());
    for &t in prefix {
        first.decode_step(t);
    }
    let anchor = first.save_anchor(prefix, split - 1).expect("save anchor");

    let mut probe = RefModel::new(cfg, w);
    probe.restore_anchor(&anchor).expect("restore anchor");
    let mut reuse_logits = Vec::new();
    for &t in delta {
        reuse_logits = probe.decode_step(t);
    }

    // Final logits must match exactly (same math, same order on the CPU ref).
    let max = max_abs_diff(&full_final, &reuse_logits);
    assert_eq!(
        max, 0.0,
        "reuse vs full-recompute final logits must be identical (max diff {max})"
    );
    assert_eq!(
        probe.pos(),
        tokens.len(),
        "restored model must end at full length"
    );

    // Anchor hidden state: logits-at-anchor equals the full recompute's logits
    // at the prefix end (position split-1).
    let at_anchor = first.logits_at_anchor();
    let max_a = max_abs_diff(&full[split - 1], &at_anchor);
    assert_eq!(
        max_a, 0.0,
        "logits_at_anchor must match full recompute at the anchor position (max diff {max_a})"
    );
}

#[test]
fn dense_restore_reproduces_full_recompute() {
    let tokens: Vec<u32> = vec![5, 9, 3, 200, 7, 42, 1, 88];
    run_reuse_vs_full(cfg_dense(), 11, &tokens, 5);
}

#[test]
fn dense_restore_short_prefix() {
    let tokens: Vec<u32> = vec![5, 9, 3, 200, 7];
    run_reuse_vs_full(cfg_dense(), 23, &tokens, 2);
}

#[test]
fn moe_restore_reproduces_full_recompute() {
    // The real P3 model (qwen3-moe-tiny) is MoE; exercise the routed path.
    let tokens: Vec<u32> = vec![5, 9, 3, 200, 7, 42, 1, 88, 300, 12];
    run_reuse_vs_full(cfg_moe(), 7, &tokens, 6);
}

#[test]
fn mla_restore_reproduces_full_recompute() {
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    run_reuse_vs_full(cfg_mla(), 31, &tokens, 4);
}

#[test]
fn anchor_store_finds_longest_matching_prefix() {
    let cfg = cfg_dense();
    let w = Weights::random(&cfg, 3).unwrap();
    let mut sr = StateReuse::new(16);

    // Build two anchors at different depths of the same token stream.
    let mut m = RefModel::new(cfg, w.clone());
    let tokens: Vec<u32> = vec![10, 20, 30, 40, 50, 60];
    let mut ids = Vec::new();
    for (i, &t) in tokens.iter().enumerate() {
        m.decode_step(t);
        if i == 1 || i == 3 {
            ids.push(sr.insert_anchor(m.save_anchor(&tokens[..=i], i).unwrap()));
        }
    }
    assert_eq!(sr.store().len(), 2);

    // A longer query sharing both prefixes must match the deeper anchor.
    let query: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 99, 100];
    let reused = sr
        .find_reusable(&query)
        .expect("must find a reusable prefix");
    assert_eq!(reused.prefix_len, 4, "longest matching prefix wins");
    assert_eq!(reused.anchor_id, ids[1]);

    let stats = sr.stats();
    assert_eq!(stats.lookups, 1);
    assert_eq!(stats.hits, 1);
    assert_eq!(stats.tokens_reused, 4);

    // A query with a different prefix must miss.
    let miss = sr.find_reusable(&[1, 2, 3, 4, 5, 6, 7]);
    assert!(miss.is_none(), "non-matching prefix must miss");
    assert_eq!(sr.stats().hits, 1);
    assert_eq!(sr.stats().lookups, 2);
}

#[test]
fn anchor_store_evicts_oldest_when_full() {
    let mut store = AnchorStore::new(2);
    let cfg = cfg_dense();
    let w = Weights::random(&cfg, 5).unwrap();
    let mut m = RefModel::new(cfg, w);
    m.decode_step(1);
    let a1 = store.insert(m.save_anchor(&[1], 0).unwrap());
    m.decode_step(2);
    let a2 = store.insert(m.save_anchor(&[1, 2], 1).unwrap());
    m.decode_step(3);
    let a3 = store.insert(m.save_anchor(&[1, 2, 3], 2).unwrap());

    assert_eq!(store.len(), 2, "capacity must be enforced");
    assert!(store.get(a1).is_none(), "oldest anchor must be evicted");
    assert!(store.get(a2).is_some());
    assert!(store.get(a3).is_some());
    assert_eq!(store.max_anchors(), 2);
}

#[test]
fn anchor_store_zero_capacity_stores_nothing() {
    // max_anchors=0 must retain nothing; before the fix the pop_front guard
    // no-op'd and the store grew unbounded.
    let mut store = AnchorStore::new(0);
    let cfg = cfg_dense();
    let w = Weights::random(&cfg, 7).unwrap();
    let mut m = RefModel::new(cfg, w);
    m.decode_step(1);
    let id = store.insert(m.save_anchor(&[1], 0).unwrap());
    m.decode_step(2);
    store.insert(m.save_anchor(&[1, 2], 1).unwrap());
    assert_eq!(store.len(), 0, "zero-capacity store must stay empty");
    assert!(store.get(id).is_none());
    assert_eq!(store.bytes_held(), 0);
}

#[test]
fn save_anchor_rejects_wrong_position() {
    let cfg = cfg_dense();
    let w = Weights::random(&cfg, 9).unwrap();
    let mut m = RefModel::new(cfg, w);
    m.decode_step(7);
    m.decode_step(8);
    // Only 2 tokens processed: token_idx 3 (4 tokens) must be rejected.
    assert!(m.save_anchor(&[7, 8, 9, 10], 3).is_err());
    // Mismatched token list length must be rejected too.
    assert!(m.save_anchor(&[7, 8], 5).is_err());
}

#[test]
fn restore_anchor_rejects_mismatched_layout() {
    let cfg = cfg_dense();
    let w = Weights::random(&cfg, 13).unwrap();
    let mut m = RefModel::new(cfg, w.clone());
    m.decode_step(4);
    let anchor = m.save_anchor(&[4], 0).unwrap();

    // Hidden-size mismatch.
    let mut bad = anchor.clone();
    bad.hidden = vec![0.0; cfg.d_model + 1];
    let mut m2 = RefModel::new(cfg, w.clone());
    assert!(m2.restore_anchor(&bad).is_err());

    // Layer-count mismatch.
    let mut bad2 = anchor.clone();
    bad2.kv.layers.pop();
    let mut m3 = RefModel::new(cfg, w);
    assert!(m3.restore_anchor(&bad2).is_err());
}

#[test]
fn restore_anchor_rejects_cross_config() {
    let w1 = Weights::random(&cfg_dense(), 2).unwrap();
    let mut m = RefModel::new(cfg_dense(), w1);
    m.decode_step(4);
    let anchor = m.save_anchor(&[4], 0).unwrap();

    // A config with a different layer count must fail the layer-count check.
    let mut cfg2 = cfg_dense();
    cfg2.n_layers = 3;
    let w2 = Weights::random(&cfg2, 2).unwrap();
    let mut m2 = RefModel::new(cfg2, w2);
    assert!(m2.restore_anchor(&anchor).is_err());
}

#[test]
fn fp16_dtype_cfg_uses_same_anchor_path() {
    // The anchor path is dtype-independent on the CPU reference (f32 weights),
    // but the GPU F16 path serializes fp16 KV bytes; make sure an F16-flagged
    // config still round-trips the CPU reference.
    let mut cfg = cfg_dense();
    cfg.dtype = ModelDType::F16;
    let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    run_reuse_vs_full(cfg, 17, &tokens, 3);
}

#[cfg(feature = "hip")]
mod gpu {
    use super::*;
    use mach_kernel_sys::hip;
    use mach_model::batched::BatchedModel;
    use mach_model::continuous::ContinuousModel;
    use mach_model::sampling::SamplingParams;
    use std::sync::Arc;

    fn hip_ctx() -> Option<Arc<hip::Hip>> {
        match hip::hip() {
            Ok(h) => match hip::device_count() {
                Ok(n) if n > 0 => Some(h),
                _ => {
                    eprintln!("skipping HIP test: no device");
                    None
                }
            },
            Err(e) => {
                eprintln!("skipping HIP test: {e}");
                None
            }
        }
    }

    /// GPU anchor restore must reproduce a full GPU recompute (and the CPU
    /// reference) within the repo's standard tolerance.
    #[ignore = "GPU (opt-in: -- --ignored --test-threads=1)"]
    #[test]
    fn gpu_anchor_restore_matches_full_recompute() {
        let Some(hip) = hip_ctx() else { return };
        let cfg = cfg_moe();
        let w = Weights::random(&cfg, 7).unwrap();
        let tokens: Vec<u32> = vec![5, 9, 3, 200, 7, 42, 1, 88];
        let split = 5;
        let (prefix, delta) = tokens.split_at(split);

        // Full recompute on GPU (batch 1).
        let mut full = BatchedModel::new(Arc::clone(&hip), cfg, &w, 1).unwrap();
        for &t in &tokens {
            full.decode_step(&[t]).unwrap();
        }
        let full_logits = full.read_logits().unwrap();
        let scale = full_logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));

        // Reuse path: prefill the prefix, save an anchor, restore it into a
        // fresh model, then prefill only the delta.
        let mut a = BatchedModel::new(Arc::clone(&hip), cfg, &w, 1).unwrap();
        for &t in prefix {
            a.decode_step(&[t]).unwrap();
        }
        let anchor = a.save_anchor(0, prefix, split - 1).unwrap();
        assert_eq!(anchor.token_idx, split - 1);
        assert_eq!(anchor.tokens, prefix);

        let mut b = BatchedModel::new(Arc::clone(&hip), cfg, &w, 1).unwrap();
        b.restore_anchor(0, &anchor).unwrap();
        for &t in delta {
            b.decode_step(&[t]).unwrap();
        }
        let reuse_logits = b.read_logits().unwrap();

        let max = max_abs_diff(&full_logits, &reuse_logits);
        assert!(
            max <= 2e-3 + 2e-3 * scale,
            "GPU reuse vs full recompute: max diff {max} (scale {scale})"
        );

        // Independent reference: CPU full recompute.
        let mut cpu = RefModel::new(cfg, w);
        let mut cpu_logits = Vec::new();
        for &t in &tokens {
            cpu_logits = cpu.decode_step(t);
        }
        let max_cpu = max_abs_diff(&full_logits, &cpu_logits);
        assert!(
            max_cpu <= 2e-3 + 2e-3 * scale,
            "GPU full vs CPU: max diff {max_cpu} (scale {scale})"
        );
    }

    /// End-to-end multi-turn reuse in ContinuousModel: turn 2 with reuse must
    /// produce the exact same greedy output as turn 2 without reuse, while
    /// skipping the shared prefix (tokens_reused == prefix length).
    #[ignore = "GPU (opt-in: -- --ignored --test-threads=1)"]
    #[test]
    fn gpu_continuous_reuse_matches_full_prefill_generation() {
        let Some(hip) = hip_ctx() else { return };
        let cfg = cfg_moe();
        let w = Weights::random(&cfg, 7).unwrap();
        let prompt1: Vec<u32> = vec![5, 9, 3, 200];
        let user2: Vec<u32> = vec![42, 1, 88];

        let run = |reuse: bool| -> (Vec<u32>, Option<mach_model::state_reuse::ReuseStats>) {
            let mut eng = if reuse {
                ContinuousModel::with_state_reuse(
                    Arc::clone(&hip),
                    cfg,
                    &w,
                    2,
                    mach_model::state_reuse::StateReuse::new(8),
                )
                .unwrap()
            } else {
                ContinuousModel::new(Arc::clone(&hip), cfg, &w, 2).unwrap()
            };
            let params = SamplingParams::default();
            // Turn 1: generate a short deterministic assistant response.
            let id1 = eng
                .add(&prompt1, 3, None, Vec::new(), Vec::new(), params)
                .unwrap();
            while !eng.is_done(id1) {
                eng.step().unwrap();
            }
            let resp1 = eng.generated(id1);
            assert_eq!(resp1.len(), 3);
            // Turn 2: full context = turn1 prompt + turn1 response + new user msg.
            let mut prompt2 = prompt1.clone();
            prompt2.extend_from_slice(&resp1);
            prompt2.extend_from_slice(&user2);
            let id2 = eng
                .add(&prompt2, 4, None, Vec::new(), Vec::new(), params)
                .unwrap();
            let mut out = Vec::new();
            while !eng.is_done(id2) {
                for (_, t) in eng.step().unwrap() {
                    out.push(t);
                }
            }
            let stats = eng.reuse_stats();
            (out, stats)
        };

        let (baseline, stats_base) = run(false);
        let (reused, stats_reuse) = run(true);
        assert_eq!(
            baseline, reused,
            "reuse must not change the generated output"
        );

        let s = stats_reuse.expect("reuse mode must report stats");
        assert!(s.hits >= 1, "turn 2 must hit an anchor");
        assert_eq!(
            s.tokens_reused, 7,
            "turn 2 must skip prompt1 + turn1 response (7 tokens), got {}",
            s.tokens_reused
        );
        // Baseline engine (no reuse mode) reports no stats.
        assert!(stats_base.is_none());
    }
}
