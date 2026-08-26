//! Batched decode correctness: a batched step must equal running each sequence
//! through the (already validated) single-sequence model.
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::batched::BatchedModel;
use mach_model::config::ModelDType;
use mach_model::model::GpuModel;
use mach_model::{Config, Weights};

/// Paged-KV decode must match the static (contiguous) decode on the GPU: the
/// paged kernels route the same KV through block tables, so logits must agree
/// tightly across a long sequence spanning multiple pages.
#[test]
fn batched_paged_decode_matches_static_gpu() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny(); // max_seq 256, tokens_per_page 64 -> 4 pages
    let w = Weights::random(&cfg, 61).unwrap();
    let mut stat = BatchedModel::new(hip.clone(), cfg, &w, 1).unwrap();
    let mut paged = BatchedModel::with_paged_kv(hip.clone(), cfg, &w, 1, 64).unwrap();

    // 70 single-token steps spans 2 pages (64 + 6), exercising the block-table
    // page boundary on the decode path.
    let tokens: Vec<u32> = (0..70u32).map(|i| (i * 37) % 1024 + 1).collect();
    for (i, &t) in tokens.iter().enumerate() {
        stat.decode_step(&[t]).unwrap();
        paged.decode_step(&[t]).unwrap();
        let sl = stat.read_logits().unwrap();
        let pl = paged.read_logits().unwrap();
        assert_eq!(sl.len(), cfg.vocab_size);
        let scale = sl.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = sl
            .iter()
            .zip(&pl)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            d <= 1e-4 + 1e-4 * scale,
            "pos {i}: paged vs static logits max diff {d} (scale {scale})"
        );
    }
}

fn hip_ctx() -> Option<std::sync::Arc<hip::Hip>> {
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

#[test]
fn batched_matches_single_seq() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 31).unwrap();
    let batch = 4usize;

    let mut batched = BatchedModel::new(hip.clone(), cfg, &w, batch).unwrap();
    // One single-seq model per sequence (independent state).
    let mut singles: Vec<GpuModel> = (0..batch)
        .map(|_| GpuModel::new(hip.clone(), cfg, &w).unwrap())
        .collect();

    // Steps: each sequence uses its own token stream.
    let steps: Vec<Vec<u32>> = vec![vec![5, 9, 33, 7], vec![12, 3, 1, 200], vec![8, 55, 4, 99]];

    for step_tokens in &steps {
        let got = batched.decode_step(step_tokens).unwrap();
        assert_eq!(got.len(), batch);
        for s in 0..batch {
            let want = singles[s].decode_step_sampled(step_tokens[s]).unwrap();
            assert_eq!(
                got[s], want,
                "step {step_tokens:?} seq {s}: batched={} single={}",
                got[s], want
            );
        }
    }
}

fn moe_cfg() -> Config {
    let mut cfg = Config::tiny();
    cfg.intermediate_size = 64;
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg
}

#[test]
fn batched_moe_matches_single_seq() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = moe_cfg();
    let w = Weights::random(&cfg, 51).unwrap();
    let batch = 4usize;

    let mut batched = BatchedModel::new(hip.clone(), cfg, &w, batch).unwrap();
    let mut singles: Vec<GpuModel> = (0..batch)
        .map(|_| GpuModel::new(hip.clone(), cfg, &w).unwrap())
        .collect();

    let steps: Vec<Vec<u32>> = vec![vec![5, 9, 33, 7], vec![12, 3, 1, 200], vec![8, 55, 4, 99]];
    for step_tokens in &steps {
        let got = batched.decode_step(step_tokens).unwrap();
        let got_logits = batched.read_logits().unwrap();
        assert_eq!(got.len(), batch);
        for s in 0..batch {
            let want_logits = singles[s].decode_step(step_tokens[s]).unwrap();
            let want = want_logits
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.partial_cmp(b).unwrap().then_with(|| j.cmp(i)))
                .map(|(i, _)| i as u32)
                .unwrap();
            assert_eq!(got[s], want, "step {step_tokens:?} seq {s}: greedy token");
            let row = &got_logits[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let scale = want_logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let max = row
                .iter()
                .zip(&want_logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max <= 2e-3 + 2e-3 * scale,
                "step {step_tokens:?} seq {s}: logits max diff {max} (scale {scale})"
            );
        }
    }
}

#[test]
fn batched_moe_f16_matches_single_seq() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = moe_cfg();
    cfg.dtype = ModelDType::F16;
    let mut w = Weights::random(&cfg, 77).unwrap();
    // Peak the router so fp16/fp32 paths select the same experts robustly
    // (fp16 router logits carry ~1e-3 rounding).
    for lw in w.layers.iter_mut() {
        for v in lw.moe_router.iter_mut() {
            *v *= 6.0;
        }
    }
    let batch = 4usize;

    let mut batched = BatchedModel::new(hip.clone(), cfg, &w, batch).unwrap();
    let mut singles: Vec<GpuModel> = (0..batch)
        .map(|_| GpuModel::new(hip.clone(), cfg, &w).unwrap())
        .collect();

    let steps: Vec<Vec<u32>> = vec![vec![5, 9, 33, 7], vec![12, 3, 1, 200], vec![8, 55, 4, 99]];
    for step_tokens in &steps {
        let got = batched.decode_step(step_tokens).unwrap();
        let got_logits = batched.read_logits().unwrap();
        for s in 0..batch {
            let want_logits = singles[s].decode_step(step_tokens[s]).unwrap();
            let want = want_logits
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.partial_cmp(b).unwrap().then_with(|| j.cmp(i)))
                .map(|(i, _)| i as u32)
                .unwrap();
            assert_eq!(got[s], want, "step {step_tokens:?} seq {s}: greedy token");
            let row = &got_logits[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let max = row
                .iter()
                .zip(&want_logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max < 0.1,
                "step {step_tokens:?} seq {s}: f16 logits max diff {max}"
            );
        }
    }
}

#[test]
fn batched_moe_small_config_matches_single_seq() {
    let Some(hip) = hip_ctx() else { return };
    // More realistic MoE shape: 8 experts / 3 active, batch spreads rows
    // across experts (grouping is exercised harder than the 4-expert tiny
    // config). Uses Config::small() dims but a smaller vocab for speed.
    let mut cfg = Config::small();
    cfg.vocab_size = 2048;
    cfg.max_seq_len = 128;
    cfg.num_experts = 8;
    cfg.num_experts_per_tok = 3;
    let w = Weights::random(&cfg, 123).unwrap();
    let batch = 8usize;

    let mut batched = BatchedModel::new(hip.clone(), cfg, &w, batch).unwrap();
    let mut singles: Vec<GpuModel> = (0..batch)
        .map(|_| GpuModel::new(hip.clone(), cfg, &w).unwrap())
        .collect();

    let steps: Vec<Vec<u32>> = vec![
        vec![5, 9, 33, 7, 200, 3, 11, 42],
        vec![12, 3, 1, 99, 55, 8, 4, 77],
    ];
    for step_tokens in &steps {
        let got = batched.decode_step(step_tokens).unwrap();
        let got_logits = batched.read_logits().unwrap();
        for s in 0..batch {
            let want_logits = singles[s].decode_step(step_tokens[s]).unwrap();
            let want = want_logits
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.partial_cmp(b).unwrap().then_with(|| j.cmp(i)))
                .map(|(i, _)| i as u32)
                .unwrap();
            assert_eq!(got[s], want, "step {step_tokens:?} seq {s}: greedy token");
            let row = &got_logits[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let scale = want_logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let max = row
                .iter()
                .zip(&want_logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max <= 2e-3 + 2e-3 * scale,
                "step {step_tokens:?} seq {s}: logits max diff {max} (scale {scale})"
            );
        }
    }
}
#[test]
fn batched_mixed_dense_moe_matches_single_seq() {
    let Some(hip) = hip_ctx() else { return };
    // Qwen3-MoE-style mixed checkpoint: layer 0 is dense (empty MoE tensors,
    // keeps its dense MLP weights), layers 1..n are routed MoE, and the
    // expert FFN width (`moe_intermediate_size`) differs from the dense MLP
    // width (`intermediate_size`). The batched path must dispatch per layer
    // (not on the config-level num_experts) and use expert_size() for the
    // expert GEMMs, otherwise it reads the wrong weight layout.
    let mut cfg = Config::tiny();
    cfg.intermediate_size = 64;
    cfg.moe_intermediate_size = 32;
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg.dtype = ModelDType::F32;
    let mut w = Weights::random(&cfg, 123).unwrap();
    w.layers[0].moe_router.clear();
    w.layers[0].moe_wg.clear();
    w.layers[0].moe_wu.clear();
    w.layers[0].moe_wd.clear();

    let batch = 4usize;
    let mut batched = BatchedModel::new(hip.clone(), cfg, &w, batch).unwrap();
    let mut singles: Vec<GpuModel> = (0..batch)
        .map(|_| GpuModel::new(hip.clone(), cfg, &w).unwrap())
        .collect();
    let steps: Vec<Vec<u32>> = vec![vec![5, 9, 33, 7], vec![12, 3, 1, 200]];
    for step_tokens in &steps {
        let got = batched.decode_step(step_tokens).unwrap();
        let got_logits = batched.read_logits().unwrap();
        assert_eq!(got.len(), batch);
        for s in 0..batch {
            let want_logits = singles[s].decode_step(step_tokens[s]).unwrap();
            let want = want_logits
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.partial_cmp(b).unwrap().then_with(|| j.cmp(i)))
                .map(|(i, _)| i as u32)
                .unwrap();
            assert_eq!(got[s], want, "step {step_tokens:?} seq {s}: greedy token");
            let row = &got_logits[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let scale = want_logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let max = row
                .iter()
                .zip(&want_logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max <= 2e-3 + 2e-3 * scale,
                "step {step_tokens:?} seq {s}: logits max diff {max} (scale {scale})"
            );
        }
    }
}

#[test]
fn batched_sequences_are_independent() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 47).unwrap();
    let batch = 2usize;

    let mut batched = BatchedModel::new(hip.clone(), cfg, &w, batch).unwrap();
    // seq0 always gets token 7, seq1 always gets token 42: outputs must differ
    // and stay stable per sequence.
    let mut prev: Option<Vec<u32>> = None;
    for _ in 0..4 {
        let got = batched.decode_step(&[7, 42]).unwrap();
        if let Some(p) = &prev {
            assert_eq!(got, *p, "per-sequence outputs must be deterministic");
        }
        assert_ne!(got[0], got[1], "different input tokens must diverge");
        prev = Some(got);
    }
}
