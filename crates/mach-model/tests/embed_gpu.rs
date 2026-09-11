//! Env-gated GPU parity for multimodal row-embedding injection.
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::batched::BatchedModel;
use mach_model::sampling::SamplingParams;
use mach_model::{Config, Weights};

#[test]
#[ignore = "GPU embedding injection parity; set MACH_TEST_EMBED_GPU=1"]
fn row_embedding_override_matches_normal_gather() {
    if std::env::var("MACH_TEST_EMBED_GPU").as_deref() != Ok("1") {
        eprintln!("skipping GPU embedding parity: MACH_TEST_EMBED_GPU is not 1");
        return;
    }
    let hip = hip::hip().expect("HIP runtime");
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 77).unwrap();
    let mut normal = BatchedModel::with_rows(hip.clone(), cfg, &w, 1, 4).unwrap();
    let mut injected = BatchedModel::with_rows(hip, cfg, &w, 1, 4).unwrap();
    let tokens = [3u32, 17, 42, 5];
    let lens = [0u32, 1, 2, 3];
    let slots = [0u32, 0, 0, 0];
    let d = cfg.d_model;
    let features: Vec<f32> = tokens
        .iter()
        .flat_map(|&t| {
            w.tok_emb[t as usize * d..(t as usize + 1) * d]
                .iter()
                .copied()
        })
        .collect();
    let mask = [1i32; 4];
    injected
        .set_row_embeddings(&features, &mask, tokens.len())
        .unwrap();
    let mut params_a = vec![SamplingParams::default(); tokens.len()];
    let mut params_b = params_a.clone();
    let counts = vec![Vec::new(); tokens.len()];
    let bias = vec![Vec::new(); tokens.len()];
    normal
        .decode_step_explicit(&tokens, &lens, &slots, &mut params_a, &counts, &bias, false)
        .unwrap();
    injected
        .decode_step_explicit(&tokens, &lens, &slots, &mut params_b, &counts, &bias, false)
        .unwrap();
    let a = normal.read_logits_rows(tokens.len()).unwrap();
    let b = injected.read_logits_rows(tokens.len()).unwrap();
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(
            x.is_finite() && y.is_finite(),
            "non-finite logit at {i}: {x} vs {y}"
        );
        let diff = (x - y).abs();
        assert!(
            diff < 1e-5,
            "embedding override mismatch at {i}: {x} vs {y}"
        );
    }
}
