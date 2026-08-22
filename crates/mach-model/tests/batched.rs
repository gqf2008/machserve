//! Batched decode correctness: a batched step must equal running each sequence
//! through the (already validated) single-sequence model.
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::batched::BatchedModel;
use mach_model::model::GpuModel;
use mach_model::{Config, Weights};

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
