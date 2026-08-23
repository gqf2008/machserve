//! Speculative decoding must produce the exact plain-greedy output of the
//! target model (argmax acceptance), even when the draft is a weak predictor.
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::batched::BatchedModel;
use mach_model::config::ModelDType;
use mach_model::sampling::SamplingParams;
use mach_model::speculative::SpeculativeDecoder;
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

/// Plain greedy generation from a target-only model (emit-then-predict).
fn plain_greedy(
    hip: &std::sync::Arc<hip::Hip>,
    cfg: Config,
    w: &Weights,
    prompt: &[u32],
    max_new: usize,
) -> Vec<u32> {
    let mut m = BatchedModel::new(hip.clone(), cfg, w, 64).unwrap();
    let lens: Vec<u32> = (0..prompt.len() as u32).collect();
    let slots = vec![0u32; prompt.len()];
    let mut gp = vec![SamplingParams::greedy(0); prompt.len()];
    m.decode_step_explicit(
        prompt,
        &lens,
        &slots,
        &mut gp,
        &vec![Vec::new(); prompt.len()],
        &vec![Vec::new(); prompt.len()],
    )
    .unwrap();
    let first = m
        .decode_step_explicit(
            &[prompt[prompt.len() - 1]],
            &[(prompt.len() - 1) as u32],
            &[0],
            &mut [SamplingParams::greedy(0)],
            &vec![Vec::new(); 1],
            &vec![Vec::new(); 1],
        )
        .unwrap()
        .0[0];
    let mut rnext = first;
    let mut out = Vec::new();
    for rpos in prompt.len()..prompt.len() + max_new {
        out.push(rnext);
        let mut p = [SamplingParams::greedy(0)];
        rnext = m
            .decode_step_explicit(
                &[rnext],
                &[rpos as u32],
                &[0],
                &mut p,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
            )
            .unwrap()
            .0[0];
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn spec_decode(
    hip: &std::sync::Arc<hip::Hip>,
    dcfg: Config,
    dw: &Weights,
    tcfg: Config,
    tw: &Weights,
    prompt: &[u32],
    max_new: usize,
    k: usize,
) -> Vec<u32> {
    let draft = BatchedModel::new(hip.clone(), dcfg, dw, 64).unwrap();
    let target = BatchedModel::new(hip.clone(), tcfg, tw, 64).unwrap();
    let mut dec = SpeculativeDecoder::new(draft, target, k, prompt).unwrap();
    let mut out = Vec::new();
    while out.len() < max_new {
        for t in dec.step().unwrap() {
            if out.len() < max_new {
                out.push(t);
            }
        }
    }
    out
}

#[test]
fn spec_decode_matches_plain_greedy() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = Config::tiny();
    cfg.dtype = ModelDType::F32;
    // Draft and target are different random models with the same shapes/vocab
    // (the draft is a weak predictor -> low acceptance, which still must keep
    // the greedy output identical).
    let dw = Weights::random(&cfg, 61).unwrap();
    let tw = Weights::random(&cfg, 73).unwrap();
    for (prompt, n) in [
        (vec![5u32, 9, 3, 200, 44, 88], 12usize),
        (vec![1u32], 20usize),
        (vec![300u32, 77, 5, 9, 3, 200, 44, 88, 1, 22, 333], 25usize),
    ] {
        let want = plain_greedy(&hip, cfg, &tw, &prompt, n);
        for k in [1usize, 2, 4] {
            let got = spec_decode(&hip, cfg, &dw, cfg, &tw, &prompt, n, k);
            assert_eq!(
                got,
                want,
                "spec-decode (prompt len {}, k={k}) must equal plain greedy, got {got:?} want {want:?}",
                prompt.len()
            );
        }
    }
}

#[test]
fn spec_decode_full_accept_matches_plain_greedy() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = Config::tiny();
    cfg.dtype = ModelDType::F32;
    // Draft == target (same weights): every draft token is accepted, so the
    // a == k full-accept path is exercised each round.
    let w = Weights::random(&cfg, 61).unwrap();
    let prompt: Vec<u32> = vec![5, 9, 3, 200, 44, 88];
    let want = plain_greedy(&hip, cfg, &w, &prompt, 12);
    for k in [1usize, 2, 4] {
        let got = spec_decode(&hip, cfg, &w, cfg, &w, &prompt, 12, k);
        assert_eq!(
            got, want,
            "spec-decode full-accept (k={k}) must equal plain greedy, got {got:?} want {want:?}"
        );
    }
}
