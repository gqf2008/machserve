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
        true,
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
            true,
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
                true,
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

#[test]
fn spec_decode_batch_matches_plain_greedy_per_sequence() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = Config::tiny();
    cfg.dtype = ModelDType::F32;
    let dw = Weights::random(&cfg, 61).unwrap();
    let tw = Weights::random(&cfg, 73).unwrap();
    let capacity = 3usize;
    let k = 4usize;
    let prompts: Vec<Vec<u32>> = vec![vec![5, 9, 3, 200], vec![44, 88, 1, 300, 77], vec![2, 3, 4]];
    let max_new = 8usize;

    // Per-sequence plain-greedy references.
    let mut wants = Vec::new();
    for p in &prompts {
        wants.push(plain_greedy(&hip, cfg, &tw, p, max_new));
    }

    let draft =
        mach_model::batched::BatchedModel::with_rows(hip.clone(), cfg, &dw, capacity, capacity)
            .unwrap();
    let target = mach_model::batched::BatchedModel::with_rows(
        hip.clone(),
        cfg,
        &tw,
        capacity,
        capacity * (k + 1),
    )
    .unwrap();
    let mut batch = mach_model::speculative::SpeculativeBatch::new(draft, target, k, capacity);
    for p in &prompts {
        batch.add(p).unwrap();
    }
    let mut got: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    while got.iter().any(|g| g.len() < max_new) {
        let accepted = batch.step().unwrap();
        for (s, seq) in accepted.iter().enumerate() {
            if let Some(seq) = seq {
                for &t in seq {
                    if got[s].len() < max_new {
                        got[s].push(t);
                    }
                }
            }
        }
    }
    for (s, p) in prompts.iter().enumerate() {
        assert_eq!(
            got[s], wants[s],
            "batched spec-decode seq {s} (prompt {:?}) must equal plain greedy, got {:?} want {:?}",
            p, got[s], wants[s]
        );
    }
}

/// EOS/max_new-aware plain greedy reference (emit-then-predict).
fn plain_greedy_eos(
    hip: &std::sync::Arc<hip::Hip>,
    cfg: Config,
    w: &Weights,
    prompt: &[u32],
    max_new: usize,
    eos: Option<u32>,
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
        true,
    )
    .unwrap();
    let mut rnext = m
        .decode_step_explicit(
            &[prompt[prompt.len() - 1]],
            &[(prompt.len() - 1) as u32],
            &[0],
            &mut [SamplingParams::greedy(0)],
            &vec![Vec::new(); 1],
            &vec![Vec::new(); 1],
            true,
        )
        .unwrap()
        .0[0];
    let mut out = Vec::new();
    for rpos in prompt.len()..prompt.len() + max_new {
        if eos.is_some_and(|e| rnext == e) {
            break;
        }
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
                true,
            )
            .unwrap()
            .0[0];
    }
    out
}

#[test]
fn spec_decode_batch_lifecycle_matches_plain_greedy() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = Config::tiny();
    cfg.dtype = ModelDType::F32;
    let dw = Weights::random(&cfg, 61).unwrap();
    let tw = Weights::random(&cfg, 73).unwrap();
    let capacity = 3usize;
    let k = 4usize;
    let eos = Some(77u32);
    let jobs: Vec<(Vec<u32>, usize)> = vec![
        (vec![5, 9, 3, 200], 10usize),
        (vec![44, 88, 1], 8usize),
        (vec![2, 3, 4, 5], 12usize),
    ];

    let mut wants = Vec::new();
    for (p, n) in &jobs {
        wants.push(plain_greedy_eos(&hip, cfg, &tw, p, *n, eos));
    }

    let draft = BatchedModel::with_rows(hip.clone(), cfg, &dw, capacity, capacity).unwrap();
    let target =
        BatchedModel::with_rows(hip.clone(), cfg, &tw, capacity, capacity * (k + 1)).unwrap();
    let mut batch = mach_model::speculative::SpeculativeBatch::new(draft, target, k, capacity);
    for (p, _) in &jobs {
        batch.add(p).unwrap();
    }
    let mut got: Vec<Vec<u32>> = vec![Vec::new(); jobs.len()];
    while batch.active() > 0 {
        let accepted = batch.step().unwrap();
        for (s, seq) in accepted.iter().enumerate() {
            let Some(seq) = seq else { continue };
            for &t in seq {
                if got[s].len() >= jobs[s].1 || eos.is_some_and(|e| t == e) {
                    batch.finish(s);
                    break;
                }
                got[s].push(t);
            }
        }
    }
    for (s, (p, _)) in jobs.iter().enumerate() {
        assert_eq!(
            got[s], wants[s],
            "batch lifecycle seq {s} (prompt {p:?}) must equal plain greedy, got {:?} want {:?}",
            got[s], wants[s]
        );
    }
}

#[test]
fn speculative_engine_matches_continuous_model() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = Config::tiny();
    cfg.dtype = ModelDType::F32;
    let dw = Weights::random(&cfg, 61).unwrap();
    let tw = Weights::random(&cfg, 73).unwrap();
    let capacity = 3usize;
    let k = 4usize;
    let eos = Some(77u32);
    let jobs: Vec<(Vec<u32>, usize)> = vec![
        (vec![5, 9, 3, 200], 10usize),
        (vec![44, 88, 1], 8usize),
        (vec![2, 3, 4, 5], 12usize),
    ];

    // Reference: the standard continuous engine (greedy).
    let mut cm =
        mach_model::continuous::ContinuousModel::new(hip.clone(), cfg, &tw, capacity).unwrap();
    let mut cm_ids = Vec::new();
    for (p, n) in &jobs {
        cm_ids.push(
            cm.add(
                p,
                *n,
                eos,
                Vec::new(),
                Vec::new(),
                SamplingParams::default(),
            )
            .unwrap(),
        );
    }
    while !cm.all_done() {
        cm.step().unwrap();
    }

    // Speculative engine with the same requests.
    let mut eng = mach_model::speculative::SpeculativeEngine::new(
        BatchedModel::with_rows(hip.clone(), cfg, &dw, capacity, capacity).unwrap(),
        BatchedModel::with_rows(hip.clone(), cfg, &tw, capacity, capacity * (k + 1)).unwrap(),
        k,
        capacity,
    );
    for (p, n) in &jobs {
        eng.add(p, *n, eos).unwrap();
    }
    while !eng.all_done() {
        eng.step().unwrap();
    }
    for (i, (_, n)) in jobs.iter().enumerate() {
        let want = cm.generated(cm_ids[i]);
        let got = eng.generated(i);
        assert_eq!(
            got, want,
            "spec engine seq {i} must match continuous engine, got {got:?} want {want:?}"
        );
        assert_eq!(eng.finish_reason(i), cm.finish_reason(cm_ids[i]));
        assert_eq!(got.len(), *n, "seq {i} generated length");
    }
}
