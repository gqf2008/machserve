//! Continuous-batching engine tests: engine generation must match the
//! single-sequence model exactly, slot reuse must work after sequences finish,
//! and EOS must stop generation.
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::continuous::ContinuousModel;
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

/// Greedy generation reference via the single-sequence model.
fn gen_ref(
    hip: &std::sync::Arc<hip::Hip>,
    cfg: Config,
    w: &Weights,
    prompt: &[u32],
    max_new: usize,
) -> Vec<u32> {
    let mut m = GpuModel::new(hip.clone(), cfg, w).unwrap();
    let mut preds = Vec::new();
    for &t in prompt {
        preds.push(m.decode_step_sampled(t).unwrap());
    }
    let mut tok = *preds.last().unwrap();
    let mut gens = vec![tok];
    for _ in 1..max_new {
        tok = m.decode_step_sampled(tok).unwrap();
        gens.push(tok);
    }
    gens
}

fn run_engine(
    hip: &std::sync::Arc<hip::Hip>,
    cfg: Config,
    w: &Weights,
    capacity: usize,
    jobs: &[(Vec<u32>, usize, Option<u32>)],
) -> Vec<Vec<u32>> {
    let mut eng = ContinuousModel::new(hip.clone(), cfg, w, capacity).unwrap();
    let mut ids = Vec::new();
    for (prompt, max_new, eos) in jobs {
        ids.push(eng.add(prompt, *max_new, *eos).unwrap());
    }
    while !eng.all_done() {
        eng.step().unwrap();
    }
    ids.iter().map(|&id| eng.generated(id)).collect()
}

#[test]
fn engine_matches_single_model() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 91).unwrap();
    let jobs = vec![
        (vec![5, 9, 3], 6usize, None),
        (vec![1, 200, 7], 5usize, None),
    ];
    let engine_out = run_engine(&hip, cfg, &w, 4, &jobs);
    for (i, (prompt, max_new, _)) in jobs.iter().enumerate() {
        let want = gen_ref(&hip, cfg, &w, prompt, *max_new);
        assert_eq!(
            engine_out[i], want,
            "seq {i}: engine={:?} ref={:?}",
            engine_out[i], want
        );
    }
}

#[test]
fn slots_reuse_after_finish() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 53).unwrap();
    // A short sequence finishes quickly, freeing its slot for a new one.
    let mut eng = ContinuousModel::new(hip.clone(), cfg, &w, 2).unwrap();
    let a = eng.add(&[3], 2, None).unwrap();
    while !eng.is_done(a) {
        eng.step().unwrap();
    }
    assert_eq!(eng.active(), 0, "only sequence finished -> engine empty");
    let a_out = eng.generated(a);

    // Slot is freed: a new sequence can join at the compacted slot.
    let b = eng.add(&[9, 7, 42], 4, None).unwrap();
    assert_ne!(b, a, "stable ids differ");
    while !eng.all_done() {
        eng.step().unwrap();
    }
    let b_out = eng.generated(b);

    assert_eq!(
        a_out,
        gen_ref(&hip, cfg, &w, &[3], 2),
        "seq A must match ref"
    );
    assert_eq!(
        b_out,
        gen_ref(&hip, cfg, &w, &[9, 7, 42], 4),
        "seq B must match ref and be isolated"
    );
}

#[test]
fn eos_stops_generation() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 17).unwrap();
    // Pick an EOS token that the greedy model is guaranteed to produce, so the
    // test deterministically exercises early stopping.
    let ref_g = gen_ref(&hip, cfg, &w, &[2, 5], 20);
    let eos_tok = ref_g[3];
    let idx = ref_g.iter().position(|&t| t == eos_tok).unwrap();

    let mut eng = ContinuousModel::new(hip.clone(), cfg, &w, 2).unwrap();
    let s = eng.add(&[2, 5], 20, Some(eos_tok)).unwrap();
    while !eng.is_done(s) {
        eng.step().unwrap();
    }
    let gens = eng.generated(s);
    assert_eq!(gens, &ref_g[..=idx], "must stop exactly at the EOS token");
    assert!(gens.len() < 20, "must stop before max_new");
}
