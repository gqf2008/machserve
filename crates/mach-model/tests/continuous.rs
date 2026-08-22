//! Continuous-batching engine tests: engine generation must match the
//! single-sequence model exactly, slot reuse must work after sequences finish,
//! and EOS must stop generation.
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::continuous::ContinuousModel;
use mach_model::model::GpuModel;
use mach_model::sampling::SamplingParams;
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
        ids.push(
            eng.add(
                prompt,
                *max_new,
                *eos,
                Vec::new(),
                SamplingParams::default(),
            )
            .unwrap(),
        );
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
    let a = eng
        .add(&[3], 2, None, Vec::new(), SamplingParams::default())
        .unwrap();
    while !eng.is_done(a) {
        eng.step().unwrap();
    }
    assert_eq!(eng.active(), 0, "only sequence finished -> engine empty");
    let a_out = eng.generated(a);

    // Slot is freed: a new sequence can join at the compacted slot.
    let b = eng
        .add(&[9, 7, 42], 4, None, Vec::new(), SamplingParams::default())
        .unwrap();
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
    let s = eng
        .add(
            &[2, 5],
            20,
            Some(eos_tok),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.is_done(s) {
        eng.step().unwrap();
    }
    let gens = eng.generated(s);
    assert_eq!(gens, &ref_g[..=idx], "must stop exactly at the EOS token");
    assert!(gens.len() < 20, "must stop before max_new");
}

#[test]
fn sampling_is_deterministic_per_seed() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 29).unwrap();
    let base = SamplingParams {
        temperature: 0.9,
        top_k: 0,
        top_p: 0.95,
        seed: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
    };
    let run = |seed: u64| -> Vec<u32> {
        let mut eng = ContinuousModel::new(hip.clone(), cfg, &w, 4).unwrap();
        let mut p = base;
        p.seed = seed;
        let id = eng.add(&[3, 9, 27], 12, None, Vec::new(), p).unwrap();
        while !eng.is_done(id) {
            eng.step().unwrap();
        }
        eng.generated(id)
    };
    assert_eq!(run(42), run(42), "same seed must reproduce the same sample");
    assert_ne!(
        run(1),
        run(2),
        "different seeds must (overwhelmingly likely) diverge"
    );
}

#[test]
fn chunked_prefill_matches_single_token() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 83).unwrap();
    // Prompt longer than the engine capacity forces multiple prefill chunks.
    let prompt: Vec<u32> = (0..80u32).map(|i| (i % 977) + 1).collect();
    let capacity = 16usize;
    let max_new = 8usize;

    // Reference: single-token prefill + greedy decode via the single-seq model.
    let want = gen_ref(&hip, cfg, &w, &prompt, max_new);

    let mut eng = ContinuousModel::new(hip.clone(), cfg, &w, capacity).unwrap();
    let id = eng
        .add(
            &prompt,
            max_new,
            None,
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.is_done(id) {
        eng.step().unwrap();
    }
    assert_eq!(
        eng.generated(id),
        want,
        "chunked prefill must match single-token greedy generation"
    );
}

#[test]
fn chunked_prefill_finishes_with_fewer_steps() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 91).unwrap();
    let prompt: Vec<u32> = (0..80u32).map(|i| (i % 977) + 1).collect();
    let capacity = 16usize;
    let mut eng = ContinuousModel::new(hip.clone(), cfg, &w, capacity).unwrap();
    let id = eng
        .add(&prompt, 4, None, Vec::new(), SamplingParams::default())
        .unwrap();
    let mut steps = 0usize;
    while !eng.is_done(id) {
        eng.step().unwrap();
        steps += 1;
    }
    // 80 prompt tokens at capacity 16 -> >= ceil(80/16) = 5 prefill steps,
    // plus 4 decode steps. Far fewer than the 80+ steps of single-token prefill.
    assert!(
        steps < 20,
        "chunked prefill should finish in few steps, took {steps}"
    );
    assert_eq!(eng.generated(id).len(), 4);
}

#[test]
fn fp16_prefill_attention_matches_single_token() {
    let Some(hip) = hip_ctx() else { return };
    // F16 engine (fp16 KV + shared-KV prefill attention) vs an F32
    // single-token greedy reference. Capacity < prompt length forces
    // multi-chunk prefill, each chunk a detected run -> prefill attention.
    let mut cfg = Config::tiny();
    cfg.dtype = mach_model::config::ModelDType::F16;
    let w = Weights::random(&cfg, 83).unwrap();
    let prompt: Vec<u32> = (0..80u32).map(|i| (i % 977) + 1).collect();
    let capacity = 16usize;
    let max_new = 8usize;

    let mut cfg32 = cfg;
    cfg32.dtype = mach_model::config::ModelDType::F32;
    let want = gen_ref(&hip, cfg32, &w, &prompt, max_new);

    let mut eng = ContinuousModel::new(hip.clone(), cfg, &w, capacity).unwrap();
    let id = eng
        .add(
            &prompt,
            max_new,
            None,
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.is_done(id) {
        eng.step().unwrap();
    }
    assert_eq!(
        eng.generated(id),
        want,
        "fp16 prefill attention must match single-token greedy"
    );
}

#[test]
fn stop_sequence_terminates_generation() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 61).unwrap();
    let prompt = vec![5u32, 9, 3];

    // Find the greedy first token T; then stop=[[T]] must finish immediately.
    let want = gen_ref(&hip, cfg, &w, &prompt, 6);
    let stop_tok = want[0];
    let mut eng = ContinuousModel::new(hip.clone(), cfg, &w, 4).unwrap();
    let id = eng
        .add(
            &prompt,
            6,
            None,
            vec![vec![stop_tok]],
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.is_done(id) {
        eng.step().unwrap();
    }
    let g = eng.generated(id);
    assert_eq!(
        g,
        vec![stop_tok],
        "stop sequence must terminate right after it is generated, got {g:?}"
    );
    assert_eq!(eng.finish_reason(id), "stop");

    // A longer stop sequence (first two generated tokens) must also stop.
    let stop2 = want[..2].to_vec();
    let mut eng2 = ContinuousModel::new(hip.clone(), cfg, &w, 4).unwrap();
    let id2 = eng2
        .add(&prompt, 6, None, vec![stop2], SamplingParams::default())
        .unwrap();
    while !eng2.is_done(id2) {
        eng2.step().unwrap();
    }
    assert_eq!(
        eng2.generated(id2),
        want[..2],
        "two-token stop sequence must stop at the pair"
    );
    assert_eq!(eng2.finish_reason(id2), "stop");

    // A run to max_new without stop/EOS reports "length".
    let mut eng3 = ContinuousModel::new(hip.clone(), cfg, &w, 4).unwrap();
    let id3 = eng3
        .add(&prompt, 6, None, Vec::new(), SamplingParams::default())
        .unwrap();
    while !eng3.is_done(id3) {
        eng3.step().unwrap();
    }
    assert_eq!(eng3.finish_reason(id3), "length");
}
