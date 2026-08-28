//! Continuous-batching engine tests: engine generation must match the
//! single-sequence model exactly, slot reuse must work after sequences finish,
//! and EOS must stop generation.
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::config::ModelDType;
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
        .add(
            &[3],
            2,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.is_done(a) {
        eng.step().unwrap();
    }
    assert_eq!(eng.active(), 0, "only sequence finished -> engine empty");
    let a_out = eng.generated(a);

    // Slot is freed: a new sequence can join at the compacted slot.
    let b = eng
        .add(
            &[9, 7, 42],
            4,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
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
        top_logprobs: 0,
    };
    let run = |seed: u64| -> Vec<u32> {
        let mut eng = ContinuousModel::new(hip.clone(), cfg, &w, 4).unwrap();
        let mut p = base;
        p.seed = seed;
        let id = eng
            .add(&[3, 9, 27], 12, None, Vec::new(), Vec::new(), p)
            .unwrap();
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
        .add(
            &prompt,
            4,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
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
            Vec::new(),
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
        .add(
            &prompt,
            6,
            None,
            vec![stop2],
            Vec::new(),
            SamplingParams::default(),
        )
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
        .add(
            &prompt,
            6,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng3.is_done(id3) {
        eng3.step().unwrap();
    }
    assert_eq!(eng3.finish_reason(id3), "length");
}

#[test]
fn prefill_rows_gives_identical_output() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 67).unwrap();
    // A prompt long enough to need several prefill steps at capacity 2.
    let prompt: Vec<u32> = (0..40).map(|i| (i % 977) as u32).collect();

    let run = |prefill_rows: usize| -> Vec<u32> {
        let mut eng =
            ContinuousModel::with_prefill_rows(hip.clone(), cfg, &w, 2, prefill_rows).unwrap();
        let id = eng
            .add(
                &prompt,
                6,
                None,
                Vec::new(),
                Vec::new(),
                SamplingParams::default(),
            )
            .unwrap();
        let mut steps = 0usize;
        while !eng.is_done(id) {
            eng.step().unwrap();
            steps += 1;
        }
        let g = eng.generated(id);
        eprintln!("prefill_rows={prefill_rows}: {steps} steps -> {g:?}");
        g
    };

    let base = run(2); // default: one row per step of capacity 2
    let big = run(8); // larger prefill rows: fewer, wider steps
    assert!(!base.is_empty());
    assert_eq!(base, big, "prefill_rows must not change generated output");
}

/// MLA (DeepSeek-V2 style) config: low-rank Q + compressed KV.
fn mla_cfg() -> Config {
    Config::mla(128, 2, 4, 1024, 64, 32, 16, 16, 8, 16)
}

#[test]
fn engine_matches_single_model_mla() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = mla_cfg();
    let w = Weights::random(&cfg, 92).unwrap();
    let jobs = vec![
        (vec![5, 9, 3], 6usize, None),
        (vec![1, 200, 7], 5usize, None),
    ];
    let engine_out = run_engine(&hip, cfg, &w, 4, &jobs);
    for (i, (prompt, max_new, _)) in jobs.iter().enumerate() {
        let want = gen_ref(&hip, cfg, &w, prompt, *max_new);
        assert_eq!(
            engine_out[i], want,
            "MLA seq {i}: engine={:?} ref={:?}",
            engine_out[i], want
        );
    }
}

#[test]
fn slots_compact_keeps_mla_sequence_intact() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = mla_cfg();
    let w = Weights::random(&cfg, 54).unwrap();
    // A (short, slot 0) finishes first; B is compacted from slot 1 to
    // slot 0 via copy_seq_kv. The MLA expanded KV must move with the slot,
    // otherwise B's continued decode attends to stale KV and diverges.
    let mut eng = ContinuousModel::new(hip.clone(), cfg, &w, 2).unwrap();
    let a = eng
        .add(
            &[3],
            2,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    let b = eng
        .add(
            &[9, 7, 42],
            6,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.is_done(a) {
        eng.step().unwrap();
    }
    assert_eq!(eng.active(), 1, "B must stay active after A compacts");
    while !eng.all_done() {
        eng.step().unwrap();
    }
    assert_eq!(
        eng.generated(a),
        gen_ref(&hip, cfg, &w, &[3], 2),
        "MLA seq A must match ref"
    );
    assert_eq!(
        eng.generated(b),
        gen_ref(&hip, cfg, &w, &[9, 7, 42], 6),
        "MLA seq B must survive slot compaction"
    );
}

#[test]
fn add_rejects_prompt_over_max_seq_len() {
    // A prompt longer than max_seq_len would write past the KV cache during
    // prefill; the engine must reject it at admission (regression for the
    // silent OOB-write path in the continuous engine).
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny(); // max_seq_len = 256
    let w = Weights::random(&cfg, 41).expect("weights");
    let mut eng = ContinuousModel::new(hip.clone(), cfg, &w, 2).unwrap();
    let long: Vec<u32> = (0..=256).map(|i| i % 32000).collect(); // 257 tokens
    assert!(
        eng.add(
            &long,
            1,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .is_err(),
        "prompt longer than max_seq_len must be rejected"
    );
    // A prompt that exactly fits is still accepted.
    let fits: Vec<u32> = (0..256).map(|i| i % 32000).collect();
    eng.add(
        &fits,
        1,
        None,
        Vec::new(),
        Vec::new(),
        SamplingParams::default(),
    )
    .expect("exactly max_seq_len fits");
}

#[test]
fn decode_stops_at_max_seq_len_without_crashing() {
    // Once a sequence fills the context, further decode rows would write past
    // the KV cache; the engine must finish it instead of stepping out of
    // bounds (regression for the hard-stop path).
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny(); // max_seq_len = 256
    let w = Weights::random(&cfg, 43).expect("weights");
    let mut eng = ContinuousModel::new(hip.clone(), cfg, &w, 1).unwrap();
    let full: Vec<u32> = (0..256).map(|i| i % 32000).collect();
    let id = eng
        .add(
            &full,
            8,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .expect("full-context prompt admitted");
    while !eng.is_done(id) {
        eng.step().unwrap();
    }
    // The prefill-completion sample is emitted, but no token may be decoded at
    // a position >= max_seq_len, so generation must stop after it.
    let gens = eng.generated(id);
    assert!(
        gens.len() <= 1,
        "decode must hard-stop at max_seq_len, got {} generated",
        gens.len()
    );
    assert!(eng.all_done(), "over-length sequence must finish cleanly");
}

// ---------- paged-KV engine (#78 C5) ----------

fn shared_prefix_prompt(tpp: usize) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let prefix: Vec<u32> = (0..tpp as u32).map(|i| (i * 29 + 5) % 1024 + 1).collect();
    let d_a = vec![7u32, 11];
    let d_b = vec![300u32, 17, 9];
    let a = prefix.iter().chain(&d_a).copied().collect();
    let b = prefix.iter().chain(&d_b).copied().collect();
    (a, b, prefix)
}

/// C5a gate: a request sharing a materialized prefix page aliases it and
/// prefills only its delta; its output equals full recompute, and the
/// prompt-token savings are counted.
#[test]
fn paged_engine_reuses_shared_prefix() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let tpp = 64usize;
    let w = Weights::random(&cfg, 51).unwrap();
    let (a, b, _prefix) = shared_prefix_prompt(tpp);

    let mut eng =
        ContinuousModel::with_paged_prefill_rows(hip.clone(), cfg, &w, 2, 2, tpp).unwrap();
    let id_a = eng
        .add(
            &a,
            3,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.is_done(id_a) {
        eng.step().unwrap();
    }
    let id_b = eng
        .add(
            &b,
            3,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.all_done() {
        eng.step().unwrap();
    }

    let stats = eng.paged_reuse_stats().expect("paged engine");
    assert_eq!(stats.requests, 2);
    assert_eq!(
        stats.reused_tokens, tpp,
        "B reuses exactly the shared prefix page"
    );
    assert_eq!(
        eng.generated(id_b),
        gen_ref(&hip, cfg, &w, &b, 3),
        "reused output must equal full recompute"
    );
    assert_eq!(
        eng.generated(id_a),
        gen_ref(&hip, cfg, &w, &a, 3),
        "writer output must be intact"
    );
    let want_ratio = tpp as f32 / (a.len() + b.len()) as f32;
    assert!(
        (stats.reuse_ratio() - want_ratio).abs() < 1e-6,
        "reuse ratio {} != {want_ratio}",
        stats.reuse_ratio()
    );
}

/// C5a gate: pages are reusable only after materialization — a request added
/// before its writer prefills falls back to full compute (never reads
/// unwritten pages) and still produces the reference output.
#[test]
fn paged_engine_defers_reuse_until_materialized() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let tpp = 64usize;
    let w = Weights::random(&cfg, 61).unwrap();
    let (a, b, _) = shared_prefix_prompt(tpp);

    let mut eng =
        ContinuousModel::with_paged_prefill_rows(hip.clone(), cfg, &w, 2, 2, tpp).unwrap();
    let id_a = eng
        .add(
            &a,
            2,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    // B is admitted before A produced any page content.
    let id_b = eng
        .add(
            &b,
            2,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.all_done() {
        eng.step().unwrap();
    }
    let stats = eng.paged_reuse_stats().expect("paged engine");
    assert_eq!(stats.reused_tokens, 0, "no materialized page may be reused");
    assert_eq!(
        eng.generated(id_a),
        gen_ref(&hip, cfg, &w, &a, 2),
        "writer output must be correct"
    );
    assert_eq!(
        eng.generated(id_b),
        gen_ref(&hip, cfg, &w, &b, 2),
        "deferred-reuse request must full-compute correctly"
    );
}

/// C5a gate: slot compaction in paged mode moves the block table (pages
/// alias), so a sequence that survives a compaction keeps reading its KV and
/// stays bit-identical to full recompute.
#[test]
fn paged_engine_compaction_moves_tables() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let tpp = 64usize;
    let w = Weights::random(&cfg, 71).unwrap();
    let (a, b, _) = shared_prefix_prompt(tpp);

    // prefill_rows == tpp: A's 66-token prompt drains in exactly two steps
    // (64 + 2), materializing the shared page before B is admitted.
    let mut eng =
        ContinuousModel::with_paged_prefill_rows(hip.clone(), cfg, &w, 2, tpp, tpp).unwrap();
    let id_a = eng
        .add(
            &a,
            6,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    eng.step().unwrap();
    eng.step().unwrap(); // A prefill complete -> registered
    let id_b = eng
        .add(
            &b,
            4,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    assert_eq!(
        eng.paged_reuse_stats().unwrap().reused_tokens,
        tpp,
        "B reuses the materialized prefix"
    );
    while !eng.all_done() {
        eng.step().unwrap();
    }
    assert_eq!(
        eng.generated(id_b),
        gen_ref(&hip, cfg, &w, &b, 4),
        "survivor of compaction must equal full recompute"
    );
    assert_eq!(
        eng.generated(id_a),
        gen_ref(&hip, cfg, &w, &a, 6),
        "writer output must be intact"
    );
}

/// Paged page-pool eviction (#80 P4): with a 2-slot / 8-page pool, three
/// distinct-content requests (4 pages each) exceed the pool — the third
/// admission must evict the oldest unreferenced retired entry instead of
/// failing, and a later request reusing the evicted prefix falls back to full
/// compute with correct output (no stale aliasing).
#[test]
fn paged_engine_evicts_cold_pages_under_pressure() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny(); // max_seq 256, tpp 64 -> 4 pages/seq; pool = 2*4 = 8
    let tpp = 64usize;
    let w = Weights::random(&cfg, 55).unwrap();

    // Each prompt fills the whole context (4 distinct content pages), so the
    // 8-page pool is fully consumed by two requests; the third must evict.
    let len = cfg.max_seq_len;
    let mk = |base: u32| -> Vec<u32> { (0..len as u32).map(|i| (base + i) % 1024 + 1).collect() };
    let prompts = [mk(0), mk(300), mk(600), mk(0)];

    // Contiguous reference outputs.
    let mut refs = Vec::new();
    for prompt in &prompts {
        let mut cm = ContinuousModel::with_prefill_rows(hip.clone(), cfg, &w, 2, len).unwrap();
        let id = cm
            .add(
                prompt,
                2,
                None,
                Vec::new(),
                Vec::new(),
                SamplingParams::default(),
            )
            .unwrap();
        while !cm.is_done(id) {
            cm.step().unwrap();
        }
        refs.push(cm.generated(id));
    }

    let mut eng =
        ContinuousModel::with_paged_prefill_rows(hip.clone(), cfg, &w, 2, len, tpp).unwrap();
    for (i, prompt) in prompts.iter().enumerate() {
        let id = eng
            .add(
                prompt,
                2,
                None,
                Vec::new(),
                Vec::new(),
                SamplingParams::default(),
            )
            .unwrap();
        while !eng.is_done(id) {
            eng.step().unwrap();
        }
        assert_eq!(
            eng.generated(id),
            refs[i],
            "request {i} output must survive pool pressure/eviction"
        );
    }
    let stats = eng.paged_reuse_stats().expect("paged");
    assert_eq!(stats.requests, 4, "all four requests admitted");
    assert_eq!(stats.reused_tokens, 0, "evicted prefix must not be reused");
    assert_eq!(
        stats.prompt_tokens,
        4 * len,
        "every prompt fully computed (no stale aliasing)"
    );
}

/// Eviction with sibling retired entries sharing content (PR #81 review):
/// two concurrent same-prompt requests both allocate fresh pages and the
/// loser's duplicates are freed at retire — both retired entries then claim
/// the SAME hashes. Eviction must release shared pages with the last
/// claimant and keep surviving chains resolvable; a later same-prompt
/// request must either alias written pages or full-compute — never skip
/// prefill over pages that no longer hold KV (the stale-reuse-boundary bug
/// produced silent garbage here).
#[test]
fn paged_engine_eviction_with_sibling_retired_entries() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny(); // max_seq 256, tpp 64 -> 4 pages/seq; pool = 8
    let tpp = 64usize;
    let w = Weights::random(&cfg, 61).unwrap();
    // 192 tokens = 3 full pages (+1 pad freed at retire): each table is 4
    // pages, so two concurrent requests exactly fill the 8-page pool, while
    // prompt + 2 generated tokens stay inside max_seq_len for the reference.
    let len = 3 * tpp;
    let mk = |base: u32| -> Vec<u32> { (0..len as u32).map(|i| (base + i) % 1024 + 1).collect() };
    let p0 = mk(0);
    let p300 = mk(300);
    let p600 = mk(600);
    let refs = [
        gen_ref(&hip, cfg, &w, &p0, 2),
        gen_ref(&hip, cfg, &w, &p300, 2),
        gen_ref(&hip, cfg, &w, &p600, 2),
    ];

    let mut eng =
        ContinuousModel::with_paged_prefill_rows(hip.clone(), cfg, &w, 2, 8, tpp).unwrap();

    // Phase 1: A and B share p0 and are admitted concurrently (both plan
    // fresh — 4 pages each, exactly the 8-page pool).
    let id_a = eng
        .add(
            &p0,
            2,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    let id_b = eng
        .add(
            &p0,
            2,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.all_done() {
        eng.step().unwrap();
    }
    assert_eq!(eng.generated(id_a), refs[0]);
    assert_eq!(eng.generated(id_b), refs[0]);

    // Phase 2: distinct content consumes the sibling-freed pages, leaving
    // less than one full table free.
    let id_c = eng
        .add(
            &p300,
            2,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.all_done() {
        eng.step().unwrap();
    }
    assert_eq!(eng.generated(id_c), refs[1]);

    // Phase 3: another distinct prompt cannot fit — eviction drains the
    // sibling pair (shared pages released with the last claimant) and c's
    // pages, then the admission retries.
    let id_d = eng
        .add(
            &p600,
            2,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.all_done() {
        eng.step().unwrap();
    }
    assert_eq!(eng.generated(id_d), refs[2]);

    // Phase 4: p0 was evicted — this request must full-compute (its output
    // equals the reference; the stale-boundary bug served garbage here).
    let id_e = eng
        .add(
            &p0,
            2,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.all_done() {
        eng.step().unwrap();
    }
    assert_eq!(
        eng.generated(id_e),
        refs[0],
        "post-eviction same-prompt request must equal full recompute"
    );

    // Phase 5: d's materialized prefix survived — reuse resumes healthily.
    let id_f = eng
        .add(
            &p600,
            2,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.all_done() {
        eng.step().unwrap();
    }
    assert_eq!(eng.generated(id_f), refs[2]);
    let stats = eng.paged_reuse_stats().expect("paged");
    assert_eq!(stats.requests, 6, "all six requests admitted");
    assert_eq!(
        stats.reused_tokens,
        len - 1,
        "only f reuses (full prefix minus the one rewound token)"
    );
}

/// Concurrent same-prompt decode must not clobber (#80 review fix): the
/// shared prompt's LAST page is partial (130 tokens, tpp 64) — the first
/// generated token of every request lands in that page's offsets. Each
/// request must get its own partial page (reuse covers full pages only), so
/// A mid-decode and B admitted after A's first token both produce the
/// reference output.
#[test]
fn paged_engine_concurrent_partial_page_does_not_clobber() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny(); // max_seq 256, tpp 64 -> 4 pages/seq; pool = 8
    let tpp = 64usize;
    let w = Weights::random(&cfg, 57).unwrap();

    // 130 tokens = 2 full pages + a 2-token partial page.
    let prompt: Vec<u32> = (0..130u32).map(|i| (i * 7 + 1) % 1024 + 1).collect();
    let want = gen_ref(&hip, cfg, &w, &prompt, 8);

    let mut eng =
        ContinuousModel::with_paged_prefill_rows(hip.clone(), cfg, &w, 2, 8, tpp).unwrap();
    let id_a = eng
        .add(
            &prompt,
            8,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    // Run A until its first generated token: A is now decoding and has
    // written generated KV into the partial page it owns.
    let mut emitted = false;
    while !emitted {
        for (id, _) in eng.step().unwrap() {
            if id == id_a {
                emitted = true;
            }
        }
    }
    // B (identical prompt) is admitted while A is mid-decode.
    let id_b = eng
        .add(
            &prompt,
            8,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.all_done() {
        eng.step().unwrap();
    }
    assert_eq!(
        eng.generated(id_a),
        want,
        "A must keep its own generated KV (no clobber)"
    );
    assert_eq!(
        eng.generated(id_b),
        want,
        "B must produce the reference output with its own partial page"
    );
}

/// First-writer-wins registration must not leak the loser's pages (#80
/// review fix): two identical prompts admitted before either materializes
/// allocate duplicate pages; the loser's pages are freed at retire, so a
/// third identical request still fits the fixed pool and stays correct.
#[test]
fn paged_engine_concurrent_identical_prompts_do_not_leak_pool() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny(); // pool = 2 slots * 4 pages = 8
    let tpp = 64usize;
    let w = Weights::random(&cfg, 59).unwrap();
    let prompt: Vec<u32> = (0..130u32).map(|i| (i * 11 + 3) % 1024 + 1).collect();
    let want = gen_ref(&hip, cfg, &w, &prompt, 6);

    let mut eng =
        ContinuousModel::with_paged_prefill_rows(hip.clone(), cfg, &w, 2, 8, tpp).unwrap();
    // Both admitted before any step: each allocates its own 4 pages (8 total).
    let id_a = eng
        .add(
            &prompt,
            6,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    let id_b = eng
        .add(
            &prompt,
            6,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.all_done() {
        eng.step().unwrap();
    }
    assert_eq!(eng.generated(id_a), want, "A output");
    assert_eq!(eng.generated(id_b), want, "B output");

    // C reuses A's registered full pages; its partial page is fresh. The
    // loser's duplicate pages must have been freed at retire, so C still fits.
    let id_c = eng
        .add(
            &prompt,
            6,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.all_done() {
        eng.step().unwrap();
    }
    assert_eq!(
        eng.generated(id_c),
        want,
        "C output (pool must not have shrunk)"
    );
    let stats = eng.paged_reuse_stats().expect("paged");
    assert_eq!(stats.requests, 3);
    assert_eq!(stats.reused_tokens, 2 * tpp, "C reuses the two full pages");
}

// ---------- paged storage-quantized engines (#80 P3) ----------

/// Two-request run helper: A fully finishes first (materializing pages),
/// then B is admitted and runs to completion. Returns outputs + the engine's
/// reused-token count (contiguous engines have no paged stats → 0).
fn run_two_requests(eng: &mut ContinuousModel, a: &[u32], b: &[u32]) -> (Vec<Vec<u32>>, usize) {
    let id_a = eng
        .add(
            a,
            3,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.is_done(id_a) {
        eng.step().unwrap();
    }
    let id_b = eng
        .add(
            b,
            3,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !eng.all_done() {
        eng.step().unwrap();
    }
    let reused = eng
        .paged_reuse_stats()
        .map(|s| s.reused_tokens)
        .unwrap_or(0);
    (vec![eng.generated(id_a), eng.generated(id_b)], reused)
}

/// Storage-quantized paged engines (#80 P3): Q4 and FP8 device-f16 paths serve
/// cross-request prefix reuse identically — the second shared-prefix request's
/// output equals its own contiguous-engine run and the shared page is reused.
#[test]
fn paged_engine_quantized_reuses_shared_prefix() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = Config::tiny(); // max_seq 256; tpp 64
    cfg.dtype = ModelDType::F16;
    let tpp = 64usize;
    let (a, b, prefix) = shared_prefix_prompt(tpp);

    for quant in 0..2 {
        match quant {
            0 => {
                let w = Weights::random(&cfg, 91).unwrap();
                let wq = mach_model::WeightsQ4::from_weights(&w, &cfg);
                let run = |paged: bool| -> (Vec<Vec<u32>>, usize) {
                    if paged {
                        let mut eng = ContinuousModel::with_paged_prefill_rows_q4(
                            hip.clone(),
                            cfg,
                            &wq,
                            2,
                            2,
                            tpp,
                        )
                        .unwrap();
                        run_two_requests(&mut eng, &a, &b)
                    } else {
                        let mut eng =
                            ContinuousModel::with_prefill_rows_q4(hip.clone(), cfg, &wq, 2, 2)
                                .unwrap();
                        run_two_requests(&mut eng, &a, &b)
                    }
                };
                let (contig_out, _) = run(false);
                let (paged_out, reused) = run(true);
                assert_eq!(
                    contig_out[0], paged_out[0],
                    "Q4 writer output must be identical across engines"
                );
                assert_eq!(
                    contig_out[1], paged_out[1],
                    "Q4 reused output must equal contiguous"
                );
                assert!(prefix.len() >= tpp);
                assert_eq!(reused, tpp, "Q4 engine reuses the shared page");
            }
            _ => {
                let w = Weights::random(&cfg, 93).unwrap();
                let wf = mach_model::WeightsFp8::from_weights(&w, &cfg);
                let run = |paged: bool| -> (Vec<Vec<u32>>, usize) {
                    if paged {
                        let mut eng = ContinuousModel::with_paged_prefill_rows_fp8(
                            hip.clone(),
                            cfg,
                            &wf,
                            2,
                            2,
                            tpp,
                        )
                        .unwrap();
                        run_two_requests(&mut eng, &a, &b)
                    } else {
                        let mut eng =
                            ContinuousModel::with_prefill_rows_fp8(hip.clone(), cfg, &wf, 2, 2)
                                .unwrap();
                        run_two_requests(&mut eng, &a, &b)
                    }
                };
                let (contig_out, _) = run(false);
                let (paged_out, reused) = run(true);
                assert_eq!(
                    contig_out[0], paged_out[0],
                    "FP8 writer output must be identical across engines"
                );
                assert_eq!(
                    contig_out[1], paged_out[1],
                    "FP8 reused output must equal contiguous"
                );
                assert_eq!(reused, tpp, "FP8 engine reuses the shared page");
            }
        }
    }
}

/// Retired-metadata cap with shared content: sequential same-prefix requests
/// pile up retired entries (admissions alias, so the pool never pressures a
/// drain). At the cap the metadata drains and the shared page releases with
/// the last claimant; the next request must full-compute correctly and reuse
/// must resume healthily afterwards (no orphaned, unreachable pages).
#[test]
fn paged_engine_retired_cap_recovers_and_reuses() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny(); // max_seq 256, tpp 64 -> 4 pages/seq; pool = 8
    let tpp = 64usize;
    let w = Weights::random(&cfg, 63).unwrap();

    // Exactly 2 full pages: reuse skips 64 tokens (the kept last page).
    let prompt: Vec<u32> = (0..128u32).map(|i| (i * 3 + 5) % 1024 + 1).collect();
    let want = gen_ref(&hip, cfg, &w, &prompt, 2);

    let mut eng =
        ContinuousModel::with_paged_prefill_rows(hip.clone(), cfg, &w, 2, 8, tpp).unwrap();
    let mut ids = Vec::new();
    for _ in 0..11 {
        let id = eng
            .add(
                &prompt,
                2,
                None,
                Vec::new(),
                Vec::new(),
                SamplingParams::default(),
            )
            .unwrap();
        while !eng.is_done(id) {
            eng.step().unwrap();
        }
        assert_eq!(eng.generated(id), want, "every request must be correct");
        ids.push(id);
    }
    let stats = eng.paged_reuse_stats().expect("paged");
    assert_eq!(stats.requests, 11);
    // r2..r9 reuse (8x), r10 full-computes after the cap drain, r11 reuses.
    // Each full-prefix hit reuses every token but the one rewound for the
    // first decode row.
    assert_eq!(
        stats.reused_tokens,
        9 * (2 * tpp - 1),
        "reuse before and after the cap drain"
    );
}
