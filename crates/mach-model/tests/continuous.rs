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

// ---------- paged storage-quantized engines (#80 P3) ----------

/// Converts f32 weights to storage-Q4 (GEMM tensors quantized, small tensors
/// copied) for the Q4 paged-engine test.
fn to_q4(w: &Weights) -> mach_model::WeightsQ4 {
    use mach_model::LayerWeightsQ4;
    use mach_model::q4::Q4Tensor;
    let q = |v: &[f32]| Q4Tensor::quantize(v);
    mach_model::WeightsQ4 {
        tok_emb: q(&w.tok_emb),
        rms_final: w.rms_final.clone(),
        lm_head: q(&w.lm_head),
        layers: w
            .layers
            .iter()
            .map(|l| LayerWeightsQ4 {
                wq: q(&l.wq),
                wk: q(&l.wk),
                wv: q(&l.wv),
                wo: q(&l.wo),
                rms_attn: l.rms_attn.clone(),
                wg: q(&l.wg),
                wu: q(&l.wu),
                wd: q(&l.wd),
                rms_mlp: l.rms_mlp.clone(),
                bq: l.bq.clone(),
                bk: l.bk.clone(),
                bv: l.bv.clone(),
                q_norm: l.q_norm.clone(),
                k_norm: l.k_norm.clone(),
                mla_q_a: q(&l.mla_q_a),
                mla_q_a_norm: l.mla_q_a_norm.clone(),
                mla_q_b: q(&l.mla_q_b),
                mla_q_rope: q(&l.mla_q_rope),
                mla_kv_a: q(&l.mla_kv_a),
                mla_kv_a_norm: l.mla_kv_a_norm.clone(),
                mla_kv_b: q(&l.mla_kv_b),
                mla_o: q(&l.mla_o),
                moe_router: l.moe_router.clone(),
                moe_wg: q(&l.moe_wg),
                moe_wu: q(&l.moe_wu),
                moe_wd: q(&l.moe_wd),
            })
            .collect(),
    }
}

/// Converts f32 weights to storage-FP8 (GEMM tensors E4M3-quantized with
/// per-tensor scales) for the FP8 paged-engine test.
fn to_fp8(w: &Weights) -> mach_model::WeightsFp8 {
    use mach_model::LayerWeightsFp8;
    use mach_model::fp8::Fp8Tensor;
    let q = |v: &[f32]| Fp8Tensor::quantize(v);
    mach_model::WeightsFp8 {
        tok_emb: q(&w.tok_emb),
        rms_final: w.rms_final.clone(),
        lm_head: q(&w.lm_head),
        layers: w
            .layers
            .iter()
            .map(|l| LayerWeightsFp8 {
                wq: q(&l.wq),
                wk: q(&l.wk),
                wv: q(&l.wv),
                wo: q(&l.wo),
                rms_attn: l.rms_attn.clone(),
                wg: q(&l.wg),
                wu: q(&l.wu),
                wd: q(&l.wd),
                rms_mlp: l.rms_mlp.clone(),
                bq: l.bq.clone(),
                bk: l.bk.clone(),
                bv: l.bv.clone(),
                q_norm: l.q_norm.clone(),
                k_norm: l.k_norm.clone(),
                mla_q_a: q(&l.mla_q_a),
                mla_q_a_norm: l.mla_q_a_norm.clone(),
                mla_q_b: q(&l.mla_q_b),
                mla_q_rope: q(&l.mla_q_rope),
                mla_kv_a: q(&l.mla_kv_a),
                mla_kv_a_norm: l.mla_kv_a_norm.clone(),
                mla_kv_b: q(&l.mla_kv_b),
                mla_o: q(&l.mla_o),
                moe_router: l.moe_router.clone(),
                moe_wg: q(&l.moe_wg),
                moe_wu: q(&l.moe_wu),
                moe_wd: q(&l.moe_wd),
            })
            .collect(),
    }
}

/// Shared-prefix helper: A fully finishes first (materializing pages), then B
/// is admitted and runs to completion. Returns outputs + reuse stats tokens.
fn run_paged_two_requests(
    eng: &mut ContinuousModel,
    a: &[u32],
    b: &[u32],
) -> (Vec<Vec<u32>>, usize) {
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
    let reused = eng.paged_reuse_stats().expect("paged").reused_tokens;
    (vec![eng.generated(id_a), eng.generated(id_b)], reused)
}

/// Contiguous counterpart of [`run_paged_two_requests`] (no reuse stats).
fn run_contiguous_two_requests(
    eng: &mut ContinuousModel,
    a: &[u32],
    b: &[u32],
) -> (Vec<Vec<u32>>, usize) {
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
    (vec![eng.generated(id_a), eng.generated(id_b)], 0)
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
                let wq = to_q4(&w);
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
                        run_paged_two_requests(&mut eng, &a, &b)
                    } else {
                        let mut eng =
                            ContinuousModel::with_prefill_rows_q4(hip.clone(), cfg, &wq, 2, 2)
                                .unwrap();
                        run_contiguous_two_requests(&mut eng, &a, &b)
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
                let wf = to_fp8(&w);
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
                        run_paged_two_requests(&mut eng, &a, &b)
                    } else {
                        let mut eng =
                            ContinuousModel::with_prefill_rows_fp8(hip.clone(), cfg, &wf, 2, 2)
                                .unwrap();
                        run_contiguous_two_requests(&mut eng, &a, &b)
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
