//! Batched decode correctness: a batched step must equal running each sequence
//! through the (already validated) single-sequence model.
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::batched::BatchedModel;
use mach_model::config::ModelDType;
use mach_model::model::GpuModel;
use mach_model::paged_kv::GpuPagedTableBuilder;
use mach_model::sampling::SamplingParams;
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

fn greedy_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.partial_cmp(b).unwrap().then_with(|| j.cmp(i)))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let d = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        d <= 1e-4 + 1e-4 * scale,
        "{ctx}: logits max diff {d} (scale {scale})"
    );
}

/// One forward over arbitrary rows; returns sampled tokens and the dense
/// logits head (`n * vocab` entries, row-major by forwarded row).
fn fwd_rows(
    m: &mut BatchedModel,
    toks: &[u32],
    poss: &[u32],
    slts: &[u32],
    vocab: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut params: Vec<SamplingParams> = (0..toks.len())
        .map(|i| SamplingParams::greedy(1000 + i as u64))
        .collect();
    let counts = vec![Vec::<(u32, u32)>::new(); toks.len()];
    let bias = vec![Vec::<(u32, f32)>::new(); toks.len()];
    let (sampled, _, _) = m
        .decode_step_explicit(toks, poss, slts, &mut params, &counts, &bias)
        .unwrap();
    let logits = m.read_logits_rows(toks.len()).unwrap();
    (sampled, logits[..toks.len() * vocab].to_vec())
}

/// Shared-prefix block tables (#78 C1): two sequences whose tables (built by
/// the content-hash `GpuPagedTableBuilder`) alias the same physical prefix
/// page must produce results identical to full recompute — and the second
/// sequence skips the shared prefix entirely (delta-only stores/decodes; the
/// prefix K/V it reads were written by the first sequence).
#[test]
fn shared_prefix_paged_reuse_matches_full_compute() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny(); // max_seq 256; tokens_per_page 64 -> 4 pages/seq
    let tpp = 64usize;
    let w = Weights::random(&cfg, 91).unwrap();
    let vocab = cfg.vocab_size;

    // Shared system-prompt page + distinct tails of EQUAL length (joint steps
    // keep both rows active through the end) + fixed continuations.
    let prefix: Vec<u32> = (0..tpp as u32).map(|i| (i * 37 + 3) % 1024 + 1).collect();
    let cont_a: Vec<u32> = vec![42, 99, 5, 250];
    let cont_b: Vec<u32> = vec![300, 17, 8, 63];
    let gen_tail = vec![7u32, 11, 13];
    let seq_a: Vec<u32> = prefix
        .iter()
        .chain(&cont_a)
        .chain(&gen_tail)
        .copied()
        .collect();
    let seq_b: Vec<u32> = prefix
        .iter()
        .chain(&cont_b)
        .chain(&gen_tail)
        .copied()
        .collect();

    let mut m = BatchedModel::with_paged_kv(hip.clone(), cfg, &w, 2, tpp).unwrap();
    let mut r0 = BatchedModel::with_paged_kv(hip.clone(), cfg, &w, 1, tpp).unwrap();
    let mut r1 = BatchedModel::with_paged_kv(hip.clone(), cfg, &w, 1, tpp).unwrap();

    // Content-hash tables: B must reuse exactly the shared prefix page(s).
    let pool_pages = (2 * cfg.max_seq_len / tpp) as u32;
    let mut builder = GpuPagedTableBuilder::new(pool_pages, tpp);
    let as_i32 = |v: &[u32]| -> Vec<i32> { v.iter().map(|&x| x as i32).collect() };
    let (ta, _ra) = builder.build_table(&as_i32(&seq_a)).unwrap();
    let (tb, rb) = builder.build_table(&as_i32(&seq_b)).unwrap();
    assert_eq!(ta.len(), 2, "ceil((64+3+3)/64) pages");
    assert_eq!(rb, tpp, "B reuses exactly the shared prefix page");
    assert!(builder.cached_pages() >= 2, "both requests cached");
    assert_eq!(ta.get(0), tb.get(0), "prefix physical page is aliased");
    let union: std::collections::HashSet<u32> =
        ta.pages().iter().chain(tb.pages()).copied().collect();
    assert!(
        union.len() < ta.len() + tb.len(),
        "physical pages must be shared across the two tables"
    );

    m.set_block_table(0, ta.pages()).unwrap();
    m.set_block_table(1, tb.pages()).unwrap();

    // Phase 1: prefix on m is computed by sequence A alone (pages get their
    // values once); references advance independently in their own pools.
    for t in 0..tpp {
        let pos = t as u32;
        let _ = fwd_rows(&mut m, &[seq_a[t]], &[pos], &[0], vocab);
        let _ = fwd_rows(&mut r0, &[seq_a[t]], &[pos], &[0], vocab);
        let _ = fwd_rows(&mut r1, &[seq_b[t]], &[pos], &[0], vocab);
    }

    // Phase 2: joint deltas/continuations. Sequence B's rows address the
    // prefix through B's table — the physical page written by A.
    for j in tpp..seq_b.len() {
        let pos = j as u32;
        let (sm, lm) = fwd_rows(&mut m, &[seq_a[j], seq_b[j]], &[pos, pos], &[0, 1], vocab);
        let (_, la) = fwd_rows(&mut r0, &[seq_a[j]], &[pos], &[0], vocab);
        let (_, lb) = fwd_rows(&mut r1, &[seq_b[j]], &[pos], &[0], vocab);
        let row_a = &lm[..vocab];
        let row_b = &lm[vocab..2 * vocab];
        assert_close(row_a, &la, &format!("shared-prefix slot A @pos {pos}"));
        assert_close(row_b, &lb, &format!("aliased-prefix slot B @pos {pos}"));
        assert_eq!(sm[0], greedy_argmax(&la), "greedy token A @pos {pos}");
        assert_eq!(sm[1], greedy_argmax(&lb), "greedy token B @pos {pos}");
    }
}

/// Paged chunked prefill (#78 C2): several prompt positions of one sequence
/// packed into distinct rows of a single step must produce the same per-
/// position logits and greedy tokens as sequential single-token stepping.
/// Exercises the per-row table-offset refresh (`row != slot`) across two
/// chunks with different row counts spanning a page boundary region.
#[test]
fn batched_paged_chunked_prefill_matches_sequential() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny(); // max_seq 256; tokens_per_page 64
    let tpp = 64usize;
    let w = Weights::random(&cfg, 71).unwrap();
    let vocab = cfg.vocab_size;

    let seq: Vec<u32> = (0..24u32).map(|i| (i * 53 + 11) % 1024 + 1).collect();

    // One slot, 16 rows: a prefill step may pack up to 16 positions at once,
    // every one of them addressed through slot 0's block table.
    let mut paged = BatchedModel::with_paged_kv_rows(hip.clone(), cfg, &w, 1, 16, tpp).unwrap();
    let mut r = GpuModel::new(hip.clone(), cfg, &w).unwrap();

    // Sequential reference: capture the logits after each single step.
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(seq.len());
    for &t in &seq {
        ref_logits.push(r.decode_step(t).unwrap());
    }

    // Chunk 1 (12 rows): intra-chunk causality — row `p` attends over exactly
    // positions 0..=p because kv_store_paged stores all rows before attention.
    let lens1: Vec<u32> = (0..12u32).collect();
    let slots1: Vec<u32> = vec![0; 12];
    let (s1, lm1) = fwd_rows(&mut paged, &seq[0..12], &lens1, &slots1, vocab);
    for &row in &[5usize, 10, 11] {
        assert_close(
            &lm1[row * vocab..(row + 1) * vocab],
            &ref_logits[row],
            &format!("chunk1 row {row} (pos {row})"),
        );
        assert_eq!(
            s1[row],
            greedy_argmax(&ref_logits[row]),
            "chunk1 greedy row {row}"
        );
    }

    // Chunk 2 (8 rows, different shape): rows cross into the next position
    // range; offsets are refreshed for the new active set.
    let lens2: Vec<u32> = (12..20u32).collect();
    let slots2: Vec<u32> = vec![0; 8];
    let (s2, lm2) = fwd_rows(&mut paged, &seq[12..20], &lens2, &slots2, vocab);
    for &row in &[3usize, 7] {
        let pos = 12 + row;
        assert_close(
            &lm2[row * vocab..(row + 1) * vocab],
            &ref_logits[pos],
            &format!("chunk2 row {row} (pos {pos})"),
        );
        assert_eq!(
            s2[row],
            greedy_argmax(&ref_logits[pos]),
            "chunk2 greedy row {row} (pos {pos})"
        );
    }
}
