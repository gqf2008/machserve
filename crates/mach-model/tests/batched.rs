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

/// Greedy-token check for tests whose batched path uses a different GEMM
/// implementation (grouped GEMV) than the single-sequence reference
/// (hipBLAS): a near-tie may legitimately flip, so the batched token is
/// accepted when its reference logit is within `tie_band` of the reference
/// argmax logit. NOTE: with `tie_band = 2 * logits_tolerance` this is
/// implied by the logits-diff assertion above (a flip with a reference gap
/// beyond the band would also break the logits bound) — the check is
/// defense-in-depth for future tolerance edits, not an independent oracle.
/// The ROUTER tie-break itself is pinned directly by
/// `moe_router_batched_matches_serial_topk` in kernels.rs.
fn assert_greedy_compatible(got: u32, want: u32, want_logits: &[f32], ctx: &str, tie_band: f32) {
    if got == want {
        return;
    }
    let got_ref_logit = want_logits[got as usize];
    let want_ref_logit = want_logits[want as usize];
    assert!(
        got_ref_logit >= want_ref_logit - tie_band,
        "{ctx}: greedy token {got} != {want} and is not a near-tie (ref logit {got_ref_logit} vs {want_ref_logit})"
    );
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
            assert_greedy_compatible(
                got[s],
                want,
                &want_logits,
                &format!("step {step_tokens:?} seq {s}"),
                4e-3 + 4e-3 * scale,
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
            // Near-tie tolerance for the grouped GEMV decode path (see the
            // dense MoE test above); f16 noise -> flat wider band.
            assert_greedy_compatible(got[s], want, &want_logits, "f16", 0.2);
        }
    }
}

/// Shared experts (DeepSeek-V2's dense `n_shared_experts` SwiGLU MLP added to
/// the routed experts' weighted sum) must contribute on EVERY dtype path.
/// Regression for #107: the shared block used to be guarded by the f32
/// `shared_w*` device pointers, which are null on every F16 upload path — the
/// weights live in the f16 table only — so F16/Q4/FP8 models silently ran
/// without their shared experts and decoded garbage (Qwen-MoE was unaffected
/// because it ships no shared experts). Compare the F32 and F16 batched
/// models on identical weights: dropping the shared MLP is a gross,
/// whole-block-sized error, far outside the f16 rounding envelope.
/// Calibration on this config: the fixed path drifts ~1.5e-3; with the
/// shared block dropped it is ~0.7 at the first step already — the bound
/// below sits between with two orders of margin on each side.
#[test]
fn batched_f16_shared_experts_match_f32() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = moe_cfg();
    cfg.n_shared_experts = 2;
    let mut w = Weights::random(&cfg, 71).unwrap();
    // Peak the router so both dtypes select the same experts robustly (same
    // rationale as the f16 batched test above).
    for lw in w.layers.iter_mut() {
        for v in lw.moe_router.iter_mut() {
            *v *= 6.0;
        }
    }

    let mut f32m = BatchedModel::new(hip.clone(), cfg, &w, 1).unwrap();
    let mut f16cfg = cfg;
    f16cfg.dtype = ModelDType::F16;
    let mut f16m = BatchedModel::new(hip, f16cfg, &w, 1).unwrap();

    for (i, &t) in [3u32, 11, 42, 7].iter().enumerate() {
        f32m.decode_step(&[t]).unwrap();
        f16m.decode_step(&[t]).unwrap();
        let a = f32m.read_logits().unwrap();
        let b = f16m.read_logits().unwrap();
        let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            d <= 2e-2 + 2e-2 * scale,
            "pos {i}: f16 vs f32 logits max diff {d} (scale {scale}) — shared experts dropped on the f16 path?"
        );
    }
}

/// Grouped-GEMV decode pinned to the CPU reference (an implementation that
/// shares no code with the GPU path): the strongest oracle for the new MoE
/// kernels — stronger than the near-tie token bands, which only bound how
/// far a flip may drift. f32 everywhere: rounding differences are GEMM-order
/// only, so the standard tolerance applies.
#[test]
fn batched_moe_grouped_matches_cpu_reference() {
    use mach_model::ref_model::RefModel;

    let Some(hip) = hip_ctx() else { return };
    let cfg = moe_cfg();
    let w = Weights::random(&cfg, 141).unwrap();
    let mut paged = BatchedModel::new(hip, cfg, &w, 1).unwrap();
    let prompt: Vec<u32> = (0..12u32).map(|i| (i * 29 + 7) % 1024 + 1).collect();
    let mut cpu = RefModel::new(cfg, w);
    for (i, &t) in prompt.iter().enumerate() {
        paged.decode_step(&[t]).unwrap();
        let got = paged.read_logits().unwrap();
        // RefModel is stateful (internal KV cache + pos counter): feed only
        // the incremental token, matching the GPU side's per-step advance.
        let want = cpu.forward(&[t]);
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff <= 2e-3 + 2e-3 * scale,
            "step {i}: grouped MoE vs CPU ref: max diff {diff} (scale {scale})"
        );
    }
}

/// The strongest oracle for the fixed shared-experts branch (#107): the f16
/// grouped-MoE GPU path vs the CPU reference — `RefModel` computes the shared
/// SwiGLU on the host weights, an implementation sharing no code with the GPU
/// path. The F16-vs-F32 test above can only catch dtype-asymmetric drops; the
/// repo convention (每条 GPU 路径都要有 CPU 参考对拍) asks for a CPU-reference
/// pin, and `moe_cfg()` leaves `n_shared_experts = 0`, so before this test the
/// restored branch was the one GPU path no CPU-reference parity test exercised.
#[test]
fn batched_moe_shared_experts_match_cpu_reference() {
    use mach_model::ref_model::RefModel;

    let Some(hip) = hip_ctx() else { return };
    let mut cfg = moe_cfg();
    cfg.n_shared_experts = 2;
    cfg.dtype = ModelDType::F16;
    let mut w = Weights::random(&cfg, 167).unwrap();
    // Peak the router so the f16 device router and the f32 CPU reference
    // select the same experts robustly (same rationale as the test above).
    for lw in w.layers.iter_mut() {
        for v in lw.moe_router.iter_mut() {
            *v *= 6.0;
        }
    }
    let mut gpu = BatchedModel::new(hip, cfg, &w, 1).unwrap();
    let mut cpu = RefModel::new(cfg, w);
    for (i, &t) in [3u32, 11, 42, 7].iter().enumerate() {
        gpu.decode_step(&[t]).unwrap();
        let got = gpu.read_logits().unwrap();
        // RefModel is stateful: feed only the incremental token.
        let want = cpu.forward(&[t]);
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let diff = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff <= 2e-2 + 2e-2 * scale,
            "step {i}: f16 grouped MoE + shared vs CPU ref: max diff {diff} (scale {scale}) — shared experts dropped or miscomputed?"
        );
    }
}

/// The MACH_MOE_GROUPED=0 fallback (hipBLAS host loop) must produce the same
/// logits as the grouped GEMV path: driven directly through
/// `decode_step_explicit`'s `decode_only` flag (which is exactly the runtime
/// switch the env var sets), no env manipulation needed.
#[test]
fn moe_grouped_fallback_matches_grouped() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = moe_cfg();
    let w = Weights::random(&cfg, 151).unwrap();
    let mut grouped = BatchedModel::new(hip.clone(), cfg, &w, 1).unwrap();
    let mut fallback = BatchedModel::new(hip.clone(), cfg, &w, 1).unwrap();
    let prompt: Vec<u32> = (0..12u32).map(|i| (i * 17 + 3) % 1024 + 1).collect();
    for (i, &t) in prompt.iter().enumerate() {
        let mut p1 = [SamplingParams::greedy(0)];
        let mut p2 = [SamplingParams::greedy(0)];
        let g = grouped
            .decode_step_explicit(
                &[t],
                &[(i) as u32],
                &[0],
                &mut p1,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
                true,
            )
            .unwrap();
        let f = fallback
            .decode_step_explicit(
                &[t],
                &[(i) as u32],
                &[0],
                &mut p2,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
                false,
            )
            .unwrap();
        let g_logits = grouped.read_logits().unwrap();
        let f_logits = fallback.read_logits().unwrap();
        let scale = g_logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let diff = g_logits
            .iter()
            .zip(&f_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff <= 2e-3 + 2e-3 * scale,
            "step {i}: grouped vs hipBLAS fallback: max diff {diff} (scale {scale})"
        );
        assert_eq!(g.0[0], f.0[0], "step {i}: greedy tokens must match");
    }
}

/// f16 variant of the fallback-vs-grouped pin. The grouped f16 kernels round
/// weights to f16 but keep activations f32; the hipBLAS f16 fallback
/// (gemm_batched_f16) additionally rounds activations and results to f16 per
/// layer — so the two paths drift by per-layer f16 rounding, bounded here at
/// the same 0.1 band as `batched_moe_f16_matches_single_seq`. Both engines
/// share the same f16 router, so expert selection is identical and only the
/// arithmetic may differ (tokens are NOT asserted equal).
#[test]
fn moe_grouped_fallback_matches_grouped_f16() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = moe_cfg();
    cfg.dtype = ModelDType::F16;
    let w = Weights::random(&cfg, 161).unwrap();
    let mut grouped = BatchedModel::new(hip.clone(), cfg, &w, 1).unwrap();
    let mut fallback = BatchedModel::new(hip.clone(), cfg, &w, 1).unwrap();
    let prompt: Vec<u32> = (0..12u32).map(|i| (i * 23 + 5) % 1024 + 1).collect();
    for (i, &t) in prompt.iter().enumerate() {
        let mut p1 = [SamplingParams::greedy(0)];
        let mut p2 = [SamplingParams::greedy(0)];
        grouped
            .decode_step_explicit(
                &[t],
                &[(i) as u32],
                &[0],
                &mut p1,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
                true,
            )
            .unwrap();
        fallback
            .decode_step_explicit(
                &[t],
                &[(i) as u32],
                &[0],
                &mut p2,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
                false,
            )
            .unwrap();
        let g_logits = grouped.read_logits().unwrap();
        let f_logits = fallback.read_logits().unwrap();
        let scale = g_logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let diff = g_logits
            .iter()
            .zip(&f_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff < 0.1,
            "step {i}: grouped vs hipBLAS fallback (f16): max diff {diff} (scale {scale})"
        );
    }
}

/// #103: a captured decode graph replays bit-identically to the eager path —
/// same greedy tokens and identical logits at every step (same kernels, same
/// buffers; the graph only folds the launches).
#[test]
fn decode_graph_matches_eager_f16() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = moe_cfg();
    cfg.dtype = ModelDType::F16;
    let w = Weights::random(&cfg, 171).unwrap();
    let mut eager = BatchedModel::new(hip.clone(), cfg, &w, 1).unwrap();
    let mut graphed = BatchedModel::new(hip.clone(), cfg, &w, 1).unwrap();
    graphed.set_decode_graph_enabled(true);
    let prompt: Vec<u32> = (0..16u32).map(|i| (i * 17 + 3) % 1024 + 1).collect();
    for (i, &t) in prompt.iter().enumerate() {
        let mut pe = [SamplingParams::greedy(0)];
        let mut pg = [SamplingParams::greedy(0)];
        let e = eager
            .decode_step_explicit(
                &[t],
                &[i as u32],
                &[0],
                &mut pe,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
                true,
            )
            .unwrap();
        let g = graphed
            .decode_step_explicit(
                &[t],
                &[i as u32],
                &[0],
                &mut pg,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
                true,
            )
            .unwrap();
        assert_eq!(e.0, g.0, "step {i}: graph vs eager sampled tokens");
        let el = eager.read_logits().unwrap();
        let gl = graphed.read_logits().unwrap();
        assert_eq!(el, gl, "step {i}: graph replay must be bitwise identical");
        assert_eq!(graphed.decode_graph_count(), 1, "one graph for n=1");
    }
}

/// #103: changing the active row count switches per-n graph buckets — each n
/// captures once and replays thereafter, outputs matching the eager twin.
#[test]
fn decode_graph_buckets_follow_active_rows() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = moe_cfg();
    cfg.dtype = ModelDType::F16;
    let w = Weights::random(&cfg, 173).unwrap();
    let mut eager = BatchedModel::new(hip.clone(), cfg, &w, 2).unwrap();
    let mut graphed = BatchedModel::new(hip.clone(), cfg, &w, 2).unwrap();
    graphed.set_decode_graph_enabled(true);
    // n = 1, 2, 2, 1: captures the n=1 and n=2 buckets, then replays both.
    for (i, &n) in [1usize, 2, 2, 1].iter().enumerate() {
        let toks: Vec<u32> = (0..n as u32)
            .map(|r| (i as u32 * 31 + r * 7 + 1) % 1024 + 1)
            .collect();
        let lens: Vec<u32> = vec![i as u32; n];
        let slots: Vec<u32> = (0..n as u32).collect();
        let counts = vec![Vec::new(); n];
        let bias = vec![Vec::new(); n];
        let mut pe = vec![SamplingParams::greedy(0); n];
        let mut pg = vec![SamplingParams::greedy(0); n];
        let e = eager
            .decode_step_explicit(&toks, &lens, &slots, &mut pe, &counts, &bias, true)
            .unwrap();
        let g = graphed
            .decode_step_explicit(&toks, &lens, &slots, &mut pg, &counts, &bias, true)
            .unwrap();
        assert_eq!(e.0, g.0, "step {i} (n={n}): graph vs eager tokens");
    }
    assert_eq!(graphed.decode_graph_count(), 2, "n=1 and n=2 buckets");
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
            assert_greedy_compatible(
                got[s],
                want,
                &want_logits,
                &format!("step {step_tokens:?} seq {s}"),
                4e-3 + 4e-3 * scale,
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
            assert_greedy_compatible(
                got[s],
                want,
                &want_logits,
                &format!("step {step_tokens:?} seq {s}"),
                4e-3 + 4e-3 * scale,
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
        .decode_step_explicit(toks, poss, slts, &mut params, &counts, &bias, true)
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
    let chain_a = builder.compute_chain(&as_i32(&seq_a));
    let (ta, _ra) = builder.build_table(&as_i32(&seq_a)).unwrap();
    // Materialize A's pages (the engine registers content only after prefill
    // completes); B's build then resolves the shared prefix through the cache.
    builder.register_chain(&chain_a, &ta);
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

/// Paged decode in F16 mode (#78 C3): the newly wired
/// `kv_store_paged_f16` / `attn_decode_paged_f16_gqa` must track the static
/// F16 path bit-tight across a page boundary, mirroring the F32 paged test.
#[test]
fn batched_paged_f16_decode_matches_static_gpu() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = Config::tiny(); // max_seq 256, tokens_per_page 64 -> 4 pages
    cfg.dtype = ModelDType::F16;
    let w = Weights::random(&cfg, 97).unwrap();
    let mut stat = BatchedModel::new(hip.clone(), cfg, &w, 1).unwrap();
    let mut paged = BatchedModel::with_paged_kv(hip.clone(), cfg, &w, 1, 64).unwrap();

    // 70 single-token steps span 2 pages, exercising the block-table page
    // boundary on the F16 store/attention path.
    let tokens: Vec<u32> = (0..70u32).map(|i| (i * 37) % 1024 + 1).collect();
    for (i, &t) in tokens.iter().enumerate() {
        let s_next = stat.decode_step(&[t]).unwrap();
        let p_next = paged.decode_step(&[t]).unwrap();
        assert_eq!(
            s_next, p_next,
            "pos {i}: f16 paged vs static greedy token diverged"
        );
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
            d <= 1e-3 + 1e-3 * scale,
            "pos {i}: f16 paged vs static logits max diff {d} (scale {scale})"
        );
    }
}

/// Paged decode in MLA mode (#78 C4): the newly wired assemble-to-scratch /
/// `kv_store_paged_mla` / `attn_decode_paged_mla_batched` path must track the
/// static contiguous MLA path across a page boundary.
#[test]
fn batched_paged_mla_decode_matches_static_gpu() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::mla(128, 2, 4, 1024, 256, 32, 16, 16, 8, 16); // tpp 64 -> 4 pages
    let w = Weights::random(&cfg, 13).unwrap();
    let mut stat = BatchedModel::new(hip.clone(), cfg, &w, 1).unwrap();
    let mut paged = BatchedModel::with_paged_kv(hip.clone(), cfg, &w, 1, 64).unwrap();

    // 70 single-token steps span 2 pages, exercising the block-table page
    // boundary on the MLA expanded-KV store/attention path.
    let tokens: Vec<u32> = (0..70u32).map(|i| (i * 37) % 1024 + 1).collect();
    for (i, &t) in tokens.iter().enumerate() {
        let s_next = stat.decode_step(&[t]).unwrap();
        let p_next = paged.decode_step(&[t]).unwrap();
        assert_eq!(
            s_next, p_next,
            "pos {i}: paged vs static MLA greedy token diverged"
        );
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
            d <= 1e-3 + 1e-3 * scale,
            "pos {i}: paged vs static MLA logits max diff {d} (scale {scale})"
        );
    }
}

/// Paged chunked prefill crossing a page boundary (#78 C2 review fix): one
/// step packs 80 rows (positions 0..79, spanning page 0 and page 1), then a
/// second chunk continues past it — the block-table page crossing on the
/// packed-prefill path must match sequential stepping.
#[test]
fn batched_paged_chunked_prefill_crosses_page_boundary() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny(); // max_seq 256; tokens_per_page 64
    let tpp = 64usize;
    let w = Weights::random(&cfg, 72).unwrap();
    let vocab = cfg.vocab_size;

    let seq: Vec<u32> = (0..96u32).map(|i| (i * 61 + 7) % 1024 + 1).collect();

    // One slot, 80 rows: the first chunk spans the 64-token page boundary.
    let mut paged = BatchedModel::with_paged_kv_rows(hip.clone(), cfg, &w, 1, 80, tpp).unwrap();
    let mut r = GpuModel::new(hip.clone(), cfg, &w).unwrap();
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(seq.len());
    for &t in &seq {
        ref_logits.push(r.decode_step(t).unwrap());
    }

    // Chunk 1: positions 0..79 (pages 0 and 1).
    let lens1: Vec<u32> = (0..80u32).collect();
    let slots1: Vec<u32> = vec![0; 80];
    let (s1, lm1) = fwd_rows(&mut paged, &seq[0..80], &lens1, &slots1, vocab);
    for &row in &[63usize, 64, 79] {
        assert_close(
            &lm1[row * vocab..(row + 1) * vocab],
            &ref_logits[row],
            &format!("boundary chunk1 row {row} (pos {row})"),
        );
        assert_eq!(
            s1[row],
            greedy_argmax(&ref_logits[row]),
            "chunk1 greedy {row}"
        );
    }

    // Chunk 2: positions 80..95 (page 1 only).
    let lens2: Vec<u32> = (80..96u32).collect();
    let slots2: Vec<u32> = vec![0; 16];
    let (s2, lm2) = fwd_rows(&mut paged, &seq[80..96], &lens2, &slots2, vocab);
    for &row in &[0usize, 15] {
        let pos = 80 + row;
        assert_close(
            &lm2[row * vocab..(row + 1) * vocab],
            &ref_logits[pos],
            &format!("boundary chunk2 row {row} (pos {pos})"),
        );
        assert_eq!(
            s2[row],
            greedy_argmax(&ref_logits[pos]),
            "chunk2 greedy {row}"
        );
    }
}

/// Paged packed prefill in F16 mode (#78 C2/C3 review fix): the f16 paged
/// branch of `run_kernels` with rows > slots (the server's prefill shape)
/// must match sequential F16 stepping.
#[test]
fn batched_paged_f16_chunked_prefill_matches_sequential() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = Config::tiny();
    cfg.dtype = ModelDType::F16;
    let tpp = 64usize;
    let w = Weights::random(&cfg, 73).unwrap();
    let vocab = cfg.vocab_size;

    let seq: Vec<u32> = (0..40u32).map(|i| (i * 13 + 5) % 1024 + 1).collect();
    let mut paged = BatchedModel::with_paged_kv_rows(hip.clone(), cfg, &w, 1, 24, tpp).unwrap();
    let mut r = GpuModel::new(hip.clone(), cfg, &w).unwrap();
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(seq.len());
    for &t in &seq {
        ref_logits.push(r.decode_step(t).unwrap());
    }

    // Cross-path bound: batched (packed) vs single-sequence GEMMs accumulate
    // in different orders, so f16 logits agree to ~1e-3 like the other f16
    // batched-vs-single tests (0.1).
    let check = |_row: usize, got: &[f32], want: &[f32], ctx: &str| {
        let max = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 0.1, "{ctx}: f16 logits max diff {max}");
    };
    let lens1: Vec<u32> = (0..16u32).collect();
    let slots1: Vec<u32> = vec![0; 16];
    let (s1, lm1) = fwd_rows(&mut paged, &seq[0..16], &lens1, &slots1, vocab);
    for &row in &[0usize, 15] {
        check(
            row,
            &lm1[row * vocab..(row + 1) * vocab],
            &ref_logits[row],
            &format!("f16 chunk1 row {row} (pos {row})"),
        );
        assert_eq!(
            s1[row],
            greedy_argmax(&ref_logits[row]),
            "f16 chunk1 greedy {row}"
        );
    }
    let lens2: Vec<u32> = (16..40u32).collect();
    let slots2: Vec<u32> = vec![0; 24];
    let (s2, lm2) = fwd_rows(&mut paged, &seq[16..40], &lens2, &slots2, vocab);
    for &row in &[0usize, 23] {
        let pos = 16 + row;
        check(
            row,
            &lm2[row * vocab..(row + 1) * vocab],
            &ref_logits[pos],
            &format!("f16 chunk2 row {row} (pos {pos})"),
        );
        assert_eq!(
            s2[row],
            greedy_argmax(&ref_logits[pos]),
            "f16 chunk2 greedy {row}"
        );
    }
}

/// Paged packed prefill in MLA mode (#78 C2/C4 review fix): the MLA paged
/// branch with rows > slots must match sequential MLA stepping.
#[test]
fn batched_paged_mla_chunked_prefill_matches_sequential() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::mla(128, 2, 4, 1024, 256, 32, 16, 16, 8, 16); // tpp 64
    let tpp = 64usize;
    let w = Weights::random(&cfg, 74).unwrap();
    let vocab = cfg.vocab_size;

    let seq: Vec<u32> = (0..40u32).map(|i| (i * 17 + 3) % 1024 + 1).collect();
    let mut paged = BatchedModel::with_paged_kv_rows(hip.clone(), cfg, &w, 1, 24, tpp).unwrap();
    let mut r = GpuModel::new(hip.clone(), cfg, &w).unwrap();
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(seq.len());
    for &t in &seq {
        ref_logits.push(r.decode_step(t).unwrap());
    }

    // Cross-path bound (batched vs single-sequence GEMM order), f32.
    let check = |_row: usize, got: &[f32], want: &[f32], ctx: &str| {
        let scale = want.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let max = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max <= 2e-3 + 2e-3 * scale,
            "{ctx}: MLA logits max diff {max} (scale {scale})"
        );
    };
    let lens1: Vec<u32> = (0..16u32).collect();
    let slots1: Vec<u32> = vec![0; 16];
    let (s1, lm1) = fwd_rows(&mut paged, &seq[0..16], &lens1, &slots1, vocab);
    for &row in &[0usize, 15] {
        check(
            row,
            &lm1[row * vocab..(row + 1) * vocab],
            &ref_logits[row],
            &format!("mla chunk1 row {row} (pos {row})"),
        );
        assert_eq!(
            s1[row],
            greedy_argmax(&ref_logits[row]),
            "mla chunk1 greedy {row}"
        );
    }
    let lens2: Vec<u32> = (16..40u32).collect();
    let slots2: Vec<u32> = vec![0; 24];
    let (s2, lm2) = fwd_rows(&mut paged, &seq[16..40], &lens2, &slots2, vocab);
    for &row in &[0usize, 23] {
        let pos = 16 + row;
        check(
            row,
            &lm2[row * vocab..(row + 1) * vocab],
            &ref_logits[pos],
            &format!("mla chunk2 row {row} (pos {pos})"),
        );
        assert_eq!(
            s2[row],
            greedy_argmax(&ref_logits[pos]),
            "mla chunk2 greedy {row}"
        );
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

/// Paged quantized paths pinned to a dequantized-weight CPU reference
/// (CLAUDE.md rule 3): both sides run the same int4/E4M3 weights — the GPU
/// dequantizes to f16, the CPU reference to f32 — so a paged-addressing
/// regression (shared by neither the contiguous GPU path nor a GPU-vs-GPU
/// pairing) surfaces as a large logit diff. Quantization error itself
/// cancels: only f16 rounding and GEMM order remain in the diff.
mod paged_quantized_cpu_parity {
    use super::*;
    use mach_model::ref_model::RefModel;
    use mach_model::weights::LayerWeights;
    use mach_model::{WeightsFp8, WeightsQ4};

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn dequantize_q4(wq: &WeightsQ4) -> Weights {
        Weights {
            tok_emb: wq.tok_emb.dequantize(),
            rms_final: wq.rms_final.clone(),
            lm_head: wq.lm_head.dequantize(),
            layers: wq
                .layers
                .iter()
                .map(|l| LayerWeights {
                    wq: l.wq.dequantize(),
                    wk: l.wk.dequantize(),
                    wv: l.wv.dequantize(),
                    wo: l.wo.dequantize(),
                    rms_attn: l.rms_attn.clone(),
                    wg: l.wg.dequantize(),
                    wu: l.wu.dequantize(),
                    wd: l.wd.dequantize(),
                    rms_mlp: l.rms_mlp.clone(),
                    bq: l.bq.clone(),
                    bk: l.bk.clone(),
                    bv: l.bv.clone(),
                    q_norm: l.q_norm.clone(),
                    k_norm: l.k_norm.clone(),
                    mla_q_a: l.mla_q_a.dequantize(),
                    mla_q_a_norm: l.mla_q_a_norm.clone(),
                    mla_q_b: l.mla_q_b.dequantize(),
                    mla_q_rope: l.mla_q_rope.dequantize(),
                    mla_kv_a: l.mla_kv_a.dequantize(),
                    mla_kv_a_norm: l.mla_kv_a_norm.clone(),
                    mla_kv_b: l.mla_kv_b.dequantize(),
                    mla_o: l.mla_o.dequantize(),
                    moe_router: l.moe_router.clone(),
                    moe_wg: l.moe_wg.dequantize(),
                    moe_wu: l.moe_wu.dequantize(),
                    moe_wd: l.moe_wd.dequantize(),
                    shared_wg: l.shared_wg.dequantize(),
                    shared_wu: l.shared_wu.dequantize(),
                    shared_wd: l.shared_wd.dequantize(),
                })
                .collect(),
        }
    }

    fn dequantize_fp8(wf: &WeightsFp8) -> Weights {
        Weights {
            tok_emb: wf.tok_emb.dequantize(),
            rms_final: wf.rms_final.clone(),
            lm_head: wf.lm_head.dequantize(),
            layers: wf
                .layers
                .iter()
                .map(|l| LayerWeights {
                    wq: l.wq.dequantize(),
                    wk: l.wk.dequantize(),
                    wv: l.wv.dequantize(),
                    wo: l.wo.dequantize(),
                    rms_attn: l.rms_attn.clone(),
                    wg: l.wg.dequantize(),
                    wu: l.wu.dequantize(),
                    wd: l.wd.dequantize(),
                    rms_mlp: l.rms_mlp.clone(),
                    bq: l.bq.clone(),
                    bk: l.bk.clone(),
                    bv: l.bv.clone(),
                    q_norm: l.q_norm.clone(),
                    k_norm: l.k_norm.clone(),
                    mla_q_a: l.mla_q_a.dequantize(),
                    mla_q_a_norm: l.mla_q_a_norm.clone(),
                    mla_q_b: l.mla_q_b.dequantize(),
                    mla_q_rope: l.mla_q_rope.dequantize(),
                    mla_kv_a: l.mla_kv_a.dequantize(),
                    mla_kv_a_norm: l.mla_kv_a_norm.clone(),
                    mla_kv_b: l.mla_kv_b.dequantize(),
                    mla_o: l.mla_o.dequantize(),
                    moe_router: l.moe_router.clone(),
                    moe_wg: l.moe_wg.dequantize(),
                    moe_wu: l.moe_wu.dequantize(),
                    moe_wd: l.moe_wd.dequantize(),
                    shared_wg: l.shared_wg.dequantize(),
                    shared_wu: l.shared_wu.dequantize(),
                    shared_wd: l.shared_wd.dequantize(),
                })
                .collect(),
        }
    }

    fn assert_paged_matches_cpu(
        paged: &mut BatchedModel,
        cfg: Config,
        want: &Weights,
        prompt: &[u32],
        tol: f32,
    ) {
        let mut cpu = RefModel::new(cfg, want.clone());
        for &t in prompt.iter() {
            paged.decode_step(&[t]).unwrap();
            let got = paged.read_logits().unwrap();
            // RefModel is stateful: feed exactly the one new token and read
            // its (last-position) logits.
            let cpu_logits = cpu.forward(&[t]);
            let d = max_abs_diff(&got, &cpu_logits);
            let scale = cpu_logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                d <= tol * (1.0 + scale),
                "step {}: paged quantized vs dequantized CPU ref: max diff {d} (scale {scale})",
                cpu.pos() - 1
            );
        }
    }

    #[test]
    fn batched_paged_q4_matches_dequantized_reference() {
        let Some(hip) = hip_ctx() else { return };
        let mut cfg = Config::tiny();
        cfg.dtype = ModelDType::F16;
        let w = Weights::random(&cfg, 101).unwrap();
        let wq = WeightsQ4::from_weights(&w, &cfg);
        let wd = dequantize_q4(&wq);
        let mut paged = BatchedModel::with_paged_kv_rows_q4(hip, cfg, &wq, 1, 4, 8).unwrap();
        // 20 tokens at tpp 8: crosses two page boundaries.
        let prompt: Vec<u32> = (0..20u32).map(|i| (i * 13 + 3) % 1024 + 1).collect();
        assert_paged_matches_cpu(&mut paged, cfg, &wd, &prompt, 5e-2);
    }

    /// Control: the contiguous Q4 engine against the same CPU reference —
    /// isolates whether a parity failure comes from the paged path or from
    /// the reference setup itself.
    #[test]
    fn batched_contiguous_q4_matches_dequantized_reference() {
        let Some(hip) = hip_ctx() else { return };
        let mut cfg = Config::tiny();
        cfg.dtype = ModelDType::F16;
        let w = Weights::random(&cfg, 101).unwrap();
        let wq = WeightsQ4::from_weights(&w, &cfg);
        let wd = dequantize_q4(&wq);
        let mut contig = BatchedModel::with_rows_q4(hip, cfg, &wq, 1, 4).unwrap();
        let prompt: Vec<u32> = (0..20u32).map(|i| (i * 13 + 3) % 1024 + 1).collect();
        assert_paged_matches_cpu(&mut contig, cfg, &wd, &prompt, 5e-2);
    }

    /// Q4-on-device expert pool (in-kernel dequant) must match the
    /// dequantized-f16 reference within the Q4-path tolerance (the reference
    /// rounds the dequantized weights to f16; the in-kernel path keeps the
    /// exact f32 scales). Exercises the `with_rows_q4_device` mode end-to-end
    /// on a MoE config: greedy decode steps (the `assert_paged_matches_cpu`
    /// loop) plus an explicit prefill step — Q4 mode runs the grouped path
    /// for EVERY step (no f16/f32 expert copy exists for the hipBLAS path),
    /// so the prefill branch of the grouped condition needs its own pin.
    #[test]
    fn batched_q4_device_moe_matches_dequantized_reference() {
        let Some(hip) = hip_ctx() else { return };
        let mut cfg = Config::tiny();
        cfg.dtype = ModelDType::F16;
        cfg.intermediate_size = 64;
        cfg.num_experts = 4;
        cfg.num_experts_per_tok = 2;
        let w = Weights::random(&cfg, 103).unwrap();
        let wq = WeightsQ4::from_weights(&w, &cfg);
        let wd = dequantize_q4(&wq);
        let mut eng = BatchedModel::with_rows_q4_device(hip.clone(), cfg, &wq, 1, 4).unwrap();
        let prompt: Vec<u32> = (0..20u32).map(|i| (i * 13 + 3) % 1024 + 1).collect();
        assert_paged_matches_cpu(&mut eng, cfg, &wd, &prompt, 5e-2);
        // Explicit prefill step (decode_only=false): Q4 mode routes it
        // through the grouped kernels too.
        eng.reset_state().unwrap();
        let mut params = vec![SamplingParams::greedy(1); 1];
        let lens = vec![0u32];
        let slots = vec![0u32];
        eng.decode_step_explicit(
            &[7],
            &lens,
            &slots,
            &mut params,
            &[Vec::new()],
            &[Vec::new()],
            false,
        )
        .unwrap();
        let got = eng.read_logits().unwrap();
        let mut ref_eng = BatchedModel::with_rows(hip.clone(), cfg, &wd, 1, 4).unwrap();
        ref_eng.reset_state().unwrap();
        let mut rp = vec![SamplingParams::greedy(1); 1];
        ref_eng
            .decode_step_explicit(
                &[7],
                &lens,
                &slots,
                &mut rp,
                &[Vec::new()],
                &[Vec::new()],
                false,
            )
            .unwrap();
        let want = ref_eng.read_logits().unwrap();
        let max = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max <= 5e-2,
            "Q4-on-device prefill must match the dequantized reference (max diff {max})"
        );
    }

    /// Control: the plain F16 engine on the unquantized f32 weights — if
    /// this fails the same way, the divergence is the harness/reference
    /// setup, not the Q4 upload path.
    #[test]
    fn batched_f16_matches_cpu_reference_control() {
        let Some(hip) = hip_ctx() else { return };
        let mut cfg = Config::tiny();
        cfg.dtype = ModelDType::F16;
        let w = Weights::random(&cfg, 101).unwrap();
        let mut eng = BatchedModel::with_rows(hip, cfg, &w, 1, 4).unwrap();
        let prompt: Vec<u32> = (0..20u32).map(|i| (i * 13 + 3) % 1024 + 1).collect();
        assert_paged_matches_cpu(&mut eng, cfg, &w, &prompt, 5e-2);
    }

    #[test]
    fn batched_paged_fp8_matches_dequantized_reference() {
        let Some(hip) = hip_ctx() else { return };
        let mut cfg = Config::tiny();
        cfg.dtype = ModelDType::F16;
        let w = Weights::random(&cfg, 103).unwrap();
        let wf = WeightsFp8::from_weights(&w, &cfg);
        let wd = dequantize_fp8(&wf);
        let mut paged = BatchedModel::with_paged_kv_rows_fp8(hip, cfg, &wf, 1, 4, 8).unwrap();
        let prompt: Vec<u32> = (0..20u32).map(|i| (i * 13 + 3) % 1024 + 1).collect();
        assert_paged_matches_cpu(&mut paged, cfg, &wd, &prompt, 5e-2);
    }
}
