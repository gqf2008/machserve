//! Qwen3.5 (Qwen3.8-27B family) plumbing: hybrid full-attention /
//! gated-DeltaNet layer split, partial rotary, the CPU reference forward on a
//! small synthetic model, and (hip feature) GPU parity for the single-seq
//! and batched models against that reference across the hybrid stack
//! (issue #112 Stage A).

use mach_model::ref_model::RefModel;
use mach_model::{Config, Weights};

/// Small but structurally faithful config: 5 layers with interval 4 means
/// ONLY layer 3 is full-attention (layers 0,1,2,4 are GDN), head_dim 16 with
/// rotary_pct 0.25 -> rotary_dim 4, gdn k/v head dims 8.
fn qwen35_small() -> Config {
    Config::qwen3_5(64, 5, 4, 2, 16, 176, 97, 64, 2, 4, 8, 4)
}

#[test]
fn config_layer_pattern_matches_interval_4() {
    let cfg = qwen35_small();
    assert!(cfg.gdn_enabled());
    assert_eq!(cfg.full_attention_interval, 4);
    // (li + 1) % 4 == 0 -> layers 3 (and 7, 11, ... in the real 64-layer
    // model; here only 3 exists below 5).
    let full: Vec<bool> = (0..cfg.n_layers)
        .map(|li| cfg.layer_is_full_attn(li))
        .collect();
    assert_eq!(full, vec![false, false, false, true, false]);
    assert_eq!(cfg.attn_rotary_dim(), 4, "0.25 * 16");
    assert_eq!(cfg.gdn_key_dim(), 2 * 8);
    assert_eq!(cfg.gdn_value_dim(), 4 * 8);
    assert_eq!(cfg.gdn_conv_kernel, 4);
    // Family constants.
    assert_eq!(cfg.rope_theta, 10_000_000.0);
    assert!(cfg.qk_norm);
    assert!(!cfg.rope_interleave, "Qwen family pairs half-split");
    assert!(!cfg.yarn());
    // Qwen3.5 family: doubled q_proj with a sigmoid attention output gate,
    // and zero-centered (`x * (1 + w)`) RMSNorm weights in the checkpoint.
    assert!(cfg.attn_output_gate);
    assert!(cfg.zero_centered_norm);
}

/// `Weights::random` must populate `linear_attn.*` tensors on GDN layers and
/// leave the standard attention tensors empty there (and vice versa on the
/// full-attention layer). The conv weight follows the checkpoint's
/// identity-like init: taps `[1, 1, 1, 2]` for kernel 4.
#[test]
fn random_weights_split_by_layer_type() {
    let cfg = qwen35_small();
    let w = Weights::random(&cfg, 7).unwrap();
    let kd = cfg.gdn_key_dim();
    let vd = cfg.gdn_value_dim();
    let conv_dim = 2 * kd + vd;
    for (li, lw) in w.layers.iter().enumerate() {
        if cfg.layer_is_full_attn(li) {
            assert!(!lw.wq.is_empty(), "layer {li} needs q_proj");
            // `attn_output_gate`: q_proj doubles (`[query | gate]` per head).
            assert_eq!(
                lw.wq.len(),
                2 * cfg.n_heads * cfg.head_dim * cfg.d_model,
                "layer {li} doubled q_proj"
            );
            assert!(!lw.q_norm.is_empty(), "layer {li} needs q_norm");
            assert!(lw.gdn_in_qkv.is_empty(), "layer {li} has no GDN");
        } else {
            assert!(lw.wq.is_empty(), "layer {li} is linear attention");
            assert!(lw.q_norm.is_empty(), "layer {li} has no q_norm");
            assert_eq!(lw.gdn_in_qkv.len(), conv_dim * cfg.d_model);
            assert_eq!(lw.gdn_in_z.len(), vd * cfg.d_model);
            assert_eq!(lw.gdn_in_a.len(), cfg.gdn_v_heads * cfg.d_model);
            assert_eq!(lw.gdn_in_b.len(), cfg.gdn_v_heads * cfg.d_model);
            assert_eq!(lw.gdn_conv_w.len(), conv_dim * 4);
            assert_eq!(lw.gdn_a_log.len(), cfg.gdn_v_heads);
            assert_eq!(lw.gdn_dt_bias.len(), cfg.gdn_v_heads);
            assert_eq!(lw.gdn_norm.len(), cfg.gdn_head_dim);
            assert_eq!(lw.gdn_out.len(), cfg.d_model * vd);
            // Identity-like depthwise init: the first k-1 taps are 1, the
            // newest tap is 2 (HF's `eye` + last-column-doubling init).
            for c in 0..conv_dim {
                assert_eq!(lw.gdn_conv_w[c * 4], 1.0, "layer {li} tap0");
                assert_eq!(lw.gdn_conv_w[c * 4 + 1], 1.0, "layer {li} tap1");
                assert_eq!(lw.gdn_conv_w[c * 4 + 2], 1.0, "layer {li} tap2");
                assert_eq!(lw.gdn_conv_w[c * 4 + 3], 2.0, "layer {li} tap3");
            }
            // A_log spans the checkpoint init range U(0.01, 16).
            for &al in &lw.gdn_a_log {
                let v = al.exp();
                assert!((0.01..=16.0).contains(&v), "A_log {al} out of init range");
            }
            // dt_bias is ones at init.
            for &db in &lw.gdn_dt_bias {
                assert_eq!(db, 1.0);
            }
        }
        // The MLP exists on every layer (Qwen3.8-27B text stack is dense).
        assert!(!lw.wg.is_empty());
    }
}

/// The CPU reference runs the hybrid stack end to end: finite logits,
/// deterministic across two identically-seeded models, and the GDN recurrent
/// state actually carries information (later positions differ from the
/// first).
#[test]
fn ref_forward_finite_and_deterministic() {
    let cfg = qwen35_small();
    let w = Weights::random(&cfg, 11).unwrap();
    let mut a = RefModel::new(cfg, w.clone());
    let mut b = RefModel::new(cfg, w);
    let tokens = [3u32, 17, 42, 5, 90];
    let mut prev: Option<Vec<f32>> = None;
    for t in tokens {
        let la = a.decode_step(t);
        let lb = b.decode_step(t);
        assert_eq!(la.len(), 97);
        assert!(la.iter().all(|v| v.is_finite()), "non-finite logits");
        assert_eq!(la, lb, "two identically-seeded models diverged");
        if let Some(p) = &prev {
            assert_ne!(la, *p, "logits frozen across steps (state not carried)");
        }
        prev = Some(la);
    }
    assert_eq!(a.pos(), tokens.len());
}

/// The attention output gate is pinned end to end by the hand-computed
/// `gated_attention_step_matches_hand_computation` unit test (doubled q_proj
/// split, sigmoid placement). Here at integration level just verify the
/// doubled q_proj flows through a full hybrid stack: random weights seed,
/// decode runs, logits stay finite.
#[test]
fn ref_forward_with_gate_is_finite_across_hybrid_stack() {
    let cfg = qwen35_small();
    let w = Weights::random(&cfg, 21).unwrap();
    let mut m = RefModel::new(cfg, w);
    for t in [3u32, 17, 42] {
        let l = m.decode_step(t);
        assert!(l.iter().all(|v| v.is_finite()));
    }
}

/// GPU parity for the hybrid stack: the single-sequence and batched models
/// (f32 + f16) vs the CPU reference, across multi-token decodes — exercises
/// the GDN kernels (conv update, l2norm, delta-rule recurrence, gated norm),
/// the attention output gate split/apply, and partial rope end to end.
#[cfg(feature = "hip")]
mod gpu {
    use super::qwen35_small;
    use mach_kernel_sys::hip;
    use mach_model::batched::BatchedModel;
    use mach_model::config::ModelDType;
    use mach_model::continuous::ContinuousModel;
    use mach_model::model::GpuModel;
    use mach_model::ref_model::RefModel;
    use mach_model::sampling::SamplingParams;
    use mach_model::{Config, Weights};
    use std::sync::Arc;

    fn hip_ctx() -> Option<Arc<hip::Hip>> {
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

    /// GPU-vs-CPU margin. F32 keeps the repo's parity bound
    /// (`tests/moe.rs`: 2e-3 + 2e-3 * scale). F16 rounds the projection
    /// weights, and that noise rides the GDN recurrence: observed peak
    /// ~1.2e-2 on random weights at step 2, DECAYING afterwards (per-token
    /// rounding noise, not state accumulation — F32 passes the tight bound
    /// at every step). Use the `tests/fp16.rs` convention instead: a loose
    /// absolute bound plus greedy-argmax agreement at every step (the
    /// functional check that actually pins the decode).
    fn gpu_tol(dtype: ModelDType, scale: f32) -> f32 {
        match dtype {
            ModelDType::F16 => 5e-2,
            _ => 2e-3 + 2e-3 * scale,
        }
    }

    fn argmax(xs: &[f32]) -> usize {
        let mut best = 0usize;
        for (i, &v) in xs.iter().enumerate() {
            if v > xs[best] {
                best = i;
            }
        }
        best
    }

    fn check_row(label: &str, dtype: ModelDType, gpu: &[f32], cpu: &[f32]) {
        assert_eq!(gpu.len(), cpu.len(), "{label}: length mismatch");
        let scale = gpu.iter().chain(cpu).fold(0.0f32, |m, &v| m.max(v.abs()));
        let tol = gpu_tol(dtype, scale);
        let diff = gpu
            .iter()
            .zip(cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            diff <= tol,
            "{label}: max diff {diff:.3e} > tol {tol:.3e} (scale {scale:.3e})"
        );
        assert_eq!(
            argmax(gpu),
            argmax(cpu),
            "{label}: greedy argmax flipped (f16 noise changed the token)"
        );
    }

    fn small_cfg(dtype: ModelDType) -> Config {
        let mut cfg = qwen35_small();
        cfg.dtype = dtype;
        cfg
    }

    /// Single-sequence decode, step by step (the GDN recurrence must track
    /// the reference across positions, not just at the first token).
    fn single_seq_matches_ref(dtype: ModelDType) {
        let Some(hip) = hip_ctx() else { return };
        let cfg = small_cfg(dtype);
        let w = Weights::random(&cfg, 31).unwrap();
        let mut cpu = RefModel::new(cfg, w.clone());
        let mut gpu = GpuModel::new(Arc::clone(&hip), cfg, &w).unwrap();
        for (step, &t) in [3u32, 17, 42, 5].iter().enumerate() {
            let cpu_logits = cpu.decode_step(t);
            let gpu_logits = gpu.decode_step(t).unwrap();
            check_row(
                &format!("{:?} step {step}", cfg.dtype),
                dtype,
                &gpu_logits,
                &cpu_logits,
            );
        }
    }

    #[test]
    fn single_seq_matches_ref_f32() {
        single_seq_matches_ref(ModelDType::F32);
    }

    #[test]
    fn single_seq_matches_ref_f16() {
        single_seq_matches_ref(ModelDType::F16);
    }

    /// Continuous-batching prefill on the hybrid stack: GDN models apply the
    /// per-slot recurrent state once per step row, so the engine must prefill
    /// sequentially (one prompt token per step per slot) — token-major packing
    /// is #112 Stage B and is rejected by `decode_step_explicit`. Two
    /// staggered sequences (different prompt lengths, so prefill and decode
    /// rows interleave) must complete and generate the same greedy tokens as
    /// the step-by-step reference chains.
    /// Greedy engine generation vs the sequential RefModel chain. Chunked
    /// GDN prefill (#112 Stage B) REASSOCIATES the recurrence's sums, so at
    /// near-tied logits the argmax may flip without a semantic error
    /// (per-position logits stay within the row tolerance — pinned by
    /// `batched_gdn_chunk_step_matches_sequential`; the repo lesson: judge
    /// synthetic-random-weight argmax flips by the top1-top2 margin). Where
    /// the reference IS decisive (margin >= `CHAIN_TIE`) the engine must
    /// follow it; after a tolerated flip the chain continues from the
    /// engine's actual token (teacher forcing), keeping later steps
    /// meaningful.
    #[test]
    fn continuous_prefill_sequential_matches_ref_f32() {
        let Some(hip) = hip_ctx() else { return };
        let cfg = small_cfg(ModelDType::F32);
        let w = Weights::random(&cfg, 43).unwrap();
        let jobs: [(Vec<u32>, usize); 2] = [(vec![3, 17, 42, 5], 3), (vec![90, 7], 2)];
        let mut eng = ContinuousModel::new(Arc::clone(&hip), cfg, &w, 2).unwrap();
        let mut ids = Vec::new();
        for (prompt, max_new) in &jobs {
            ids.push(
                eng.add(
                    prompt,
                    *max_new,
                    None,
                    Vec::new(),
                    Vec::new(),
                    SamplingParams::default(),
                )
                .unwrap(),
            );
        }
        while !eng.all_done() {
            // Chunk-contract violations would surface as the engine-step
            // error here.
            eng.step().unwrap();
        }
        const CHAIN_TIE: f32 = 2e-2;
        for ((prompt, max_new), &id) in jobs.iter().zip(&ids) {
            let got = eng.generated(id);
            let mut cpu = RefModel::new(cfg, w.clone());
            let mut logits = None;
            for &t in prompt {
                logits = Some(cpu.decode_step(t));
            }
            for (i, &tok) in got.iter().enumerate() {
                let l = logits.as_ref().expect("prompt consumed");
                let want = argmax(l);
                if tok as usize != want {
                    let mut sorted: Vec<f32> = l.clone();
                    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    let margin = sorted[0] - sorted[1];
                    assert!(
                        margin < CHAIN_TIE,
                        "seq {prompt:?} step {i}: engine={got:?} diverged at a DECISIVE step (margin {margin:.4}, want {want})",
                    );
                }
                logits = Some(cpu.decode_step(tok));
            }
            assert_eq!(got.len(), *max_new);
        }
    }

    /// Slot reuse must not leak GDN recurrent state: the recurrence reads the
    /// per-slot state unconditionally, so a sequence admitted onto a retired
    /// slot would inherit its context (observed on the real 27B: an
    /// arithmetic answer bleeding into the next request). The reused-slot
    /// generation must match a fresh-engine generation token for token.
    #[test]
    fn gdn_slot_reuse_clears_recurrent_state() {
        let Some(hip) = hip_ctx() else { return };
        let cfg = small_cfg(ModelDType::F32);
        let w = Weights::random(&cfg, 47).unwrap();
        let prompt = vec![3u32, 17, 42, 5];
        let max_new = 4;
        // Control: the prompt on a fresh engine.
        let control = {
            let mut eng = ContinuousModel::new(Arc::clone(&hip), cfg, &w, 2).unwrap();
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
            while !eng.all_done() {
                eng.step().unwrap();
            }
            eng.generated(id)
        };
        // Retire an unrelated sequence, then run the same prompt on the
        // reused (compacted) slot.
        let mut eng = ContinuousModel::new(Arc::clone(&hip), cfg, &w, 2).unwrap();
        let x = eng
            .add(
                &[90u32, 7, 64],
                2,
                None,
                Vec::new(),
                Vec::new(),
                SamplingParams::default(),
            )
            .unwrap();
        while !eng.is_done(x) {
            eng.step().unwrap();
        }
        assert_eq!(eng.active(), 0, "unrelated sequence retired");
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
        while !eng.all_done() {
            eng.step().unwrap();
        }
        let reused = eng.generated(id);
        assert_eq!(
            control, reused,
            "slot reuse leaked GDN state: fresh={control:?} reused={reused:?}"
        );
    }

    /// Compaction must move the GDN recurrent state with the sequence (#112):
    /// when a lower slot retires mid-life of a later sequence, the moved
    /// sequence keeps its recurrence. Caught live by the chunked-prefill
    /// engine test (chunking changed prefill pacing so an earlier slot
    /// finished first and the later one compacted onto its stale state —
    /// a DECISIVE 0.07-margin argmax flip, not tolerance noise). This test
    /// pins the same scenario directly: the compacted run must equal an
    /// uncompacted control token for token.
    #[test]
    fn gdn_compaction_moves_recurrent_state() {
        let Some(hip) = hip_ctx() else { return };
        let cfg = small_cfg(ModelDType::F32);
        let w = Weights::random(&cfg, 59).unwrap();
        let prompt_b = vec![90u32, 7, 42, 5];
        let max_new_b = 5usize;
        // Control: sequence B alone (capacity 2, one admission — no
        // compaction ever).
        let control = {
            let mut eng = ContinuousModel::new(Arc::clone(&hip), cfg, &w, 2).unwrap();
            let id = eng
                .add(
                    &prompt_b,
                    max_new_b,
                    None,
                    Vec::new(),
                    Vec::new(),
                    SamplingParams::default(),
                )
                .unwrap();
            while !eng.all_done() {
                eng.step().unwrap();
            }
            eng.generated(id)
        };
        // Sequence A (slot 0) finishes after one token; B (slot 1) is still
        // prefilling/decoding, so B compacts onto slot 0 mid-life.
        let mut eng = ContinuousModel::new(Arc::clone(&hip), cfg, &w, 2).unwrap();
        let a = eng
            .add(
                &[3u32, 17],
                1,
                None,
                Vec::new(),
                Vec::new(),
                SamplingParams::default(),
            )
            .unwrap();
        let b = eng
            .add(
                &prompt_b,
                max_new_b,
                None,
                Vec::new(),
                Vec::new(),
                SamplingParams::default(),
            )
            .unwrap();
        while !eng.all_done() {
            eng.step().unwrap();
        }
        assert_eq!(eng.generated(a).len(), 1, "A finished after one token");
        assert!(
            !eng.is_done(b) || eng.generated(b).len() == max_new_b,
            "B ran to its budget"
        );
        assert_eq!(
            eng.generated(b),
            control,
            "compaction lost/corrupted B's GDN state: compacted={:?} control={:?}",
            eng.generated(b),
            control
        );
    }

    /// Batched decode of two interleaved sequences: each row's logits vs its
    /// own single-sequence reference — pins that the SLOT-indexed GDN state
    /// keeps the sequences isolated while they share steps.
    fn batched_matches_ref(dtype: ModelDType) {
        let Some(hip) = hip_ctx() else { return };
        let cfg = small_cfg(dtype);
        let w = Weights::random(&cfg, 37).unwrap();
        let batch = 2usize;
        let mut m = BatchedModel::with_rows(Arc::clone(&hip), cfg, &w, batch, 8).unwrap();
        let mut rows: Vec<RefModel> = (0..batch).map(|_| RefModel::new(cfg, w.clone())).collect();
        let streams = [[3u32, 17, 42, 5], [90u32, 7, 64, 21]];
        for step in 0..streams[0].len() {
            let toks: Vec<u32> = streams.iter().map(|s| s[step]).collect();
            m.decode_step(&toks).unwrap();
            let logits = m.read_logits().unwrap();
            let vocab = cfg.vocab_size;
            for (i, row) in rows.iter_mut().enumerate() {
                let cpu_logits = row.decode_step(toks[i]);
                let gpu_row = &logits[i * vocab..(i + 1) * vocab];
                check_row(
                    &format!("{:?} batched row {i} step {step}", cfg.dtype),
                    dtype,
                    gpu_row,
                    &cpu_logits,
                );
            }
        }
    }

    #[test]
    fn batched_matches_ref_f32() {
        batched_matches_ref(ModelDType::F32);
    }

    #[test]
    fn batched_matches_ref_f16() {
        batched_matches_ref(ModelDType::F16);
    }

    /// Stage B chunked scan (#112): one `decode_step_explicit` carrying a
    /// 5-row GDN chunk of slot 0 PLUS a single decode row of slot 1 must
    /// reproduce the sequential per-token recurrence — every chunk row's
    /// logits match the RefModel chain at the same position (the chunk
    /// reassociates the recurrence's sums, so the tolerance is the same
    /// reassociation class the layer/kernel tests pin, not bit equality).
    /// Also pins the mixed-step contract: runs partition ALL rows, so the
    /// lone decode row of slot 1 rides along correctly.
    #[test]
    fn batched_gdn_chunk_step_matches_sequential() {
        let Some(hip) = hip_ctx() else { return };
        let cfg = small_cfg(ModelDType::F32);
        let w = Weights::random(&cfg, 51).unwrap();
        let chunk: [u32; 5] = [3, 17, 42, 5, 11];
        let lone = 90u32;
        let batch = 2usize;
        let mut m = BatchedModel::with_rows(Arc::clone(&hip), cfg, &w, batch, 8).unwrap();
        // rows: slot 0 chunk (positions 0..5), then slot 1 single row.
        let tokens: Vec<u32> = chunk.iter().copied().chain([lone]).collect();
        let lens: Vec<u32> = [0u32, 1, 2, 3, 4].iter().copied().chain([0]).collect();
        let slots: Vec<u32> = [0u32, 0, 0, 0, 0].iter().copied().chain([1]).collect();
        let params: Vec<SamplingParams> = (0..tokens.len())
            .map(|_| SamplingParams::default())
            .collect();
        let counts: Vec<Vec<(u32, u32)>> = (0..tokens.len()).map(|_| Vec::new()).collect();
        let bias: Vec<Vec<(u32, f32)>> = (0..tokens.len()).map(|_| Vec::new()).collect();
        let mut params = params;
        m.decode_step_explicit(&tokens, &lens, &slots, &mut params, &counts, &bias, false)
            .unwrap();
        let logits = m.read_logits_rows(tokens.len()).unwrap();
        let vocab = cfg.vocab_size;
        // Slot 0: sequential chain, every position compared.
        let mut cpu = RefModel::new(cfg, w.clone());
        for (t, tok) in chunk.iter().enumerate() {
            let cpu_logits = cpu.decode_step(*tok);
            let gpu_row = &logits[t * vocab..(t + 1) * vocab];
            check_row(
                &format!("chunk row {t}"),
                ModelDType::F32,
                gpu_row,
                &cpu_logits,
            );
            assert_eq!(
                argmax(gpu_row),
                argmax(&cpu_logits),
                "chunk row {t} argmax flipped"
            );
        }
        // Slot 1's lone row: its own reference (fresh state).
        let mut cpu1 = RefModel::new(cfg, w.clone());
        let cpu1_logits = cpu1.decode_step(lone);
        check_row(
            "lone decode row",
            ModelDType::F32,
            &logits[5 * vocab..6 * vocab],
            &cpu1_logits,
        );
    }

    /// The chunk contract must be enforced loudly: a slot split across two
    /// blocks (the #116 double-application shape) is rejected instead of
    /// silently corrupting the recurrent state.
    #[test]
    fn gdn_chunk_rejects_split_slot_blocks() {
        let Some(hip) = hip_ctx() else { return };
        let cfg = small_cfg(ModelDType::F32);
        let w = Weights::random(&cfg, 53).unwrap();
        let mut m = BatchedModel::with_rows(Arc::clone(&hip), cfg, &w, 2, 8).unwrap();
        type RowExtras = (
            Vec<SamplingParams>,
            Vec<Vec<(u32, u32)>>,
            Vec<Vec<(u32, f32)>>,
        );
        let mk = |n: usize| -> RowExtras {
            (
                (0..n).map(|_| SamplingParams::default()).collect(),
                (0..n).map(|_| Vec::new()).collect(),
                (0..n).map(|_| Vec::new()).collect(),
            )
        };
        // slot 0, slot 1, slot 0 again: two blocks on slot 0.
        let (mut p3, c3, b3) = mk(3);
        let err = m
            .decode_step_explicit(
                &[3u32, 90, 17],
                &[0u32, 0, 1],
                &[0u32, 1, 0],
                &mut p3,
                &c3,
                &b3,
                false,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("contiguous"),
            "expected a contiguity rejection, got: {err}"
        );
        // Non-consecutive positions inside one block are rejected too.
        let (mut p2, c2, b2) = mk(2);
        let err = m
            .decode_step_explicit(
                &[3u32, 17],
                &[0u32, 2],
                &[0u32, 0],
                &mut p2,
                &c2,
                &b2,
                false,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("consecutive"),
            "expected a position-order rejection, got: {err}"
        );
    }

    /// Dequantizes storage-Q4 weights back to exact f32 (`nibble * group
    /// scale` — the same values `gemv_q4`/`embed_gather_q4` compute in
    /// kernel), giving the dense-Q4 oracle a bit-identical weight source.
    fn dequantize_q4(wq: &mach_model::WeightsQ4) -> Weights {
        use mach_model::weights::LayerWeights;
        mach_model::Weights {
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
                    gdn_in_qkv: l.gdn_in_qkv.dequantize(),
                    gdn_in_z: l.gdn_in_z.dequantize(),
                    gdn_in_a: l.gdn_in_a.clone(),
                    gdn_in_b: l.gdn_in_b.clone(),
                    gdn_conv_w: l.gdn_conv_w.clone(),
                    gdn_a_log: l.gdn_a_log.clone(),
                    gdn_dt_bias: l.gdn_dt_bias.clone(),
                    gdn_norm: l.gdn_norm.clone(),
                    gdn_out: l.gdn_out.dequantize(),
                })
                .collect(),
        }
    }

    /// Dense Q4-on-device (server `MACH_Q4_DEVICE=2` class): every big tensor
    /// stays raw Q4 on device, dequantized in-kernel by `gemv_q4` /
    /// `embed_gather_q4`, vs the CPU reference on the SAME dequantized f32
    /// weights — the big GEMMs share a bit-identical source, so the residual
    /// diff is the f16-only tensors (GDN a/b ride the recurrence) plus GEMM
    /// order: the F16 convention (loose bound + greedy argmax per step).
    #[test]
    fn batched_q4_all_matches_dequantized_ref() {
        let Some(hip) = hip_ctx() else { return };
        let cfg = small_cfg(ModelDType::F16);
        let w = Weights::random(&cfg, 43).unwrap();
        let wq = mach_model::WeightsQ4::from_weights(&w, &cfg);
        let mut cpu = RefModel::new(cfg, dequantize_q4(&wq));
        let mut m = BatchedModel::with_rows_q4_all(Arc::clone(&hip), cfg, &wq, 1, 1).unwrap();
        for (step, &t) in [3u32, 17, 42, 5].iter().enumerate() {
            let cpu_logits = cpu.decode_step(t);
            m.decode_step(&[t]).unwrap();
            let gpu_logits = m.read_logits().unwrap();
            check_row(
                &format!("q4-all step {step}"),
                ModelDType::F16,
                &gpu_logits,
                &cpu_logits,
            );
        }
    }
}
