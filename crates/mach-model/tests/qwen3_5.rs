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
    use mach_model::model::GpuModel;
    use mach_model::ref_model::RefModel;
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

    /// Batched decode of two interleaved sequences: each row's logits vs its
    /// own single-sequence reference — pins that the SLOT-indexed GDN state
    /// keeps the sequences isolated while they share steps.
    fn batched_matches_ref(dtype: ModelDType) {
        let Some(hip) = hip_ctx() else { return };
        let cfg = small_cfg(dtype);
        let w = Weights::random(&cfg, 37).unwrap();
        let batch = 2usize;
        let mut m = BatchedModel::new(Arc::clone(&hip), cfg, &w, batch).unwrap();
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
}
