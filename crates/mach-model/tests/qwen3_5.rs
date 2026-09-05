//! Qwen3.5 (Qwen3.8-27B family) config plumbing: hybrid full-attention /
//! gated-DeltaNet layer split, partial rotary, and the CPU reference forward
//! on a small synthetic model. Kernel/GPU parity lands with the HIP kernels;
//! these pin the host-side foundation (issue #112 Stage A).

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
