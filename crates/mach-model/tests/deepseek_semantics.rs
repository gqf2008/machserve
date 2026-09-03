//! DeepSeek-V2 semantics: shared experts, the `norm_topk_prob=false` top-k
//! weighting convention, and the fused `q_proj` split — all pinned on the CPU
//! with synthetic weights, independently of any real checkpoint.
//!
//! Every assertion here is a transcription of HF `modeling_deepseek.py`
//! (`DeepseekV2MoE.forward`, `MoEGate.forward`, `DeepseekV2Attention`) rather
//! than a restatement of the code under test: the f32 CPU reference
//! (`ref_model`) and the f64 reference (`fp64_ref`) are two separate
//! implementations, so cross-checking them catches a misreading that a
//! single-path test would pass.

use mach_model::fp64_ref::{LayerWeights64, moe_route};
use mach_model::q4::Q4Tensor;
use mach_model::ref_model::RefModel;
use mach_model::{Config, Weights};

/// Shared-expert config: small enough to run fast, with `n_shared_experts > 0`
/// so the always-routed branch is live.
fn shared_cfg(norm_topk: bool, rscale: f32, topk: usize) -> Config {
    let mut cfg = Config::tiny();
    cfg.intermediate_size = 64;
    cfg.moe_intermediate_size = 32;
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = topk;
    cfg.n_shared_experts = 2;
    cfg.moe_norm_topk = norm_topk;
    cfg.moe_routed_scale = rscale;
    cfg
}

/// Shared experts are one fused MLP of width `n_shared_experts *
/// moe_intermediate_size` (HF: `intermediate_size = moe_intermediate_size *
/// n_shared_experts`), not per-expert tensors.
#[test]
fn shared_experts_have_fused_width() {
    let cfg = shared_cfg(false, 1.0, 2);
    let w = Weights::random(&cfg, 42).unwrap();
    let l = &w.layers[0];
    let d = cfg.d_model;
    let sh = cfg.shared_size();
    assert_eq!(sh, 2 * 32, "shared width = n_shared_experts * expert_size");
    assert_eq!(l.shared_wg.len(), sh * d, "shared gate [sh, d]");
    assert_eq!(l.shared_wu.len(), sh * d, "shared up [sh, d]");
    assert_eq!(l.shared_wd.len(), d * sh, "shared down [d, sh]");
    // The routed experts keep their own (per-expert) width.
    assert_eq!(l.moe_wg.len(), cfg.num_experts * cfg.expert_size() * d);
    // The device FFN scratch is sized `intermediate_size.max(shared_size())`,
    // so the fused shared width must never exceed `intermediate_size` — a
    // checkpoint where it did would need that max() to be revisited.
    assert!(
        sh <= cfg.intermediate_size,
        "shared width {sh} must fit the FFN scratch ({})",
        cfg.intermediate_size
    );
    // Same invariant at DeepSeek-V2-Lite scale: 2 * 1408 = 2816 <= 10944.
    let mut lite = cfg;
    lite.intermediate_size = 10_944;
    lite.moe_intermediate_size = 1408;
    lite.n_shared_experts = 2;
    assert_eq!(lite.shared_size(), 2 * 1408);
    assert!(lite.shared_size() <= lite.intermediate_size);
    // Without `n_shared_experts` the tensors stay empty.
    let mut none = cfg;
    none.n_shared_experts = 0;
    let w = Weights::random(&none, 42).unwrap();
    assert!(w.layers[0].shared_wg.is_empty());
    assert!(w.layers[0].shared_wu.is_empty());
    assert!(w.layers[0].shared_wd.is_empty());
}

/// The shared MLP must actually contribute: zeroing its weights while keeping
/// every other tensor byte-identical has to change the logits. A wired-but-dead
/// branch (or one whose output is dropped) would pass a finiteness test.
#[test]
fn shared_experts_change_the_logits() {
    let cfg = shared_cfg(false, 1.0, 2);
    let w = Weights::random(&cfg, 7).unwrap();
    let mut zeroed = w.clone();
    for l in &mut zeroed.layers {
        l.shared_wg.fill(0.0);
        l.shared_wu.fill(0.0);
        l.shared_wd.fill(0.0);
    }
    let tokens = [5u32, 9, 3, 200];
    let mut a = RefModel::new(cfg, w);
    let mut b = RefModel::new(cfg, zeroed);
    let la = a.forward(&tokens);
    let lb = b.forward(&tokens);
    let max: f32 = la
        .iter()
        .zip(&lb)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    assert!(
        max > 1e-4,
        "shared experts must move the logits (max {max})"
    );
}

/// The f32 CPU reference and the f64 reference must agree on the shared-experts
/// MoE layer, so the addition — not the routed sum alone — is what both
/// compute. fp64 differs from fp32 only by the f32 path's rounding.
#[test]
fn shared_experts_f32_ref_matches_fp64_ref() {
    let cfg = shared_cfg(false, 1.0, 2);
    let w = Weights::random(&cfg, 11).unwrap();
    let tokens = [5u32, 9, 3, 200];
    let mut f32m = RefModel::new(cfg, w.clone());
    let a = f32m.forward(&tokens);
    let mut f64m = mach_model::fp64_ref::Fp64RefModel::new(cfg, &w);
    let b = f64m.forward(&tokens);
    let scale: f64 = a
        .iter()
        .fold(0.0f64, |m, &v| m.max(f64::from(v.abs())))
        .max(1.0);
    let max: f64 = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (f64::from(*x) - *y).abs())
        .fold(0.0, f64::max);
    assert!(
        max <= 1e-3 * scale,
        "shared-experts f32 vs f64 ref: max diff {max} (scale {scale})"
    );
}

/// Softmax over the router logits, used to derive the expected top-k weights
/// independently of `moe_route`.
fn softmax(v: &[f64]) -> Vec<f64> {
    let m = v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let e: Vec<f64> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

/// `MoEGate.forward`: renormalize only when `top_k > 1 && norm_topk_prob`, in
/// which case `routed_scaling_factor` is NOT applied; otherwise keep the raw
/// softmax score and multiply by `routed_scaling_factor`.
#[test]
fn moe_route_weights_follow_the_hf_branch() {
    let cfg = shared_cfg(false, 2.0, 3);
    let w = Weights::random(&cfg, 3).unwrap();
    let lw = LayerWeights64::from(&w.layers[0]);
    // Any nonzero input picks experts deterministically; the convention under
    // test does not depend on which ones.
    let xn: Vec<f64> = (0..cfg.d_model)
        .map(|i| ((i as f64) * 0.31).sin())
        .collect();

    // DeepSeek-V2 (`norm_topk_prob: false`): raw scores * rscale.
    let (ids, ws) = moe_route(&xn, &lw, &cfg);
    let probs = softmax(
        &lw.moe_router[..cfg.num_experts]
            .iter()
            .enumerate()
            .map(|(e, _)| {
                let row = &lw.moe_router[e * cfg.d_model..(e + 1) * cfg.d_model];
                xn.iter().zip(row).map(|(a, b)| a * b).sum::<f64>()
            })
            .collect::<Vec<_>>(),
    );
    assert_eq!(ids.len(), 3);
    let sum: f64 = ws.iter().sum();
    let want: f64 = ids.iter().map(|&e| probs[e as usize]).sum::<f64>() * 2.0;
    assert!(
        (sum - want).abs() < 1e-9,
        "norm_topk=false: weights must be p*rscale (got {sum}, want {want})"
    );
    assert!(
        (sum - 1.0).abs() > 1e-6,
        "norm_topk=false must NOT renormalize to 1 (got {sum})"
    );

    // Qwen-MoE (`norm_topk_prob: true`): renormalized, rscale ignored.
    let mut norm_cfg = cfg;
    norm_cfg.moe_norm_topk = true;
    let (ids, ws) = moe_route(&xn, &lw, &norm_cfg);
    let sum: f64 = ws.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-9,
        "norm_topk=true must renormalize to 1 (got {sum})"
    );
    let norm: f64 = ids.iter().map(|&e| probs[e as usize]).sum::<f64>();
    for (j, &e) in ids.iter().enumerate() {
        let want = probs[e as usize] / norm;
        assert!(
            (ws[j] - want).abs() < 1e-9,
            "norm_topk=true slot {j}: got {} want {want}",
            ws[j]
        );
    }

    // HF skips renormalization when `top_k == 1`, even with norm_topk_prob.
    let mut one_cfg = cfg;
    one_cfg.moe_norm_topk = true;
    one_cfg.num_experts_per_tok = 1;
    let (ids, ws) = moe_route(&xn, &lw, &one_cfg);
    assert_eq!(ids.len(), 1);
    let want = probs[ids[0] as usize] * 2.0;
    assert!(
        (ws[0] - want).abs() < 1e-9,
        "top_k=1 must not renormalize: got {} want {want}",
        ws[0]
    );
}

/// A real DeepSeek checkpoint stores ONE fused `q_proj` of shape
/// `[heads*(nope+rope), kk]`; the runtime keeps the two halves apart. Splitting
/// the quantized fused tensor by row blocks must be byte-identical to
/// quantizing the per-head halves on their own — nothing may be requantized
/// along the way, or the weights the GPU sees would drift from the checkpoint.
#[test]
fn q4_fused_q_row_split_is_byte_identical_to_halves() {
    let heads = 4usize;
    let nope = 128usize;
    let rope = 64usize;
    let kk = 96usize; // 3 groups/row: a multiple of Q4_GROUP, like d_model.
    let rows = heads * (nope + rope);
    let fused: Vec<f32> = (0..rows * kk)
        .map(|i| ((i as f32) * 0.017).sin() * 2.0)
        .collect();
    let qfused = Q4Tensor::quantize(&fused);

    let mut nope_blocks = Vec::new();
    let mut rope_blocks = Vec::new();
    let mut nope_rows = Vec::new();
    let mut rope_rows = Vec::new();
    for h in 0..heads {
        let base = h * (nope + rope);
        nope_blocks.push((base, nope));
        rope_blocks.push((base + nope, rope));
        nope_rows.extend_from_slice(&fused[base * kk..(base + nope) * kk]);
        rope_rows.extend_from_slice(&fused[(base + nope) * kk..(base + nope + rope) * kk]);
    }
    let got_nope = Q4Tensor::concat_many(&qfused.split_row_blocks(kk, &nope_blocks));
    let got_rope = Q4Tensor::concat_many(&qfused.split_row_blocks(kk, &rope_blocks));
    let want_nope = Q4Tensor::quantize(&nope_rows);
    let want_rope = Q4Tensor::quantize(&rope_rows);
    assert_eq!(got_nope.q_bytes(), want_nope.q_bytes(), "nope packed bytes");
    assert_eq!(got_nope.scales(), want_nope.scales(), "nope scales");
    assert_eq!(got_rope.q_bytes(), want_rope.q_bytes(), "rope packed bytes");
    assert_eq!(got_rope.scales(), want_rope.scales(), "rope scales");
    assert_eq!(got_nope.len(), heads * nope * kk);
    assert_eq!(got_rope.len(), heads * rope * kk);
}

/// The split is only sound while `kk` keeps rows on group boundaries; a
/// non-multiple would silently mix two rows' nibbles into one byte and must be
/// rejected rather than produce plausible-looking garbage.
#[test]
#[should_panic(expected = "not a multiple of Q4_GROUP")]
fn q4_fused_q_row_split_rejects_unaligned_width() {
    let t = Q4Tensor::quantize(&[0.5f32; 128]);
    let _ = t.split_row_blocks(20, &[(0, 2)]);
}
