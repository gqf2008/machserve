//! MoE foundation: config + weights + CPU reference forward (synthetic model).
//!
//! The GPU MoE kernels and real-model loading are later slices; this verifies
//! the data plumbing (router / per-expert tensors) and the reference forward.

use mach_model::ref_model::RefModel;
use mach_model::{Config, Weights};

fn moe_cfg() -> Config {
    let mut cfg = Config::tiny();
    cfg.intermediate_size = 64;
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg
}

#[test]
fn moe_weights_have_expected_shapes() {
    let cfg = moe_cfg();
    let w = Weights::random(&cfg, 42).unwrap();
    let l = &w.layers[0];
    let d = cfg.d_model;
    let inter = cfg.intermediate_size;
    let ne = cfg.num_experts;
    assert_eq!(l.moe_router.len(), ne * d, "router [ne, d]");
    assert_eq!(l.moe_wg.len(), ne * inter * d, "expert gate [ne, inter, d]");
    assert_eq!(l.moe_wu.len(), ne * inter * d, "expert up [ne, inter, d]");
    assert_eq!(l.moe_wd.len(), ne * d * inter, "expert down [ne, d, inter]");
    // Dense model has empty MoE tensors.
    let dense = Weights::random(&Config::tiny(), 42).unwrap();
    assert!(dense.layers[0].moe_router.is_empty());
    assert!(dense.layers[0].moe_wg.is_empty());
}

#[test]
fn moe_ref_forward_is_finite_and_deterministic() {
    let cfg = moe_cfg();
    let w = Weights::random(&cfg, 7).unwrap();
    let tokens = [5u32, 9, 3, 200];
    let mut m1 = RefModel::new(cfg, w.clone());
    let l1 = m1.forward(&tokens);
    assert!(
        l1.iter().all(|v| v.is_finite()),
        "MoE logits must be finite"
    );
    let mut m2 = RefModel::new(cfg, w);
    let l2 = m2.forward(&tokens);
    assert_eq!(l1, l2, "MoE forward must be deterministic");
}
