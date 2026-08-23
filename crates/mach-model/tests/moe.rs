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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(feature = "hip")]
#[test]
fn moe_gpu_forward_matches_cpu_reference() {
    use mach_kernel_sys::hip;
    use mach_model::model::GpuModel;
    use mach_model::ref_model::RefModel;

    let hip = match hip::hip() {
        Ok(h) => match hip::device_count() {
            Ok(n) if n > 0 => h,
            _ => {
                eprintln!("skipping HIP test: no device");
                return;
            }
        },
        Err(e) => {
            eprintln!("skipping HIP test: {e}");
            return;
        }
    };

    let cfg = moe_cfg();
    let w = Weights::random(&cfg, 7).unwrap();
    let tokens = [5u32, 9, 3, 200];

    let mut gpu = GpuModel::new(hip, cfg, &w).unwrap();
    let gpu_logits = gpu.forward(&tokens).unwrap();
    let mut cpu = RefModel::new(cfg, w);
    let cpu_logits = cpu.forward(&tokens);

    let max = max_abs_diff(&gpu_logits, &cpu_logits);
    let scale = cpu_logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        max <= 2e-3 + 2e-3 * scale,
        "MoE GPU vs CPU: max diff {max} (scale {scale})"
    );
}
