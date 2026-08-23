//! Qwen3 dense-family support: QK-norm (per-head RMSNorm on q/k) plus the
//! Qwen3 config shape (3x hidden MLP, theta=1e6, head_dim = hidden / heads).
//!
//! Verifies the CPU reference forward with QK-norm enabled, then compares the
//! GPU paths (single-seq and batched) against it.

use mach_model::ref_model::RefModel;
use mach_model::{Config, Weights};

fn qwen3_cfg() -> Config {
    Config::qwen3(256, 2, 4, 2, 2048, 128)
}

#[cfg(feature = "hip")]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn qwen3_config_shape_matches_qwen3_8b_proportions() {
    let cfg = Config::qwen3(4096, 36, 32, 8, 151936, 4096);
    assert!(cfg.qk_norm, "Qwen3 enables QK-norm");
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.intermediate_size, 3 * 4096, "Qwen3 uses 3x hidden MLP");
    assert_eq!(cfg.rope_theta, 1_000_000.0);
}

#[test]
fn qk_norm_weights_have_expected_shapes() {
    let cfg = qwen3_cfg();
    let w = Weights::random(&cfg, 42).unwrap();
    let l = &w.layers[0];
    assert_eq!(l.q_norm.len(), cfg.n_heads * cfg.head_dim);
    assert_eq!(l.k_norm.len(), cfg.n_kv_heads * cfg.head_dim);
    // Dense llama-style config has no QK-norm tensors.
    let dense = Weights::random(&Config::tiny(), 42).unwrap();
    assert!(dense.layers[0].q_norm.is_empty());
    assert!(dense.layers[0].k_norm.is_empty());
}

#[test]
fn qwen3_ref_forward_is_finite_and_deterministic() {
    let cfg = qwen3_cfg();
    let w = Weights::random(&cfg, 7).unwrap();
    let tokens = [5u32, 9, 3, 200];
    let mut m1 = RefModel::new(cfg, w.clone());
    let l1 = m1.forward(&tokens);
    assert!(l1.iter().all(|v| v.is_finite()), "logits must be finite");
    let mut m2 = RefModel::new(cfg, w);
    let l2 = m2.forward(&tokens);
    assert_eq!(l1, l2, "forward must be deterministic");
}

#[cfg(feature = "hip")]
#[test]
fn qwen3_gpu_matches_cpu_reference() {
    use mach_kernel_sys::hip;
    use mach_model::model::GpuModel;

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

    let cfg = qwen3_cfg();
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
        "Qwen3 GPU vs CPU: max diff {max} (scale {scale})"
    );
}

#[cfg(feature = "hip")]
#[test]
fn qwen3_batched_matches_cpu_reference() {
    use mach_kernel_sys::hip;
    use mach_model::batched::BatchedModel;

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

    let cfg = qwen3_cfg();
    let w = Weights::random(&cfg, 11).unwrap();
    let batch = 2usize;
    // Same-length token streams so every decode step advances all sequences.
    let seqs: Vec<Vec<u32>> = vec![vec![5, 9, 3], vec![200, 7, 11]];

    let mut batched = BatchedModel::new(hip, cfg, &w, batch).unwrap();
    for step in 0..seqs[0].len() {
        let tokens: Vec<u32> = seqs.iter().map(|s| s[step]).collect();
        batched.decode_step(&tokens).unwrap();
        let got = batched.read_logits().unwrap();
        for s in 0..batch {
            let mut cpu = RefModel::new(cfg, w.clone());
            let cpu_logits = cpu.forward(&seqs[s][..=step]);
            let row = &got[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let max = max_abs_diff(row, &cpu_logits);
            let scale = cpu_logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                max <= 2e-3 + 2e-3 * scale,
                "step {step} seq {s} Qwen3 batched GPU vs CPU: max diff {max} (scale {scale})"
            );
        }
    }
}
