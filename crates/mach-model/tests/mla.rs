//! MLA (DeepSeek-V2 style) foundation: config + weights + CPU reference forward.
//!
//! Verifies the data plumbing (low-rank Q / compressed KV tensors) and the
//! reference forward math against a synthetic model; GPU kernels land in a
//! later slice.

use mach_model::ref_model::RefModel;
use mach_model::{Config, Weights};

fn mla_cfg() -> Config {
    Config::mla(128, 2, 4, 1024, 64, 32, 16, 16, 8, 16)
}

#[test]
fn mla_weights_have_expected_shapes() {
    let cfg = mla_cfg();
    let w = Weights::random(&cfg, 42).unwrap();
    let l = &w.layers[0];
    let d = cfg.d_model;
    let heads = cfg.n_heads;
    assert_eq!(l.mla_q_a.len(), cfg.q_lora_rank * d, "q_a [q_lora, d]");
    assert_eq!(l.mla_q_a_norm.len(), cfg.q_lora_rank, "q_a_norm");
    assert_eq!(
        l.mla_q_b.len(),
        heads * cfg.qk_nope_head_dim * cfg.q_lora_rank,
        "q_b [heads*nope, q_lora]"
    );
    assert_eq!(
        l.mla_q_rope.len(),
        heads * cfg.qk_rope_head_dim * d,
        "q_rope [heads*rope, d]"
    );
    assert_eq!(
        l.mla_kv_a.len(),
        (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * d,
        "kv_a [kv_lora+rope, d]"
    );
    assert_eq!(l.mla_kv_a_norm.len(), cfg.kv_lora_rank, "kv_a_norm");
    assert_eq!(
        l.mla_kv_b.len(),
        heads * (cfg.qk_nope_head_dim + cfg.v_head_dim) * cfg.kv_lora_rank,
        "kv_b [heads*(nope+v), kv_lora]"
    );
    assert_eq!(l.mla_o.len(), d * heads * cfg.v_head_dim, "o [d, heads*v]");
    // Standard attention tensors stay empty on the MLA path.
    assert!(l.wq.is_empty());
    assert!(l.wk.is_empty());
    assert!(l.wv.is_empty());
    assert!(l.wo.is_empty());
    // A dense config has no MLA tensors.
    let dense = Weights::random(&Config::tiny(), 42).unwrap();
    assert!(dense.layers[0].mla_q_a.is_empty());
    assert!(dense.layers[0].mla_kv_b.is_empty());
}

#[test]
fn mla_ref_forward_is_finite_and_deterministic() {
    let cfg = mla_cfg();
    let w = Weights::random(&cfg, 7).unwrap();
    let tokens = [5u32, 9, 3, 200];
    let mut m1 = RefModel::new(cfg, w.clone());
    let l1 = m1.forward(&tokens);
    assert!(
        l1.iter().all(|v| v.is_finite()),
        "MLA logits must be finite"
    );
    let mut m2 = RefModel::new(cfg, w);
    let l2 = m2.forward(&tokens);
    assert_eq!(l1, l2, "MLA forward must be deterministic");
}

#[cfg(feature = "hip")]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(feature = "hip")]
#[test]
fn mla_gpu_matches_cpu_reference() {
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

    let cfg = mla_cfg();
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
        "MLA GPU vs CPU: max diff {max} (scale {scale})"
    );
}

#[cfg(feature = "hip")]
#[test]
fn mla_batched_matches_cpu_reference() {
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

    let cfg = mla_cfg();
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
                "step {step} seq {s} MLA batched GPU vs CPU: max diff {max} (scale {scale})"
            );
        }
    }
}
