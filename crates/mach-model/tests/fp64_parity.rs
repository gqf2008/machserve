//! Three-way fp64 parity (issue #22): GPU(f32 logits) == CPU(f32) == fp64 ref.
//!
//! The fp64 reference (`mach_model::fp64_ref`) computes the same forward in
//! f64 with the same f32 weights (exact f32 → f64 conversion), so the
//! difference against the f32 CPU/GPU paths is exactly the f32 rounding error
//! of those paths. These opt-in GPU tests verify, per MoE offload mode
//! (full / slots / adaptive, and the batched cpu-backend), that all three
//! agree within an accumulated f32 rounding bound and that the argmax is
//! identical across all three.
//!
//! Run serially (AMD Windows ROCm GPU setup deadlocks on concurrent init):
//!   cargo test -p mach-model --features hip --test fp64_parity -- --ignored --test-threads=1

#[cfg(feature = "hip")]
use mach_model::fp64_ref::Fp64RefModel;
#[cfg(feature = "hip")]
use mach_model::ref_model::RefModel;
#[cfg(feature = "hip")]
use mach_model::{Config, Weights};

/// Synthetic MoE config shared by the GPU parity tests (mirrors `tests/moe.rs`).
#[cfg(feature = "hip")]
fn moe_cfg() -> Config {
    let mut cfg = Config::tiny();
    cfg.intermediate_size = 64;
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg
}

#[cfg(feature = "hip")]
fn f32_max_abs(a: &[f32]) -> f64 {
    a.iter().fold(0.0f64, |m, &v| m.max(f64::from(v.abs())))
}

#[cfg(feature = "hip")]
fn f32_max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0f64, f64::max)
}

#[cfg(feature = "hip")]
fn f32_f64_max_abs_diff(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - y).abs())
        .fold(0.0f64, f64::max)
}

#[cfg(feature = "hip")]
fn argmax_f32(a: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in a.iter().enumerate() {
        if v > a[best] {
            best = i;
        }
    }
    best
}

#[cfg(feature = "hip")]
fn argmax_f64(a: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in a.iter().enumerate() {
        if v > a[best] {
            best = i;
        }
    }
    best
}

/// Accumulated f32 rounding bound for a full forward (CPU f32 vs fp64): L
/// layers x ~7 chained dot products (attention q/k/v/o + MLP g/u/d), each with
/// typical relative error sqrt(n)*eps32; 8x margin over the typical estimate.
#[cfg(feature = "hip")]
fn cpu_fp64_tol(cfg: &Config, scale: f64) -> f64 {
    let eps = f64::from(f32::EPSILON);
    let n = cfg.d_model.max(cfg.expert_size()) as f64;
    8.0 * 7.0 * cfg.n_layers as f64 * n.sqrt() * eps * scale + eps
}

/// GPU-vs-anything bound: matches the repo's existing GPU-vs-CPU parity margin
/// (`tests/moe.rs`: `2e-3 + 2e-3 * scale`) — GPU GEMMs use tiled orderings and
/// `__expf`, so allow a 2e-3 relative + absolute margin over fp64.
#[cfg(feature = "hip")]
fn gpu_tol(scale: f64) -> f64 {
    2e-3 + 2e-3 * scale
}

#[cfg(feature = "hip")]
fn hip_ctx() -> Option<std::sync::Arc<mach_kernel_sys::hip::Hip>> {
    use mach_kernel_sys::hip;
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

/// Asserts the three-way parity of one logits triple and prints the measured
/// errors (recorded in `docs/benchmark-fp64-parity.md`).
#[cfg(feature = "hip")]
fn check_three_way(label: &str, cfg: &Config, gpu: &[f32], cpu: &[f32], fp64: &[f64]) {
    let scale = f32_max_abs(gpu).max(f32_max_abs(cpu));
    let gpu_cpu = f32_max_abs_diff(gpu, cpu);
    let gpu_fp64 = f32_f64_max_abs_diff(gpu, fp64);
    let cpu_fp64 = f32_f64_max_abs_diff(cpu, fp64);

    assert!(
        gpu_cpu <= gpu_tol(scale),
        "{label}: GPU vs CPU f32 max diff {gpu_cpu:.3e} > tol {:.3e} (scale {scale:.3e})",
        gpu_tol(scale)
    );
    assert!(
        gpu_fp64 <= gpu_tol(scale),
        "{label}: GPU f32 vs fp64 max diff {gpu_fp64:.3e} > tol {:.3e} (scale {scale:.3e})",
        gpu_tol(scale)
    );
    assert!(
        cpu_fp64 <= cpu_fp64_tol(cfg, scale),
        "{label}: CPU f32 vs fp64 max diff {cpu_fp64:.3e} > tol {:.3e} (scale {scale:.3e})",
        cpu_fp64_tol(cfg, scale)
    );

    let (am_gpu, am_cpu, am_fp64) = (argmax_f32(gpu), argmax_f32(cpu), argmax_f64(fp64));
    assert_eq!(
        (am_gpu, am_cpu),
        (am_fp64, am_fp64),
        "{label}: argmax mismatch GPU={am_gpu} CPU={am_cpu} fp64={am_fp64}"
    );

    let rel = |d: f64| if scale > 0.0 { d / scale } else { 0.0 };
    eprintln!(
        "{label}: scale={scale:.4e} gpu_vs_cpu={gpu_cpu:.3e} (rel {:.2e}) \
         gpu_vs_fp64={gpu_fp64:.3e} (rel {:.2e}) cpu_vs_fp64={cpu_fp64:.3e} (rel {:.2e}) \
         argmax=({am_gpu},{am_cpu},{am_fp64})",
        rel(gpu_cpu),
        rel(gpu_fp64),
        rel(cpu_fp64)
    );
}

#[cfg(feature = "hip")]
#[ignore]
#[test]
fn gpu_full_slots_adaptive_three_way_fp64_parity() {
    use mach_model::model::GpuModel;
    use std::sync::Arc;

    let Some(hip) = hip_ctx() else { return };
    let cfg = moe_cfg();
    let w = Weights::random(&cfg, 7).unwrap();
    let tokens = [5u32, 9, 3, 200];

    let mut cpu = RefModel::new(cfg, w.clone());
    let cpu_logits = cpu.forward(&tokens);
    let mut fp64 = Fp64RefModel::new(cfg, &w);
    let fp64_logits = fp64.forward(&tokens);

    // full (all experts GPU-resident).
    let mut full = GpuModel::new(Arc::clone(&hip), cfg, &w).unwrap();
    let full_logits = full.forward(&tokens).unwrap();
    check_three_way("gpu full", &cfg, &full_logits, &cpu_logits, &fp64_logits);

    // slots=1 < ne=4: bounded GPU slots + CPU overflow.
    let mut slots = GpuModel::with_expert_slots(Arc::clone(&hip), cfg, &w, 1).unwrap();
    let slots_logits = slots.forward(&tokens).unwrap();
    check_three_way(
        "gpu slots=1",
        &cfg,
        &slots_logits,
        &cpu_logits,
        &fp64_logits,
    );

    // adaptive q*: per-miss GPU-vs-CPU choice from measured bandwidth.
    let mut adaptive = GpuModel::with_adaptive(Arc::clone(&hip), cfg, &w, 2).unwrap();
    let adaptive_logits = adaptive.forward(&tokens).unwrap();
    check_three_way(
        "gpu adaptive",
        &cfg,
        &adaptive_logits,
        &cpu_logits,
        &fp64_logits,
    );
}

#[cfg(feature = "hip")]
#[ignore]
#[test]
fn batched_cpu_backend_three_way_fp64_parity() {
    use mach_model::batched::BatchedModel;
    use std::sync::Arc;

    let Some(hip) = hip_ctx() else { return };
    let cfg = moe_cfg();
    let w = Weights::random(&cfg, 7).unwrap();
    let batch = 2usize;
    let tokens = [5u32, 9];

    // Full-resident GPU batch vs cpu-backend batch (experts in host RAM).
    let mut full = BatchedModel::new(Arc::clone(&hip), cfg, &w, batch).unwrap();
    let mut off =
        BatchedModel::with_expert_slots(Arc::clone(&hip), cfg, &w, batch, batch, 1).unwrap();
    let _ = full.decode_step(&tokens).unwrap();
    let _ = off.decode_step(&tokens).unwrap();
    let full_logits = full.read_logits().unwrap();
    let off_logits = off.read_logits().unwrap();

    // Placement invariance (batch): full vs cpu-backend within f32 rounding.
    let scale = f32_max_abs(&full_logits);
    let full_off = f32_max_abs_diff(&full_logits, &off_logits);
    eprintln!(
        "batched full vs cpu-backend placement: scale={scale:.4e} max_diff={full_off:.3e} (rel {:.2e})",
        if scale > 0.0 { full_off / scale } else { 0.0 }
    );
    assert!(
        full_off <= gpu_tol(scale),
        "batched full vs cpu-backend max diff {full_off:.3e} > tol {:.3e} (scale {scale:.3e})",
        gpu_tol(scale)
    );

    // Each batch row is a single-token decode at position 0, so the CPU f32
    // and fp64 references are `RefModel::forward(&[t])` / `Fp64RefModel` for
    // each token.
    for i in 0..batch {
        let t = tokens[i];
        let row_gpu = &full_logits[i * cfg.vocab_size..(i + 1) * cfg.vocab_size];
        let row_off = &off_logits[i * cfg.vocab_size..(i + 1) * cfg.vocab_size];
        let mut cpu_row = RefModel::new(cfg, w.clone());
        let cpu_logits = cpu_row.forward(&[t]);
        let mut fp64_row = Fp64RefModel::new(cfg, &w);
        let fp64_logits = fp64_row.forward(&[t]);

        check_three_way(
            &format!("batched row {i} (token {t}) full"),
            &cfg,
            row_gpu,
            &cpu_logits,
            &fp64_logits,
        );
        check_three_way(
            &format!("batched row {i} (token {t}) cpu-backend"),
            &cfg,
            row_off,
            &cpu_logits,
            &fp64_logits,
        );
    }
}

/// Real qwen3-moe-tiny three-way parity on the 7900 XTX: full + slots + adaptive
/// vs the f32 CPU reference and the fp64 reference. Skipped when the checkpoint
/// is not present. Opt-in (`#[ignore]`); run with `--test-threads=1`.
#[cfg(feature = "hip")]
#[ignore]
#[test]
fn moe_real_three_way_fp64_parity() {
    use mach_model::config::ModelDType;
    use mach_model::loader::load_safetensors;
    use mach_model::model::GpuModel;
    use std::path::PathBuf;
    use std::sync::Arc;

    let path = [
        PathBuf::from("../../.models/model.safetensors"),
        PathBuf::from(".models/model.safetensors"),
    ]
    .into_iter()
    .find(|p| p.exists());
    let Some(path) = path else {
        eprintln!("skipping qwen3-moe-tiny fp64 parity: .models/model.safetensors not present");
        return;
    };
    let Some(hip) = hip_ctx() else { return };

    let cfg = Config {
        dtype: ModelDType::F32,
        vocab_size: 151936,
        d_model: 1024,
        n_layers: 24,
        n_heads: 16,
        n_kv_heads: 4,
        head_dim: 64,
        intermediate_size: 2048,
        moe_intermediate_size: 256,
        num_experts: 16,
        num_experts_per_tok: 4,
        max_seq_len: 2048,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
        qk_norm: true,
        q_lora_rank: 0,
        kv_lora_rank: 0,
        qk_nope_head_dim: 0,
        qk_rope_head_dim: 0,
        v_head_dim: 0,
    };
    let w: Weights = load_safetensors(&path, &cfg, true).expect("load qwen3-moe-tiny");
    let tokens = [1u32, 100, 200, 300];

    let mut cpu = RefModel::new(cfg, w.clone());
    let cpu_logits = cpu.forward(&tokens);
    let mut fp64 = Fp64RefModel::new(cfg, &w);
    let fp64_logits = fp64.forward(&tokens);

    let mut full = GpuModel::new(Arc::clone(&hip), cfg, &w).unwrap();
    let full_logits = full.forward(&tokens).unwrap();
    check_three_way(
        "qwen3-moe-tiny full",
        &cfg,
        &full_logits,
        &cpu_logits,
        &fp64_logits,
    );

    let mut slots = GpuModel::with_expert_slots(Arc::clone(&hip), cfg, &w, 4).unwrap();
    let slots_logits = slots.forward(&tokens).unwrap();
    check_three_way(
        "qwen3-moe-tiny slots=4",
        &cfg,
        &slots_logits,
        &cpu_logits,
        &fp64_logits,
    );

    let mut adaptive = GpuModel::with_adaptive(Arc::clone(&hip), cfg, &w, 4).unwrap();
    let adaptive_logits = adaptive.forward(&tokens).unwrap();
    check_three_way(
        "qwen3-moe-tiny adaptive",
        &cfg,
        &adaptive_logits,
        &cpu_logits,
        &fp64_logits,
    );
}
