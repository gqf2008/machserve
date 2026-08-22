//! P1 decode-slice acceptance tests (run on the 7900 XTX when `--features hip`).
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::model::GpuModel;
use mach_model::ref_model::RefModel;
use mach_model::{Config, Weights};

/// Skips the test when no HIP device is available.
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn assert_close(got: &[f32], want: &[f32], atol: f32, rtol: f32, what: &str) {
    let max = max_abs_diff(got, want);
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let bound = atol + rtol * scale;
    assert!(
        max <= bound,
        "{what}: max abs diff {max:.6} exceeds bound {bound:.6} (scale {scale:.3})"
    );
}

#[test]
fn cpu_reference_advances_state() {
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 42).unwrap();
    let mut m = RefModel::new(cfg, w.clone());
    let l1 = m.forward(&[3, 7, 11]);
    let mut m2 = RefModel::new(cfg, w);
    let l2 = m2.forward(&[3, 7, 12]);
    assert_ne!(
        max_abs_diff(&l1, &l2),
        0.0,
        "different last tokens must produce different logits"
    );
}

#[test]
fn gpu_matches_cpu_reference() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 7).unwrap();
    let tokens: Vec<u32> = vec![5, 9, 1, 33, 8, 2, 90, 4];

    let mut gpu = GpuModel::new(hip, cfg, &w).unwrap();
    let gpu_logits = gpu.forward(&tokens).unwrap();

    let mut cpu = RefModel::new(cfg, w);
    let cpu_logits = cpu.forward(&tokens);

    assert_close(
        &gpu_logits,
        &cpu_logits,
        2e-3,
        2e-3,
        "gpu vs cpu final logits",
    );
}

#[test]
fn graph_replay_matches_eager() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 11).unwrap();
    let tokens = [17u32, 3, 255];

    // Eager path on a fresh model.
    let mut eager = GpuModel::new(hip.clone(), cfg, &w).unwrap();
    let mut eager_logits = Vec::new();
    for &t in &tokens {
        eager_logits.push(eager.decode_step(t).unwrap());
    }

    // Graph path: warmup + reset + capture, then replay the same tokens.
    let mut graph_model = GpuModel::new(hip, cfg, &w).unwrap();
    let graph = graph_model.capture_decode().unwrap();
    let mut graph_logits = Vec::new();
    for &t in &tokens {
        graph_logits.push(graph_model.decode_step_graph(&*graph, t).unwrap());
    }

    for (i, (g, e)) in graph_logits.iter().zip(&eager_logits).enumerate() {
        assert_close(g, e, 5e-3, 5e-3, &format!("graph vs eager step {i}"));
    }
}

#[test]
fn kv_cache_is_positional() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 3).unwrap();
    let mut m = GpuModel::new(hip.clone(), cfg, &w).unwrap();
    let a = m.forward(&[1, 2, 3]).unwrap();
    let mut m2 = GpuModel::new(hip.clone(), cfg, &w).unwrap();
    let b = m2.forward(&[1, 2, 4]).unwrap();
    let d = max_abs_diff(&a, &b);
    assert!(
        d > 1e-3,
        "different last tokens must diverge (max diff {d})"
    );
}
