//! Qwen2-1.5B real-model smoke test (skippable).
//!
//! Loads the already-downloaded `Qwen2-1.5B` checkpoint from
//! `.models/qwen-1.5b.safetensors` (single BF16 file, tie_word_embeddings)
//! and verifies the GPU decodes a few tokens with finite, deterministic
//! logits. Exercises the BF16 loader, GQA with rope_theta=1e6, tied LM head
//! and the F16 device path on real weights.

#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::config::ModelDType;
use mach_model::loader::load_safetensors;
use mach_model::model::GpuModel;
use mach_model::{Config, Weights};
use std::path::PathBuf;

fn model_path() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("../../.models/qwen-1.5b.safetensors"),
        PathBuf::from(".models/qwen-1.5b.safetensors"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

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

/// Qwen2-1.5B hyperparameters (from .models/qwen-1.5b-config.json).
fn qwen2_1_5b_cfg() -> Config {
    Config {
        dtype: ModelDType::F16,
        vocab_size: 151936,
        d_model: 1536,
        n_layers: 28,
        n_heads: 12,
        n_kv_heads: 2,
        head_dim: 128,
        intermediate_size: 8960,
        num_experts: 0,
        num_experts_per_tok: 0,
        max_seq_len: 2048,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
        qk_norm: false,
    }
}

#[test]
fn qwen2_1_5b_decodes_finite_and_deterministic() {
    let Some(path) = model_path() else {
        eprintln!("skipping qwen2_1_5b: .models/qwen-1.5b.safetensors not present");
        return;
    };
    let Some(hip) = hip_ctx() else { return };

    let cfg = qwen2_1_5b_cfg();
    // Qwen2-1.5B ties the LM head to the embedding matrix.
    let w: Weights = load_safetensors(&path, &cfg, true).expect("load Qwen2-1.5B");

    let mut m = GpuModel::new(hip.clone(), cfg, &w).expect("build model");
    let tokens = [1u32, 100, 200, 300, 400];
    let a = m.forward(&tokens).expect("decode");
    assert!(a.iter().all(|v| v.is_finite()), "logits must be finite");
    assert_eq!(a.len(), cfg.vocab_size, "logits must cover the vocab");

    let mut fresh = GpuModel::new(hip.clone(), cfg, &w).expect("fresh");
    let b = fresh.forward(&tokens).expect("decode fresh");
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(max, 0.0, "decode must be deterministic");
    eprintln!(
        "qwen2_1_5b OK: {} logits, max|x| {:.3}",
        a.len(),
        a.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    );
}
