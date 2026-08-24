//! Qwen2.5-MoE-A3B real-model smoke test (skippable).
//!
//! Loads `Qwen/Qwen2.5-MoE-A3B` (BF16, ~3.4GB, MoE: 64 experts / 8 active per
//! token) from `.models/qwen2.5-moe-a3b/` and verifies the GPU decodes a few
//! tokens with finite, deterministic logits. Exercises the loader's MoE branch
//! (router + per-expert gate/up/down) and the GpuModel MoE forward on real
//! weights.
//!
//! Download:
//!   curl -L -o .models/qwen2.5-moe-a3b/model-00001-of-00002.safetensors \
//!     https://huggingface.co/Qwen/Qwen2.5-MoE-A3B/resolve/main/model-00001-of-00002.safetensors
//!   curl -L -o .models/qwen2.5-moe-a3b/model-00002-of-00002.safetensors \
//!     https://huggingface.co/Qwen/Qwen2.5-MoE-A3B/resolve/main/model-00002-of-00002.safetensors
//!   curl -L -o .models/qwen2.5-moe-a3b/config.json \
//!     https://huggingface.co/Qwen/Qwen2.5-MoE-A3B/resolve/main/config.json

#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::config::ModelDType;
use mach_model::loader::load_safetensors_dir;
use mach_model::model::GpuModel;
use mach_model::{Config, Weights};
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("../../.models/qwen2.5-moe-a3b"),
        PathBuf::from(".models/qwen2.5-moe-a3b"),
    ];
    // Require config.json AND at least one weight shard, so a half-downloaded
    // checkpoint skips instead of hard-failing on load.
    candidates.into_iter().find(|p| {
        p.join("config.json").exists()
            && std::fs::read_dir(p)
                .map(|mut it| {
                    it.any(|e| {
                        e.as_ref().is_ok_and(|e| {
                            e.file_name().to_string_lossy().ends_with(".safetensors")
                        })
                    })
                })
                .unwrap_or(false)
    })
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

/// Qwen2.5-MoE-A3B hyperparameters (from config.json; max_seq_len is a test
/// choice, the checkpoint supports 32768).
fn moe_a3b_cfg() -> Config {
    Config {
        dtype: ModelDType::F32,
        vocab_size: 151936,
        d_model: 2048,
        n_layers: 24,
        n_heads: 16,
        n_kv_heads: 8,
        head_dim: 128,
        intermediate_size: 1024,
        num_experts: 64,
        num_experts_per_tok: 8,
        max_seq_len: 2048,
        rms_eps: 1e-6,
        rope_theta: 1_000_000.0,
        qk_norm: false,
        q_lora_rank: 0,
        kv_lora_rank: 0,
        qk_nope_head_dim: 0,
        qk_rope_head_dim: 0,
        v_head_dim: 0,
    }
}

#[test]
fn qwen25_moe_a3b_decodes_finite_and_deterministic() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping qwen25_moe_a3b: .models/qwen2.5-moe-a3b not present (see doc comment)");
        return;
    };
    let Some(hip) = hip_ctx() else { return };

    let cfg = moe_a3b_cfg();
    // Qwen2.5-MoE-A3B ties the LM head to the embedding matrix.
    let w: Weights = load_safetensors_dir(&dir, &cfg, true).expect("load Qwen2.5-MoE-A3B");

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
        "qwen25_moe_a3b OK: {} logits, max|x| {:.3}",
        a.len(),
        a.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    );
}
