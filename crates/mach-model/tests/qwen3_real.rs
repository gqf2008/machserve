//! Qwen3-8B real-model smoke test (skippable).
//!
//! Loads `Qwen/Qwen3-8B` (5 BF16 shards) from `.models/qwen3-8b/` and
//! verifies the GPU decodes a few tokens with finite, deterministic logits.
//!
//! Download (E: drive has room; ~16GB):
//!   curl -L -o .models/qwen3-8b/model-00001-of-00005.safetensors \
//!     https://huggingface.co/Qwen/Qwen3-8B/resolve/main/model-00001-of-00005.safetensors
//!   ... (shards 2..5) + config.json

#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::config::ModelDType;
use mach_model::loader::load_safetensors_dir;
use mach_model::model::GpuModel;
use mach_model::{Config, Weights};
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from("../../.models/qwen3-8b"),
        PathBuf::from(".models/qwen3-8b"),
    ];
    candidates
        .into_iter()
        .find(|p| p.join("config.json").exists())
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

#[test]
fn qwen3_8b_decodes_finite_and_deterministic() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping qwen3_8b: .models/qwen3-8b not present (see doc comment)");
        return;
    };
    let Some(hip) = hip_ctx() else { return };

    // Qwen3-8B: hidden 4096, 36 layers, 32 q / 8 kv heads, vocab 151936.
    // F16 path keeps only fp16 weights on device (~16GB, fits 24GB VRAM).
    let mut cfg = Config::qwen3(4096, 36, 32, 8, 151936, 2048);
    cfg.dtype = ModelDType::F16;
    let w: Weights = load_safetensors_dir(&dir, &cfg, false).expect("load Qwen3-8B shards");

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
        "qwen3_8b OK: {} logits, max|x| {:.3}",
        a.len(),
        a.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    );
}
