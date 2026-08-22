//! Real-model smoke test (skippable).
//!
//! Loads `hf-internal-testing/tiny-random-LlamaForCausalLM` from
//! `.models/tiny-llama.safetensors` (download with the commands below) and
//! verifies the GPU decodes a few tokens with finite, deterministic logits.
//! Numeric cross-validation against an independent fp64 reference lives in
//! `tools/ref_llama.py` (Rust GPU vs ref: rel err ~2e-7).
//!
//! Download:
//!   curl -L -o .models/tiny-llama.safetensors \
//!     https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM/resolve/main/model.safetensors

#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::loader::load_safetensors;
use mach_model::model::GpuModel;
use mach_model::{Config, Weights};
use std::path::PathBuf;

#[test]
fn real_model_decodes_finite_and_deterministic() {
    // Cargo test runs from the crate dir; the model lives at the repo root.
    let candidates = [
        PathBuf::from("../../.models").join("tiny-llama.safetensors"),
        PathBuf::from(".models").join("tiny-llama.safetensors"),
    ];
    let Some(model_path) = candidates.into_iter().find(|p| p.exists()) else {
        eprintln!("skipping real_model: model not present (see doc comment)");
        return;
    };
    let hip = match hip::hip() {
        Ok(h) => match hip::device_count() {
            Ok(n) if n > 0 => h,
            _ => {
                eprintln!("skipping real_model: no device");
                return;
            }
        },
        Err(e) => {
            eprintln!("skipping real_model: {e}");
            return;
        }
    };

    let cfg = Config::llama(16, 2, 4, 4, 32000, 2048);
    let w: Weights = load_safetensors(&model_path, &cfg, false).expect("load weights");
    let mut model = GpuModel::new(hip.clone(), cfg, &w).expect("build model");

    let tokens = [1u32, 2, 3, 4, 5];
    let a = model.forward(&tokens).expect("decode");
    assert!(a.iter().all(|v| v.is_finite()), "logits must be finite");
    // Deterministic: a fresh model decoding the same tokens must be identical.
    // state advanced after the first forward, so compare against a fresh model.
    let mut fresh = GpuModel::new(hip.clone(), cfg, &w).expect("fresh");
    let b2 = fresh.forward(&tokens).expect("decode fresh");
    let max = a
        .iter()
        .zip(&b2)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(max, 0.0, "decode must be deterministic");
    eprintln!(
        "real_model OK: {} logits, max|x| {:.3}",
        a.len(),
        a.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    );
}
