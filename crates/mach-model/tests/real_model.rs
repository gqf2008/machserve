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
use mach_model::continuous::ContinuousModel;
use mach_model::loader::load_safetensors;
use mach_model::model::GpuModel;
use mach_model::sampling::SamplingParams;
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

#[test]
fn real_model_samples_with_seed_deterministically() {
    let candidates = [
        PathBuf::from("../../.models").join("tiny-llama.safetensors"),
        PathBuf::from(".models").join("tiny-llama.safetensors"),
    ];
    let Some(model_path) = candidates.into_iter().find(|p| p.exists()) else {
        eprintln!("skipping real_model sampling: model not present (see doc comment)");
        return;
    };
    let hip = match hip::hip() {
        Ok(h) => match hip::device_count() {
            Ok(n) if n > 0 => h,
            _ => {
                eprintln!("skipping real_model sampling: no device");
                return;
            }
        },
        Err(e) => {
            eprintln!("skipping real_model sampling: {e}");
            return;
        }
    };

    let cfg = Config::llama(16, 2, 4, 4, 32000, 2048);
    let w: Weights = load_safetensors(&model_path, &cfg, false).expect("load weights");
    let params = SamplingParams {
        temperature: 0.8,
        top_k: 0,
        top_p: 0.9,
        seed: 99,
    };
    let run = || -> Vec<u32> {
        let mut eng = ContinuousModel::new(hip.clone(), cfg, &w, 4).expect("engine");
        let id = eng.add(&[1, 2, 3], 10, None, params).expect("add");
        while !eng.is_done(id) {
            eng.step().expect("step");
        }
        eng.generated(id)
    };
    let a = run();
    assert!(!a.is_empty(), "sampling must generate tokens");
    assert!(
        a.iter().all(|&t| t < cfg.vocab_size as u32),
        "sampled tokens must be in vocabulary"
    );
    let b = run();
    assert_eq!(
        a, b,
        "same seed must reproduce the same sample on the real model"
    );
    eprintln!("real_model sampling OK: {a:?}");
}
