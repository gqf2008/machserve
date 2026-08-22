//! fp16 compute path: logits must be close to the f32 path and greedy token
//! selection must be stable for well-separated logits (random weights).
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::batched::BatchedModel;
use mach_model::config::ModelDType;
use mach_model::model::GpuModel;
use mach_model::{Config, Weights};

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

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
fn single_seq_fp16_matches_fp32() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = Config::tiny();
    cfg.dtype = ModelDType::F32;
    let w = Weights::random(&cfg, 41).unwrap();
    let mut m32 = GpuModel::new(hip.clone(), cfg, &w).unwrap();
    cfg.dtype = ModelDType::F16;
    let mut m16 = GpuModel::new(hip.clone(), cfg, &w).unwrap();

    let tokens = [5u32, 9, 3, 200];
    let l32 = m32.forward(&tokens).unwrap();
    let l16 = m16.forward(&tokens).unwrap();
    assert_eq!(l32.len(), l16.len());
    let max_abs = l32
        .iter()
        .zip(&l16)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("single-seq fp16 vs fp32: max |logit diff| = {max_abs:.6}");
    assert!(
        max_abs < 0.1,
        "fp16 vs fp32 logit diff too large: {max_abs}"
    );
    // Greedy token must agree (weights are random, logits well separated).
    assert_eq!(
        argmax(&l32),
        argmax(&l16),
        "fp16 greedy argmax must match fp32"
    );
}

#[test]
fn batched_fp16_matches_fp32() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = Config::tiny();
    cfg.dtype = ModelDType::F32;
    let w = Weights::random(&cfg, 61).unwrap();
    let batch = 4usize;
    let mut b32 = BatchedModel::new(hip.clone(), cfg, &w, batch).unwrap();
    cfg.dtype = ModelDType::F16;
    let mut b16 = BatchedModel::new(hip.clone(), cfg, &w, batch).unwrap();

    // Run several steps; compare greedy tokens and read back logits once.
    let steps: Vec<Vec<u32>> = vec![vec![5, 9, 33, 7], vec![12, 3, 1, 200], vec![8, 55, 4, 99]];
    let mut tok32 = vec![0u32; batch];
    let mut tok16 = vec![0u32; batch];
    let mut logits32 = Vec::new();
    let mut logits16 = Vec::new();
    for (i, step_tokens) in steps.iter().enumerate() {
        let t32: Vec<u32> = step_tokens.iter().zip(&tok32).map(|(a, b)| a + b).collect();
        let t16: Vec<u32> = step_tokens.iter().zip(&tok16).map(|(a, b)| a + b).collect();
        tok32 = b32.decode_step(&t32).unwrap();
        tok16 = b16.decode_step(&t16).unwrap();
        assert_eq!(tok32, tok16, "batched greedy tokens must agree at step {i}");
        if i == steps.len() - 1 {
            logits32 = b32.read_logits().unwrap();
            logits16 = b16.read_logits().unwrap();
        }
    }
    let max_abs = logits32
        .iter()
        .zip(&logits16)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("batched fp16 vs fp32: max |logit diff| = {max_abs:.6}");
    assert!(
        max_abs < 0.1,
        "batched fp16 vs fp32 logit diff too large: {max_abs}"
    );
}
