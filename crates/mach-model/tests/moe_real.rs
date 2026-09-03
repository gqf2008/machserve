//! Qwen3-MoE real-model smoke test (skippable).
//!
//! Loads `PrimeIntellect/qwen3-moe-tiny` (BF16, single ~1.34GB file; MoE: 16
//! experts / 4 active per token; layer 0 is dense via `mlp_only_layers`) from
//! `.models/` and verifies the GPU decodes a few tokens with finite,
//! deterministic logits. Exercises the loader's mixed dense/MoE layers +
//! `moe_intermediate_size` + shared QK-norm tiling, and the GpuModel MoE
//! forward on real weights.
//!
//! Download (use hf-mirror.com if huggingface.co is unreachable):
//!   curl -L -o .models/model.safetensors \
//!     https://huggingface.co/PrimeIntellect/qwen3-moe-tiny/resolve/main/model.safetensors
//!
//! Note: Qwen2.5-MoE-A3B (64 experts / 8 active, shared-expert) is not yet
//! supported - the loader does not read shared_expert tensors, and the
//! checkpoint is too large for consumer GPUs in fp32. qwen3-moe-tiny is the
//! smallest real Qwen-MoE checkpoint exercising the same loader/forward paths.

#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::config::ModelDType;
use mach_model::loader::load_safetensors;
use mach_model::model::GpuModel;
use mach_model::{Config, Weights};
use std::path::PathBuf;

fn model_path() -> Option<PathBuf> {
    [
        PathBuf::from("../../.models/model.safetensors"),
        PathBuf::from(".models/model.safetensors"),
    ]
    .into_iter()
    .find(|p| p.exists())
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

/// qwen3-moe-tiny hyperparameters (from config.json; max_seq_len is a test
/// choice, the checkpoint supports 32768).
fn qwen3_moe_tiny_cfg() -> Config {
    Config {
        dtype: ModelDType::F32,
        vocab_size: 151936,
        d_model: 1024,
        n_layers: 24,
        n_heads: 16,
        n_kv_heads: 4,
        head_dim: 64,
        // Dense layer 0 (`mlp_only_layers`) uses intermediate_size; the 15
        // routed-expert layers use moe_intermediate_size.
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
        n_shared_experts: 0,
        moe_norm_topk: true,
        moe_routed_scale: 1.0,
        rope_yarn_factor: 0.0,
        rope_yarn_orig_len: 0,
        rope_yarn_beta_fast: 0.0,
        rope_yarn_beta_slow: 0.0,
        rope_yarn_mscale: 1.0,
        rope_yarn_mscale_all_dim: 0.0,
        rope_interleave: false,
        moe_grouped: true,
        step_profile: false,
    }
}

#[test]
fn qwen3_moe_tiny_decodes_finite_and_deterministic() {
    let Some(path) = model_path() else {
        eprintln!(
            "skipping qwen3_moe_tiny: .models/model.safetensors not present (see doc comment)"
        );
        return;
    };
    let Some(hip) = hip_ctx() else { return };

    let cfg = qwen3_moe_tiny_cfg();
    let w: Weights = load_safetensors(&path, &cfg, true).expect("load qwen3-moe-tiny");

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
        "qwen3_moe_tiny OK: {} logits, max|x| {:.3}",
        a.len(),
        a.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    );
}
