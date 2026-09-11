//! Env-gated GPU parity harness for the Qwen3.5/Qwen3.8 vision tower.
//!
//! Run only in a dedicated GPU window:
//! `$env:MACH_TEST_VISION_GPU='1'; cargo test -p mach-model --features hip --test vision_gpu -- --ignored --test-threads 1`
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::vision::{
    VisionActivation, VisionConfig, VisionLayerWeights, VisionLinear, VisionWeights, vision_forward,
};
use mach_model::vision_gpu::VisionGpu;

fn fill(name: &str, n: usize) -> Vec<f32> {
    let s = name.bytes().map(u32::from).sum::<u32>() * 31 + 17;
    (0..n)
        .map(|j| ((s + 13 * j as u32) % 29) as f32 / 29.0 - 0.5)
        .collect()
}

fn linear(name: &str, out_dim: usize, in_dim: usize) -> VisionLinear {
    VisionLinear {
        weight: fill(&format!("{name}.weight"), out_dim * in_dim),
        bias: fill(&format!("{name}.bias"), out_dim),
    }
}

fn tiny_cfg() -> VisionConfig {
    VisionConfig {
        depth: 1,
        hidden_size: 4,
        intermediate_size: 8,
        num_heads: 1,
        in_channels: 1,
        patch_size: 2,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        num_position_embeddings: 4,
        out_hidden_size: 6,
        image_token_id: 101,
        video_token_id: 102,
        vision_start_token_id: 103,
        vision_end_token_id: 104,
        mrope_section: [1, 1, 1],
        mrope_interleaved: true,
        hidden_act: VisionActivation::GeluPytorchTanh,
    }
}

fn tiny_weights() -> VisionWeights {
    let layer = VisionLayerWeights {
        norm1_weight: fill("blocks.0.norm1.weight", 4),
        norm1_bias: fill("blocks.0.norm1.bias", 4),
        qkv: linear("blocks.0.attn.qkv", 12, 4),
        attn_proj: linear("blocks.0.attn.proj", 4, 4),
        norm2_weight: fill("blocks.0.norm2.weight", 4),
        norm2_bias: fill("blocks.0.norm2.bias", 4),
        mlp_fc1: linear("blocks.0.mlp.linear_fc1", 8, 4),
        mlp_fc2: linear("blocks.0.mlp.linear_fc2", 4, 8),
    };
    VisionWeights {
        patch_embed_weight: fill("patch_embed.proj.weight", 4 * 8),
        patch_embed_bias: fill("patch_embed.proj.bias", 4),
        pos_embed_weight: fill("pos_embed.weight", 4 * 4),
        layers: vec![layer],
        merger_norm_weight: fill("merger.norm.weight", 4),
        merger_norm_bias: fill("merger.norm.bias", 4),
        merger_fc1: linear("merger.linear_fc1", 16, 16),
        merger_fc2: linear("merger.linear_fc2", 6, 16),
    }
}

#[test]
#[ignore = "GPU vision parity; set MACH_TEST_VISION_GPU=1 and run explicitly"]
fn gpu_vision_forward_matches_cpu() {
    if std::env::var("MACH_TEST_VISION_GPU").as_deref() != Ok("1") {
        return;
    }
    let hip = hip::hip().expect("HIP runtime");
    let cfg = tiny_cfg();
    let w = tiny_weights();
    let pixel = fill("pixel", 16 * 8);
    let grids = [[1, 2, 4], [2, 2, 2]];
    let prep = VisionGpu::prepare(&cfg, &w, &grids).unwrap();
    let cpu = vision_forward(&cfg, &w, &pixel, &grids).unwrap();
    let mut gpu = VisionGpu::new(hip, cfg, &w, prep.tokens).unwrap();
    let got = gpu.forward(&pixel, &prep).unwrap();
    assert_eq!(cpu.len(), got.len());
    let mut max_diff = 0.0f32;
    for (a, b) in cpu.iter().zip(&got) {
        max_diff = max_diff.max((a - b).abs());
    }
    eprintln!("vision GPU max_diff={max_diff}");
    assert!(max_diff < 2e-4, "vision GPU mismatch: {max_diff}");
}
