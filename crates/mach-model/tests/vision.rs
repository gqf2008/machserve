//! Qwen3.5/Qwen3.8 vision metadata and header-validation tests.

use mach_model::loader::{load_vision_weights, validate_vision_checkpoint};
use mach_model::vision::{
    VisionActivation, VisionConfig, VisionLayerWeights, VisionLinear, VisionWeights, vision_forward,
};
use std::path::{Path, PathBuf};

fn tmp_dir(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("machserve-vision-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Write a header-only BF16 safetensors file with correctly-sized zero payload.
fn write_bf16(path: &Path, tensors: &[(String, Vec<usize>)]) {
    let mut data_len = 0usize;
    let mut entries = Vec::with_capacity(tensors.len());
    for (name, shape) in tensors {
        let elems = shape.iter().product::<usize>();
        let bytes = elems * 2;
        let start = data_len;
        let end = start + bytes;
        data_len = end;
        let shape = shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        entries.push(format!(
            r#""{name}": {{"dtype":"BF16","shape":[{shape}],"data_offsets":[{start},{end}]}}"#
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut out = Vec::with_capacity(8 + header.len() + data_len);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.resize(8 + header.len() + data_len, 0);
    std::fs::write(path, out).unwrap();
}

fn tiny_cfg() -> VisionConfig {
    VisionConfig {
        depth: 2,
        hidden_size: 8,
        intermediate_size: 16,
        num_heads: 2,
        in_channels: 3,
        patch_size: 2,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        num_position_embeddings: 4,
        out_hidden_size: 12,
        image_token_id: 101,
        video_token_id: 102,
        vision_start_token_id: 103,
        vision_end_token_id: 104,
        mrope_section: [1, 1, 2],
        mrope_interleaved: true,
        hidden_act: VisionActivation::GeluPytorchTanh,
    }
}

#[test]
fn parses_qwen38_vision_config() {
    let json: serde_json::Value = serde_json::from_str(
        r#"{
            "vision_config": {
                "depth": 27, "hidden_size": 1152, "intermediate_size": 4304,
                "num_heads": 16, "in_channels": 3, "patch_size": 16,
                "temporal_patch_size": 2, "spatial_merge_size": 2,
                "num_position_embeddings": 2304, "out_hidden_size": 5120
            },
            "image_token_id": 248056, "video_token_id": 248057,
            "vision_start_token_id": 248053, "vision_end_token_id": 248054,
            "text_config": {
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10]
                }
            }
        }"#,
    )
    .unwrap();
    let cfg = VisionConfig::from_hf_json(&json).unwrap();
    assert_eq!(cfg.depth, 27);
    assert_eq!(cfg.hidden_size, 1152);
    assert_eq!(cfg.head_dim(), 72);
    assert_eq!(cfg.merger_input_dim(), 4608);
    assert_eq!(cfg.image_token_id, 248056);
    assert_eq!(cfg.mrope_section, [11, 11, 10]);
    assert!(cfg.mrope_interleaved);
}

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

#[test]
#[allow(clippy::excessive_precision)]
fn cpu_vision_forward_matches_hf_golden() {
    let cfg = VisionConfig {
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
    };
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
    let w = VisionWeights {
        patch_embed_weight: fill("patch_embed.proj.weight", 4 * 8),
        patch_embed_bias: fill("patch_embed.proj.bias", 4),
        pos_embed_weight: fill("pos_embed.weight", 4 * 4),
        layers: vec![layer],
        merger_norm_weight: fill("merger.norm.weight", 4),
        merger_norm_bias: fill("merger.norm.bias", 4),
        merger_fc1: linear("merger.linear_fc1", 16, 16),
        merger_fc2: linear("merger.linear_fc2", 6, 16),
    };
    let pixel = fill("pixel", 16 * 8);
    let got = vision_forward(&cfg, &w, &pixel, &[[1, 2, 4], [2, 2, 2]]).unwrap();
    let want = [
        0.11009461432695389f32,
        -0.0033003389835357666,
        -0.4586601257324219,
        0.2488490492105484,
        -0.03923000395298004,
        0.8268396854400635,
        0.10213955491781235,
        0.23839175701141357,
        -0.2461928129196167,
        0.2944979667663574,
        -0.1495908498764038,
        0.5739130973815918,
        0.144168421626091,
        0.1647706925868988,
        -0.2099294662475586,
        0.22346705198287964,
        -0.12787874042987823,
        0.6426060199737549,
        0.10436826199293137,
        0.008504003286361694,
        -0.42983460426330566,
        0.26751384139060974,
        -0.04159429669380188,
        0.8149771690368652,
    ];
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() < 5e-7, "token {i}: {a} vs {b}");
    }
}
fn multi_head_cfg() -> VisionConfig {
    VisionConfig {
        depth: 1,
        hidden_size: 8,
        intermediate_size: 16,
        num_heads: 2,
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

fn multi_head_weights() -> VisionWeights {
    let layer = VisionLayerWeights {
        norm1_weight: fill("blocks.0.norm1.weight", 8),
        norm1_bias: fill("blocks.0.norm1.bias", 8),
        qkv: linear("blocks.0.attn.qkv", 24, 8),
        attn_proj: linear("blocks.0.attn.proj", 8, 8),
        norm2_weight: fill("blocks.0.norm2.weight", 8),
        norm2_bias: fill("blocks.0.norm2.bias", 8),
        mlp_fc1: linear("blocks.0.mlp.linear_fc1", 16, 8),
        mlp_fc2: linear("blocks.0.mlp.linear_fc2", 8, 16),
    };
    VisionWeights {
        patch_embed_weight: fill("patch_embed.proj.weight", 8 * 8),
        patch_embed_bias: fill("patch_embed.proj.bias", 8),
        pos_embed_weight: fill("pos_embed.weight", 4 * 8),
        layers: vec![layer],
        merger_norm_weight: fill("merger.norm.weight", 8),
        merger_norm_bias: fill("merger.norm.bias", 8),
        merger_fc1: linear("merger.linear_fc1", 32, 32),
        merger_fc2: linear("merger.linear_fc2", 6, 32),
    }
}

#[test]
#[allow(clippy::excessive_precision)]
fn cpu_vision_forward_matches_hf_golden_multi_head() {
    let cfg = multi_head_cfg();
    let w = multi_head_weights();
    let pixel = fill("pixel", 16 * 8);
    let got = vision_forward(&cfg, &w, &pixel, &[[1, 2, 4], [2, 2, 2]]).unwrap();
    let want = [
        0.12758752703666687f32,
        -0.3536241948604584,
        0.7869868278503418,
        -0.21890529990196228,
        -0.9496842622756958,
        1.3021525144577026,
        0.09965535253286362,
        -0.3464723527431488,
        0.7794058322906494,
        0.1719481498003006,
        -0.8959850072860718,
        1.328220009803772,
        0.8674502372741699,
        0.0795118510723114,
        0.8524913191795349,
        0.9201756715774536,
        -0.4743712246417999,
        1.4105879068374634,
        0.39053773880004883,
        -0.18814316391944885,
        0.8579662442207336,
        -0.008011549711227417,
        -0.8432198762893677,
        1.3198504447937012,
    ];
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() < 5e-7, "token {i}: {a} vs {b}");
    }
}
#[test]
fn validates_complete_vision_header() {
    let cfg = tiny_cfg();
    let dir = tmp_dir("complete");
    write_bf16(&dir.join("model.safetensors"), &cfg.expected_tensors());
    let layout = validate_vision_checkpoint(&dir, &cfg).unwrap();
    assert_eq!(layout.shards, 1);
    assert_eq!(layout.tensors, cfg.expected_tensors().len());
    assert!(layout.payload_bytes > 0);
    let _ = std::fs::remove_dir_all(dir);
}

fn write_index(dir: &Path, map: &[(String, String)]) {
    let mut weight_map = serde_json::Map::new();
    for (name, file) in map {
        weight_map.insert(name.clone(), serde_json::Value::String(file.clone()));
    }
    let value = serde_json::json!({ "weight_map": weight_map });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
}

#[test]
fn rejects_missing_indexed_non_visual_shard() {
    let cfg = tiny_cfg();
    let dir = tmp_dir("missing-indexed-shard");
    let tensors = cfg.expected_tensors();
    write_bf16(&dir.join("model-00001-of-00002.safetensors"), &tensors);
    let mut map: Vec<(String, String)> = tensors
        .iter()
        .map(|(name, _)| (name.clone(), "model-00001-of-00002.safetensors".into()))
        .collect();
    map[0].1 = "model-00002-of-00002.safetensors".into();
    write_index(&dir, &map);
    let err = validate_vision_checkpoint(&dir, &cfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("missing"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn rejects_duplicate_same_file_tensor_key() {
    let cfg = tiny_cfg();
    let dir = tmp_dir("duplicate-key");
    let mut tensors = cfg.expected_tensors();
    tensors.push(tensors[0].clone());
    write_bf16(&dir.join("model.safetensors"), &tensors);
    let err = validate_vision_checkpoint(&dir, &cfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("duplicate"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}
#[test]
fn loads_complete_vision_weights() {
    let cfg = tiny_cfg();
    let dir = tmp_dir("load");
    write_bf16(&dir.join("model.safetensors"), &cfg.expected_tensors());
    let w = load_vision_weights(&dir, &cfg).unwrap();
    assert_eq!(w.layers.len(), cfg.depth);
    assert_eq!(w.patch_embed_weight.len(), cfg.hidden_size * 3 * 2 * 2 * 2);
    assert_eq!(
        w.pos_embed_weight.len(),
        cfg.num_position_embeddings * cfg.hidden_size
    );
    assert_eq!(
        w.layers[0].qkv.weight.len(),
        3 * cfg.hidden_size * cfg.hidden_size
    );
    assert_eq!(
        w.layers[0].mlp_fc1.weight.len(),
        cfg.intermediate_size * cfg.hidden_size
    );
    assert_eq!(
        w.merger_fc2.weight.len(),
        cfg.out_hidden_size * cfg.merger_input_dim()
    );
    let _ = std::fs::remove_dir_all(dir);
}
#[test]
fn rejects_missing_visual_tensor() {
    let cfg = tiny_cfg();
    let dir = tmp_dir("missing");
    let mut tensors = cfg.expected_tensors();
    tensors.retain(|(name, _)| name != "model.visual.merger.linear_fc2.bias");
    write_bf16(&dir.join("model.safetensors"), &tensors);
    let err = validate_vision_checkpoint(&dir, &cfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("missing"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn rejects_shape_mismatch() {
    let cfg = tiny_cfg();
    let dir = tmp_dir("shape");
    let mut tensors = cfg.expected_tensors();
    let (_, shape) = tensors
        .iter_mut()
        .find(|(name, _)| name.ends_with("blocks.0.attn.proj.weight"))
        .unwrap();
    *shape = vec![cfg.hidden_size];
    write_bf16(&dir.join("model.safetensors"), &tensors);
    let err = validate_vision_checkpoint(&dir, &cfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("shape"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn rejects_unexpected_visual_tensor() {
    let cfg = tiny_cfg();
    let dir = tmp_dir("extra");
    let mut tensors = cfg.expected_tensors();
    tensors.push(("model.visual.extra.weight".into(), vec![cfg.hidden_size]));
    write_bf16(&dir.join("model.safetensors"), &tensors);
    let err = validate_vision_checkpoint(&dir, &cfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("unexpected"), "{err}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn validates_real_qwen38_vision_header() {
    let Some(path) = mach_model::real_test_model_path() else {
        eprintln!("skipping real Qwen3.8 header validation: MACH_TEST_MODEL is not set");
        return;
    };
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path.join("config.json")).unwrap()).unwrap();
    let cfg = VisionConfig::from_hf_json(&config).unwrap();
    let layout = validate_vision_checkpoint(&path, &cfg).unwrap();
    println!(
        "vision checkpoint: {} shards, {} tensors, {} bytes",
        layout.shards, layout.tensors, layout.payload_bytes
    );
    assert_eq!(layout.tensors, cfg.expected_tensors().len());
}
