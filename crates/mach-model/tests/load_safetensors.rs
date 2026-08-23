//! Safetensors loading regression tests.
//!
//! Writes a synthetic safetensors checkpoint (Llama-style tensor names, F32),
//! loads it back through `mach_model::loader`, and verifies both the parsed
//! weights and (with the `hip` feature) that a GPU model built from the loaded
//! weights decodes identically to the CPU reference.

use mach_model::loader::{load_safetensors, load_safetensors_dir};
use mach_model::{Config, Weights};
use std::path::PathBuf;

fn tmp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("machserve-{name}.safetensors"))
}

/// Serializes `tensors` (name, f32 data, shape) into the safetensors format.
fn write_safetensors(path: &PathBuf, tensors: &[(&str, &[f32], &[usize])]) {
    let mut offsets: Vec<(usize, usize)> = Vec::new();
    let mut data: Vec<u8> = Vec::new();
    let mut entries: Vec<String> = Vec::new();
    for (name, values, shape) in tensors {
        let start = data.len();
        for v in *values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len();
        offsets.push((start, end));
        let shape_json = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        entries.push(format!(
            r#""{name}": {{"dtype": "F32", "shape": [{shape_json}], "data_offsets": [{start}, {end}]}}"#
        ));
    }
    let header = format!("{{{}}}", entries.join(", "));
    let header_bytes = header.as_bytes();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&data);
    std::fs::write(path, out).expect("write safetensors");
}

fn tensor_names(cfg: &Config) -> Vec<(String, Vec<f32>, Vec<usize>)> {
    let w = Weights::random(cfg, 99).unwrap();
    let d = cfg.d_model;
    let nq = cfg.n_heads * cfg.head_dim;
    let nkv = cfg.n_kv_heads * cfg.head_dim;
    let mut t: Vec<(String, Vec<f32>, Vec<usize>)> = Vec::new();
    t.push((
        "model.embed_tokens.weight".into(),
        w.tok_emb.clone(),
        vec![cfg.vocab_size, d],
    ));
    t.push(("model.norm.weight".into(), w.rms_final.clone(), vec![d]));
    t.push((
        "lm_head.weight".into(),
        w.lm_head.clone(),
        vec![cfg.vocab_size, d],
    ));
    for (i, lw) in w.layers.iter().enumerate() {
        let p = |s: &str| format!("model.layers.{i}.{s}");
        t.push((p("self_attn.q_proj.weight"), lw.wq.clone(), vec![nq, d]));
        t.push((p("self_attn.k_proj.weight"), lw.wk.clone(), vec![nkv, d]));
        t.push((p("self_attn.v_proj.weight"), lw.wv.clone(), vec![nkv, d]));
        t.push((p("self_attn.o_proj.weight"), lw.wo.clone(), vec![d, nq]));
        t.push((p("input_layernorm.weight"), lw.rms_attn.clone(), vec![d]));
        t.push((
            p("mlp.gate_proj.weight"),
            lw.wg.clone(),
            vec![cfg.intermediate_size, d],
        ));
        t.push((
            p("mlp.up_proj.weight"),
            lw.wu.clone(),
            vec![cfg.intermediate_size, d],
        ));
        t.push((
            p("mlp.down_proj.weight"),
            lw.wd.clone(),
            vec![d, cfg.intermediate_size],
        ));
        t.push((
            p("post_attention_layernorm.weight"),
            lw.rms_mlp.clone(),
            vec![d],
        ));
        if cfg.num_experts > 0 {
            let ne = cfg.num_experts;
            let inter = cfg.intermediate_size;
            t.push((p("mlp.gate.weight"), lw.moe_router.clone(), vec![ne, d]));
            for e in 0..ne {
                let ep = |s: &str| format!("model.layers.{i}.mlp.experts.{e}.{s}");
                let wg = &lw.moe_wg[e * inter * d..(e + 1) * inter * d];
                let wu = &lw.moe_wu[e * inter * d..(e + 1) * inter * d];
                let wd = &lw.moe_wd[e * d * inter..(e + 1) * d * inter];
                t.push((ep("gate_proj.weight"), wg.to_vec(), vec![inter, d]));
                t.push((ep("up_proj.weight"), wu.to_vec(), vec![inter, d]));
                t.push((ep("down_proj.weight"), wd.to_vec(), vec![d, inter]));
            }
        }
    }
    t
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn roundtrip_load_matches_original_weights() {
    let cfg = Config::tiny();
    let path = tmp_path("roundtrip");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let loaded = load_safetensors(&path, &cfg, false).unwrap();
    let original = Weights::random(&cfg, 99).unwrap();

    assert_eq!(max_abs_diff(&loaded.tok_emb, &original.tok_emb), 0.0);
    assert_eq!(max_abs_diff(&loaded.rms_final, &original.rms_final), 0.0);
    assert_eq!(max_abs_diff(&loaded.lm_head, &original.lm_head), 0.0);
    for (li, (a, b)) in loaded.layers.iter().zip(&original.layers).enumerate() {
        assert_eq!(max_abs_diff(&a.wq, &b.wq), 0.0, "layer {li} wq");
        assert_eq!(max_abs_diff(&a.wk, &b.wk), 0.0, "layer {li} wk");
        assert_eq!(max_abs_diff(&a.wv, &b.wv), 0.0, "layer {li} wv");
        assert_eq!(max_abs_diff(&a.wo, &b.wo), 0.0, "layer {li} wo");
        assert_eq!(max_abs_diff(&a.wg, &b.wg), 0.0, "layer {li} wg");
        assert_eq!(max_abs_diff(&a.wu, &b.wu), 0.0, "layer {li} wu");
        assert_eq!(max_abs_diff(&a.wd, &b.wd), 0.0, "layer {li} wd");
        assert_eq!(max_abs_diff(&a.rms_attn, &b.rms_attn), 0.0);
        assert_eq!(max_abs_diff(&a.rms_mlp, &b.rms_mlp), 0.0);
    }
    let _ = std::fs::remove_file(&path);
}

/// A MoE config: dense `tiny()` plus 4 experts, 2 active per token.
fn moe_cfg() -> Config {
    let mut cfg = Config::tiny();
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg
}

#[test]
fn roundtrip_moe_load_matches_original_weights() {
    let cfg = moe_cfg();
    let path = tmp_path("roundtrip_moe");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let loaded = load_safetensors(&path, &cfg, false).unwrap();
    let original = Weights::random(&cfg, 99).unwrap();

    for (li, (a, b)) in loaded.layers.iter().zip(&original.layers).enumerate() {
        assert!(!a.moe_router.is_empty(), "layer {li} router loaded");
        assert_eq!(
            max_abs_diff(&a.moe_router, &b.moe_router),
            0.0,
            "layer {li} router"
        );
        assert_eq!(max_abs_diff(&a.moe_wg, &b.moe_wg), 0.0, "layer {li} moe_wg");
        assert_eq!(max_abs_diff(&a.moe_wu, &b.moe_wu), 0.0, "layer {li} moe_wu");
        assert_eq!(max_abs_diff(&a.moe_wd, &b.moe_wd), 0.0, "layer {li} moe_wd");
        // Dense fields must still round-trip under a MoE config (loader reads them too).
        assert_eq!(max_abs_diff(&a.wg, &b.wg), 0.0, "layer {li} dense wg");
        assert_eq!(max_abs_diff(&a.wu, &b.wu), 0.0, "layer {li} dense wu");
        assert_eq!(max_abs_diff(&a.wd, &b.wd), 0.0, "layer {li} dense wd");
    }
    let _ = std::fs::remove_file(&path);
}

#[cfg(feature = "hip")]
#[test]
fn loaded_weights_decode_matches_cpu_reference() {
    use mach_kernel_sys::hip;
    use mach_model::model::GpuModel;
    use mach_model::ref_model::RefModel;

    let hip = match hip::hip() {
        Ok(h) => match hip::device_count() {
            Ok(n) if n > 0 => h,
            _ => {
                eprintln!("skipping HIP test: no device");
                return;
            }
        },
        Err(e) => {
            eprintln!("skipping HIP test: {e}");
            return;
        }
    };

    let cfg = Config::tiny();
    let path = tmp_path("decode");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let w = load_safetensors(&path, &cfg, false).unwrap();
    let tokens = [3u32, 7, 1, 22];

    let mut gpu = GpuModel::new(hip, cfg, &w).unwrap();
    let gpu_logits = gpu.forward(&tokens).unwrap();
    let mut cpu = RefModel::new(cfg, w);
    let cpu_logits = cpu.forward(&tokens);

    let max = max_abs_diff(&gpu_logits, &cpu_logits);
    let scale = cpu_logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(
        max <= 2e-3 + 2e-3 * scale,
        "loaded-weights GPU vs CPU: max diff {max} (scale {scale})"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn sharded_load_matches_single_file() {
    let cfg = Config::tiny();
    let tensors = tensor_names(&cfg);
    // Split the tensor list across two shard files; offsets are per-file and
    // must be rebased by the directory loader.
    let mid = tensors.len() / 2;
    let (first, second) = tensors.split_at(mid);
    let dir = std::env::temp_dir().join("machserve-shards");
    if dir.exists() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    std::fs::create_dir_all(&dir).unwrap();
    let p1 = dir.join("model-00001-of-00002.safetensors");
    let p2 = dir.join("model-00002-of-00002.safetensors");
    let flat1: Vec<(&str, &[f32], &[usize])> = first
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    let flat2: Vec<(&str, &[f32], &[usize])> = second
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&p1, &flat1);
    write_safetensors(&p2, &flat2);

    let loaded = load_safetensors_dir(&dir, &cfg, false).unwrap();
    let original = Weights::random(&cfg, 99).unwrap();

    assert_eq!(max_abs_diff(&loaded.tok_emb, &original.tok_emb), 0.0);
    assert_eq!(max_abs_diff(&loaded.rms_final, &original.rms_final), 0.0);
    assert_eq!(max_abs_diff(&loaded.lm_head, &original.lm_head), 0.0);
    for (li, (a, b)) in loaded.layers.iter().zip(&original.layers).enumerate() {
        assert_eq!(max_abs_diff(&a.wq, &b.wq), 0.0, "layer {li} wq");
        assert_eq!(max_abs_diff(&a.wk, &b.wk), 0.0, "layer {li} wk");
        assert_eq!(max_abs_diff(&a.wv, &b.wv), 0.0, "layer {li} wv");
        assert_eq!(max_abs_diff(&a.wo, &b.wo), 0.0, "layer {li} wo");
        assert_eq!(max_abs_diff(&a.wg, &b.wg), 0.0, "layer {li} wg");
        assert_eq!(max_abs_diff(&a.wu, &b.wu), 0.0, "layer {li} wu");
        assert_eq!(max_abs_diff(&a.wd, &b.wd), 0.0, "layer {li} wd");
        assert_eq!(max_abs_diff(&a.rms_attn, &b.rms_attn), 0.0);
        assert_eq!(max_abs_diff(&a.rms_mlp, &b.rms_mlp), 0.0);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
