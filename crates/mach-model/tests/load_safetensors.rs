//! Safetensors loading regression tests.
//!
//! Writes a synthetic safetensors checkpoint (Llama-style tensor names, F32),
//! loads it back through `mach_model::loader`, and verifies both the parsed
//! weights and (with the `hip` feature) that a GPU model built from the loaded
//! weights decodes identically to the CPU reference.

#[cfg(feature = "hip")]
use mach_model::config::ModelDType;
use mach_model::loader::{
    load_safetensors, load_safetensors_dir, load_safetensors_fp8, load_safetensors_q4,
};
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
        if cfg.kv_lora_rank > 0 {
            // MLA (DeepSeek-V2 style) replaces q/k/v/o projections.
            let nope = cfg.qk_nope_head_dim;
            let rope = cfg.qk_rope_head_dim;
            let v_hd = cfg.v_head_dim;
            let heads = cfg.n_heads;
            t.push((
                p("self_attn.q_a_proj.weight"),
                lw.mla_q_a.clone(),
                vec![cfg.q_lora_rank, d],
            ));
            t.push((
                p("self_attn.q_a_layernorm.weight"),
                lw.mla_q_a_norm.clone(),
                vec![cfg.q_lora_rank],
            ));
            t.push((
                p("self_attn.q_b_proj.weight"),
                lw.mla_q_b.clone(),
                vec![heads * nope, cfg.q_lora_rank],
            ));
            t.push((
                p("self_attn.q_rope_proj.weight"),
                lw.mla_q_rope.clone(),
                vec![heads * rope, d],
            ));
            t.push((
                p("self_attn.kv_a_proj_with_mqa.weight"),
                lw.mla_kv_a.clone(),
                vec![cfg.kv_lora_rank + rope, d],
            ));
            t.push((
                p("self_attn.kv_a_layernorm.weight"),
                lw.mla_kv_a_norm.clone(),
                vec![cfg.kv_lora_rank],
            ));
            t.push((
                p("self_attn.kv_b_proj.weight"),
                lw.mla_kv_b.clone(),
                vec![heads * (nope + v_hd), cfg.kv_lora_rank],
            ));
            t.push((
                p("self_attn.o_proj.weight"),
                lw.mla_o.clone(),
                vec![d, heads * v_hd],
            ));
        } else {
            t.push((p("self_attn.q_proj.weight"), lw.wq.clone(), vec![nq, d]));
            t.push((p("self_attn.k_proj.weight"), lw.wk.clone(), vec![nkv, d]));
            t.push((p("self_attn.v_proj.weight"), lw.wv.clone(), vec![nkv, d]));
            t.push((p("self_attn.o_proj.weight"), lw.wo.clone(), vec![d, nq]));
        }
        t.push((p("input_layernorm.weight"), lw.rms_attn.clone(), vec![d]));
        if cfg.num_experts == 0 {
            // MoE layers have no dense MLP tensors (expert tensors replace them).
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
        }
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
    assert_eq!(a.len(), b.len(), "max_abs_diff: length mismatch");
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
        // MoE layers carry no dense MLP weights (expert tensors replace them).
        assert!(a.wg.is_empty(), "layer {li} dense wg must be empty for MoE");
        assert!(a.wu.is_empty(), "layer {li} dense wu must be empty for MoE");
        assert!(a.wd.is_empty(), "layer {li} dense wd must be empty for MoE");
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

/// Multi-shard Q4 load must equal the single-file result bit-for-bit: 30B-class
/// checkpoints (e.g. Qwen3-30B-A3B) ship as 16+ shards and `load_safetensors_q4`
/// streams them, rebasing per-file offsets. Splitting the same tensors across
/// shards is a pure layout change, so the packed Q4 output is identical.
#[test]
fn q4_sharded_load_matches_single_file() {
    let cfg = Config::tiny();
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();

    let single = tmp_path("q4shard-single");
    write_safetensors(&single, &flat);
    let q4_single = load_safetensors_q4(&single, &cfg, false).unwrap();

    let dir = std::env::temp_dir().join("machserve-q4-shards");
    if dir.exists() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    std::fs::create_dir_all(&dir).unwrap();
    let n_shards = 4usize;
    let per = tensors.len().div_ceil(n_shards);
    for sh in 0..n_shards {
        let lo = sh * per;
        let hi = (lo + per).min(tensors.len());
        if lo >= hi {
            break;
        }
        let part: Vec<(&str, &[f32], &[usize])> = tensors[lo..hi]
            .iter()
            .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
            .collect();
        let path = dir.join(format!(
            "model-{:02}-of-{:02}.safetensors",
            sh + 1,
            n_shards
        ));
        write_safetensors(&path, &part);
    }
    let q4_shards = load_safetensors_q4(&dir, &cfg, false).unwrap();

    // Same source values -> identical packed Q4 tensors (dequantize bit-equal).
    assert_eq!(
        q4_single.tok_emb.dequantize(),
        q4_shards.tok_emb.dequantize(),
        "tok_emb"
    );
    assert_eq!(q4_single.rms_final, q4_shards.rms_final, "rms_final");
    assert_eq!(
        q4_single.lm_head.dequantize(),
        q4_shards.lm_head.dequantize(),
        "lm_head"
    );
    for (li, (a, b)) in q4_single.layers.iter().zip(&q4_shards.layers).enumerate() {
        assert_eq!(a.rms_attn, b.rms_attn, "layer {li} rms_attn");
        assert_eq!(a.rms_mlp, b.rms_mlp, "layer {li} rms_mlp");
        assert_eq!(a.wq.dequantize(), b.wq.dequantize(), "layer {li} wq");
        assert_eq!(a.wk.dequantize(), b.wk.dequantize(), "layer {li} wk");
        assert_eq!(a.wv.dequantize(), b.wv.dequantize(), "layer {li} wv");
        assert_eq!(a.wo.dequantize(), b.wo.dequantize(), "layer {li} wo");
        assert_eq!(a.wg.dequantize(), b.wg.dequantize(), "layer {li} wg");
        assert_eq!(a.wu.dequantize(), b.wu.dequantize(), "layer {li} wu");
        assert_eq!(a.wd.dequantize(), b.wd.dequantize(), "layer {li} wd");
        assert_eq!(a.moe_router, b.moe_router, "layer {li} moe_router");
        assert_eq!(
            a.moe_wg.dequantize(),
            b.moe_wg.dequantize(),
            "layer {li} moe_wg"
        );
        assert_eq!(
            a.moe_wu.dequantize(),
            b.moe_wu.dequantize(),
            "layer {li} moe_wu"
        );
        assert_eq!(
            a.moe_wd.dequantize(),
            b.moe_wd.dequantize(),
            "layer {li} moe_wd"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&single);
}

#[test]
fn q4_load_matches_f32_within_tolerance() {
    let cfg = Config::tiny();
    let path = tmp_path("q4");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let q4 = load_safetensors_q4(&path, &cfg, false).unwrap();
    let original = Weights::random(&cfg, 99).unwrap();

    // Norms/biases stay exact; quantized GEMM weights stay within ~half scale.
    assert_eq!(q4.rms_final, original.rms_final);
    assert_eq!(q4.layers[0].rms_attn, original.layers[0].rms_attn);
    assert_eq!(q4.layers[0].rms_mlp, original.layers[0].rms_mlp);

    // tiny random weights scale ~= 1/sqrt(d) ~ 0.09; int4 error <= scale/2.
    let tol = 0.05;
    assert!(
        max_abs_diff(&q4.tok_emb.dequantize(), &original.tok_emb) < tol,
        "tok_emb dequant must be close"
    );
    assert!(
        max_abs_diff(&q4.lm_head.dequantize(), &original.lm_head) < tol,
        "lm_head dequant must be close"
    );
    for (li, (a, b)) in q4.layers.iter().zip(&original.layers).enumerate() {
        assert!(
            max_abs_diff(&a.wq.dequantize(), &b.wq) < tol,
            "layer {li} wq"
        );
        assert!(
            max_abs_diff(&a.wk.dequantize(), &b.wk) < tol,
            "layer {li} wk"
        );
        assert!(
            max_abs_diff(&a.wv.dequantize(), &b.wv) < tol,
            "layer {li} wv"
        );
        assert!(
            max_abs_diff(&a.wo.dequantize(), &b.wo) < tol,
            "layer {li} wo"
        );
        assert!(
            max_abs_diff(&a.wg.dequantize(), &b.wg) < tol,
            "layer {li} wg"
        );
        assert!(
            max_abs_diff(&a.wu.dequantize(), &b.wu) < tol,
            "layer {li} wu"
        );
        assert!(
            max_abs_diff(&a.wd.dequantize(), &b.wd) < tol,
            "layer {li} wd"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn fp8_load_matches_f32_within_tolerance() {
    let cfg = Config::tiny();
    let path = tmp_path("fp8");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let fp8 = load_safetensors_fp8(&path, &cfg, false).unwrap();
    let original = Weights::random(&cfg, 99).unwrap();

    // Norms/biases stay exact; quantized GEMM weights stay within E4M3's
    // ~6% relative precision (per-tensor scale). tiny random weights scale
    // ~= 1/sqrt(d) ~ 0.09, so max abs error ~= 0.09 * 2^-4 ~ 0.006.
    assert_eq!(fp8.rms_final, original.rms_final);
    assert_eq!(fp8.layers[0].rms_attn, original.layers[0].rms_attn);
    assert_eq!(fp8.layers[0].rms_mlp, original.layers[0].rms_mlp);
    let tol = 0.01;
    assert!(
        max_abs_diff(&fp8.tok_emb.dequantize(), &original.tok_emb) < tol,
        "tok_emb dequant must be close"
    );
    assert!(
        max_abs_diff(&fp8.lm_head.dequantize(), &original.lm_head) < tol,
        "lm_head dequant must be close"
    );
    for (li, (a, b)) in fp8.layers.iter().zip(&original.layers).enumerate() {
        assert!(
            max_abs_diff(&a.wq.dequantize(), &b.wq) < tol,
            "layer {li} wq"
        );
        assert!(
            max_abs_diff(&a.wk.dequantize(), &b.wk) < tol,
            "layer {li} wk"
        );
        assert!(
            max_abs_diff(&a.wv.dequantize(), &b.wv) < tol,
            "layer {li} wv"
        );
        assert!(
            max_abs_diff(&a.wo.dequantize(), &b.wo) < tol,
            "layer {li} wo"
        );
        assert!(
            max_abs_diff(&a.wg.dequantize(), &b.wg) < tol,
            "layer {li} wg"
        );
        assert!(
            max_abs_diff(&a.wu.dequantize(), &b.wu) < tol,
            "layer {li} wu"
        );
        assert!(
            max_abs_diff(&a.wd.dequantize(), &b.wd) < tol,
            "layer {li} wd"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn fp8_load_matches_f32_moe() {
    let cfg = moe_cfg();
    let path = tmp_path("fp8moe");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let fp8 = load_safetensors_fp8(&path, &cfg, false).unwrap();
    let original = Weights::random(&cfg, 99).unwrap();
    let tol = 0.01;
    // Router stays f32 (exact); expert GEMMs quantized within tolerance and
    // concatenated with per-expert scales (block = expert size).
    assert_eq!(fp8.layers[0].moe_router, original.layers[0].moe_router);
    let ne = cfg.num_experts;
    let einter = cfg.expert_size();
    let d = cfg.d_model;
    assert_eq!(fp8.layers[0].moe_wg.block(), einter * d, "wg expert block");
    assert_eq!(
        fp8.layers[0].moe_wg.scales().len(),
        ne,
        "wg per-expert scales"
    );
    assert_eq!(fp8.layers[0].moe_wd.block(), d * einter, "wd expert block");
    assert_eq!(
        fp8.layers[0].moe_wd.scales().len(),
        ne,
        "wd per-expert scales"
    );
    for (li, (a, b)) in fp8.layers.iter().zip(&original.layers).enumerate() {
        assert!(
            max_abs_diff(&a.moe_wg.dequantize(), &b.moe_wg) < tol,
            "layer {li} wg"
        );
        assert!(
            max_abs_diff(&a.moe_wu.dequantize(), &b.moe_wu) < tol,
            "layer {li} wu"
        );
        assert!(
            max_abs_diff(&a.moe_wd.dequantize(), &b.moe_wd) < tol,
            "layer {li} wd"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn fp8_load_matches_f32_mla() {
    let cfg = Config::mla(128, 2, 4, 1024, 64, 32, 16, 16, 8, 16);
    let path = tmp_path("fp8mla");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let fp8 = load_safetensors_fp8(&path, &cfg, false).unwrap();
    let original = Weights::random(&cfg, 99).unwrap();
    let tol = 0.01;
    // MLA norms stay exact; quantized projections stay close.
    assert_eq!(fp8.layers[0].mla_q_a_norm, original.layers[0].mla_q_a_norm);
    assert_eq!(
        fp8.layers[0].mla_kv_a_norm,
        original.layers[0].mla_kv_a_norm
    );
    for (li, (a, b)) in fp8.layers.iter().zip(&original.layers).enumerate() {
        assert!(
            max_abs_diff(&a.mla_q_a.dequantize(), &b.mla_q_a) < tol,
            "layer {li} q_a"
        );
        assert!(
            max_abs_diff(&a.mla_q_b.dequantize(), &b.mla_q_b) < tol,
            "layer {li} q_b"
        );
        assert!(
            max_abs_diff(&a.mla_q_rope.dequantize(), &b.mla_q_rope) < tol,
            "layer {li} q_rope"
        );
        assert!(
            max_abs_diff(&a.mla_kv_a.dequantize(), &b.mla_kv_a) < tol,
            "layer {li} kv_a"
        );
        assert!(
            max_abs_diff(&a.mla_kv_b.dequantize(), &b.mla_kv_b) < tol,
            "layer {li} kv_b"
        );
        assert!(
            max_abs_diff(&a.mla_o.dequantize(), &b.mla_o) < tol,
            "layer {li} o"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn q4_load_matches_f32_mla() {
    let cfg = Config::mla(128, 2, 4, 1024, 64, 32, 16, 16, 8, 16);
    let path = tmp_path("q4mla");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let q4 = load_safetensors_q4(&path, &cfg, false).unwrap();
    let original = Weights::random(&cfg, 99).unwrap();
    let tol = 0.05;
    // MLA norms stay exact; quantized projections stay close.
    assert_eq!(q4.layers[0].mla_q_a_norm, original.layers[0].mla_q_a_norm);
    assert_eq!(q4.layers[0].mla_kv_a_norm, original.layers[0].mla_kv_a_norm);
    for (li, (a, b)) in q4.layers.iter().zip(&original.layers).enumerate() {
        assert!(
            max_abs_diff(&a.mla_q_a.dequantize(), &b.mla_q_a) < tol,
            "layer {li} q_a"
        );
        assert!(
            max_abs_diff(&a.mla_q_b.dequantize(), &b.mla_q_b) < tol,
            "layer {li} q_b"
        );
        assert!(
            max_abs_diff(&a.mla_q_rope.dequantize(), &b.mla_q_rope) < tol,
            "layer {li} q_rope"
        );
        assert!(
            max_abs_diff(&a.mla_kv_a.dequantize(), &b.mla_kv_a) < tol,
            "layer {li} kv_a"
        );
        assert!(
            max_abs_diff(&a.mla_kv_b.dequantize(), &b.mla_kv_b) < tol,
            "layer {li} kv_b"
        );
        assert!(
            max_abs_diff(&a.mla_o.dequantize(), &b.mla_o) < tol,
            "layer {li} o"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn q4_load_matches_f32_moe() {
    let cfg = moe_cfg();
    let path = tmp_path("q4moe");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let q4 = load_safetensors_q4(&path, &cfg, false).unwrap();
    let original = Weights::random(&cfg, 99).unwrap();
    let tol = 0.05;
    // Router stays f32 (exact); expert GEMMs quantized within tolerance.
    assert_eq!(q4.layers[0].moe_router, original.layers[0].moe_router);
    for (li, (a, b)) in q4.layers.iter().zip(&original.layers).enumerate() {
        assert!(
            max_abs_diff(&a.moe_wg.dequantize(), &b.moe_wg) < tol,
            "layer {li} wg"
        );
        assert!(
            max_abs_diff(&a.moe_wu.dequantize(), &b.moe_wu) < tol,
            "layer {li} wu"
        );
        assert!(
            max_abs_diff(&a.moe_wd.dequantize(), &b.moe_wd) < tol,
            "layer {li} wd"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[cfg(feature = "hip")]
#[test]
fn q4_gpu_matches_f32() {
    use mach_kernel_sys::hip;
    use mach_model::model::GpuModel;

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

    let mut cfg = Config::tiny();
    cfg.dtype = ModelDType::F16;
    let path = tmp_path("q4gpu");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let w32 = load_safetensors(&path, &cfg, false).unwrap();
    let wq4 = load_safetensors_q4(&path, &cfg, false).unwrap();
    let tokens = [3u32, 7, 1, 22];

    let mut m32 = GpuModel::new(hip.clone(), cfg, &w32).unwrap();
    let mut mq4 = GpuModel::from_q4(hip, cfg, &wq4).unwrap();
    let l32 = m32.forward(&tokens).unwrap();
    let lq4 = mq4.forward(&tokens).unwrap();

    let max = max_abs_diff(&l32, &lq4);
    let scale = l32.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    eprintln!("q4 GPU vs f32 GPU: max logit diff {max:.4} (scale {scale:.3})");
    assert!(
        max <= 0.2 + 0.2 * scale,
        "q4 GPU vs f32 GPU logits diverged: {max} vs scale {scale}"
    );
    let _ = std::fs::remove_file(&path);
}

/// Batched Q4 vs batched f16 on synthetic weights: Q4 is a storage format, so
/// dequantizing to f16 and running the same compute path stays within the int4
/// quantization error (per-group scale) plus fp16 rounding. Argmax may flip on
/// near-uniform synthetic logits, so only the logits bound is asserted here;
/// the greedy-token behavior on real weights is measured by `q4_bench`.
#[cfg(feature = "hip")]
#[ignore]
#[test]
fn q4_batched_gpu_matches_f16() {
    use mach_kernel_sys::hip;
    use mach_model::batched::BatchedModel;

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

    let mut cfg = Config::tiny();
    cfg.dtype = ModelDType::F16;
    let path = tmp_path("q4batched");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let w32 = load_safetensors(&path, &cfg, false).unwrap();
    let wq4 = load_safetensors_q4(&path, &cfg, false).unwrap();

    let batch = 4usize;
    let steps = [[3u32, 7, 1, 22], [9, 2, 200, 5], [4, 8, 15, 16]];

    let mut m16 = BatchedModel::new(hip.clone(), cfg, &w32, batch).unwrap();
    let mut mq4 = BatchedModel::from_q4(hip.clone(), cfg, &wq4, batch).unwrap();
    for (si, step_tokens) in steps.iter().enumerate() {
        let _ = m16.decode_step(step_tokens).unwrap();
        let _ = mq4.decode_step(step_tokens).unwrap();
        let l16 = m16.read_logits().unwrap();
        let lq4 = mq4.read_logits().unwrap();
        for s in 0..batch {
            let row16 = &l16[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let rowq4 = &lq4[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let max = max_abs_diff(row16, rowq4);
            let scale = row16.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            eprintln!("q4 batched vs f16: step {si} seq {s} max diff {max:.4} (scale {scale:.3})");
            // Same convention as the single-seq q4_gpu_matches_f32 test.
            assert!(
                max <= 0.2 + 0.2 * scale,
                "step {si} seq {s}: q4 batched vs f16 logits max diff {max} (scale {scale})"
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// Batched MoE Q4 vs f16: per-expert gate/up/down are quantized to int4, the
/// router stays f32 (Q4 host layout) with the same f16 device copy as the f16
/// path, so expert selection is identical. The logits bound is looser than the
/// dense case: MoE routing composes per-expert quantization error across the
/// active experts.
#[cfg(feature = "hip")]
#[ignore]
#[test]
fn q4_batched_moe_gpu_matches_f16() {
    use mach_kernel_sys::hip;
    use mach_model::batched::BatchedModel;

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

    let mut cfg = moe_cfg();
    cfg.dtype = ModelDType::F16;
    let path = tmp_path("q4batchedmoe");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let w32 = load_safetensors(&path, &cfg, false).unwrap();
    let wq4 = load_safetensors_q4(&path, &cfg, false).unwrap();
    let batch = 4usize;
    let steps = [[3u32, 7, 1, 22], [9, 2, 200, 5], [4, 8, 15, 16]];

    let mut m16 = BatchedModel::new(hip.clone(), cfg, &w32, batch).unwrap();
    let mut mq4 = BatchedModel::from_q4(hip.clone(), cfg, &wq4, batch).unwrap();
    for (si, step_tokens) in steps.iter().enumerate() {
        let _ = m16.decode_step(step_tokens).unwrap();
        let _ = mq4.decode_step(step_tokens).unwrap();
        let l16 = m16.read_logits().unwrap();
        let lq4 = mq4.read_logits().unwrap();
        for s in 0..batch {
            let row16 = &l16[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let rowq4 = &lq4[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let max = max_abs_diff(row16, rowq4);
            let scale = row16.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            eprintln!(
                "q4 batched MoE vs f16: step {si} seq {s} max diff {max:.4} (scale {scale:.3})"
            );
            // Per-expert int4 error composes through the top-2 route.
            assert!(
                max <= 0.5 + 0.5 * scale,
                "step {si} seq {s}: q4 batched MoE vs f16 logits max diff {max} (scale {scale})"
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// The upload-path correctness gate: `BatchedModel::from_q4` must produce the
/// same device f16 bits as an f16 model built from the *dequantized* Q4
/// weights (Q4 -> f16 -> identical compute path). This is bit-exact, so any
/// diff above fp slack is a real bug in the Q4 upload, independent of how big
/// the quantization error is vs the original f32 weights.
#[cfg(feature = "hip")]
#[ignore]
#[test]
fn q4_batched_matches_dequantized_f16() {
    use mach_kernel_sys::hip;
    use mach_model::batched::BatchedModel;
    use mach_model::q4::Q4Tensor;
    use mach_model::{LayerWeights, Weights};

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

    let to_f32 = |q: &Q4Tensor| q.dequantize();
    for (label, mut cfg) in [("dense", Config::tiny()), ("moe", moe_cfg())] {
        cfg.dtype = ModelDType::F16;
        let path = tmp_path(&format!("q4deq{label}"));
        let tensors = tensor_names(&cfg);
        let flat: Vec<(&str, &[f32], &[usize])> = tensors
            .iter()
            .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
            .collect();
        write_safetensors(&path, &flat);

        let wq4 = load_safetensors_q4(&path, &cfg, false).unwrap();
        // Reference: f32 weights reconstructed from the Q4 storage (GEMM
        // tensors dequantized; norms/biases/router copied exactly).
        let wref = Weights {
            tok_emb: to_f32(&wq4.tok_emb),
            rms_final: wq4.rms_final.clone(),
            lm_head: to_f32(&wq4.lm_head),
            layers: wq4
                .layers
                .iter()
                .map(|l| LayerWeights {
                    wq: to_f32(&l.wq),
                    wk: to_f32(&l.wk),
                    wv: to_f32(&l.wv),
                    wo: to_f32(&l.wo),
                    rms_attn: l.rms_attn.clone(),
                    wg: to_f32(&l.wg),
                    wu: to_f32(&l.wu),
                    wd: to_f32(&l.wd),
                    rms_mlp: l.rms_mlp.clone(),
                    bq: l.bq.clone(),
                    bk: l.bk.clone(),
                    bv: l.bv.clone(),
                    q_norm: l.q_norm.clone(),
                    k_norm: l.k_norm.clone(),
                    mla_q_a: to_f32(&l.mla_q_a),
                    mla_q_a_norm: l.mla_q_a_norm.clone(),
                    mla_q_b: to_f32(&l.mla_q_b),
                    mla_q_rope: to_f32(&l.mla_q_rope),
                    mla_kv_a: to_f32(&l.mla_kv_a),
                    mla_kv_a_norm: l.mla_kv_a_norm.clone(),
                    mla_kv_b: to_f32(&l.mla_kv_b),
                    mla_o: to_f32(&l.mla_o),
                    moe_router: l.moe_router.clone(),
                    moe_wg: to_f32(&l.moe_wg),
                    moe_wu: to_f32(&l.moe_wu),
                    moe_wd: to_f32(&l.moe_wd),
                })
                .collect(),
        };

        let batch = 4usize;
        let steps = [[3u32, 7, 1, 22], [9, 2, 200, 5], [4, 8, 15, 16]];
        let mut mref = BatchedModel::new(hip.clone(), cfg, &wref, batch).unwrap();
        let mut mq4 = BatchedModel::from_q4(hip.clone(), cfg, &wq4, batch).unwrap();
        for (si, step_tokens) in steps.iter().enumerate() {
            let _ = mref.decode_step(step_tokens).unwrap();
            let _ = mq4.decode_step(step_tokens).unwrap();
            let lref = mref.read_logits().unwrap();
            let lq4 = mq4.read_logits().unwrap();
            let max = max_abs_diff(&lref, &lq4);
            let scale = lref.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                max <= 1e-4 * (1.0 + scale),
                "{label} step {si}: q4 batched vs dequantized-f16 reference max diff {max} (scale {scale}) — upload path diverged"
            );
        }
        let _ = std::fs::remove_file(&path);
    }
}

/// Batched FP8 vs batched f16 on synthetic weights: FP8 is a storage format,
/// so dequantizing to f16 and running the same compute path stays within the
/// E4M3 per-tensor-scale error plus fp16 rounding. Argmax may flip on
/// near-uniform synthetic logits, so only the logits bound is asserted here;
/// the greedy-token behavior on real weights is measured by `fp8_bench`.
#[cfg(feature = "hip")]
#[ignore]
#[test]
fn fp8_batched_gpu_matches_f16() {
    use mach_kernel_sys::hip;
    use mach_model::batched::BatchedModel;

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

    let mut cfg = Config::tiny();
    cfg.dtype = ModelDType::F16;
    let path = tmp_path("fp8batched");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let w32 = load_safetensors(&path, &cfg, false).unwrap();
    let wfp8 = load_safetensors_fp8(&path, &cfg, false).unwrap();

    let batch = 4usize;
    let steps = [[3u32, 7, 1, 22], [9, 2, 200, 5], [4, 8, 15, 16]];

    let mut m16 = BatchedModel::new(hip.clone(), cfg, &w32, batch).unwrap();
    let mut mfp8 = BatchedModel::from_fp8(hip.clone(), cfg, &wfp8, batch).unwrap();
    for (si, step_tokens) in steps.iter().enumerate() {
        let _ = m16.decode_step(step_tokens).unwrap();
        let _ = mfp8.decode_step(step_tokens).unwrap();
        let l16 = m16.read_logits().unwrap();
        let lfp8 = mfp8.read_logits().unwrap();
        for s in 0..batch {
            let row16 = &l16[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let rowfp8 = &lfp8[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let max = max_abs_diff(row16, rowfp8);
            let scale = row16.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            eprintln!("fp8 batched vs f16: step {si} seq {s} max diff {max:.4} (scale {scale:.3})");
            // E4M3 (3 mantissa bits) is ~2.7x more precise than int4 on this path, so the
            // bound is tighter than the Q4 test.s 0.2 + 0.2*scale.
            assert!(
                max <= 0.1 + 0.1 * scale,
                "step {si} seq {s}: fp8 batched vs f16 logits max diff {max} (scale {scale})"
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// Batched MoE FP8 vs f16: per-expert gate/up/down are quantized to E4M3 with
/// per-expert scales, the router stays f32 (FP8 host layout) with the same f16
/// device copy as the f16 path, so expert selection is identical. The logits
/// bound is looser than the dense case: MoE routing composes per-expert
/// quantization error across the active experts.
#[cfg(feature = "hip")]
#[ignore]
#[test]
fn fp8_batched_moe_gpu_matches_f16() {
    use mach_kernel_sys::hip;
    use mach_model::batched::BatchedModel;

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

    let mut cfg = moe_cfg();
    cfg.dtype = ModelDType::F16;
    let path = tmp_path("fp8batchedmoe");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let w32 = load_safetensors(&path, &cfg, false).unwrap();
    let wfp8 = load_safetensors_fp8(&path, &cfg, false).unwrap();
    let batch = 4usize;
    let steps = [[3u32, 7, 1, 22], [9, 2, 200, 5], [4, 8, 15, 16]];

    let mut m16 = BatchedModel::new(hip.clone(), cfg, &w32, batch).unwrap();
    let mut mfp8 = BatchedModel::from_fp8(hip.clone(), cfg, &wfp8, batch).unwrap();
    for (si, step_tokens) in steps.iter().enumerate() {
        let _ = m16.decode_step(step_tokens).unwrap();
        let _ = mfp8.decode_step(step_tokens).unwrap();
        let l16 = m16.read_logits().unwrap();
        let lfp8 = mfp8.read_logits().unwrap();
        for s in 0..batch {
            let row16 = &l16[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let rowfp8 = &lfp8[s * cfg.vocab_size..(s + 1) * cfg.vocab_size];
            let max = max_abs_diff(row16, rowfp8);
            let scale = row16.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            eprintln!(
                "fp8 batched MoE vs f16: step {si} seq {s} max diff {max:.4} (scale {scale:.3})"
            );
            assert!(
                max <= 0.25 + 0.2 * scale,
                "step {si} seq {s}: fp8 batched MoE vs f16 logits max diff {max} (scale {scale})"
            );
        }
    }
    let _ = std::fs::remove_file(&path);
}

/// The upload-path correctness gate: `BatchedModel::from_fp8` must produce the
/// same device f16 bits as an f16 model built from the *dequantized* FP8
/// weights (FP8 -> f16 -> identical compute path). This is bit-exact, so any
/// diff above fp slack is a real bug in the FP8 upload, independent of how big
/// the quantization error is vs the original f32 weights.
#[cfg(feature = "hip")]
#[ignore]
#[test]
fn fp8_batched_matches_dequantized_f16() {
    use mach_kernel_sys::hip;
    use mach_model::batched::BatchedModel;
    use mach_model::fp8::Fp8Tensor;
    use mach_model::{LayerWeights, Weights};

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

    let to_f32 = |q: &Fp8Tensor| q.dequantize();
    for (label, mut cfg) in [("dense", Config::tiny()), ("moe", moe_cfg())] {
        cfg.dtype = ModelDType::F16;
        let path = tmp_path(&format!("fp8deq{label}"));
        let tensors = tensor_names(&cfg);
        let flat: Vec<(&str, &[f32], &[usize])> = tensors
            .iter()
            .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
            .collect();
        write_safetensors(&path, &flat);

        let wfp8 = load_safetensors_fp8(&path, &cfg, false).unwrap();
        // Reference: f32 weights reconstructed from the FP8 storage (GEMM
        // tensors dequantized; norms/biases/router copied exactly).
        let wref = Weights {
            tok_emb: to_f32(&wfp8.tok_emb),
            rms_final: wfp8.rms_final.clone(),
            lm_head: to_f32(&wfp8.lm_head),
            layers: wfp8
                .layers
                .iter()
                .map(|l| LayerWeights {
                    wq: to_f32(&l.wq),
                    wk: to_f32(&l.wk),
                    wv: to_f32(&l.wv),
                    wo: to_f32(&l.wo),
                    rms_attn: l.rms_attn.clone(),
                    wg: to_f32(&l.wg),
                    wu: to_f32(&l.wu),
                    wd: to_f32(&l.wd),
                    rms_mlp: l.rms_mlp.clone(),
                    bq: l.bq.clone(),
                    bk: l.bk.clone(),
                    bv: l.bv.clone(),
                    q_norm: l.q_norm.clone(),
                    k_norm: l.k_norm.clone(),
                    mla_q_a: to_f32(&l.mla_q_a),
                    mla_q_a_norm: l.mla_q_a_norm.clone(),
                    mla_q_b: to_f32(&l.mla_q_b),
                    mla_q_rope: to_f32(&l.mla_q_rope),
                    mla_kv_a: to_f32(&l.mla_kv_a),
                    mla_kv_a_norm: l.mla_kv_a_norm.clone(),
                    mla_kv_b: to_f32(&l.mla_kv_b),
                    mla_o: to_f32(&l.mla_o),
                    moe_router: l.moe_router.clone(),
                    moe_wg: to_f32(&l.moe_wg),
                    moe_wu: to_f32(&l.moe_wu),
                    moe_wd: to_f32(&l.moe_wd),
                })
                .collect(),
        };

        let batch = 4usize;
        let steps = [[3u32, 7, 1, 22], [9, 2, 200, 5], [4, 8, 15, 16]];
        let mut mref = BatchedModel::new(hip.clone(), cfg, &wref, batch).unwrap();
        let mut mfp8 = BatchedModel::from_fp8(hip.clone(), cfg, &wfp8, batch).unwrap();
        for (si, step_tokens) in steps.iter().enumerate() {
            let _ = mref.decode_step(step_tokens).unwrap();
            let _ = mfp8.decode_step(step_tokens).unwrap();
            let lref = mref.read_logits().unwrap();
            let lfp8 = mfp8.read_logits().unwrap();
            let max = max_abs_diff(&lref, &lfp8);
            let scale = lref.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                max <= 1e-4 * (1.0 + scale),
                "{label} step {si}: fp8 batched vs dequantized-f16 reference max diff {max} (scale {scale}) — upload path diverged"
            );
        }
        let _ = std::fs::remove_file(&path);
    }
}

/// Single-sequence `GpuModel::from_fp8` vs the f32-loaded GPU model on
/// synthetic weights: FP8 is a storage format, so the logits stay within the
/// E4M3 per-tensor-scale error. Ignored: GPU test, run serially.
#[cfg(feature = "hip")]
#[ignore]
#[test]
fn fp8_gpu_matches_f32() {
    use mach_kernel_sys::hip;
    use mach_model::model::GpuModel;

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

    let mut cfg = Config::tiny();
    cfg.dtype = ModelDType::F16;
    let path = tmp_path("fp8gpu");
    let tensors = tensor_names(&cfg);
    let flat: Vec<(&str, &[f32], &[usize])> = tensors
        .iter()
        .map(|(n, d, s)| (n.as_str(), d.as_slice(), s.as_slice()))
        .collect();
    write_safetensors(&path, &flat);

    let w32 = load_safetensors(&path, &cfg, false).unwrap();
    let wfp8 = load_safetensors_fp8(&path, &cfg, false).unwrap();
    let tokens = [3u32, 7, 1, 22];

    let mut m32 = GpuModel::new(hip.clone(), cfg, &w32).unwrap();
    let mut mfp8 = GpuModel::from_fp8(hip, cfg, &wfp8).unwrap();
    let l32 = m32.forward(&tokens).unwrap();
    let lfp8 = mfp8.forward(&tokens).unwrap();

    let max = max_abs_diff(&l32, &lfp8);
    let scale = l32.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    eprintln!("fp8 GPU vs f32 GPU: max logit diff {max:.4} (scale {scale:.3})");
    assert!(
        max <= 0.1 + 0.1 * scale,
        "fp8 GPU vs f32 GPU logits diverged: {max} vs scale {scale}"
    );
    let _ = std::fs::remove_file(&path);
}
