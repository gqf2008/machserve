//! Startup-time helpers that must be testable without the `hip` feature.
//!
//! The VRAM preflight, weight-size accounting and checkpoint-config parsing
//! are pure `Config`/filesystem/env logic, but they used to live in `main.rs`
//! behind `#[cfg(feature = "hip")]`, so `cargo test -p mach-server --bin
//! mach-server` reported "0 tests" for them — a green run that covered
//! nothing. Keeping them here (and their unit tests below) puts those cases on
//! the default test面; the binary still drives them from its hip path.

use mach_model::config::{Config, ModelDType};
use std::path::{Path, PathBuf};

/// Collapse a model-family name to a lowercase alphanumeric key so the two
/// places it can come from agree: `model_type` is snake_case (`deepseek_v2`)
/// while the `architectures[0]` fallback is a PascalCase class name
/// (`DeepseekV2ForCausalLM`). Both must yield `deepseekv2`, or a checkpoint
/// that omits `model_type` silently misses the `rope_interleave` allowlist.
fn normalize_model_family(raw: Option<&str>) -> String {
    raw.unwrap_or_default()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

pub fn config_from_json(path: &std::path::Path) -> Config {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read config"))
            .expect("parse config");
    // Multi-modal checkpoints (Qwen3.5 `Qwen3_5ForConditionalGeneration`) nest
    // the text stack under `text_config`; single-modality configs are flat.
    // Every text-model parameter below reads from `t`, never `v` directly.
    let t = v.get("text_config").filter(|x| x.is_object()).unwrap_or(&v);
    let hidden = t["hidden_size"].as_u64().unwrap_or(896) as usize;
    let layers = t["num_hidden_layers"].as_u64().unwrap_or(24) as usize;
    let heads = t["num_attention_heads"].as_u64().unwrap_or(14) as usize;
    let kv = t["num_key_value_heads"].as_u64().unwrap_or(heads as u64) as usize;
    let vocab = t["vocab_size"].as_u64().unwrap_or(151936) as usize;
    let inter = t["intermediate_size"].as_u64().unwrap_or(4 * hidden as u64) as usize;
    // DeepSeek-V2-Lite declares `max_position_embeddings: 163840`, but the
    // preallocated KV cache is `max_seq_len`-sized per slot (MLA: ~552 KB per
    // token across 27 layers), so honoring it verbatim would try to reserve
    // gigabytes per slot. Clamp to 8192 unless MACH_MAX_SEQ says otherwise.
    let max_seq = match std::env::var("MACH_MAX_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        Some(n) => n,
        None => t["max_position_embeddings"]
            .as_u64()
            .unwrap_or(2048)
            .min(8192) as usize,
    };
    let eps = t["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    // Qwen3.5 ships theta inside `rope_parameters` instead of the flat key.
    let theta = t["rope_theta"]
        .as_f64()
        .or_else(|| t["rope_parameters"]["rope_theta"].as_f64())
        .unwrap_or(10000.0) as f32;
    let mut cfg = Config::llama(hidden, layers, heads, kv, vocab, max_seq);
    // Some configs (e.g. Qwen3-30B-A3B) ship an explicit `head_dim` that
    // differs from hidden/n_heads (q/o width = n_heads*head_dim is wider
    // than hidden). Honor it, or the loader under-sizes q/o projections.
    if let Some(hd) = t["head_dim"].as_u64() {
        cfg.head_dim = hd as usize;
    }
    cfg.intermediate_size = inter;
    cfg.rms_eps = eps;
    cfg.rope_theta = theta;
    // RoPE pairing convention is a property of the checkpoint's own modeling
    // code, not of any hyper-parameter, so it has to be keyed off the model
    // family. DeepSeek-V2 rotates ADJACENT coordinates (its
    // `apply_rotary_pos_emb` permutes `view(d//2, 2).transpose(4, 3)` before
    // `rotate_half`; current transformers rotates `view_as_complex` pairs),
    // while Llama/Qwen2/Qwen3 apply `rotate_half` straight to split halves.
    // Getting this wrong leaves `pos == 0` bit-identical and corrupts every
    // later position, so it is invisible to a first-token-only comparison.
    // DeepSeek-V3/R1 share the V2 lineage and the same convention.
    let family = normalize_model_family(
        v["model_type"]
            .as_str()
            .or_else(|| v["architectures"].get(0).and_then(|a| a.as_str())),
    );
    // Allowlist, NOT a `starts_with("deepseek")` prefix test: the bare
    // `model_type: "deepseek"` is DeepSeek-**V1** / DeepSeekMoE, whose
    // modeling code is a Llama copy (`rotate_half` on split halves, no
    // permute). Those checkpoints are dense-shaped, so they load without
    // complaint and then get silently corrupted at every `pos > 0` — the very
    // bug this flag exists to prevent, with no error path to catch it.
    // Only the MLA lineage (V2/V3/R1/VL2) rotates adjacent coordinates.
    cfg.rope_interleave = ["deepseekv2", "deepseekv3", "deepseekvlv2"]
        .iter()
        .any(|p| family.starts_with(p));
    // MLA (DeepSeek-V2 style): compressed KV + low-rank Q replace q/k/v/o.
    cfg.q_lora_rank = t["q_lora_rank"].as_u64().unwrap_or(0) as usize;
    cfg.kv_lora_rank = t["kv_lora_rank"].as_u64().unwrap_or(0) as usize;
    cfg.qk_nope_head_dim = t["qk_nope_head_dim"].as_u64().unwrap_or(0) as usize;
    cfg.qk_rope_head_dim = t["qk_rope_head_dim"].as_u64().unwrap_or(0) as usize;
    cfg.v_head_dim = t["v_head_dim"].as_u64().unwrap_or(0) as usize;
    if cfg.kv_lora_rank > 0 {
        // MLA: per-head q is (nope + rope); the expanded KV cache is per-head
        // f32. head_dim from hidden/heads would be too small and under-size the
        // q scratch (mla_assemble_q_batched writes nope+rope per head).
        if cfg.qk_nope_head_dim == 0 || cfg.qk_rope_head_dim == 0 || cfg.v_head_dim == 0 {
            panic!(
                "MLA config (kv_lora_rank={}) requires qk_nope_head_dim, qk_rope_head_dim and v_head_dim to be > 0",
                cfg.kv_lora_rank
            );
        }
        cfg.head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        cfg.n_kv_heads = cfg.n_heads;
    }
    // MoE (Qwen2.5-MoE style): num_experts / num_experts_per_tok.
    // DeepSeek-V2 names the routed experts `n_routed_experts`.
    cfg.num_experts = t["num_experts"]
        .as_u64()
        .or_else(|| t["n_routed_experts"].as_u64())
        .unwrap_or(0) as usize;
    cfg.num_experts_per_tok = t["num_experts_per_tok"].as_u64().unwrap_or(0) as usize;
    // Qwen-MoE expert FFN width (moe_intermediate_size); 0 = use intermediate_size.
    cfg.moe_intermediate_size = t["moe_intermediate_size"].as_u64().unwrap_or(0) as usize;
    // DeepSeek-V2 always-routed "shared experts" (`n_shared_experts`), fused
    // into a single `moe_intermediate_size * n_shared_experts`-wide MLP whose
    // output is added to the routed sum. 0 = none (Qwen-MoE).
    cfg.n_shared_experts = t["n_shared_experts"].as_u64().unwrap_or(0) as usize;
    // Top-k weighting convention. HF `MoEGate` renormalizes the selected
    // probabilities when `norm_topk_prob` is set (Qwen-MoE); DeepSeek-V2 sets
    // it false and multiplies by `routed_scaling_factor` instead.
    cfg.moe_norm_topk = v["norm_topk_prob"].as_bool().unwrap_or(true);
    cfg.moe_routed_scale = v["routed_scaling_factor"].as_f64().unwrap_or(1.0) as f32;
    // The router the kernels implement is softmax-scored greedy top-k. HF
    // supports `sigmoid` scoring and `group_limited_greedy` selection
    // (DeepSeek-V3 / Qwen3), which would silently decode garbage, so fail fast
    // instead. DeepSeek-V2-Lite is `softmax` + `greedy` with `n_group: 1`.
    let scoring = t["scoring_func"].as_str().unwrap_or("softmax");
    if scoring != "softmax" {
        panic!("unsupported MoE scoring_func {scoring:?}: only softmax is implemented");
    }
    let topk_method = t["topk_method"].as_str().unwrap_or("greedy");
    if topk_method != "greedy" {
        panic!("unsupported MoE topk_method {topk_method:?}: only greedy is implemented");
    }
    // YaRN RoPE (`rope_scaling.type == "yarn"`, DeepSeek-V2). The kernel takes
    // the raw parameters and reproduces HF's ramp mask + cos/sin
    // `attention_factor`; the `mscale^2` logit correction is derived from
    // `mscale_all_dim` in `Config::attn_scale`.
    if let Some(rs) = t["rope_scaling"].as_object() {
        match rs.get("type").and_then(|t| t.as_str()) {
            Some("yarn") => {
                cfg.rope_yarn_factor = rs["factor"].as_f64().unwrap_or(1.0) as f32;
                cfg.rope_yarn_orig_len =
                    rs["original_max_position_embeddings"].as_u64().unwrap_or(0) as usize;
                cfg.rope_yarn_beta_fast = rs["beta_fast"].as_f64().unwrap_or(32.0) as f32;
                cfg.rope_yarn_beta_slow = rs["beta_slow"].as_f64().unwrap_or(1.0) as f32;
                // HF `MoEGate`-style config defaults: `mscale` falls back to 1
                // and `mscale_all_dim` to 0 (which disables the logit fixup).
                cfg.rope_yarn_mscale = rs["mscale"].as_f64().unwrap_or(1.0) as f32;
                cfg.rope_yarn_mscale_all_dim = rs["mscale_all_dim"].as_f64().unwrap_or(0.0) as f32;
                if !cfg.yarn() {
                    panic!(
                        "rope_scaling.type=yarn requires factor > 1 and \
                         original_max_position_embeddings > 0 (got factor={}, orig={})",
                        cfg.rope_yarn_factor, cfg.rope_yarn_orig_len
                    );
                }
            }
            Some(other) => panic!("unsupported rope_scaling type {other:?}: only yarn"),
            None => {}
        }
    }
    // Qwen3 QK-norm: HF configs express it via `model_type` ("qwen3" /
    // "qwen3_moe") rather than an explicit flag, so default to ON for those
    // and honor an explicit `use_qk_norm` / `qk_norm` key when present.
    cfg.qk_norm = t["model_type"]
        .as_str()
        .or_else(|| v["model_type"].as_str())
        .is_some_and(|mt| mt.starts_with("qwen3"));
    if let Some(qk) = t["use_qk_norm"]
        .as_bool()
        .or_else(|| t["qk_norm"].as_bool())
    {
        cfg.qk_norm = qk;
    }
    // Qwen3.5 (Qwen3.8 family): hybrid full-attention / gated-DeltaNet stack.
    // Like qk_norm above, the family constants are modeling-code facts keyed
    // off model_type: full-attn layers carry a sigmoid attention output gate
    // (doubled q_proj), and RMSNorm weights ship zero-centered (`x * (1+w)`)
    // — the loader assumes both, so they must be set here, not left at the
    // Llama defaults.
    let qwen35 = family.starts_with("qwen35");
    cfg.attn_output_gate = t["attn_output_gate"].as_bool().unwrap_or(qwen35);
    if qwen35 {
        cfg.zero_centered_norm = true;
    }
    if let Some(z) = t["zero_centered_norm"].as_bool() {
        cfg.zero_centered_norm = z;
    }
    if qwen35 {
        // Only the leading `partial_rotary_factor` slice of each head
        // rotates; the tail passes through (Config::attn_rotary_dim).
        cfg.rope_rotary_pct = t["partial_rotary_factor"].as_f64().unwrap_or(0.25) as f32;
        cfg.full_attention_interval = t["full_attention_interval"].as_u64().unwrap_or(4) as usize;
        cfg.gdn_k_heads = t["linear_num_key_heads"].as_u64().unwrap_or(16) as usize;
        cfg.gdn_v_heads = t["linear_num_value_heads"].as_u64().unwrap_or(48) as usize;
        // The GDN kernels implement ONE shared head dim for k and v.
        let kd = t["linear_key_head_dim"].as_u64().unwrap_or(128) as usize;
        let vd = t["linear_value_head_dim"].as_u64().unwrap_or(128) as usize;
        if kd != vd {
            panic!(
                "qwen3_5 with linear_key_head_dim={kd} != linear_value_head_dim={vd}: \
                 a single shared GDN head dim is implemented"
            );
        }
        cfg.gdn_head_dim = kd;
        cfg.gdn_conv_kernel = t["linear_conv_kernel_dim"].as_u64().unwrap_or(4) as usize;
        if !cfg.gdn_enabled() {
            panic!(
                "qwen3_5 config must set linear_num_value_heads > 0 and \
                 full_attention_interval > 0 (GDN is not optional in this family)"
            );
        }
        // The checkpoint spells out the hybrid layout in `layer_types`; the
        // engine derives it from the interval. A mismatch would silently load
        // every layer as the wrong kind, so verify instead of trusting.
        if let Some(types) = t["layer_types"].as_array() {
            if types.len() != cfg.n_layers {
                panic!(
                    "layer_types has {} entries but num_hidden_layers is {}",
                    types.len(),
                    cfg.n_layers
                );
            }
            for (li, ty) in types.iter().enumerate() {
                let expect = if cfg.layer_is_full_attn(li) {
                    "full_attention"
                } else {
                    "linear_attention"
                };
                let got = ty.as_str().unwrap_or_default();
                if got != expect {
                    panic!(
                        "layer_types[{li}] = {got:?} but the interval-derived \
                         pattern says {expect:?}"
                    );
                }
            }
        }
        // `rope_parameters.rope_type` must be the plain default: other
        // scalings would silently decode garbage. (M-RoPE degenerates to
        // standard sequential rope for text-only input, so `default` needs
        // no special handling.)
        if let Some(rt) = t["rope_parameters"]["rope_type"].as_str() {
            assert!(
                rt == "default",
                "unsupported rope_parameters.rope_type {rt:?}: only default"
            );
        }
    }
    // MACH_MOE_GROUPED=0 (default on): batched-MoE decode falls back to the
    // hipBLAS host loop. Parsed HERE, in the server's env-knob area, not in
    // the library (the library reads no MoE env; the field lives on Config).
    cfg.moe_grouped = std::env::var("MACH_MOE_GROUPED")
        .map(|x| x != "0")
        .unwrap_or(true);
    // MACH_STEP_PROFILE=1 (diagnostic): per-layer attention/MoE HIP event
    // bracketing, reported after each decode step.
    cfg.step_profile = std::env::var("MACH_STEP_PROFILE").is_ok_and(|v| v != "0");
    cfg
}

/// Total weight-payload size for the VRAM preflight: a single checkpoint file,
/// or the sum of every `*.safetensors` shard when `path` is a directory of
/// shards (Qwen-8B+ checkpoints ship as 5..65 files). A bare directory's
/// `metadata().len()` is ~0 on Windows, so using it directly would let the
/// preflight pass while the actual upload OOMs. Best-effort: unreadable shard
/// entries are skipped, and a directory with no shards sums to 0.
pub fn model_file_bytes(path: &Path) -> u64 {
    if let Ok(entries) = std::fs::read_dir(path) {
        // Directory of shards: sum every `*.safetensors`; non-shard files and
        // unreadable entries are skipped (best-effort preflight estimate).
        entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum()
    } else {
        // Single checkpoint file (read_dir on a file path fails).
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    }
}

/// Rough device-memory estimate for the preflight: weight file + KV cache +
/// 256MiB scratch margin (+ draft model in spec mode). MLA uses the expanded
/// per-head KV cache (always f32); dense uses the GQA formula with the dtype's
/// element size. Sharded weight files are counted via [`model_file_bytes`];
/// hipBLAS workspace and compiled kernels are not counted; the margin covers
/// today's scenarios.
pub fn estimate_vram(
    cfg: &Config,
    capacity: usize,
    file_bytes: u64,
    draft: Option<(&Config, u64)>,
    fp8: bool,
    q4_device: bool,
    kv_int8: bool,
) -> u64 {
    let kv_elem = if cfg.dtype == ModelDType::F16 { 2 } else { 4 };
    let kv = if cfg.kv_lora_rank > 0 {
        capacity
            * cfg.max_seq_len
            * cfg.n_heads
            * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim)
            * 4
    } else if kv_int8 {
        // Contiguous INT8 KV stores K and V as i8 payloads plus one f32 scale
        // per token/head for each of K and V: 2*head_dim + 2*4 bytes per
        // token/head. MLA and paged combinations are not wired for INT8 and
        // retain their existing estimates above/below.
        capacity * cfg.max_seq_len * cfg.n_kv_heads * (cfg.head_dim * 2 + 8)
    } else {
        capacity * cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim * kv_elem * 2
    };
    // MACH_Q4 (host-side storage): the server loads standard BF16/F16
    // safetensors and quantizes at load; the device holds dequantized f16 =
    // the same bytes as the file. (The x4 device multiplier would only apply
    // to checkpoints already stored as packed int4, which the server never
    // reads.) MACH_Q4_DEVICE keeps the expert pool packed int4 on the device
    // (~0.28x the BF16 file bytes incl. scales) while non-expert weights
    // dequantize to f16 (~1.0x); MoE checkpoints are expert-dominated, so
    // x0.3 is a safe estimate (the 30B measured ~16.5GB device for a 61GB
    // file). FP8 stores packed E4M3 (1 byte/weight) but the device holds
    // dequantized f16 (2 bytes/weight); x2 is the exact device multiplier.
    let weight = if q4_device {
        file_bytes * 3 / 10
    } else if fp8 {
        file_bytes * 2
    } else {
        file_bytes
    };
    // Qwen3.5 hybrid: only the full-attention layers carry a dense KV cache
    // (the GDN layers' "KV" is the recurrent state below), and that state is
    // a device allocation the preflight must count — [capacity, vh, hd, hd]
    // per GDN layer (3.1MB/layer/slot on the 27B: 1.2GB at capacity 8,
    // 9.7GB at 64) plus the small conv windows. Sizing by kd*vd instead of
    // per-head hd*hd would over-count 768x (#112 build-time OOM class).
    let (kv_layers, gdn_layers) = if cfg.gdn_enabled() {
        let full = (0..cfg.n_layers)
            .filter(|&li| cfg.layer_is_full_attn(li))
            .count();
        (full, cfg.n_layers - full)
    } else {
        (cfg.n_layers, 0)
    };
    let gdn_state = if gdn_layers > 0 {
        (capacity as u64)
            * gdn_layers as u64
            * (cfg.gdn_v_heads * cfg.gdn_head_dim * cfg.gdn_head_dim
                + (2 * cfg.gdn_key_dim() + cfg.gdn_value_dim()) * (cfg.gdn_conv_kernel - 1))
                as u64
            * 4
    } else {
        0
    };
    let mut est = weight + (kv * kv_layers) as u64 + gdn_state + (256 << 20);
    if let Some((dcfg, dfb)) = draft {
        let dkv_elem = if dcfg.dtype == ModelDType::F16 { 2 } else { 4 };
        let dkv = if dcfg.kv_lora_rank > 0 {
            capacity
                * dcfg.max_seq_len
                * dcfg.n_heads
                * (dcfg.qk_nope_head_dim + dcfg.qk_rope_head_dim + dcfg.v_head_dim)
                * 4
        } else {
            capacity * dcfg.max_seq_len * dcfg.n_kv_heads * dcfg.head_dim * dkv_elem * 2
        };
        est += dfb + (dkv * dcfg.n_layers) as u64;
    }
    est
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_cfg() -> Config {
        Config::llama(128, 2, 4, 2, 1024, 64)
    }

    fn mla_cfg() -> Config {
        Config::mla(128, 2, 4, 1024, 64, 32, 16, 16, 8, 16)
    }

    #[test]
    fn dense_estimate_includes_weights_kv_and_margin() {
        let cfg = dense_cfg();
        let est = estimate_vram(&cfg, 8, 1_000_000, None, false, false, false);
        // KV (f32) = capacity*max_seq*kv_heads*head_dim*4 per layer.
        let kv =
            (8 * cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim * 4 * 2 * cfg.n_layers) as u64;
        assert_eq!(est, 1_000_000 + kv + (256 << 20));
    }

    #[test]
    fn f16_dense_uses_two_byte_kv() {
        let mut cfg = dense_cfg();
        cfg.dtype = ModelDType::F16;
        let f16 = estimate_vram(&cfg, 8, 0, None, false, false, false);
        cfg.dtype = ModelDType::F32;
        let f32 = estimate_vram(&cfg, 8, 0, None, false, false, false);
        // KV diff = layers * capacity*max_seq*kv_heads*head_dim*(4-2).
        let kv_diff =
            (cfg.n_layers * 8 * cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim * 2 * 2) as u64;
        assert_eq!(
            f32 - f16,
            kv_diff,
            "f32 KV must exceed f16 KV by the elem diff"
        );
    }

    #[test]
    fn int8_dense_accounts_payload_and_scales() {
        let mut cfg = dense_cfg();
        cfg.dtype = ModelDType::F16;
        for cap in [1usize, 8] {
            for max_seq in [64usize, 128] {
                cfg.max_seq_len = max_seq;
                let f16 = estimate_vram(&cfg, cap, 0, None, false, false, false);
                let int8 = estimate_vram(&cfg, cap, 0, None, false, false, true);
                let f16_kv =
                    (cfg.n_layers * cap * max_seq * cfg.n_kv_heads * cfg.head_dim * 2 * 2) as u64;
                let int8_kv =
                    (cfg.n_layers * cap * max_seq * cfg.n_kv_heads * (cfg.head_dim * 2 + 8)) as u64;
                assert_eq!(int8, f16 - f16_kv + int8_kv, "cap={cap} max_seq={max_seq}");
            }
        }
    }
    #[test]
    fn mla_estimate_uses_expanded_per_head_kv() {
        let cfg = mla_cfg();
        // MLA KV/layer is f32: capacity*max_seq*heads*(nope+rope+v_hd)*4.
        let kv = (8
            * cfg.max_seq_len
            * cfg.n_heads
            * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim + cfg.v_head_dim)
            * 4
            * cfg.n_layers) as u64;
        assert_eq!(
            estimate_vram(&cfg, 8, 0, None, false, false, false),
            kv + (256 << 20)
        );
    }

    #[test]
    fn spec_adds_draft_weights_and_kv() {
        let tcfg = dense_cfg();
        let dcfg = dense_cfg();
        let base = estimate_vram(&tcfg, 8, 1_000, None, false, false, false);
        let spec = estimate_vram(&tcfg, 8, 1_000, Some((&dcfg, 500)), false, false, false);
        let dkv =
            (dcfg.n_layers * 8 * dcfg.max_seq_len * dcfg.n_kv_heads * dcfg.head_dim * 4 * 2) as u64;
        assert_eq!(spec - base, 500 + dkv);
    }

    /// Removes the temp file on drop (also on test panic).
    struct TempFile(std::path::PathBuf);
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn parse_json(json: &str) -> Config {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "machserve_cfg_test_{}_{}.json",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, json).unwrap();
        let _guard = TempFile(path.clone());
        config_from_json(&path)
    }

    #[test]
    fn config_parses_dense_defaults() {
        let cfg = parse_json(
            r#"{"hidden_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":1024,"intermediate_size":512,"max_position_embeddings":64}"#,
        );
        assert_eq!(cfg.kv_lora_rank, 0);
        assert_eq!(cfg.q_lora_rank, 0);
        assert_eq!(cfg.num_experts, 0);
        assert_eq!(cfg.head_dim, 32, "dense head_dim = hidden/heads");
        assert_eq!(cfg.n_kv_heads, 2);
    }

    #[test]
    fn config_parses_mla_hyperparams() {
        let cfg = parse_json(
            r#"{"hidden_size":5120,"num_hidden_layers":2,"num_attention_heads":128,"vocab_size":102400,"max_position_embeddings":4096,"q_lora_rank":1536,"kv_lora_rank":512,"qk_nope_head_dim":128,"qk_rope_head_dim":64,"v_head_dim":128}"#,
        );
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.q_lora_rank, 1536);
        assert_eq!(
            cfg.head_dim,
            128 + 64,
            "MLA head_dim must be qk_nope_head_dim + qk_rope_head_dim"
        );
        assert_eq!(cfg.n_kv_heads, 128, "MLA n_kv_heads == n_heads");
        assert_eq!(cfg.num_experts, 0);
    }

    /// The allowlist in `config_from_json` is the ONLY thing that decides which
    /// real checkpoints get the interleaved convention, and getting it wrong is
    /// invisible at `pos == 0`. Pin it in both directions: the MLA lineage must
    /// be interleaved, and everything else — especially the bare
    /// `model_type: "deepseek"` (V1 / DeepSeekMoE, a Llama copy that uses plain
    /// `rotate_half`) — must keep the split-halves default.
    #[test]
    fn rope_interleave_is_allowlisted_per_family() {
        let parse = |model_type: &str| {
            parse_json(&format!(
                r#"{{"model_type":"{model_type}","hidden_size":2048,"num_hidden_layers":2,"num_attention_heads":16,"vocab_size":102400,"max_position_embeddings":4096}}"#
            ))
        };
        for t in ["deepseek_v2", "deepseek_v3", "deepseek_vl_v2"] {
            assert!(
                parse(t).rope_interleave,
                "MLA lineage {t} must use adjacent-pair RoPE"
            );
        }
        // `deepseek` (V1/DeepSeekMoE) is a Llama copy and must NOT be flipped:
        // a prefix test silently corrupts it at every pos > 0.
        for t in [
            "deepseek",
            "llama",
            "qwen2",
            "qwen3",
            "qwen3_moe",
            "deepseek_v1",
        ] {
            assert!(
                !parse(t).rope_interleave,
                "{t} must keep the split-halves default"
            );
        }
    }

    /// `architectures[0]` is the fallback when `model_type` is absent; it must
    /// classify the same way so a checkpoint with only `architectures` is not
    /// silently misrouted.
    #[test]
    fn rope_interleave_falls_back_to_architectures() {
        let cfg = parse_json(
            r#"{"architectures":["DeepseekV2ForCausalLM"],"hidden_size":2048,"num_hidden_layers":2,"num_attention_heads":16,"vocab_size":102400,"max_position_embeddings":4096}"#,
        );
        assert!(
            cfg.rope_interleave,
            "DeepseekV2ForCausalLM must be interleaved via the architectures fallback"
        );
    }

    #[test]
    fn config_parses_explicit_head_dim() {
        // Qwen3-30B-A3B-style: n_heads*head_dim (32*128=4096) is wider than
        // hidden (2048); the explicit head_dim must win over hidden/n_heads.
        let cfg = parse_json(
            r#"{"hidden_size":2048,"num_hidden_layers":48,"num_attention_heads":32,"num_key_value_heads":4,"vocab_size":151936,"intermediate_size":6144,"moe_intermediate_size":768,"num_experts":128,"num_experts_per_tok":8,"head_dim":128,"max_position_embeddings":40960}"#,
        );
        assert_eq!(cfg.head_dim, 128, "explicit head_dim wins");
        assert_eq!(cfg.n_heads * cfg.head_dim, 4096);
        assert_eq!(cfg.n_kv_heads * cfg.head_dim, 512);
        assert_eq!(cfg.num_experts, 128);
        assert_eq!(cfg.expert_size(), 768);
    }

    /// DeepSeek-V2-Lite: `n_routed_experts` (not `num_experts`), shared
    /// experts, `norm_topk_prob: false` + `routed_scaling_factor`, YaRN rope
    /// scaling, and `q_lora_rank: null` (the fused-`q_proj` shape).
    #[test]
    fn config_parses_deepseek_v2_lite() {
        let cfg = parse_json(
            r#"{
              "model_type": "deepseek_v2",
              "hidden_size": 2048,
              "num_hidden_layers": 27,
              "num_attention_heads": 16,
              "num_key_value_heads": 16,
              "vocab_size": 102400,
              "intermediate_size": 10944,
              "moe_intermediate_size": 1408,
              "n_routed_experts": 64,
              "n_shared_experts": 2,
              "num_experts_per_tok": 6,
              "norm_topk_prob": false,
              "routed_scaling_factor": 1.0,
              "scoring_func": "softmax",
              "topk_method": "greedy",
              "first_k_dense_replace": 1,
              "q_lora_rank": null,
              "kv_lora_rank": 512,
              "qk_nope_head_dim": 128,
              "qk_rope_head_dim": 64,
              "v_head_dim": 128,
              "max_position_embeddings": 163840,
              "rope_theta": 10000,
              "rope_scaling": {
                "type": "yarn",
                "factor": 40,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 0.707,
                "mscale_all_dim": 0.707
              }
            }"#,
        );
        assert_eq!(cfg.num_experts, 64, "n_routed_experts -> num_experts");
        assert_eq!(cfg.num_experts_per_tok, 6);
        assert_eq!(cfg.n_shared_experts, 2);
        assert_eq!(cfg.shared_size(), 2 * 1408);
        assert!(!cfg.moe_norm_topk, "norm_topk_prob: false");
        assert_eq!(cfg.moe_routed_scale, 1.0);
        // `q_lora_rank: null` -> 0, so the loader reads the fused q_proj.
        assert_eq!(cfg.q_lora_rank, 0);
        assert_eq!(cfg.kv_lora_rank, 512);
        assert_eq!(cfg.head_dim, 128 + 64);
        assert!(cfg.yarn(), "yarn scaling must engage");
        assert_eq!(cfg.rope_yarn_factor, 40.0);
        assert_eq!(cfg.rope_yarn_orig_len, 4096);
        assert_eq!(cfg.rope_yarn_beta_fast, 32.0);
        assert_eq!(cfg.rope_yarn_beta_slow, 1.0);
        // Both mscales are 0.707, so the cos/sin attention_factor is exactly
        // 1.0; the logit correction is what actually bites.
        assert!((cfg.yarn_attention_factor() - 1.0).abs() < 1e-6);
        let mscale = 0.1 * 0.707 * 40.0f32.ln() + 1.0;
        assert!((cfg.attn_scale(192) - mscale * mscale / 192.0f32.sqrt()).abs() < 1e-5);
        // 163840 would reserve ~90 GB of MLA KV per slot; clamped without
        // MACH_MAX_SEQ.
        assert_eq!(cfg.max_seq_len, 8192);
    }

    /// Qwen3.5 (Qwen3.8-27B): multimodal checkpoints nest the text stack
    /// under `text_config`, theta lives in `rope_parameters`, and the hybrid
    /// GDN hyperparameters are `linear_*` keys. Mirror the real config shape.
    #[test]
    fn config_parses_qwen3_5_nested_text() {
        let cfg = parse_json(
            r#"{
              "model_type": "qwen3_5",
              "architectures": ["Qwen3_5ForConditionalGeneration"],
              "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 128,
                "num_hidden_layers": 8,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "intermediate_size": 512,
                "vocab_size": 1024,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "attn_output_gate": true,
                "full_attention_interval": 4,
                "partial_rotary_factor": 0.25,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "layer_types": ["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"],
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000000}
              },
              "vision_config": {"model_type": "qwen3_5"}
            }"#,
        );
        assert_eq!(cfg.d_model, 128, "text_config.hidden_size");
        assert_eq!(cfg.n_layers, 8);
        assert_eq!(cfg.head_dim, 16);
        assert!(cfg.qk_norm, "qwen3_5_text starts with qwen3");
        assert!(cfg.attn_output_gate);
        assert!(cfg.zero_centered_norm);
        assert_eq!(cfg.rope_theta, 10_000_000.0, "theta from rope_parameters");
        assert_eq!(cfg.rope_rotary_pct, 0.25);
        assert_eq!(cfg.attn_rotary_dim(), 4, "0.25 * 16");
        assert_eq!(cfg.full_attention_interval, 4);
        assert!(cfg.gdn_enabled());
        assert_eq!(cfg.gdn_k_heads, 2);
        assert_eq!(cfg.gdn_v_heads, 4);
        assert_eq!(cfg.gdn_head_dim, 8);
        assert_eq!(cfg.gdn_conv_kernel, 4);
        assert_eq!(cfg.gdn_key_dim(), 16);
        assert_eq!(cfg.gdn_value_dim(), 32);
        assert!(cfg.layer_is_full_attn(3));
        assert!(!cfg.layer_is_full_attn(0));
        assert_eq!(cfg.max_seq_len, 8192, "262144 clamped");
        assert!(!cfg.rope_interleave, "Qwen family pairs half-split");
    }

    /// A pure-text Qwen3.5 checkpoint (flat config, `model_type:
    /// qwen3_5_text`) must classify identically — the family detection must
    /// not depend on the multimodal nesting.
    #[test]
    fn config_parses_qwen3_5_flat_text() {
        let cfg = parse_json(
            r#"{"model_type":"qwen3_5_text","hidden_size":128,"num_hidden_layers":8,"num_attention_heads":4,"vocab_size":1024,"intermediate_size":512,"head_dim":16,"max_position_embeddings":4096,"full_attention_interval":4,"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4}"#,
        );
        assert!(cfg.attn_output_gate);
        assert!(cfg.zero_centered_norm);
        assert!(cfg.gdn_enabled());
        assert_eq!(
            cfg.attn_rotary_dim(),
            4,
            "default partial_rotary_factor 0.25"
        );
    }

    /// `layer_types` disagreeing with the interval must fail loudly: a silent
    /// mismatch would load every layer as the wrong kind.
    #[test]
    #[should_panic(expected = "layer_types[1]")]
    fn config_rejects_layer_types_mismatch() {
        parse_json(
            r#"{"model_type":"qwen3_5","text_config":{"hidden_size":128,"num_hidden_layers":8,"num_attention_heads":4,"vocab_size":1024,"head_dim":16,"max_position_embeddings":4096,"full_attention_interval":4,"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":8,"linear_value_head_dim":8,"linear_conv_kernel_dim":4,"layer_types":["linear_attention","full_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"]}}"#,
        );
    }

    #[test]
    fn config_parses_moe_hyperparams() {
        let cfg = parse_json(
            r#"{"hidden_size":1024,"num_hidden_layers":2,"num_attention_heads":8,"num_key_value_heads":2,"vocab_size":151936,"intermediate_size":512,"max_position_embeddings":2048,"num_experts":64,"num_experts_per_tok":8,"moe_intermediate_size":256}"#,
        );
        assert_eq!(cfg.num_experts, 64);
        assert_eq!(cfg.num_experts_per_tok, 8);
        assert_eq!(cfg.moe_intermediate_size, 256);
        assert_eq!(
            cfg.expert_size(),
            256,
            "Qwen-MoE experts use moe_intermediate_size"
        );
        assert_eq!(cfg.kv_lora_rank, 0);
    }

    #[test]
    #[should_panic(expected = "requires qk_nope_head_dim")]
    fn config_rejects_mla_missing_dims() {
        parse_json(
            r#"{"hidden_size":5120,"num_hidden_layers":2,"num_attention_heads":128,"vocab_size":102400,"max_position_embeddings":4096,"kv_lora_rank":512}"#,
        );
    }

    #[test]
    fn model_file_bytes_sums_shard_dir() {
        let dir =
            std::env::temp_dir().join(format!("mach_preflight_shard_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // A single checkpoint file: metadata length, not a dir sum.
        let single = dir.join("model.safetensors");
        std::fs::write(&single, vec![0u8; 1234]).unwrap();
        assert_eq!(model_file_bytes(&single), 1234);
        // A directory of shards: every *.safetensors counted, non-shards not.
        std::fs::write(
            dir.join("model-00001-of-00002.safetensors"),
            vec![0u8; 1000],
        )
        .unwrap();
        std::fs::write(
            dir.join("model-00002-of-00002.safetensors"),
            vec![0u8; 2000],
        )
        .unwrap();
        std::fs::write(dir.join("config.json"), vec![0u8; 999]).unwrap();
        assert_eq!(model_file_bytes(&dir), 4234); // 1234 single + 1000 + 2000 shards
        // A directory with no shards sums to 0, not the dir's own size.
        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert_eq!(model_file_bytes(&empty), 0);
        std::fs::write(empty.join("config.json"), vec![0u8; 64]).unwrap();
        assert_eq!(model_file_bytes(&empty), 0);
        // Missing path falls back to 0.
        assert_eq!(model_file_bytes(&dir.join("nope.safetensors")), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn q4_and_q4_device_weight_terms() {
        let cfg = dense_cfg();
        let base = estimate_vram(&cfg, 8, 1_000_000, None, false, false, false);
        // MACH_Q4 loads standard BF16/F16 files and quantizes at load: the
        // device holds f16 = the same bytes as the file, so the weight term
        // is unchanged vs dense.
        let q4 = estimate_vram(&cfg, 8, 1_000_000, None, false, false, false);
        assert_eq!(q4, base);
        // MACH_Q4_DEVICE keeps the expert pool packed on the device: the
        // weight term is 0.3x the file bytes.
        let q4d = estimate_vram(&cfg, 8, 1_000_000, None, false, true, false);
        assert_eq!(q4d, base - 700_000);
    }

    /// Hybrid (Qwen3.5) VRAM accounting: KV only for the full-attention
    /// layers, plus the GDN recurrent state sized PER HEAD (`vh * hd * hd` +
    /// conv windows) — the #112 over-allocation class (kd*vd = 768x on the
    //  27B) must not come back through the estimate.
    #[test]
    fn gdn_hybrid_counts_state_and_skips_gdn_kv() {
        let mut cfg = mach_model::Config::qwen3_5(64, 5, 4, 2, 16, 176, 97, 4096, 2, 4, 8, 4);
        cfg.dtype = ModelDType::F32;
        let cap = 8usize;
        let got = estimate_vram(&cfg, cap, 0, None, false, false, false);
        // qwen35_small shape: interval 4 over 5 layers -> full-attn ONLY
        // layer 3 (li where (li+1)%4==0); layers 0,1,2,4 are GDN.
        let full_layers = (0..cfg.n_layers)
            .filter(|&li| cfg.layer_is_full_attn(li))
            .count();
        assert_eq!(full_layers, 1);
        let kv_per_layer = cap * cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim * 2 * 4;
        let per_gdn_layer = cfg.gdn_v_heads * cfg.gdn_head_dim * cfg.gdn_head_dim
            + (2 * cfg.gdn_key_dim() + cfg.gdn_value_dim()) * (cfg.gdn_conv_kernel - 1);
        let want = (kv_per_layer * full_layers
            + cap * (cfg.n_layers - full_layers) * per_gdn_layer * 4
            + (256 << 20)) as u64;
        assert_eq!(got, want);

        let int8 = estimate_vram(&cfg, cap, 0, None, false, false, true);
        let int8_kv_per_layer = cap * cfg.max_seq_len * cfg.n_kv_heads * (cfg.head_dim * 2 + 8);
        let want_int8 = (int8_kv_per_layer * full_layers
            + cap * (cfg.n_layers - full_layers) * per_gdn_layer * 4
            + (256 << 20)) as u64;
        assert_eq!(
            int8, want_int8,
            "INT8 KV accounting must cover only full-attention layers"
        );
    }

    #[test]
    fn fp8_scales_weight_term_by_two() {
        let cfg = dense_cfg();
        let base = estimate_vram(&cfg, 8, 1_000_000, None, false, false, false);
        let fp8 = estimate_vram(&cfg, 8, 1_000_000, None, true, false, false);
        // FP8 stores E4M3 (1 byte/weight) but the device holds dequantized f16
        // (2 bytes/weight): the weight term must be x2, or the preflight can
        // pass while the upload OOMs (regression).
        assert_eq!(fp8 - base, 1_000_000);
    }
}

/// Resolve `preprocessor_config.json` beside a checkpoint directory or shard.
pub fn resolve_preprocessor_path(checkpoint: &Path) -> Option<PathBuf> {
    let direct = checkpoint.join("preprocessor_config.json");
    if direct.exists() {
        return Some(direct);
    }
    checkpoint
        .parent()
        .map(|p| p.join("preprocessor_config.json"))
        .filter(|p| p.exists())
}

#[cfg(test)]
mod vision_path_tests {
    use super::*;

    #[test]
    fn resolves_preprocessor_config_for_dir_and_shard() {
        let dir = std::env::temp_dir().join(format!("mach-pre-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("preprocessor_config.json"), b"{}").unwrap();
        let want = dir.join("preprocessor_config.json");
        assert_eq!(resolve_preprocessor_path(&dir), Some(want.clone()));
        let shard = dir.join("model-00001-of-00002.safetensors");
        std::fs::write(&shard, b"x").unwrap();
        assert_eq!(resolve_preprocessor_path(&shard), Some(want));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
