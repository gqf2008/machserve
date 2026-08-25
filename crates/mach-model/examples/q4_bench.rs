//! Reproducible A/B benchmark: f16 vs storage-Q4 weights on a real
//! checkpoint, measuring host RAM, TTFT, TPOT and the GPU logits gap.
//!
//! Q4 is a *storage* format: host weights stay packed int4 (~4x smaller than
//! f32) and are dequantized to f16 per tensor during upload, so the device
//! compute path is identical to f16 — the only expected difference is the
//! int4 quantization error in the weights.
//!
//! Run (needs a real checkpoint, e.g. PrimeIntellect/qwen3-moe-tiny):
//!   cargo run -p mach-model --release --features hip --example q4_bench
//!
//! Env: MACH_MODELS (default ".models"), MACH_MODEL (default
//! "model.safetensors"), MACH_CONFIG (default "config.json"),
//! MACH_Q4 (unset = run both legs + A/B, 0 = f16 only, 1 = Q4 only),
//! MACH_BATCH (default 8), MACH_BENCH_TOKENS (default 32),
//! MACH_PROMPT_LEN (default 128), MACH_CAPACITY (default 4).

#[cfg(feature = "hip")]
use mach_kernel_sys::hip;
#[cfg(feature = "hip")]
use mach_model::continuous::ContinuousModel;
#[cfg(feature = "hip")]
use mach_model::sampling::SamplingParams;
#[cfg(feature = "hip")]
use mach_model::{Config, Weights, WeightsQ4};
#[cfg(feature = "hip")]
use std::sync::Arc;
#[cfg(feature = "hip")]
use std::time::Instant;

#[cfg(feature = "hip")]
fn main() {
    use mach_model::config::ModelDType;
    use mach_model::loader::{load_safetensors, load_safetensors_q4};
    use std::path::PathBuf;

    let root = PathBuf::from(std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into()));
    let model_name = std::env::var("MACH_MODEL").unwrap_or_else(|_| "model.safetensors".into());
    let config_name = std::env::var("MACH_CONFIG").unwrap_or_else(|_| "config.json".into());
    let model_path = root.join(&model_name);
    let cfg_path = root.join(&config_name);
    assert!(
        model_path.exists(),
        "missing {model_path:?} (set MACH_MODEL)"
    );
    assert!(cfg_path.exists(), "missing {cfg_path:?}");

    let batch: usize = std::env::var("MACH_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let tokens: usize = std::env::var("MACH_BENCH_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    let prompt_len: usize = std::env::var("MACH_PROMPT_LEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let capacity: usize = std::env::var("MACH_CAPACITY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let q4_mode: i32 = std::env::var("MACH_Q4")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(-1);

    let mut cfg = config_from_json(&cfg_path);
    cfg.dtype = ModelDType::F16;
    let model_bytes = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);

    let hip = hip::hip().expect("HIP runtime");
    assert!(hip::device_count().expect("devices") > 0, "no HIP device");
    let base_ram = host_ram_bytes();
    let gib = |b: f64| b / (1024.0 * 1024.0 * 1024.0);

    println!("=== Q4 vs f16 benchmark ===");
    println!(
        "model: {model_name} ({:.2} GiB file) | d_model={} layers={} heads={} kv={} vocab={} experts={} topk={} moe_inter={}",
        gib(model_bytes as f64),
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.vocab_size,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.expert_size()
    );
    println!(
        "host RAM baseline (after HIP init): {:.2} GiB",
        gib(base_ram as f64)
    );

    // ---- f16 leg (f32 host weights -> f16 device compute) ----
    let mut f16: Option<Meas> = None;
    if q4_mode != 1 {
        let t0 = Instant::now();
        let w: Weights = load_safetensors(&model_path, &cfg, true).expect("load f32 weights");
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let host_ram_mb = (host_ram_bytes().saturating_sub(base_ram)) as f64 / 1e6;
        println!(
            "\n[f16] loaded f32 host weights in {load_ms:.0} ms; host RAM delta {:.2} GiB",
            gib(host_ram_mb * 1e6)
        );

        let ttft_ms = bench_ttft_f16(&hip, cfg, &w, prompt_len, capacity);
        let (tpot_ms, logits) = bench_tpot_f16(&hip, cfg, &w, batch, tokens);
        f16 = Some(Meas {
            label: "f16".into(),
            host_ram_mb,
            ttft_ms,
            tpot_ms,
            logits,
        });
        drop(w);
    }

    // ---- Q4 leg (packed int4 host weights -> f16 device compute) ----
    let mut q4: Option<Meas> = None;
    if q4_mode != 0 {
        let t0 = Instant::now();
        let wq4: WeightsQ4 = load_safetensors_q4(&model_path, &cfg, true).expect("load q4 weights");
        let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let host_ram_mb = (host_ram_bytes().saturating_sub(base_ram)) as f64 / 1e6;
        println!(
            "\n[q4] loaded packed-int4 host weights in {load_ms:.0} ms; host RAM delta {:.2} GiB",
            gib(host_ram_mb * 1e6)
        );

        let ttft_ms = bench_ttft_q4(&hip, cfg, &wq4, prompt_len, capacity);
        let (tpot_ms, logits) = bench_tpot_q4(&hip, cfg, &wq4, batch, tokens);
        q4 = Some(Meas {
            label: "q4".into(),
            host_ram_mb,
            ttft_ms,
            tpot_ms,
            logits,
        });
        drop(wq4);
    }

    println!("\n=== A/B summary (f16 compute path, 7900 XTX) ===");
    println!(
        "leg  | host RAM after load | TTFT({prompt_len} tok, cap {capacity}) | TPOT(batch {batch}) | seq tok/s"
    );
    for m in [&f16, &q4].into_iter().flatten() {
        let seq_tok_s = 1000.0 / (m.tpot_ms / batch as f64);
        println!(
            "{:<4} | {:16.2} MiB | {:25.2} ms | {:18.3} ms/step | {:10.0}",
            m.label, m.host_ram_mb, m.ttft_ms, m.tpot_ms, seq_tok_s
        );
    }
    if f16.is_some() && q4.is_some() {
        println!(
            "note: in-process A/B host RAM for the 2nd leg may include allocator-retained pages from the 1st;"
        );
        println!("      use MACH_Q4=0 / MACH_Q4=1 single-leg runs for clean host-RAM numbers.");
    }
    if let (Some(a), Some(b)) = (&f16, &q4) {
        let max = max_abs_diff(&a.logits, &b.logits);
        let scale = a
            .logits
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(1e-9);
        let same = argmax(&a.logits) == argmax(&b.logits);
        println!("\nGPU logits (same fixed input stream, final step):");
        println!("  max|f16 - q4| = {max:.6} (scale {scale:.3}) | greedy argmax match: {same}");
        println!("  -> Q4 only changes storage precision; compute runs the identical f16 path.");
    }
}

#[cfg(feature = "hip")]
struct Meas {
    label: String,
    /// Working-set delta vs the post-HIP baseline, in MiB.
    host_ram_mb: f64,
    ttft_ms: f64,
    tpot_ms: f64,
    logits: Vec<f32>,
}

#[cfg(feature = "hip")]
fn bench_ttft_f16(
    hip: &Arc<hip::Hip>,
    cfg: Config,
    w: &Weights,
    prompt_len: usize,
    capacity: usize,
) -> f64 {
    let prompt: Vec<u32> = (0..prompt_len).map(|i| (i % 977) as u32).collect();
    let mut eng =
        ContinuousModel::with_prefill_rows(hip.clone(), cfg, w, capacity, capacity).unwrap();
    let id = eng
        .add(
            &prompt,
            16,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    let t = Instant::now();
    while eng.generated(id).is_empty() {
        eng.step().unwrap();
    }
    t.elapsed().as_secs_f64() * 1000.0
}

#[cfg(feature = "hip")]
fn bench_ttft_q4(
    hip: &Arc<hip::Hip>,
    cfg: Config,
    w: &WeightsQ4,
    prompt_len: usize,
    capacity: usize,
) -> f64 {
    let prompt: Vec<u32> = (0..prompt_len).map(|i| (i % 977) as u32).collect();
    let mut eng =
        ContinuousModel::with_prefill_rows_q4(hip.clone(), cfg, w, capacity, capacity).unwrap();
    let id = eng
        .add(
            &prompt,
            16,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    let t = Instant::now();
    while eng.generated(id).is_empty() {
        eng.step().unwrap();
    }
    t.elapsed().as_secs_f64() * 1000.0
}

#[cfg(feature = "hip")]
fn bench_tpot_f16(
    hip: &Arc<hip::Hip>,
    cfg: Config,
    w: &Weights,
    batch: usize,
    tokens: usize,
) -> (f64, Vec<f32>) {
    use mach_model::batched::BatchedModel;
    let mut bm = BatchedModel::new(hip.clone(), cfg, w, batch).unwrap();
    let seq: Vec<u32> = (0..batch).map(|i| (i % 977) as u32).collect();
    bm.reset_state().unwrap();
    for _ in 0..3 {
        bm.decode_step(&seq).unwrap();
    }
    bm.reset_state().unwrap();
    let t = Instant::now();
    for _ in 0..tokens {
        bm.decode_step(&seq).unwrap();
    }
    let tpot_ms = t.elapsed().as_secs_f64() * 1000.0 / tokens as f64;
    let logits = bm.read_logits().unwrap();
    (tpot_ms, logits)
}

#[cfg(feature = "hip")]
fn bench_tpot_q4(
    hip: &Arc<hip::Hip>,
    cfg: Config,
    w: &WeightsQ4,
    batch: usize,
    tokens: usize,
) -> (f64, Vec<f32>) {
    use mach_model::batched::BatchedModel;
    let mut bm = BatchedModel::from_q4(hip.clone(), cfg, w, batch).unwrap();
    let seq: Vec<u32> = (0..batch).map(|i| (i % 977) as u32).collect();
    bm.reset_state().unwrap();
    for _ in 0..3 {
        bm.decode_step(&seq).unwrap();
    }
    bm.reset_state().unwrap();
    let t = Instant::now();
    for _ in 0..tokens {
        bm.decode_step(&seq).unwrap();
    }
    let tpot_ms = t.elapsed().as_secs_f64() * 1000.0 / tokens as f64;
    let logits = bm.read_logits().unwrap();
    (tpot_ms, logits)
}

#[cfg(feature = "hip")]
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(feature = "hip")]
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Current process host RAM (Windows working set / Linux VmRSS) in bytes.
#[cfg(all(windows, feature = "hip"))]
fn host_ram_bytes() -> u64 {
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
        private_usage: usize,
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(
            process: *mut core::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
    }
    unsafe {
        let mut c = std::mem::MaybeUninit::<ProcessMemoryCounters>::zeroed();
        if GetProcessMemoryInfo(
            GetCurrentProcess(),
            c.as_mut_ptr(),
            std::mem::size_of::<ProcessMemoryCounters>() as u32,
        ) == 0
        {
            return 0;
        }
        c.assume_init().working_set_size as u64
    }
}

#[cfg(all(unix, not(target_os = "macos"), feature = "hip"))]
fn host_ram_bytes() -> u64 {
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                if let Ok(kb) = rest.trim().trim_end_matches("kB").trim().parse::<u64>() {
                    return kb * 1024;
                }
            }
        }
    }
    0
}

#[cfg(all(
    not(any(windows, all(unix, not(target_os = "macos")))),
    feature = "hip"
))]
fn host_ram_bytes() -> u64 {
    0
}

#[cfg(feature = "hip")]
fn config_from_json(path: &std::path::Path) -> Config {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read config"))
            .expect("parse config");
    let hidden = v["hidden_size"].as_u64().unwrap_or(896) as usize;
    let layers = v["num_hidden_layers"].as_u64().unwrap_or(24) as usize;
    let heads = v["num_attention_heads"].as_u64().unwrap_or(14) as usize;
    let kv = v["num_key_value_heads"].as_u64().unwrap_or(heads as u64) as usize;
    let vocab = v["vocab_size"].as_u64().unwrap_or(151936) as usize;
    let inter = v["intermediate_size"].as_u64().unwrap_or(4 * hidden as u64) as usize;
    let max_seq = v["max_position_embeddings"]
        .as_u64()
        .unwrap_or(2048)
        .min(2048) as usize;
    let eps = v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    let theta = v["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let ne = v["num_experts"]
        .as_u64()
        .or_else(|| v["n_routed_experts"].as_u64())
        .unwrap_or(0) as usize;
    let topk = v["num_experts_per_tok"]
        .as_u64()
        .or_else(|| v["top_k"].as_u64())
        .unwrap_or(0) as usize;
    let mut cfg = Config::llama(hidden, layers, heads, kv, vocab, max_seq);
    // Some configs (e.g. Qwen3-30B-A3B) ship an explicit `head_dim` that
    // differs from hidden/n_heads (q/o width = n_heads*head_dim is wider
    // than hidden). Honor it, or the loader under-sizes q/o projections.
    if let Some(hd) = v["head_dim"].as_u64() {
        cfg.head_dim = hd as usize;
    }
    cfg.intermediate_size = inter;
    cfg.rms_eps = eps;
    cfg.rope_theta = theta;
    cfg.num_experts = ne;
    cfg.num_experts_per_tok = topk;
    cfg.moe_intermediate_size = v["moe_intermediate_size"].as_u64().unwrap_or(0) as usize;
    cfg.qk_norm = v["use_qk_norm"].as_bool().unwrap_or(false);
    cfg
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!("q4_bench requires the `hip` feature");
}
