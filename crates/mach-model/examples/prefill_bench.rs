//! Reproducible A/B benchmark for full-layer double-buffered prefill.
//!
//! Compares three modes on the SAME model + prompt:
//!   full     - all experts GPU-resident, no per-layer H2D (sequential ceiling);
//!   buffered - experts streamed per layer with double buffering (the next MoE
//!              layer's experts are prefetched on a separate stream while the
//!              current layer computes);
//!   cpu      - P1 offload baseline: per-layer D2H + CPU MoE (non-buffered).
//!
//! Metrics: TTFT (long-context chunked prefill) and TPOT (avg per generated
//! token after prefill). Env-driven for reproducibility; greedy token streams
//! are cross-checked (buffered/cpu must match full-resident when logits agree).
//!
//! Run on a GPU box (needs a MoE checkpoint, e.g. qwen3-moe-tiny):
//!   MACH_MODEL=model.safetensors MACH_CONFIG=config.json MACH_CTX=2048 \
//!     cargo run -p mach-model --release --features hip --example prefill_bench

#[cfg(feature = "hip")]
use mach_model::{Config, Weights};
#[cfg(feature = "hip")]
use std::sync::Arc;
#[cfg(feature = "hip")]
use std::time::Instant;

#[cfg(feature = "hip")]
fn main() {
    use mach_kernel_sys::hip;
    use mach_model::loader::load_safetensors;
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

    let cfg = config_from_json(&cfg_path);
    assert!(
        cfg.num_experts > 0,
        "benchmark requires a MoE checkpoint (num_experts > 0)"
    );
    let ctx: usize = env_or("MACH_CTX", 2048);
    let rows: usize = env_or("MACH_PREFILL_ROWS", 512);
    let decode: usize = env_or("MACH_DECODE", 16);
    assert!(ctx >= 1 && rows >= 1 && decode >= 1);

    let w: Weights = load_safetensors(&model_path, &cfg, true).expect("load weights");
    let hip = hip::hip().expect("HIP runtime");
    assert!(hip::device_count().expect("devices") > 0, "no HIP device");

    // Deterministic prompt (in-vocab ids).
    let seq: Vec<u32> = (0..ctx)
        .map(|i| (i as u32 * 131 + 7) % cfg.vocab_size as u32)
        .collect();

    let mode_filter = std::env::var("MACH_BENCH_MODE").unwrap_or_else(|_| "all".into());
    let want = |m: &str| mode_filter == "all" || mode_filter == m;
    let full = want("full").then(|| run_mode(&hip, cfg, &w, Mode::Full, rows, &seq, decode));
    let buffered =
        want("buffered").then(|| run_mode(&hip, cfg, &w, Mode::Buffered, rows, &seq, decode));
    let cpu = want("cpu").then(|| run_mode(&hip, cfg, &w, Mode::Cpu, rows, &seq, decode));

    println!();
    println!("=== double-buffered prefill benchmark (MoE) ===");
    println!(
        "model: {model_name} | d_model={} layers={} experts={} topk={} moe_inter={} | ctx={ctx} rows={rows} decode={decode}",
        cfg.d_model,
        cfg.n_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.expert_size()
    );
    println!("mode           | TTFT(ms) | TPOT(ms/tok) | tok/s | greedy tokens match full");
    if let Some(full) = &full {
        println!(
            "full          | {:9.2} | {:11.2} | {:7.1} | -",
            full.ttft_ms,
            full.tpot_ms,
            1000.0 / full.tpot_ms
        );
    }
    if let (Some(full), Some(buffered)) = (&full, &buffered) {
        println!(
            "buffered      | {:9.2} | {:11.2} | {:7.1} | {}",
            buffered.ttft_ms,
            buffered.tpot_ms,
            1000.0 / buffered.tpot_ms,
            buffered.tokens == full.tokens
        );
    } else if let Some(buffered) = &buffered {
        println!(
            "buffered      | {:9.2} | {:11.2} | {:7.1} | -",
            buffered.ttft_ms,
            buffered.tpot_ms,
            1000.0 / buffered.tpot_ms
        );
    }
    if let (Some(full), Some(cpu)) = (&full, &cpu) {
        println!(
            "cpu-offload   | {:9.2} | {:11.2} | {:7.1} | {}",
            cpu.ttft_ms,
            cpu.tpot_ms,
            1000.0 / cpu.tpot_ms,
            cpu.tokens == full.tokens
        );
    } else if let Some(cpu) = &cpu {
        println!(
            "cpu-offload   | {:9.2} | {:11.2} | {:7.1} | -",
            cpu.ttft_ms,
            cpu.tpot_ms,
            1000.0 / cpu.tpot_ms
        );
    }
    println!("note: TTFT = chunked prefill of ctx tokens (rows/chunk); TPOT = avg decode");
    println!("      step after prefill. buffered streams the next layer's experts on a");
    println!("      separate stream; full keeps them resident (ceiling); cpu computes the");
    println!("      MoE on the CPU per layer (P1 non-buffered baseline).");
    println!();
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!(
        "prefill_bench requires the hip feature: \
         cargo run -p mach-model --release --features hip --example prefill_bench"
    );
}

#[cfg(feature = "hip")]
enum Mode {
    Full,
    Buffered,
    Cpu,
}

#[cfg(feature = "hip")]
struct Meas {
    ttft_ms: f64,
    tpot_ms: f64,
    /// Greedy output tokens (last prefill token + decode tokens) for parity.
    tokens: Vec<u32>,
}

#[cfg(feature = "hip")]
fn run_mode(
    hip: &Arc<mach_kernel_sys::hip::Hip>,
    cfg: Config,
    w: &Weights,
    mode: Mode,
    rows: usize,
    seq: &[u32],
    decode: usize,
) -> Meas {
    use mach_model::batched::BatchedModel;
    use mach_model::sampling::SamplingParams;
    let mut model = match mode {
        Mode::Full => BatchedModel::with_rows(Arc::clone(hip), cfg, w, 1, rows).unwrap(),
        Mode::Buffered => {
            BatchedModel::with_prefill_buffer(Arc::clone(hip), cfg, w, 1, rows).unwrap()
        }
        Mode::Cpu => BatchedModel::with_expert_slots(Arc::clone(hip), cfg, w, 1, rows, 0).unwrap(),
    };

    // Warmup: a short prefill to settle lazy hipBLAS init and cached state.
    let warm = seq.len().min(rows).max(1);
    prefill(&mut model, &seq[..warm], 0).unwrap();
    model.reset_state().unwrap();

    // TTFT: chunked prefill of the full prompt.
    let t0 = Instant::now();
    let mut last = 0u32;
    for (ci, chunk) in seq.chunks(rows).enumerate() {
        last = prefill(&mut model, chunk, ci * rows).unwrap();
    }
    let ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // TPOT: decode `decode` tokens after the prefill.
    let mut tokens = vec![last];
    let t1 = Instant::now();
    for i in 0..decode {
        let lens = (seq.len() + i) as u32;
        let out = model
            .decode_step_explicit(
                &[tokens[tokens.len() - 1]],
                &[lens],
                &[0],
                &mut [SamplingParams::greedy(1)],
                &[Vec::new()],
                &[Vec::new()],
            )
            .unwrap();
        tokens.push(out.0[0]);
    }
    let tpot_ms = t1.elapsed().as_secs_f64() * 1000.0 / decode as f64;

    Meas {
        ttft_ms,
        tpot_ms,
        tokens,
    }
}

/// Runs one chunked-prefill forward over `chunk` rows (positions `base..`) that
/// all share KV slot 0; returns the greedy token of the last row.
#[cfg(feature = "hip")]
fn prefill(
    model: &mut mach_model::batched::BatchedModel,
    chunk: &[u32],
    base: usize,
) -> Result<u32, mach_model::Error> {
    use mach_model::sampling::SamplingParams;
    let n = chunk.len();
    let lens: Vec<u32> = (0..n).map(|i| (base + i) as u32).collect();
    let slots = vec![0u32; n];
    let mut params = vec![SamplingParams::greedy(1); n];
    let out = model.decode_step_explicit(
        chunk,
        &lens,
        &slots,
        &mut params,
        &vec![Vec::new(); n],
        &vec![Vec::new(); n],
    )?;
    Ok(out.0[n - 1])
}

#[cfg(feature = "hip")]
fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
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
        .min(8192) as usize;
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
    cfg.intermediate_size = inter;
    cfg.rms_eps = eps;
    cfg.rope_theta = theta;
    cfg.num_experts = ne;
    cfg.num_experts_per_tok = topk;
    cfg.moe_intermediate_size = v["moe_intermediate_size"].as_u64().unwrap_or(0) as usize;
    cfg.qk_norm = v["use_qk_norm"].as_bool().unwrap_or(false);
    cfg
}
