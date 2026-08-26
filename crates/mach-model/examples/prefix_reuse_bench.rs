//! CPU-only reference benchmark: cross-request prefix (KV-cache) reuse.
//!
//! This is the **CPU reference** leg of the prefix-reuse A/B plan: it drives
//! the scheduling + reuse stack (`CpuEngine`: scheduler FSM + reuse planner +
//! prefix KV cache) end-to-end with a tiny random model, so the reuse
//! accounting (reused / computed / savings) can be validated and pinned before
//! the GPU on-real-hardware A/B leg is run in a later batch.
//!
//! Workload: N requests that share an 8-token "system prompt" and differ only
//! in a per-request random tail token. The first request prefills the whole
//! prompt; every later request should reuse the cached 8-token prefix and only
//! compute its delta (the tail), so the expected savings fraction is
//!   reused / total_prompt_tokens = 8*(N-1) / (9*N).
//!
//! Run (CPU-only, no HIP/GPU needed):
//!   cargo run -p mach-model --example prefix_reuse_bench
//!
//! Env:
//!   MACH_BENCH_REQUESTS (default 8)  number of requests sharing the prefix
//!   MACH_BENCH_TOKENS   (default 8)  max_new decode tokens per request
//!   MACH_SEED           (default 42) seed for weights + tail tokens (fixed by
//!                                    default so runs are reproducible)

use mach_model::cpu_engine::CpuEngine;
use mach_model::{Config, Weights};

/// Shared 8-token "system prompt" every request starts with.
const SYSTEM_PROMPT: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

fn main() {
    let requests: usize = env_or("MACH_BENCH_REQUESTS", 8);
    let max_new: usize = env_or("MACH_BENCH_TOKENS", 8);
    let seed: u64 = env_or("MACH_SEED", 42);
    assert!(requests >= 1, "MACH_BENCH_REQUESTS must be >= 1");
    assert!(max_new >= 1, "MACH_BENCH_TOKENS must be >= 1");

    let cfg = Config::tiny();
    let w = Weights::random(&cfg, seed).expect("random weights");

    println!(
        "=== {} | prefix-reuse CPU reference benchmark ===",
        env!("CARGO_PKG_NAME")
    );
    println!(
        "config: d_model={} layers={} heads={} kv_heads={} head_dim={} max_seq={} vocab={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.max_seq_len,
        cfg.vocab_size
    );
    println!(
        "workload: system_prompt={:?} requests={requests} tail_len=1 max_new={max_new} seed={seed}",
        SYSTEM_PROMPT
    );

    // One distinct random (in-vocab) tail token per request, deduped by the LCG.
    let tails = tail_tokens(seed, requests, cfg.vocab_size as u64);

    // capacity=4 concurrent slots, 32-page KV pool, 4 tokens per page.
    let mut engine = CpuEngine::new(cfg, w, 4, 32, 4);
    for &tail in &tails {
        let mut prompt = SYSTEM_PROMPT.to_vec();
        prompt.push(tail);
        engine.add(&prompt, max_new).expect("queue request");
    }
    engine.step_until_done().expect("run engine to idle");
    assert!(engine.is_idle(), "engine must drain all requests");

    let finished = engine.finished();
    assert_eq!(finished.len(), requests, "all requests finish");
    let stats = *engine.stats();

    // Per-request reuse/compute accounting (from `finished()`).
    println!("\n--- per-request accounting ---");
    println!(
        "{:>4} | {:>6} | {:>6} | {:>8} | {:>9}",
        "id", "prompt", "reused", "computed", "generated"
    );
    for f in finished {
        println!(
            "{:>4} | {:>6} | {:>6} | {:>8} | {:>9}",
            f.id,
            f.total_prompt_tokens,
            f.reused_tokens,
            f.computed_tokens,
            f.generated.len()
        );
    }

    let n = finished.len() as f64;
    let avg_reused = finished.iter().map(|f| f.reused_tokens).sum::<usize>() as f64 / n;
    let avg_computed = finished.iter().map(|f| f.computed_tokens).sum::<usize>() as f64 / n;

    let savings_pct = stats.savings() * 100.0;
    println!("\n--- summary (human) ---");
    println!("requests                    : {}", stats.requests);
    println!(
        "prompt tokens total         : {}",
        stats.prompt_tokens_total
    );
    println!(
        "prompt tokens reused        : {}",
        stats.prompt_tokens_reused
    );
    println!(
        "prompt tokens computed      : {}",
        stats.prompt_tokens_computed
    );
    println!("savings                     : {savings_pct:.1}%");
    println!("decoded tokens              : {}", stats.decoded_tokens);
    println!("avg reused per request      : {avg_reused:.2}");
    println!("avg computed per request    : {avg_computed:.2}");

    println!("\n--- summary (machine readable) ---");
    println!("requests={}", stats.requests);
    println!("prompt_tokens_total={}", stats.prompt_tokens_total);
    println!("prompt_tokens_reused={}", stats.prompt_tokens_reused);
    println!("prompt_tokens_computed={}", stats.prompt_tokens_computed);
    println!("savings_pct={savings_pct:.1}");
    println!("decoded_tokens={}", stats.decoded_tokens);
    println!("avg_reused_per_request={avg_reused:.2}");
    println!("avg_computed_per_request={avg_computed:.2}");
    println!("steps={}", stats.steps);
    println!("note: CPU reference only; GPU on-device A/B is a later batch");
}

/// Parses a `MACH_*`-style env var, falling back to a default when unset or
/// unparseable (keeps the benchmark runnable out of the box).
fn env_or<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Deterministic, distinct in-vocab tail tokens via an LCG (deduped so every
/// request differs from the others beyond the shared prefix).
fn tail_tokens(seed: u64, n: usize, vocab: u64) -> Vec<u32> {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let tok = ((state >> 33) % vocab) as u32;
        if !out.contains(&tok) {
            out.push(tok);
        }
    }
    out
}
