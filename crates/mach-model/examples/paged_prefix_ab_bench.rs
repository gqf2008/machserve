//! GPU A/B benchmark: cross-request prefix (KV-cache) reuse (#78 C6).
//!
//! The **on-device leg** of the prefix-reuse plan (the CPU reference leg is
//! `prefix_reuse_bench.rs`): same workload through the contiguous engine and
//! the paged engine on the same GPU, so the reuse payoff (TTFT / prompt-token
//! savings) can be measured end-to-end per `docs/benchmark-protocol.md`.
//!
//! Workload: N requests sharing a full-page "system prompt" and differing only
//! in a per-request tail token. The contiguous engine recomputes every prompt
//! token; the paged engine materializes the shared page once and every later
//! request prefills only its delta, so
//!   expected savings = (N-1) * page_tokens / (N * prompt_len)
//!
//! Run (needs ROCm + GPU):
//!   cargo run -p mach-model --release --features hip --example paged_prefix_ab_bench
//!
//! Env:
//!   MACH_BENCH_REQUESTS (default 5)  requests sharing the prefix
//!   MACH_BENCH_TOKENS   (default 8)  max_new decode tokens per request
//!   MACH_BENCH_TPP      (default 64) KV page size in tokens
//!   MACH_SEED           (default 42) seed for weights + tail tokens

#[cfg(feature = "hip")]
use mach_kernel_sys::hip;
#[cfg(feature = "hip")]
use mach_model::continuous::ContinuousModel;
#[cfg(feature = "hip")]
use mach_model::sampling::SamplingParams;
#[cfg(feature = "hip")]
use mach_model::{Config, Weights};
#[cfg(feature = "hip")]
use std::sync::Arc;
#[cfg(feature = "hip")]
use std::time::Instant;

/// Parses a `MACH_*`-style env var, falling back to a default when unset or
/// unparseable (keeps the benchmark runnable out of the box).
#[cfg(feature = "hip")]
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
#[cfg(feature = "hip")]
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

/// Runs the workload through one engine with **staggered admission** (the
/// production arrival pattern): request `i+1` is admitted as soon as request
/// `i` emits its first token, so the paged engine's materialization gate lets
/// later requests actually reuse the shared page. Returns (wall_ms,
/// per-request first-token latency ms, prompt tokens computed).
#[cfg(feature = "hip")]
#[allow(clippy::too_many_arguments)]
fn run_one(
    hip: Arc<hip::Hip>,
    cfg: Config,
    w: &Weights,
    paged: bool,
    tpp: usize,
    prefix: &[u32],
    tails: &[u32],
    max_new: usize,
) -> (f64, Vec<f64>, usize) {
    let n = tails.len();
    let mut eng = if paged {
        ContinuousModel::with_paged_prefill_rows(hip, cfg, w, n, n, tpp).expect("paged engine")
    } else {
        ContinuousModel::with_prefill_rows(hip, cfg, w, n, n).expect("engine")
    };
    let t0 = Instant::now();
    let mut first_emit: Vec<Option<Instant>> = vec![None; n];
    let mut outputs: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut done = 0usize;
    let mut admitted_at = Vec::with_capacity(n);
    // Staggered admission: request `i+1` is admitted only after request `i`
    // emitted its first token, so the paged engine's materialization gate has
    // fired before the next request arrives (the real server arrival shape).
    for i in 0..n {
        let mut prompt = prefix.to_vec();
        prompt.push(tails[i]);
        eng.add(
            &prompt,
            max_new,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .expect("add request");
        admitted_at.push(Instant::now());
        if i + 1 < n {
            // Run until THIS request emits its first token, then admit next.
            while first_emit[i].is_none() {
                let produced = eng.step().expect("engine step");
                for (id, tok) in produced {
                    let j = (id as usize) - 1;
                    if first_emit[j].is_none() {
                        first_emit[j] = Some(Instant::now());
                    }
                    outputs[j].push(tok);
                    if eng.is_done(id) {
                        done += 1;
                    }
                }
            }
        }
    }
    // Drain the remaining generation.
    while done < n {
        let produced = eng.step().expect("engine step");
        for (id, tok) in produced {
            let j = (id as usize) - 1;
            if first_emit[j].is_none() {
                first_emit[j] = Some(Instant::now());
            }
            outputs[j].push(tok);
            if eng.is_done(id) {
                done += 1;
            }
        }
    }
    let wall = t0.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms: Vec<f64> = (0..n)
        .map(|i| {
            let f = first_emit[i].expect("request emits a token");
            (f - admitted_at[i]).as_secs_f64() * 1000.0
        })
        .collect();
    let computed = eng
        .paged_reuse_stats()
        .map(|s| s.prompt_tokens - s.reused_tokens)
        .unwrap_or_else(|| prefix.len() * n + n);
    (wall, ttft_ms, computed)
}

#[cfg(feature = "hip")]
fn main() {
    let requests: usize = env_or("MACH_BENCH_REQUESTS", 5);
    let max_new: usize = env_or("MACH_BENCH_TOKENS", 8);
    let tpp: usize = env_or("MACH_BENCH_TPP", 64);
    let seed: u64 = env_or("MACH_SEED", 42);
    assert!(requests >= 1, "MACH_BENCH_REQUESTS must be >= 1");

    let h = hip::hip().expect("ROCm runtime (MACH_HIP_PATH?)");
    assert!(
        hip::device_count().is_ok_and(|n| n > 0),
        "benchmark needs a GPU"
    );

    let cfg = Config::tiny(); // max_seq 256; tpp 64 -> 4 pages/seq
    assert!(
        cfg.max_seq_len.is_multiple_of(tpp),
        "MACH_BENCH_TPP must divide max_seq_len {}",
        cfg.max_seq_len
    );
    let w = Weights::random(&cfg, seed).expect("random weights");
    let prefix: Vec<u32> = (0..tpp as u32).map(|i| (i * 37 + 3) % 1024 + 1).collect();
    let tails = tail_tokens(seed ^ 0x5EED, requests, cfg.vocab_size as u64);

    println!("=== paged prefix-reuse A/B (GPU) ===");
    println!(
        "config: d_model={} layers={} heads={} kv_heads={} head_dim={} max_seq={} vocab={} tpp={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.max_seq_len,
        cfg.vocab_size,
        tpp
    );
    println!(
        "workload: system_prompt_tokens={} requests={} prompt_len={} max_new={} seed={}",
        prefix.len(),
        requests,
        prefix.len() + 1,
        max_new,
        seed
    );
    let expected_pct =
        (requests - 1) as f64 * tpp as f64 / (requests as f64 * (prefix.len() + 1) as f64) * 100.0;
    println!("expected savings: {expected_pct:.1}% (one shared page, delta-only tails)");

    // Warm-up: compile all kernels (process-wide hiprtc cache) AND pay the
    // one-time hipBLAS/driver lazy init by running a real step, so neither
    // measured run below carries startup overhead.
    let mut warm =
        ContinuousModel::with_paged_prefill_rows(h.clone(), cfg, &w, 1, 1, tpp).expect("warmup");
    let wid = warm
        .add(
            &[7, 9],
            1,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .expect("warmup request");
    while !warm.is_done(wid) {
        warm.step().expect("warmup step");
    }
    drop(warm);

    // Contiguous engine, then paged (both post-warmup, same workload).
    let (wall_c, ttft_c, computed_c) =
        run_one(h.clone(), cfg, &w, false, tpp, &prefix, &tails, max_new);
    let (wall_p, ttft_p, computed_p) =
        run_one(h.clone(), cfg, &w, true, tpp, &prefix, &tails, max_new);

    let avg = |v: &[f64], skip: usize| -> f64 {
        let tail: Vec<f64> = v.iter().skip(skip).copied().collect();
        tail.iter().sum::<f64>() / tail.len().max(1) as f64
    };
    let tpot_c = wall_c / (requests * max_new) as f64;
    let tpot_p = wall_p / (requests * max_new) as f64;
    let prompt_saved = computed_c - computed_p;

    println!("\n--- contiguous engine ---");
    println!(
        "wall_ms={wall_c:.1} ttft_ms={:?} tpot_ms={tpot_c:.2}",
        ttft_c
    );
    println!("prompt tokens computed={computed_c} (full recompute; all {requests} requests)");
    println!("\n--- paged engine ---");
    println!(
        "wall_ms={wall_p:.1} ttft_ms={:?} tpot_ms={tpot_p:.2}",
        ttft_p
    );
    println!("prompt tokens computed={computed_p} (shared page materialized once; reuse stats)");

    println!("\n--- summary (machine readable) ---");
    println!("requests={requests}");
    println!("prompt_tokens_per_request={}", prefix.len() + 1);
    println!("wall_ms_contiguous={wall_c:.1}");
    println!("wall_ms_paged={wall_p:.1}");
    println!("wall_speedup={:.2}x", wall_c / wall_p);
    println!("ttft_first_ms_contiguous={:.1}", ttft_c[0]);
    println!("ttft_first_ms_paged={:.1}", ttft_p[0]);
    println!("ttft_avg_later_ms_contiguous={:.1}", avg(&ttft_c, 1));
    println!("ttft_avg_later_ms_paged={:.1}", avg(&ttft_p, 1));
    println!(
        "ttft_later_speedup={:.2}x",
        avg(&ttft_c, 1) / avg(&ttft_p, 1)
    );
    println!("tpot_ms_contiguous={tpot_c:.2}");
    println!("tpot_ms_paged={tpot_p:.2}");
    println!("prompt_tokens_computed_contiguous={computed_c}");
    println!("prompt_tokens_computed_paged={computed_p}");
    println!("prompt_tokens_saved={prompt_saved}");
    println!(
        "savings_pct={:.1}",
        prompt_saved as f64 / computed_c as f64 * 100.0
    );
    println!("expected_savings_pct={expected_pct:.1}");
    println!(
        "note: wall clock per engine on the same GPU, same weights/request distribution (docs/benchmark-protocol.md)"
    );
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!(
        "paged_prefix_ab_bench requires the `hip` feature: cargo run -p mach-model --release --features hip --example paged_prefix_ab_bench"
    );
}
