//! Decode-slice benchmark: eager per-token latency on the GPU.
//!
//! Run with:  cargo run -p mach-model --release --features hip --example decode_bench
#[cfg(feature = "hip")]
use mach_kernel_sys::hip;
#[cfg(feature = "hip")]
use mach_model::model::GpuModel;
#[cfg(feature = "hip")]
use mach_model::{Config, Weights};
#[cfg(feature = "hip")]
use std::time::Instant;

#[cfg(feature = "hip")]
fn main() {
    let h = hip::hip().expect("HIP runtime");
    let n = hip::device_count().expect("device count");
    assert!(n > 0, "no HIP device");
    let name = hip::device_name(0).expect("device name");
    println!("device: {name} (count={n})");

    let cfg = Config::small();
    let w = Weights::random(&cfg, 2026).expect("weights");
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
    println!("weights: {:.1} MB", w.byte_size() as f64 / 1e6);

    let mut model = GpuModel::new(h, cfg, &w).expect("model");

    let n_tokens = 200usize;

    // ---- Full decode step (input copy + kernels + logits readback) ----
    model.reset_state().expect("reset");
    for _ in 0..10 {
        model.decode_step(1).expect("warmup");
    }
    model.reset_state().expect("reset");
    let t0 = Instant::now();
    for i in 0..n_tokens {
        model.decode_step((i % 977) as u32).expect("eager step");
    }
    let eager_full_us = t0.elapsed().as_micros() as f64 / n_tokens as f64;

    // ---- Launch-only path (no per-token sync/readback; stream-ordered) ----
    model.reset_state().expect("reset");
    let t2 = Instant::now();
    for i in 0..n_tokens {
        model.step_eager((i % 977) as u32).expect("eager launch");
    }
    let eager_launch_us = t2.elapsed().as_micros() as f64 / n_tokens as f64;

    println!("\nfull step (input copy + kernels + logits readback):");
    println!("  eager : {eager_full_us:8.1} us/token");

    println!("\nlaunch-only path (no per-token sync/readback):");
    println!("  eager : {eager_launch_us:8.1} us/token");
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!(
        "decode_bench requires the `hip` feature: cargo run -p mach-model --features hip --example decode_bench"
    );
}
