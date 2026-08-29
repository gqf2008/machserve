//! Batched MoE decode microbenchmark (#70 P1/P3/P4): measures end-to-end
//! decode throughput of a synthetic batched-MoE model on the GPU, i.e. the
//! layer loop that used to serialize on per-expert host GEMMs.
//!
//! `MACH_MOE_GROUPED=0` disables the device-side grouped GEMV path (falls
//! back to the hipBLAS host loop + counts readback) — the A/B switch for
//! this benchmark and an ops lever.
//!
//! Run: cargo run -p mach-model --release --features hip --example moe_batched_bench

#[cfg(feature = "hip")]
use mach_kernel_sys::hip;
#[cfg(feature = "hip")]
use mach_model::batched::BatchedModel;
#[cfg(feature = "hip")]
use mach_model::config::ModelDType;
#[cfg(feature = "hip")]
use mach_model::{Config, Weights};
#[cfg(feature = "hip")]
use std::time::Instant;

#[cfg(feature = "hip")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let h = match hip::hip() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("HIP unavailable: {e}");
            return Ok(());
        }
    };
    if hip::device_count().map(|n| n <= 0).unwrap_or(true) {
        eprintln!("no HIP device");
        return Ok(());
    }

    let mut cfg = Config::small();
    cfg.dtype = ModelDType::F16;
    cfg.num_experts = 64;
    cfg.num_experts_per_tok = 8;
    cfg.moe_intermediate_size = 256; // routed-layer width (Qwen-MoE style)
    let batch = 32usize;
    let steps = 64usize;

    let w = Weights::random(&cfg, 77)?;
    let mut model = BatchedModel::new(h, cfg, &w, batch)?;

    // Warmup (allocator + module caches settle).
    let mut tokens: Vec<u32> = (1..=batch as u32).collect();
    for t in &mut tokens {
        *t = (*t * 31 % cfg.vocab_size as u32).max(1);
    }
    for _ in 0..8 {
        tokens = model.decode_step(&tokens)?;
    }

    let t0 = Instant::now();
    for _ in 0..steps {
        tokens = model.decode_step(&tokens)?;
    }
    let dt = t0.elapsed();
    let per_step = dt.as_secs_f64() * 1e3 / steps as f64;
    println!(
        "config: d={} layers={} experts={} topk={} batch={} dtype=F16 grouped={}",
        cfg.d_model,
        cfg.n_layers,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        batch,
        std::env::var("MACH_MOE_GROUPED")
            .map(|v| v != "0")
            .unwrap_or(true)
    );
    println!(
        "decode: {per_step:.3} ms/step, {:.0} tok/s (steps {steps})",
        batch as f64 / per_step * 1e3
    );
    Ok(())
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!("moe_batched_bench requires the `hip` feature");
}
