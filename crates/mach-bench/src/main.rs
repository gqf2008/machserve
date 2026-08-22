//! Host-side micro-benchmarks.
//!
//! These measure the *floor* of per-op host overhead: registry lookup and
//! dispatch routing. GPU kernel time is excluded by design — the point is to
//! quantify what Rust saves relative to a Python hot loop (no interpreter, no
//! object churn, deterministic dispatch).

use mach_engine::{Allocation, DType, Device, Shape};
use mach_kernel::KernelRegistry;
use mach_kernel::buffer::Buffer;
use mach_kernel::ops;
use std::time::Instant;

/// Runs `f` `iters` times and returns nanoseconds per call.
fn bench_ns<F: FnMut()>(label: &str, iters: usize, mut f: F) {
    // Warmup.
    for _ in 0..(iters / 10).max(1) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed().as_nanos() as f64;
    let per = elapsed / iters as f64;
    println!("{label:<44} {per:>10.2} ns/op  ({elapsed:.0} ns total / {iters} iters)");
}

/// Simulated engine dispatch: registry lookup + family routing + no-op.
fn dispatch(reg: &KernelRegistry, family: &str, name: &str) -> Result<(), mach_engine::Error> {
    let k = reg
        .get(family, name)
        .map_err(|e| mach_engine::Error::InvalidArgument(e.to_string()))?;
    match k.family() {
        "attention" | "gemm" | "moe" | "quant" | "sampling" => Ok(()),
        other => Err(mach_engine::Error::InvalidArgument(format!(
            "unknown family {other}"
        ))),
    }
}

fn main() {
    let reg = KernelRegistry::new();
    ops::register_cpu_reference(&reg).expect("register CPU reference kernels");

    let n = 1_000_000;
    println!("=== machserve host dispatch floor (CPU, single thread) ===");

    bench_ns("registry.get(attention/cpu.reference)", n, || {
        let _ = std::hint::black_box(reg.get("attention", "cpu.reference").map(|k| k.caps()));
    });

    bench_ns("simulated engine dispatch (lookup+routing)", n, || {
        let _ = std::hint::black_box(dispatch(&reg, "attention", "cpu.reference"));
    });

    // A tiny buffer construction, as the engine would do per-op.
    let alloc = Allocation {
        pool_id: 0,
        offset: 0,
        bytes: 4096,
    };
    bench_ns("buffer metadata construction", n, || {
        std::hint::black_box(Buffer::new(
            Device::Cuda(0),
            DType::F32,
            Shape::new([1024]),
            alloc,
        ));
    });

    println!("\nNote: GPU kernel time is NOT included; this is the host floor only.");
}
