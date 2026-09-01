//! Per-kernel shape benchmark for the 30B decode path: times gemv_f16 at the
//! four attention projection shapes and the Q4 grouped pair at the routed
//! MoE shape. Reports BOTH modes per kernel:
//! - "sync":   launch + device sync per iteration — latency floor (one
//!   kernel in flight, launch overhead included);
//! - "stream": N back-to-back launches with one final sync — the in-stream
//!   throughput the engine actually sees.
//!
//!   cargo run -p mach-model --release --features hip --example gemv_shape_bench
#[cfg(feature = "hip")]
use mach_kernel_sys::hip::{self};
#[cfg(feature = "hip")]
use std::sync::Arc;

fn main() {
    #[cfg(feature = "hip")]
    run();
    #[cfg(not(feature = "hip"))]
    eprintln!("gemv_shape_bench requires the `hip` feature");
}

#[cfg(feature = "hip")]
fn run() {
    let Ok(h) = hip::hip() else {
        eprintln!("no HIP runtime");
        return;
    };
    if hip::device_count().map(|n| n <= 0).unwrap_or(true) {
        eprintln!("no HIP device");
        return;
    }
    let k = Arc::new(mach_model::kernels::HipKernels::new(h.clone()).expect("HipKernels"));

    // 30B decode shapes.
    let d = 2048usize;
    let nq = 4096usize; // 32 heads * 128
    let nkv = 512usize; // 4 heads * 128
    let einter = 768usize;
    let topk = 8usize;
    let batch = 1usize;
    let rows = batch * topk; // routed rows

    let iters = 200usize;

    let time = |label: &str, bytes: usize, f_stream: &mut dyn FnMut()| {
        // Warmup (launches + final sync).
        for _ in 0..10 {
            f_stream();
        }
        k.sync().unwrap();

        // Sync mode: launch + device sync per iteration (latency floor).
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            f_stream();
            k.sync().unwrap();
        }
        let sync_us = t0.elapsed().as_secs_f64() / iters as f64 * 1e6;

        // Stream mode: back-to-back launches, one final sync (in-stream).
        let t1 = std::time::Instant::now();
        for _ in 0..iters {
            f_stream();
        }
        k.sync().unwrap();
        let stream_us = t1.elapsed().as_secs_f64() / iters as f64 * 1e6;
        let stream_bw = bytes as f64 / (stream_us / 1e6) / 1e9;

        println!(
            "{label:26} sync {sync_us:8.1} us | stream {stream_us:9.1} us ({stream_bw:7.1} GB/s, {:6.2} MiB)",
            bytes as f64 / (1024.0 * 1024.0)
        );
    };

    // ---- attention GEMV shapes ----
    for (name, n) in [
        ("q (4096x2048)", nq),
        ("k (512x2048)", nkv),
        ("v (512x2048)", nkv),
        ("o (2048x2048)", d),
    ] {
        let x = hip::malloc(&h, batch * d * 4).unwrap() as *mut f32;
        let w = hip::malloc(&h, n * d * 2).unwrap() as *mut u16;
        let out = hip::malloc(&h, batch * n * 4).unwrap() as *mut f32;
        let kg = k.clone();
        let mut f = Box::new(move || {
            kg.launch_gemv_f16(out, x, w, n as i32, d as i32, batch as i32)
                .unwrap();
        });
        time(name, n * d * 2 + batch * d * 4, &mut f);
        hip::free(&h, x as *mut core::ffi::c_void).unwrap();
        hip::free(&h, w as *mut core::ffi::c_void).unwrap();
        hip::free(&h, out as *mut core::ffi::c_void).unwrap();
    }

    // ---- Q4 grouped MoE shapes ----
    let ebytes = einter * d;
    let wg_bytes = topk * ebytes / 2;
    let wg_sbytes = topk * (ebytes / 32) * 4;
    let wg_q = hip::malloc(&h, wg_bytes).unwrap() as *mut u8;
    let wg_s = hip::malloc(&h, wg_sbytes).unwrap() as *mut f32;
    let wu_q = hip::malloc(&h, wg_bytes).unwrap() as *mut u8;
    let wu_s = hip::malloc(&h, wg_sbytes).unwrap() as *mut f32;
    let wd_bytes = topk * d * einter / 2;
    let wd_s = hip::malloc(&h, topk * (d * einter / 32) * 4).unwrap() as *mut f32;
    let wd_q = hip::malloc(&h, wd_bytes).unwrap() as *mut u8;
    let gate = hip::malloc(&h, rows * einter * 4).unwrap() as *mut f32;
    let up = hip::malloc(&h, rows * einter * 4).unwrap() as *mut f32;
    let down = hip::malloc(&h, rows * d * 4).unwrap() as *mut f32;
    let ids = hip::malloc(&h, rows * 4).unwrap() as *mut i32;
    let xrow = hip::malloc(&h, batch * d * 4).unwrap() as *mut f32;
    let ehrow = hip::malloc(&h, rows * einter * 4).unwrap() as *mut f32;
    let ids_v: Vec<i32> = (0..rows as i32).collect();
    hip::memcpy(
        &h,
        ids as *mut core::ffi::c_void,
        ids_v.as_ptr() as *const core::ffi::c_void,
        rows * 4,
        hip::HIP_MEMCPY_HOST_TO_DEVICE,
    )
    .unwrap();

    // gate_up_q4 reads 2 Q4 tensors (topk rows x ebytes/2 each) + scales.
    let kg = k.clone();
    let mut gate_up = Box::new(move || {
        kg.launch_moe_grouped_gate_up_q4(
            xrow,
            ids,
            wg_q,
            wg_s,
            wu_q,
            wu_s,
            gate,
            up,
            rows as i32,
            einter as i32,
            d as i32,
            topk as i32,
        )
        .unwrap();
    });
    // down_q4 reads 1 Q4 tensor (topk rows x d*einter/2 each) + scales.
    let kg = k.clone();
    let mut down_q4 = Box::new(move || {
        kg.launch_moe_grouped_down_q4(
            xrow,
            ids,
            wd_q,
            wd_s,
            down,
            rows as i32,
            d as i32,
            einter as i32,
        )
        .unwrap();
    });

    time(
        "gate_up_q4 (8x768x2048)",
        2 * topk * ebytes / 2,
        &mut gate_up,
    );
    time("down_q4 (8x2048x768)", topk * d * einter / 2, &mut down_q4);

    for p in [
        wg_q as *mut core::ffi::c_void,
        wg_s as *mut core::ffi::c_void,
        wu_q as *mut core::ffi::c_void,
        wu_s as *mut core::ffi::c_void,
        wd_q as *mut core::ffi::c_void,
        wd_s as *mut core::ffi::c_void,
        gate as *mut core::ffi::c_void,
        up as *mut core::ffi::c_void,
        down as *mut core::ffi::c_void,
        ids as *mut core::ffi::c_void,
        xrow as *mut core::ffi::c_void,
        ehrow as *mut core::ffi::c_void,
    ] {
        hip::free(&h, p).unwrap();
    }
}
