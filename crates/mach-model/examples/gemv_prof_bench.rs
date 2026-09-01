//! In-kernel occupancy/timing profiler for the 30B decode hot kernels.
//!
//! RGP cannot trace hiprtc module-launch kernels on Windows
//! (ROCm/rocm-systems#395), so the four hot kernels carry an optional
//! per-block profiling out-param instead (see the `prof` contract on
//! GEMV_F16 in kernels.rs): each block's thread 0 records globaltimer ns
//! stamps [entry, loop_done, end] and the main loop's clock64 cycles.
//!
//! This example launches each kernel once (with the prof buffer) amid
//! back-to-back in-stream iterations, reads the buffer back, and reports:
//! - span / busy / achieved block-level parallelism (busy / span);
//! - main-loop fraction of block time (load loop vs staging + reduce tail);
//! - block duration p50/p90/max and the tail (time after 90% blocks done);
//! - effective globaltimer-derived bandwidth over the span.
//!
//!   cargo run -p mach-model --release --features hip --example gemv_prof_bench
#[cfg(feature = "hip")]
use mach_kernel_sys::hip::{self};

fn main() {
    #[cfg(feature = "hip")]
    run();
    #[cfg(not(feature = "hip"))]
    eprintln!("gemv_prof_bench requires the `hip` feature");
}

/// One block's profile record (must match the kernel-side layout).
#[cfg(feature = "hip")]
#[derive(Clone, Copy, Default)]
struct Rec {
    entry_c: u64,  // clock64 at block entry (per-SIMD cycles)
    loop_c: u64,   // clock64 after the main load loop
    end_c: u64,    // clock64 at block end
    rt_entry: u64, // globaltimer ticks (10 ns) — sampled blocks only (id % 16 == 0)
    rt_end: u64,
}

#[cfg(feature = "hip")]
struct Stats {
    blocks: usize,
    span_us: f64,
    busy_us: f64,
    par: f64,
    loop_frac: f64,
    dur_p50_us: f64,
    dur_p90_us: f64,
    dur_max_us: f64,
    tail_us: f64,
    ghz: f64,
}

#[cfg(feature = "hip")]
fn analyze(recs: &[Rec]) -> Option<Stats> {
    const NS_PER_TICK: f64 = 10.0; // gfx1100 globaltimer = 100 MHz (calibrated)
    let v: Vec<&Rec> = recs
        .iter()
        .filter(|r| r.end_c > r.entry_c && r.entry_c > 0)
        .collect();
    if v.is_empty() {
        return None;
    }
    // Calibrate cycles -> ns from the sampled blocks (both clocks present).
    let mut ghz: Vec<f64> = v
        .iter()
        .filter(|r| r.rt_end > r.rt_entry && r.rt_entry > 0)
        .map(|r| (r.end_c - r.entry_c) as f64 / ((r.rt_end - r.rt_entry) as f64 * NS_PER_TICK))
        .collect();
    if ghz.is_empty() {
        return None;
    }
    ghz.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ghz = ghz[ghz.len() / 2]; // median shader clock, GHz
    let cyc_to_us = |c: u64| c as f64 / ghz / 1e3;

    // Global timeline from the sampled blocks only.
    let smp: Vec<&&Rec> = v
        .iter()
        .filter(|r| r.rt_end > r.rt_entry && r.rt_entry > 0)
        .collect();
    let t0 = smp.iter().map(|r| r.rt_entry).min().unwrap();
    let t1 = smp.iter().map(|r| r.rt_end).max().unwrap();
    let span_ns = (t1 - t0) as f64 * NS_PER_TICK;
    // busy = sum of ALL block durations (cycles -> ns via calibration).
    let busy_ns: f64 = v.iter().map(|r| (r.end_c - r.entry_c) as f64 / ghz).sum();
    let loop_c: f64 = v
        .iter()
        .map(|r| (r.loop_c.saturating_sub(r.entry_c)) as f64)
        .sum();
    let total_c: f64 = v.iter().map(|r| (r.end_c - r.entry_c) as f64).sum();
    let mut durs_us: Vec<f64> = v.iter().map(|r| cyc_to_us(r.end_c - r.entry_c)).collect();
    durs_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| durs_us[((durs_us.len() - 1) as f64 * p) as usize];
    // Tail: globaltimer span remaining once 90% of sampled blocks ended.
    let mut ends: Vec<u64> = smp.iter().map(|r| r.rt_end).collect();
    ends.sort_unstable();
    let end90 = ends[(ends.len() as f64 * 0.9) as usize];
    let tail_ns = (t1 - end90) as f64 * NS_PER_TICK;
    Some(Stats {
        blocks: v.len(),
        span_us: span_ns / 1e3,
        busy_us: busy_ns / 1e3,
        par: busy_ns / span_ns.max(1.0),
        loop_frac: loop_c / total_c.max(1.0),
        dur_p50_us: pct(0.5),
        dur_p90_us: pct(0.9),
        dur_max_us: pct(1.0),
        tail_us: tail_ns / 1e3,
        ghz,
    })
}

#[cfg(feature = "hip")]
fn report(label: &str, bytes: usize, s: &Stats) {
    let bw = bytes as f64 / (s.span_us * 1e-6) / 1e9;
    println!(
        "{label:22} blocks {:5} | span {:7.1} us ({:6.1} GB/s) | busy {:8.1} us | par {:6.1} | loop {:4.0}% | dur p50/p90/max {:5.1}/{:5.1}/{:6.1} us | tail {:6.1} us | {:4.2} GHz",
        s.blocks,
        s.span_us,
        bw,
        s.busy_us,
        s.par,
        s.loop_frac * 100.0,
        s.dur_p50_us,
        s.dur_p90_us,
        s.dur_max_us,
        s.tail_us,
        s.ghz,
    );
}

/// Zero the prof buffer, then run `iters` back-to-back launches with the
/// instrumented one in the MIDDLE (prof armed) so it runs at steady-state
/// clocks with warm L2 and a saturated pipeline — a lone launch after a
/// draining sync runs at idle clocks and reads ~2x slow.
#[cfg(feature = "hip")]
fn profiled(
    h: &hip::Hip,
    k: &mach_model::kernels::HipKernels,
    label: &str,
    bytes: usize,
    n_blocks: usize,
    iters: usize,
    mut launch: impl FnMut(*mut u64),
) {
    let prof = hip::malloc(h, n_blocks * 5 * 8).unwrap() as *mut u64;
    let zeros = vec![0u64; n_blocks * 5];
    hip::memcpy(
        h,
        prof as *mut core::ffi::c_void,
        zeros.as_ptr() as *const core::ffi::c_void,
        n_blocks * 5 * 8,
        hip::HIP_MEMCPY_HOST_TO_DEVICE,
    )
    .unwrap();
    for i in 0..iters {
        launch(if i == iters / 2 {
            prof
        } else {
            std::ptr::null_mut()
        });
    }
    k.sync().unwrap();
    let mut host = vec![0u64; n_blocks * 5];
    hip::memcpy(
        h,
        host.as_mut_ptr() as *mut core::ffi::c_void,
        prof as *const core::ffi::c_void,
        n_blocks * 5 * 8,
        hip::HIP_MEMCPY_DEVICE_TO_HOST,
    )
    .unwrap();
    let recs: Vec<Rec> = host
        .chunks_exact(5)
        .map(|c| Rec {
            entry_c: c[0],
            loop_c: c[1],
            end_c: c[2],
            rt_entry: c[3],
            rt_end: c[4],
        })
        .collect();
    match analyze(&recs) {
        Some(s) => report(label, bytes, &s),
        None => println!("{label:22} no prof data (all-zero buffer)"),
    }
    hip::free(h, prof as *mut core::ffi::c_void).unwrap();
}

#[cfg(feature = "hip")]
#[allow(clippy::too_many_lines)]
fn run() {
    let Ok(h) = hip::hip() else {
        eprintln!("no HIP runtime");
        return;
    };
    if hip::device_count().map(|n| n <= 0).unwrap_or(true) {
        eprintln!("no HIP device");
        return;
    }
    let k = mach_model::kernels::HipKernels::new(h.clone()).expect("HipKernels");

    // 30B decode shapes (same as gemv_shape_bench).
    let d = 2048usize;
    let nq = 4096usize;
    let nkv = 512usize;
    let einter = 768usize;
    let topk = 8usize;
    let batch = 1usize;
    let rows = batch * topk;
    let iters = 50usize;
    let wpb = 256 / 32; // warps per block in gemv_f16 launchers

    // ---- gemv_f16 at the four projection shapes ----
    for (name, n) in [
        ("q (4096x2048)", nq),
        ("k (512x2048)", nkv),
        ("v (512x2048)", nkv),
        ("o (2048x2048)", d),
    ] {
        let x = hip::malloc(&h, batch * d * 4).unwrap() as *mut f32;
        let w = hip::malloc(&h, n * d * 2).unwrap() as *mut u16;
        let out = hip::malloc(&h, batch * n * 4).unwrap() as *mut f32;
        let blocks = n.div_ceil(wpb) * batch;
        profiled(
            &h,
            &k,
            name,
            n * d * 2 + batch * d * 4,
            blocks,
            iters,
            |prof| {
                k.launch_gemv_f16(out, x, w, n as i32, d as i32, batch as i32, prof)
                    .unwrap();
            },
        );
        hip::free(&h, x as *mut core::ffi::c_void).unwrap();
        hip::free(&h, w as *mut core::ffi::c_void).unwrap();
        hip::free(&h, out as *mut core::ffi::c_void).unwrap();
    }

    // ---- fused QKV ----
    {
        let rows_qkv = nq + 2 * nkv;
        let x = hip::malloc(&h, batch * d * 4).unwrap() as *mut f32;
        let wq = hip::malloc(&h, nq * d * 2).unwrap() as *mut u16;
        let wk = hip::malloc(&h, nkv * d * 2).unwrap() as *mut u16;
        let wv = hip::malloc(&h, nkv * d * 2).unwrap() as *mut u16;
        let q = hip::malloc(&h, batch * nq * 4).unwrap() as *mut f32;
        let kb = hip::malloc(&h, batch * nkv * 4).unwrap() as *mut f32;
        let vb = hip::malloc(&h, batch * nkv * 4).unwrap() as *mut f32;
        let blocks = rows_qkv.div_ceil(wpb) * batch;
        let bytes = (nq + 2 * nkv) * d * 2 + batch * d * 4;
        profiled(&h, &k, "qkv fused", bytes, blocks, iters, |prof| {
            k.launch_gemv_f16_qkv(
                x,
                wq,
                wk,
                wv,
                q,
                kb,
                vb,
                nq as i32,
                nkv as i32,
                d as i32,
                batch as i32,
                prof,
            )
            .unwrap();
        });
        for p in [x, q, kb, vb] {
            hip::free(&h, p as *mut core::ffi::c_void).unwrap();
        }
        for p in [wq, wk, wv] {
            hip::free(&h, p as *mut core::ffi::c_void).unwrap();
        }
    }

    // ---- Q4 grouped MoE ----
    let ebytes = einter * d;
    let wg_bytes = topk * ebytes / 2;
    let wg_sbytes = topk * (ebytes / 32) * 4;
    let wg_q = hip::malloc(&h, wg_bytes).unwrap() as *mut u8;
    let wg_s = hip::malloc(&h, wg_sbytes).unwrap() as *mut f32;
    let wu_q = hip::malloc(&h, wg_bytes).unwrap() as *mut u8;
    let wu_s = hip::malloc(&h, wg_sbytes).unwrap() as *mut f32;
    let wd_bytes = topk * d * einter / 2;
    let wd_q = hip::malloc(&h, wd_bytes).unwrap() as *mut u8;
    let wd_s = hip::malloc(&h, topk * (d * einter / 32) * 4).unwrap() as *mut f32;
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

    profiled(
        &h,
        &k,
        "gate_up_q4 (8x768x2048)",
        2 * topk * ebytes / 2,
        rows * einter,
        iters,
        |prof| {
            k.launch_moe_grouped_gate_up_q4(
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
                prof,
            )
            .unwrap();
        },
    );
    profiled(
        &h,
        &k,
        "down_q4 (8x2048x768)",
        topk * d * einter / 2,
        rows * d,
        iters,
        |prof| {
            k.launch_moe_grouped_down_q4(
                xrow,
                ids,
                wd_q,
                wd_s,
                down,
                rows as i32,
                d as i32,
                einter as i32,
                prof,
            )
            .unwrap();
        },
    );

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
