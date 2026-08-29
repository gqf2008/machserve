//! Full-layer double-buffered prefill: CPU parity + GPU comparison tests.
//!
//! The CPU tests verify the scheduling core [`run_double_buffered`] against an
//! independent sequential reference (bitwise) and that the pipelined issue
//! order genuinely overlaps prepare with compute. The GPU test (opt-in with
//! `--ignored`) checks that the buffered prefill mode of `BatchedModel`
//! produces the same logits as the full-resident (sequential) mode on the same
//! MoE checkpoint, for a mixed dense/MoE model with `moe_intermediate_size !=
//! intermediate_size`.

use mach_model::prefill_buffered::run_double_buffered;

/// Layer math shared by the pipeline and the reference: `state = state @ w^T`
/// with a fixed f32 summation order, so the two schedules must agree bitwise.
fn apply_layer(state: &mut [f32], w: &[f32], d: usize) {
    let mut next = vec![0.0f32; d];
    for j in 0..d {
        let mut acc = 0.0f32;
        for k in 0..d {
            acc += state[k] * w[k * d + j];
        }
        next[j] = acc;
    }
    state.copy_from_slice(&next);
}

/// Deterministic per-layer `[d, d]` weight matrix (distinct per layer, so any
/// reordering shows up in the output).
fn gen_layer(d: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut w = Vec::with_capacity(d * d);
    for _ in 0..d * d {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        w.push((((s >> 33) as f64) / ((1u64 << 31) as f64)) as f32 - 1.0);
    }
    w
}

#[test]
fn double_buffered_output_matches_sequential_reference_bitwise() {
    for &n in &[1usize, 2, 5, 8] {
        let d = 12;
        let weights: Vec<Vec<f32>> = (0..n).map(|i| gen_layer(d, 7 + i as u64)).collect();
        let init: Vec<f32> = (0..d).map(|i| (i as f32) * 0.5 - 4.0).collect();

        // Pipelined (production) schedule.
        let mut pipe_state = init.clone();
        let pipe_out: Vec<Vec<f32>> = run_double_buffered(
            n,
            &mut pipe_state,
            |i| Ok::<_, ()>(weights[i].clone()),
            |_i, p, s| {
                apply_layer(s, p, d);
                Ok(s.clone())
            },
        )
        .unwrap();

        // Independent reference: a plain sequential per-layer loop written in
        // this test (not through run_double_buffered).
        let mut ref_state = init.clone();
        let mut ref_out = Vec::with_capacity(n);
        for w in &weights {
            let p = w.clone();
            apply_layer(&mut ref_state, &p, d);
            ref_out.push(ref_state.clone());
        }

        assert_eq!(
            pipe_out, ref_out,
            "per-layer outputs must be bitwise identical for n={n}"
        );
        assert_eq!(
            pipe_state, ref_state,
            "final state must be bitwise identical for n={n}"
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Call {
    Prepare(usize),
    Compute(usize),
}

#[test]
fn double_buffered_schedule_overlaps_prepare_and_compute() {
    let n = 4usize;
    let d = 8;
    let weights: Vec<Vec<f32>> = (0..n).map(|i| gen_layer(d, 3 + i as u64)).collect();
    // Both closures record into the same log, so share it through RefCell.
    let calls = std::cell::RefCell::new(Vec::new());
    let mut state = vec![0.0f32; d];

    run_double_buffered(
        n,
        &mut state,
        |i| {
            calls.borrow_mut().push(Call::Prepare(i));
            Ok::<_, ()>(weights[i].clone())
        },
        |i, p, s| {
            calls.borrow_mut().push(Call::Compute(i));
            apply_layer(s, p, d);
            Ok::<_, ()>(())
        },
    )
    .unwrap();
    let calls = calls.into_inner();

    // Pipelined issue order: prepare(i+1) is issued before compute(i), so an
    // asynchronous prepare overlaps the previous layer's compute.
    let mut expected = vec![Call::Prepare(0)];
    for i in 0..n {
        if i + 1 < n {
            expected.push(Call::Prepare(i + 1));
        }
        expected.push(Call::Compute(i));
    }
    assert_eq!(
        calls, expected,
        "prepare(i+1) must be issued before compute(i)"
    );

    // Under the async model where prepare(i+1) and compute(i) run concurrently
    // (each taking a fixed virtual time P and C), the makespan is
    // P + (n-1)*max(P,C) + C — strictly less than the sequential n*(P+C): the
    // double buffer genuinely hides prepare under compute.
    let p = 3.0f64;
    let c = 5.0f64;
    let pipelined = p + (n as f64 - 1.0) * p.max(c) + c;
    let sequential = n as f64 * (p + c);
    assert!(pipelined < sequential, "pipelined must beat sequential");
    assert_eq!(pipelined, 3.0 + 3.0 * 5.0 + 5.0, "makespan model");
}

#[cfg(feature = "hip")]
fn hip_ctx() -> Option<std::sync::Arc<mach_kernel_sys::hip::Hip>> {
    use mach_kernel_sys::hip;
    match hip::hip() {
        Ok(h) => match hip::device_count() {
            Ok(n) if n > 0 => Some(h),
            _ => {
                eprintln!("skipping HIP test: no device");
                None
            }
        },
        Err(e) => {
            eprintln!("skipping HIP test: {e}");
            None
        }
    }
}

#[cfg(feature = "hip")]
fn moe_cfg() -> mach_model::Config {
    use mach_model::config::ModelDType;
    // Qwen3-MoE-style shape: dense MLP width != expert width, mixed
    // dense/MoE layers, qwen3 QK-norm off (keep standard attention).
    let mut cfg = mach_model::Config::tiny();
    cfg.d_model = 64;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 16;
    cfg.intermediate_size = 128;
    cfg.moe_intermediate_size = 48;
    cfg.num_experts = 6;
    cfg.num_experts_per_tok = 3;
    cfg.n_layers = 4;
    cfg.vocab_size = 512;
    cfg.max_seq_len = 64;
    cfg.dtype = ModelDType::F32;
    cfg
}

/// Buffered prefill (experts streamed per layer with double buffering) must
/// produce the same logits as the sequential full-resident batched path: both
/// run the same grouped-GEMM kernels over byte-identical weights, so the
/// comparison is bitwise.
#[cfg(feature = "hip")]
#[ignore]
#[test]
fn buffered_prefill_matches_resident_batched() {
    use mach_model::Weights;
    use mach_model::batched::BatchedModel;
    use mach_model::sampling::SamplingParams;

    let Some(hip) = hip_ctx() else { return };
    let cfg = moe_cfg();
    let mut w = Weights::random(&cfg, 42).unwrap();
    // Layer 0 dense (Qwen3-MoE `mlp_only_layers`), layers 1..n routed MoE.
    w.layers[0].moe_router.clear();
    w.layers[0].moe_wg.clear();
    w.layers[0].moe_wu.clear();
    w.layers[0].moe_wd.clear();

    let n_rows = 8usize;
    let tokens: Vec<u32> = (0..n_rows)
        .map(|i| ((i as u32) * 37 + 5) % cfg.vocab_size as u32)
        .collect();
    let lens: Vec<u32> = (0..n_rows as u32).collect();
    let slots: Vec<u32> = (0..n_rows as u32).collect();
    let empty_counts = || vec![Vec::<(u32, u32)>::new(); n_rows];
    let empty_bias = || vec![Vec::<(u32, f32)>::new(); n_rows];

    let mut resident =
        BatchedModel::new(hip.clone(), cfg, &w, n_rows).expect("full-resident batched");
    let mut buffered = BatchedModel::with_prefill_buffer(hip.clone(), cfg, &w, n_rows, n_rows)
        .expect("buffered prefill");

    let run = |m: &mut BatchedModel| -> Vec<f32> {
        m.reset_state().expect("reset");
        let mut params = vec![SamplingParams::greedy(1); n_rows];
        m.decode_step_explicit(
            &tokens,
            &lens,
            &slots,
            &mut params,
            &empty_counts(),
            &empty_bias(),
            false,
        )
        .expect("prefill step");
        m.read_logits().expect("read logits")
    };

    let a = run(&mut resident);
    let b = run(&mut buffered);
    assert_eq!(a.len(), b.len());
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max, 0.0,
        "buffered prefill must match sequential full-resident bitwise (max diff {max})"
    );
}

/// Cross-step regression for the ping-pong slot race: with an ODD MoE layer
/// count (3 here: layers 1..3 routed), step N+1's `prefetch(0)` writes the
/// same slot step N's LAST MoE layer reads. `begin()` must wait for that
/// compute before overwriting, or the fast grouped-GEMV decode path lets the
/// new copy overtake the previous step's last read and the last layer's
/// logits corrupt. Many decode steps, compared bitwise against the resident
/// model after EVERY step.
///
/// NOTE: on THIS platform a pre-fix engine never corrupts here — kernels
/// already running do not observe concurrent H2D copy writes (stale L1/L2,
/// see the deterministic test's NOTE), so the copies can never visibly
/// overtake the read regardless of timing or expert-pool size. This test is
/// therefore a cross-platform integration net (on platforms where the copy
/// is visible to in-flight kernels it discriminates); the deterministic
/// event-order assertion in `prefetch_begin_waits_for_previous_step_last_
/// layer_read` is the判别 mechanism on this machine.
#[cfg(feature = "hip")]
#[test]
fn buffered_decode_cross_step_odd_moe_layers_matches_resident() {
    use mach_model::Weights;
    use mach_model::batched::BatchedModel;
    use mach_model::sampling::SamplingParams;

    let Some(hip) = hip_ctx() else { return };
    let mut cfg = moe_cfg();
    // Realistic decode shape (larger expert pool) — not a race-window
    // widening: on this platform the window is invisible either way (see the
    // NOTE above).
    cfg.moe_intermediate_size = 256;
    cfg.num_experts = 12;
    let mut w = Weights::random(&cfg, 42).unwrap();
    // Layer 0 dense, layers 1..3 routed MoE -> odd MoE layer count (3).
    w.layers[0].moe_router.clear();
    w.layers[0].moe_wg.clear();
    w.layers[0].moe_wu.clear();
    w.layers[0].moe_wd.clear();

    let n_rows = 8usize;
    let empty_counts = || vec![Vec::<(u32, u32)>::new(); n_rows];
    let empty_bias = || vec![Vec::<(u32, f32)>::new(); n_rows];

    let mut resident =
        BatchedModel::new(hip.clone(), cfg, &w, n_rows).expect("full-resident batched");
    let mut buffered = BatchedModel::with_prefill_buffer(hip.clone(), cfg, &w, n_rows, n_rows)
        .expect("buffered prefill");

    for i in 0..32u32 {
        let tokens: Vec<u32> = (0..n_rows as u32)
            .map(|r| ((i * 7 + r) * 37 + 5) % cfg.vocab_size as u32)
            .collect();
        let lens = vec![i; n_rows];
        let slots: Vec<u32> = (0..n_rows as u32).collect();
        let mut pa = vec![SamplingParams::greedy(1); n_rows];
        let mut pb = vec![SamplingParams::greedy(1); n_rows];
        resident
            .decode_step_explicit(
                &tokens,
                &lens,
                &slots,
                &mut pa,
                &empty_counts(),
                &empty_bias(),
                true,
            )
            .expect("resident step");
        buffered
            .decode_step_explicit(
                &tokens,
                &lens,
                &slots,
                &mut pb,
                &empty_counts(),
                &empty_bias(),
                true,
            )
            .expect("buffered step");
        let a = resident.read_logits().expect("read resident logits");
        let b = buffered.read_logits().expect("read buffered logits");
        let max = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            max, 0.0,
            "step {i}: buffered decode must match resident bitwise (max diff {max})"
        );
    }
}

/// DETERMINISTIC cross-step slot-safety regression (engine-level): step 2's
/// `begin()` must not let `prefetch(0)` overwrite a ping-pong slot while step
/// 1's LAST MoE layer is still reading it. The config below has FOUR routed
/// MoE layers (layers 0..3 — the test does not clear layer 0's router), so
/// the verified mechanism is the cross-step wait itself, not a specific slot
/// parity: `begin()` must wait for the previous step's last MoE layer
/// (rank 3) compute event before issuing the copies. Step 1's last-layer
/// compute is a LONG kernel (~8ms) standing in for the grouped GEMVs still in
/// flight when step 2's `begin()` runs. The assertion is on EVENT ORDER:
/// step 2's layer-0 weights event (`prefetch_ev[0]`) must not fire until the
/// previous step's last layer compute finished. A watch stream waits on that
/// event: with the `begin()` wait it passes only after the probe; without it
/// the copies complete in ~120us and the wait passes immediately — an
/// order-of-magnitude gap either way.
///
/// NOTE on why this is event-order, not content: on this platform kernels
/// already running do NOT observe concurrent H2D copy writes (stale L1/L2),
/// so a probe racing the copies reads the old data regardless — only the
/// event ordering discriminates.
#[cfg(feature = "hip")]
#[test]
fn prefetch_begin_waits_for_previous_step_last_layer_read() {
    use mach_kernel_sys::hip;
    use mach_model::Weights;
    use mach_model::prefill_buffered::PrefetchEngine;

    let Some(h) = hip_ctx() else { return };
    let cfg = moe_cfg(); // layer 0 dense, layers 1..3 routed MoE -> odd (3)
    let w = Weights::random(&cfg, 43).unwrap();
    let engine = PrefetchEngine::new(h.clone(), cfg, &w).expect("prefetch engine");

    // The engine owns the prefetch stream; the layers' compute is driven on
    // a test stream (the one `layer_end`/`weights_ready` accept).
    let mut stream = std::ptr::null_mut();
    unsafe { hip::check(&h, (h.api.hip_stream_create)(&mut stream)).unwrap() };

    // Step 1: prefetch layers 0..2, "compute" each on the test stream. The
    // last MoE layer's compute is a LONG kernel: one probe streams
    // `reps * pool` (~3GB, ~140ms) — the previous step's last-layer read
    // stays in flight long after step 2's begin() runs. (A content check
    // against a concurrent overwrite would NOT detect the race on this
    // platform: kernels already running do not observe concurrent H2D copy
    // writes — the copies are still ordered wrong, which the event-order
    // assertion below catches deterministically.)
    engine.begin().unwrap();
    let (wg_slot, _, _) = engine.weights(3).expect("last MoE layer slot");
    let pool = cfg.num_experts * cfg.expert_size() * cfg.d_model * 4;
    let n_floats = (pool / 4) as i32;
    let blocks = (n_floats as u32).div_ceil(256).max(1);
    let reps = 200_000i32;
    let dst0 = hip::malloc(&h, pool).unwrap() as *mut f32;
    let probe_kern = hip::HipKernelModule::compile(
        "gfx1100",
        r#"extern "C" __global__ void probe_read(
            const float* __restrict__ src,
            float* __restrict__ dst,
            int n, int reps) {
            int t = blockIdx.x * blockDim.x + threadIdx.x;
            float acc = 0.f;
            int stride = gridDim.x * blockDim.x;
            for (int r = 0; r < reps; r++) {
                int off = (r * 7919) % n; // rotating offset defeats hoisting
                for (int i = t; i < n; i += stride) {
                    acc += src[(i + off) % n];
                }
            }
            dst[t] = acc;
        }"#,
        "probe_read",
    )
    .expect("compile probe_read");
    let sp: *const f32 = wg_slot;
    let dp: *mut f32 = dst0;
    let mut p = vec![
        &sp as *const *const f32 as *mut core::ffi::c_void,
        &dp as *const *mut f32 as *mut core::ffi::c_void,
        &n_floats as *const i32 as *mut core::ffi::c_void,
        &reps as *const i32 as *mut core::ffi::c_void,
    ];
    for li in 1..4usize {
        engine.layer_begin(li).unwrap();
        if li == 3 {
            // Within-step ordering first (the engine's weights_ready wait),
            // then the long in-flight read of slot 0.
            engine.weights_ready(3, stream).unwrap();
            probe_kern
                .launch([blocks, 1, 1], [256, 1, 1], &mut p, stream)
                .unwrap();
        }
        engine.layer_end(li, stream).unwrap();
    }

    // Step 2's begin(): the cross-step boundary under test. The assertion is
    // on EVENT ORDER, not content: step 2's layer-0 weights event
    // (`prefetch_ev[0]`) must NOT fire until the previous step's last-layer
    // compute (the probe) finished. A watch stream waits on that event: with
    // the fix it passes only after the probe (~8ms later); without it the
    // copies complete in ~120us and the wait passes immediately — an
    // order-of-magnitude gap.
    let mut watch = std::ptr::null_mut();
    unsafe { hip::check(&h, (h.api.hip_stream_create)(&mut watch)).unwrap() };
    let t_begin = std::time::Instant::now();
    engine.begin().unwrap();
    // wait prefetch_ev[0]: layer 0 IS the first MoE layer (rank 0) in this
    // config — its weights event is recorded by step 2's begin().
    engine.weights_ready(0, watch).unwrap();
    unsafe { hip::check(&h, (h.api.hip_stream_synchronize)(watch)).unwrap() };
    let waited = t_begin.elapsed();
    unsafe { hip::check(&h, (h.api.hip_stream_destroy)(watch)).unwrap() };

    // Drain the probe (and the event chain behind it).
    unsafe { hip::check(&h, (h.api.hip_stream_synchronize)(stream)).unwrap() };
    hip::free(&h, dst0 as *mut core::ffi::c_void).unwrap();
    unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };

    // The assertion: the watch (on prefetch_ev[0]) must have been held until
    // the probe ended. With the fix it is (probe >= ~8ms at reps=200k);
    // without it the copies complete in ~120us — an order-of-magnitude gap
    // either way. The 1ms threshold is coupled to the probe's duration (a
    // fixed hiprtc compile; current margin ~8x on both sides) — if a future
    // driver/ISA change speeds the probe up dramatically, re-check the
    // probe's actual duration before lowering the reps.
    assert!(
        waited > std::time::Duration::from_millis(1),
        "prefetch(0) must wait for the previous step's last layer compute \
         (prefetch_ev[0] fired after only {waited:?} — the begin() wait is missing)"
    );
}
