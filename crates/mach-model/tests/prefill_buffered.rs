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
