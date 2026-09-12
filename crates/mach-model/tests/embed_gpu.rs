//! Env-gated GPU parity for multimodal row-embedding injection.
//!
//! The tests are `#[ignore]`d and need `MACH_TEST_EMBED_GPU=1`; selecting them
//! explicitly without that opt-in fails loudly (panic) before any device is
//! touched, rather than reporting "ok" (libtest drops passing tests' output).
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::batched::BatchedModel;
use mach_model::kernels::HipKernels;
use mach_model::sampling::SamplingParams;
use mach_model::{Config, Weights};
use std::ffi::c_void;
use std::sync::Arc;

struct DevBuf {
    h: Arc<hip::Hip>,
    ptr: *mut c_void,
}

impl DevBuf {
    fn new<T: Copy>(h: Arc<hip::Hip>, k: &HipKernels, data: &[T]) -> Self {
        let bytes = std::mem::size_of_val(data).max(4);
        let ptr = hip::malloc(&h, bytes).unwrap();
        hip::memcpy_async(
            &h,
            ptr,
            data.as_ptr() as *const c_void,
            std::mem::size_of_val(data),
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            k.stream,
        )
        .unwrap();
        Self { h, ptr }
    }

    fn download<T: Copy + Default>(&self, k: &HipKernels, len: usize) -> Vec<T> {
        let mut out = vec![T::default(); len];
        hip::memcpy_async(
            &self.h,
            out.as_mut_ptr() as *mut c_void,
            self.ptr,
            len * std::mem::size_of::<T>(),
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
            k.stream,
        )
        .unwrap();
        unsafe {
            hip::check(&self.h, (self.h.api.hip_stream_synchronize)(k.stream)).unwrap();
        }
        out
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        let _ = hip::free(&self.h, self.ptr);
    }
}

fn gpu_enabled() -> bool {
    if std::env::var("MACH_TEST_EMBED_GPU").as_deref() != Ok("1") {
        // `#[ignore]`d + explicitly selected => a missing opt-in must fail
        // loudly (libtest drops the output of passing tests).
        panic!("MACH_TEST_EMBED_GPU=1 is required to run this GPU embedding parity test");
    }
    true
}

#[test]
#[ignore = "GPU embedding scatter parity; set MACH_TEST_EMBED_GPU=1"]
fn embed_scatter_rows_matches_cpu() {
    if !gpu_enabled() {
        return;
    }
    let hip = hip::hip().expect("HIP runtime");
    let k = HipKernels::new(Arc::clone(&hip)).unwrap();
    let rows = 4usize;
    let cols = 3usize;
    let x: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    let features: Vec<f32> = (0..rows * cols).map(|i| 100.0 + i as f32).collect();
    let mask = [0i32, 1, 0, 1];
    let dx = DevBuf::new(Arc::clone(&hip), &k, &x);
    let df = DevBuf::new(Arc::clone(&hip), &k, &features);
    let dm = DevBuf::new(Arc::clone(&hip), &k, &mask);
    k.launch_embed_scatter_rows(
        dx.ptr as *mut f32,
        df.ptr as *const f32,
        dm.ptr as *const i32,
        rows as i32,
        cols as i32,
    )
    .unwrap();
    let got: Vec<f32> = dx.download(&k, x.len());
    for (r, &m) in mask.iter().enumerate() {
        for c in 0..cols {
            let i = r * cols + c;
            let want = if m != 0 { features[i] } else { x[i] };
            assert_eq!(got[i], want, "row={r} col={c}");
        }
    }
}

#[test]
#[ignore = "GPU embedding injection parity; set MACH_TEST_EMBED_GPU=1"]
fn row_embedding_partial_override_matches_replaced_token_reference() {
    if !gpu_enabled() {
        return;
    }
    let hip = hip::hip().expect("HIP runtime");
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 77).unwrap();
    let mut injected = BatchedModel::with_rows(Arc::clone(&hip), cfg, &w, 1, 4).unwrap();
    let mut reference = BatchedModel::with_rows(Arc::clone(&hip), cfg, &w, 1, 4).unwrap();
    let mut normal = BatchedModel::with_rows(hip, cfg, &w, 1, 4).unwrap();
    let tokens = [3u32, 17, 42, 5];
    // Distinct per-row features catch layout mistakes: row r must read
    // features[r*d..(r+1)*d], not just replicate the first row.
    let feature_tokens = [99u32, 100, 101, 102];
    let mask = [0i32, 1, 0, 1];
    // Masked rows 1 and 3 use explicit embeddings; rows 0 and 2 keep the
    // normal gather result, so the reference swaps only the masked tokens.
    let ref_tokens = [tokens[0], feature_tokens[1], tokens[2], feature_tokens[3]];
    let lens = [0u32, 1, 2, 3];
    let slots = [0u32, 0, 0, 0];
    let d = cfg.d_model;
    let features: Vec<f32> = feature_tokens
        .iter()
        .flat_map(|&t| {
            w.tok_emb[t as usize * d..(t as usize + 1) * d]
                .iter()
                .copied()
        })
        .collect();
    injected
        .set_row_embeddings(&features, &mask, tokens.len())
        .unwrap();
    let mut params_a = vec![SamplingParams::default(); tokens.len()];
    let mut params_b = params_a.clone();
    let mut params_c = params_a.clone();
    let counts = vec![Vec::new(); tokens.len()];
    let bias = vec![Vec::new(); tokens.len()];
    reference
        .decode_step_explicit(
            &ref_tokens,
            &lens,
            &slots,
            &mut params_a,
            &counts,
            &bias,
            false,
        )
        .unwrap();
    injected
        .decode_step_explicit(&tokens, &lens, &slots, &mut params_b, &counts, &bias, false)
        .unwrap();
    normal
        .decode_step_explicit(&tokens, &lens, &slots, &mut params_c, &counts, &bias, false)
        .unwrap();
    let want = reference.read_logits_rows(tokens.len()).unwrap();
    let got = injected.read_logits_rows(tokens.len()).unwrap();
    let baseline = normal.read_logits_rows(tokens.len()).unwrap();
    assert_eq!(got.len(), want.len());
    assert_eq!(got.len(), baseline.len());
    let mut max_diff = 0.0f32;
    let mut baseline_diff = 0.0f32;
    for (i, ((g, w), b)) in got.iter().zip(&want).zip(&baseline).enumerate() {
        assert!(
            g.is_finite() && w.is_finite() && b.is_finite(),
            "non-finite logit at {i}: {g} {w} {b}"
        );
        max_diff = max_diff.max((g - w).abs());
        baseline_diff = baseline_diff.max((g - b).abs());
    }
    assert!(
        max_diff < 1e-5,
        "partial injection vs replaced-token ref mismatch: {max_diff}"
    );
    assert!(
        baseline_diff > 1e-3,
        "partial override test is vacuous: injected and normal logits differ by only {baseline_diff}"
    );
}

#[test]
#[ignore = "GPU embedding injection guard; set MACH_TEST_EMBED_GPU=1"]
fn embed_scatter_rows_rejects_i32_overflow() {
    if !gpu_enabled() {
        return;
    }
    let hip = hip::hip().expect("HIP runtime");
    let k = HipKernels::new(hip).unwrap();
    let err = k
        .launch_embed_scatter_rows(
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            65536,
            32768,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds i32"), "{err}");
}

#[test]
#[ignore = "GPU embedding injection guard; set MACH_TEST_EMBED_GPU=1"]
fn row_embedding_row_mismatch_fails_fast() {
    if !gpu_enabled() {
        return;
    }
    let hip = hip::hip().expect("HIP runtime");
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 55).unwrap();
    let mut model = BatchedModel::with_rows(hip, cfg, &w, 1, 4).unwrap();
    let d = cfg.d_model;
    let features = w.tok_emb[99 * d..100 * d].to_vec();
    let mask = [1i32];
    model.set_row_embeddings(&features, &mask, 1).unwrap();
    let tokens = [3u32, 17, 42, 5];
    let lens = [0u32, 1, 2, 3];
    let slots = [0u32, 0, 0, 0];
    let mut params = vec![SamplingParams::default(); tokens.len()];
    let counts = vec![Vec::new(); tokens.len()];
    let bias = vec![Vec::new(); tokens.len()];
    let err = model
        .decode_step_explicit(&tokens, &lens, &slots, &mut params, &counts, &bias, false)
        .unwrap_err()
        .to_string();
    assert!(err.contains("row embeddings"), "{err}");
}
