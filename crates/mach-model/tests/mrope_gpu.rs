//! Env-gated GPU parity for the M-RoPE table/delta RoPE kernels.
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::batched::BatchedModel;
use mach_model::kernels::{HipKernels, RopeParams};
use mach_model::mrope::MropePositions;
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
        if !data.is_empty() {
            hip::memcpy_async(
                &h,
                ptr,
                data.as_ptr() as *const c_void,
                std::mem::size_of_val(data),
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
                k.stream,
            )
            .unwrap();
        }
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

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
}

#[allow(clippy::too_many_arguments)]
fn cpu_tables(
    q: &mut [f32],
    k: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    batch: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    rot: usize,
) {
    let half = rot / 2;
    for s in 0..batch {
        for h in 0..heads {
            let p = &mut q[(s * heads + h) * hd..(s * heads + h + 1) * hd];
            for d in 0..half {
                let a = p[d];
                let b = p[d + half];
                let c = cos[s * rot + d];
                let sn = sin[s * rot + d];
                p[d] = a * c - b * sn;
                p[d + half] = a * sn + b * c;
            }
        }
        for h in 0..kv_heads {
            let p = &mut k[(s * kv_heads + h) * hd..(s * kv_heads + h + 1) * hd];
            for d in 0..half {
                let a = p[d];
                let b = p[d + half];
                let c = cos[s * rot + d];
                let sn = sin[s * rot + d];
                p[d] = a * c - b * sn;
                p[d + half] = a * sn + b * c;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_delta(
    q: &mut [f32],
    k: &mut [f32],
    pos: &[i32],
    batch: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    rot: usize,
    theta: f32,
    delta: i32,
) {
    let half = rot / 2;
    let freq = |d: usize| 1.0 / theta.powf((2 * d) as f32 / rot as f32);
    for s in 0..batch {
        let p0 = (pos[s] + delta) as f32;
        for h in 0..heads {
            let p = &mut q[(s * heads + h) * hd..(s * heads + h + 1) * hd];
            for d in 0..half {
                let ang = p0 * freq(d);
                let (c, sn) = (ang.cos(), ang.sin());
                let a = p[d];
                let b = p[d + half];
                p[d] = a * c - b * sn;
                p[d + half] = a * sn + b * c;
            }
        }
        for h in 0..kv_heads {
            let p = &mut k[(s * kv_heads + h) * hd..(s * kv_heads + h + 1) * hd];
            for d in 0..half {
                let ang = p0 * freq(d);
                let (c, sn) = (ang.cos(), ang.sin());
                let a = p[d];
                let b = p[d + half];
                p[d] = a * c - b * sn;
                p[d + half] = a * sn + b * c;
            }
        }
    }
}

#[test]
#[ignore = "GPU M-RoPE model parity; set MACH_TEST_MROPE_GPU=1 and run explicitly"]
fn batched_model_mrope_tables_match_scalar_text_only() {
    if std::env::var("MACH_TEST_MROPE_GPU").as_deref() != Ok("1") {
        return;
    }
    let hip = hip::hip().expect("HIP runtime");
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 123).unwrap();
    let mut scalar = BatchedModel::with_rows(Arc::clone(&hip), cfg, &w, 1, 4).unwrap();
    let mut tabled = BatchedModel::with_rows(Arc::clone(&hip), cfg, &w, 1, 4).unwrap();
    let tokens = [3u32, 17, 42, 5];
    let lens = [0u32, 1, 2, 3];
    let slots = [0u32, 0, 0, 0];
    let mut params_a = vec![SamplingParams::default(); tokens.len()];
    let mut params_b = params_a.clone();
    let counts = vec![Vec::new(); tokens.len()];
    let bias = vec![Vec::new(); tokens.len()];
    let positions = MropePositions {
        pos: vec![[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
        delta: 0,
    };
    let (cos, sin) = positions.cos_sin(&cfg, [1, 1, 1]).unwrap();
    tabled.set_mrope_tables(&cos, &sin, tokens.len()).unwrap();
    scalar
        .decode_step_explicit(&tokens, &lens, &slots, &mut params_a, &counts, &bias, false)
        .unwrap();
    tabled
        .decode_step_explicit(&tokens, &lens, &slots, &mut params_b, &counts, &bias, false)
        .unwrap();
    let a = scalar.read_logits_rows(tokens.len()).unwrap();
    let b = tabled.read_logits_rows(tokens.len()).unwrap();
    let mut max_diff = 0.0f32;
    for (x, y) in a.iter().zip(&b) {
        max_diff = max_diff.max((x - y).abs());
    }
    assert!(max_diff < 1e-5, "M-RoPE table model mismatch: {max_diff}");
}
#[test]
#[ignore = "GPU M-RoPE parity; set MACH_TEST_MROPE_GPU=1 and run explicitly"]
fn rope_batched_tables_matches_cpu() {
    if std::env::var("MACH_TEST_MROPE_GPU").as_deref() != Ok("1") {
        eprintln!("skipping GPU M-RoPE parity: MACH_TEST_MROPE_GPU is not 1");
        return;
    }
    let h = hip::hip().expect("HIP runtime");
    let k = HipKernels::new(Arc::clone(&h)).unwrap();
    let cfg = Config::llama(8, 4, 1, 1, 32, 32);
    let positions = MropePositions {
        pos: vec![[2, 3, 4], [3, 4, 2], [4, 2, 3]],
        delta: 0,
    };
    let (cos, sin) = positions.cos_sin(&cfg, [11, 11, 10]).unwrap();
    let batch = 3usize;
    let heads = 1usize;
    let kv_heads = 1usize;
    let hd = 8usize;
    let rot = cfg.attn_rotary_dim();
    assert_eq!(rot, 8);
    let mut seed = 7u64;
    let q: Vec<f32> = (0..batch * heads * hd).map(|_| lcg(&mut seed)).collect();
    let kv: Vec<f32> = (0..batch * kv_heads * hd).map(|_| lcg(&mut seed)).collect();
    let mut want_q = q.clone();
    let mut want_k = kv.clone();
    cpu_tables(
        &mut want_q,
        &mut want_k,
        &cos,
        &sin,
        batch,
        heads,
        kv_heads,
        hd,
        rot,
    );
    let dq = DevBuf::new(Arc::clone(&h), &k, &q);
    let dk = DevBuf::new(Arc::clone(&h), &k, &kv);
    let dc = DevBuf::new(Arc::clone(&h), &k, &cos);
    let ds = DevBuf::new(Arc::clone(&h), &k, &sin);
    k.launch_rope_batched_tables(
        dq.ptr as *mut f32,
        dk.ptr as *mut f32,
        dc.ptr as *const f32,
        ds.ptr as *const f32,
        batch as i32,
        heads as i32,
        kv_heads as i32,
        hd as i32,
        rot as i32,
    )
    .unwrap();
    let got_q: Vec<f32> = dq.download(&k, q.len());
    let got_k: Vec<f32> = dk.download(&k, kv.len());
    for (a, b) in got_q.iter().zip(&want_q) {
        assert!((a - b).abs() < 1e-6, "q table mismatch: {a} vs {b}");
    }
    for (a, b) in got_k.iter().zip(&want_k) {
        assert!((a - b).abs() < 1e-6, "k table mismatch: {a} vs {b}");
    }
}

#[test]
#[ignore = "GPU M-RoPE parity; set MACH_TEST_MROPE_GPU=1 and run explicitly"]
fn rope_batched_delta_matches_cpu() {
    if std::env::var("MACH_TEST_MROPE_GPU").as_deref() != Ok("1") {
        return;
    }
    let h = hip::hip().expect("HIP runtime");
    let k = HipKernels::new(Arc::clone(&h)).unwrap();
    let cfg = Config::llama(8, 4, 1, 1, 32, 32);
    let batch = 3usize;
    let heads = 1usize;
    let kv_heads = 1usize;
    let hd = 8usize;
    let rot = cfg.attn_rotary_dim();
    let pos = [0i32, 1, 2];
    let delta = -2i32;
    let mut seed = 11u64;
    let q: Vec<f32> = (0..batch * heads * hd).map(|_| lcg(&mut seed)).collect();
    let kv: Vec<f32> = (0..batch * kv_heads * hd).map(|_| lcg(&mut seed)).collect();
    let mut want_q = q.clone();
    let mut want_k = kv.clone();
    cpu_delta(
        &mut want_q,
        &mut want_k,
        &pos,
        batch,
        heads,
        kv_heads,
        hd,
        rot,
        cfg.rope_theta,
        delta,
    );
    let dq = DevBuf::new(Arc::clone(&h), &k, &q);
    let dk = DevBuf::new(Arc::clone(&h), &k, &kv);
    let dp = DevBuf::new(Arc::clone(&h), &k, &pos);
    k.launch_rope_batched(
        dq.ptr as *mut f32,
        dk.ptr as *mut f32,
        dp.ptr as *const i32,
        batch as i32,
        heads as i32,
        kv_heads as i32,
        hd as i32,
        rot as i32,
        RopeParams::from(cfg),
        delta,
    )
    .unwrap();
    let got_q: Vec<f32> = dq.download(&k, q.len());
    let got_k: Vec<f32> = dk.download(&k, kv.len());
    for (a, b) in got_q.iter().zip(&want_q) {
        assert!((a - b).abs() < 1e-6, "q delta mismatch: {a} vs {b}");
    }
    for (a, b) in got_k.iter().zip(&want_k) {
        assert!((a - b).abs() < 1e-6, "k delta mismatch: {a} vs {b}");
    }
}
