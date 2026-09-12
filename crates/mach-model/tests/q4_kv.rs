//! Env-gated GPU parity harness for Q4 KV kernels.
//!
//! These tests are `#[ignore]` by default and additionally require
//! `MACH_TEST_Q4_KV=1`. Selecting them explicitly (`--ignored`) without that
//! opt-in fails loudly before any device is touched, rather than reporting
//! "ok" without running the kernel parity path.
//!
//! `$env:MACH_TEST_Q4_KV='1'; cargo test -p mach-model --features hip --test q4_kv -- --ignored --test-threads 1`
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::kernels::HipKernels;
use mach_model::kv_quant::{PagedQ4KvLayout, Q4Kv, attention_decode_q4, scatter_paged_q4};
use std::ffi::c_void;
use std::sync::Arc;

struct DevBuf {
    h: Arc<hip::Hip>,
    ptr: *mut c_void,
}

impl DevBuf {
    fn new<T: Copy>(h: Arc<hip::Hip>, data: &[T]) -> Self {
        let bytes = std::mem::size_of_val(data);
        let ptr = hip::malloc(&h, bytes.max(1)).unwrap();
        if !data.is_empty() {
            hip::memcpy(
                &h,
                ptr,
                data.as_ptr() as *const c_void,
                bytes,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )
            .unwrap();
        }
        Self { h, ptr }
    }

    fn download<T: Copy + Default>(&self, len: usize) -> Vec<T> {
        let mut out = vec![T::default(); len];
        if len != 0 {
            hip::memcpy(
                &self.h,
                out.as_mut_ptr() as *mut c_void,
                self.ptr,
                len * std::mem::size_of::<T>(),
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )
            .unwrap();
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

fn random_vec(seed: &mut u64, len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|_| lcg(seed) * scale).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut max = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        assert!(d.is_finite(), "non-finite diff: {x} vs {y}");
        max = max.max(d);
    }
    max
}

fn gpu_ctx() -> (Arc<hip::Hip>, HipKernels) {
    if std::env::var("MACH_TEST_Q4_KV").as_deref() != Ok("1") {
        panic!("MACH_TEST_Q4_KV=1 is required to run these Q4 KV parity tests");
    }
    let h = hip::hip()
        .unwrap_or_else(|e| panic!("MACH_TEST_Q4_KV=1 but HIP runtime is unavailable: {e}"));
    let devices = hip::device_count()
        .unwrap_or_else(|e| panic!("MACH_TEST_Q4_KV=1 but hipGetDeviceCount failed: {e}"));
    assert!(
        devices > 0,
        "MACH_TEST_Q4_KV=1 but no HIP device is present"
    );
    let k = HipKernels::new(Arc::clone(&h)).expect("compile Q4 KV kernels");
    (h, k)
}

#[test]
#[ignore = "GPU parity; run explicitly with MACH_TEST_Q4_KV=1"]
fn paged_q4_store_matches_cpu_oracle() {
    let (h, k) = gpu_ctx();
    let mut seed = 0x4b56_5f51_345f_5354u64;
    let batch = 2usize;
    let kv_heads = 2usize;
    let dim = 65usize; // odd head_dim exercises the padded high nibble
    let tpp = 4usize;
    let layout = PagedQ4KvLayout::new(4, tpp, kv_heads, dim).unwrap();

    let kv = random_vec(&mut seed, batch * kv_heads * dim, 2.0);
    let pos = [6i32, 2];
    let table = [2i32, 0, 1, 3];
    let offsets = [0i32, 2];
    let payload = vec![0u8; layout.payload_len()];
    let scales = vec![0.0f32; layout.scale_len()];

    let dkv = DevBuf::new(Arc::clone(&h), &kv);
    let dp = DevBuf::new(Arc::clone(&h), &payload);
    let ds = DevBuf::new(Arc::clone(&h), &scales);
    let dpos = DevBuf::new(Arc::clone(&h), &pos);
    let doffs = DevBuf::new(Arc::clone(&h), &offsets);
    let dtable = DevBuf::new(Arc::clone(&h), &table);

    k.launch_kv_store_paged_q4(
        dkv.ptr as *const f32,
        dp.ptr as *mut u8,
        ds.ptr as *mut f32,
        dpos.ptr as *const i32,
        doffs.ptr as *const i32,
        dtable.ptr as *const i32,
        batch as i32,
        kv_heads as i32,
        dim as i32,
        tpp as i32,
    )
    .unwrap();
    k.sync().unwrap();

    let got_p = dp.download::<u8>(layout.payload_len());
    let got_s = ds.download::<f32>(layout.scale_len());
    for s in 0..batch {
        let cpu = Q4Kv::quantize(
            &kv[s * kv_heads * dim..(s + 1) * kv_heads * dim],
            kv_heads,
            dim,
        )
        .unwrap();
        let logical = pos[s] as usize / tpp;
        let off = pos[s] as usize % tpp;
        let page = table[s * 2 + logical] as usize;
        for head in 0..kv_heads {
            let pi = layout.packed_index(page, off, head, 0).unwrap();
            assert_eq!(
                &got_p[pi..pi + layout.packed_dim()],
                cpu.head_row(0, head).unwrap(),
                "payload mismatch seq {s} head {head}"
            );
            let si = layout.scale_index(page, off, head).unwrap();
            assert_eq!(got_s[si], cpu.scale(0, head).unwrap());
        }
    }
}

fn run_paged_q4_attention_parity(
    h: &Arc<hip::Hip>,
    k: &HipKernels,
    dim: usize,
    permuted_table: bool,
) {
    let mut seed = 0x4154_544e_5f51_3400u64 ^ dim as u64;
    let batch = 2usize;
    let n_heads = 4usize;
    let kv_heads = 2usize;
    let tokens = 11usize;
    let tpp = 4usize;
    let logical_pages = tokens.div_ceil(tpp);
    let layout = PagedQ4KvLayout::new(batch * logical_pages, tpp, kv_heads, dim).unwrap();
    let page_table: Vec<usize> = if permuted_table {
        vec![3, 0, 5, 2, 1, 4]
    } else {
        (0..batch * logical_pages).collect()
    };
    let table: Vec<i32> = page_table.iter().map(|&page| page as i32).collect();
    let offsets = [0i32, logical_pages as i32];
    let pos = [tokens as i32 - 1; 2];
    let scale = 1.0 / (dim as f32).sqrt();

    let k_host = random_vec(&mut seed, batch * tokens * kv_heads * dim, 1.5);
    let v_host = random_vec(&mut seed, batch * tokens * kv_heads * dim, 1.5);
    let mut payload_k = vec![0u8; layout.payload_len()];
    let mut payload_v = vec![0u8; layout.payload_len()];
    let mut scales_k = vec![0.0f32; layout.scale_len()];
    let mut scales_v = vec![0.0f32; layout.scale_len()];
    let mut kq = Vec::with_capacity(batch);
    let mut vq = Vec::with_capacity(batch);
    for s in 0..batch {
        let base = s * tokens * kv_heads * dim;
        let seq_k =
            Q4Kv::quantize(&k_host[base..base + tokens * kv_heads * dim], kv_heads, dim).unwrap();
        let seq_v =
            Q4Kv::quantize(&v_host[base..base + tokens * kv_heads * dim], kv_heads, dim).unwrap();
        let seq_table = &page_table[s * logical_pages..(s + 1) * logical_pages];
        scatter_paged_q4(&seq_k, &layout, seq_table, &mut payload_k, &mut scales_k).unwrap();
        scatter_paged_q4(&seq_v, &layout, seq_table, &mut payload_v, &mut scales_v).unwrap();
        kq.push(seq_k);
        vq.push(seq_v);
    }

    let q = random_vec(&mut seed, batch * n_heads * dim, 1.0);
    let dq = DevBuf::new(Arc::clone(h), &q);
    let dkp = DevBuf::new(Arc::clone(h), &payload_k);
    let dks = DevBuf::new(Arc::clone(h), &scales_k);
    let dvp = DevBuf::new(Arc::clone(h), &payload_v);
    let dvs = DevBuf::new(Arc::clone(h), &scales_v);
    let dtable = DevBuf::new(Arc::clone(h), &table);
    let doffs = DevBuf::new(Arc::clone(h), &offsets);
    let dpos = DevBuf::new(Arc::clone(h), &pos);
    let dout = DevBuf::new(Arc::clone(h), &vec![0.0f32; batch * n_heads * dim]);

    k.launch_attn_decode_paged_q4_gqa(
        dq.ptr as *const f32,
        dkp.ptr as *const u8,
        dks.ptr as *const f32,
        dvp.ptr as *const u8,
        dvs.ptr as *const f32,
        dtable.ptr as *const i32,
        dout.ptr as *mut f32,
        dpos.ptr as *const i32,
        doffs.ptr as *const i32,
        batch as i32,
        n_heads as i32,
        kv_heads as i32,
        dim as i32,
        scale,
        tpp as i32,
    )
    .unwrap();
    k.sync().unwrap();
    let got = dout.download::<f32>(batch * n_heads * dim);
    for s in 0..batch {
        let qs = &q[s * n_heads * dim..(s + 1) * n_heads * dim];
        let want = attention_decode_q4(qs, n_heads, &kq[s], &vq[s], scale).unwrap();
        let diff = max_abs_diff(&got[s * n_heads * dim..(s + 1) * n_heads * dim], &want);
        assert!(
            diff < 2e-4,
            "paged Q4 attention dim={dim} seq {s} diff {diff}"
        );
    }
}

#[test]
#[ignore = "GPU parity; run explicitly with MACH_TEST_Q4_KV=1"]
fn paged_q4_attention_matches_cpu_oracle() {
    let (h, k) = gpu_ctx();
    // 65 exercises the odd packed-nibble tail; the permuted table ensures the
    // kernel follows logical -> physical page remapping instead of identity.
    run_paged_q4_attention_parity(&h, &k, 65, true);
    // 128 is the production head_dim for the current Qwen checkpoints.
    run_paged_q4_attention_parity(&h, &k, 128, false);
}
