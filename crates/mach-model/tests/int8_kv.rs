//! Env-gated GPU parity harness for INT8 KV kernels.
//!
//! These tests are `#[ignore]` by default and additionally require
//! `MACH_TEST_INT8_KV=1`. With the env var unset they return before touching a
//! HIP device. Run only in a dedicated GPU window:
//!
//! `$env:MACH_TEST_INT8_KV='1'; cargo test -p mach-model --features hip --test int8_kv -- --ignored --test-threads 1`
#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::kernels::HipKernels;
use mach_model::kv_quant::{Int8Kv, PagedInt8KvLayout, attention_decode_int8, scatter_paged_int8};
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

fn quantize_contiguous_cache(
    cache: &[f32],
    slots: usize,
    max_seq: usize,
    heads: usize,
    dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let row = heads * dim;
    let mut payload = Vec::with_capacity(slots * max_seq * row);
    let mut scales = Vec::with_capacity(slots * max_seq * heads);
    for t in 0..slots * max_seq {
        let kv = Int8Kv::quantize(&cache[t * row..(t + 1) * row], heads, dim).unwrap();
        payload.extend_from_slice(kv.quantized());
        scales.extend_from_slice(kv.scales());
    }
    (payload, scales)
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

fn gpu_ctx() -> Option<(Arc<hip::Hip>, HipKernels)> {
    if std::env::var("MACH_TEST_INT8_KV").as_deref() != Ok("1") {
        eprintln!("skipping: MACH_TEST_INT8_KV != 1");
        return None;
    }
    let h = hip::hip()
        .unwrap_or_else(|e| panic!("MACH_TEST_INT8_KV=1 but HIP runtime is unavailable: {e}"));
    let devices = hip::device_count()
        .unwrap_or_else(|e| panic!("MACH_TEST_INT8_KV=1 but hipGetDeviceCount failed: {e}"));
    assert!(
        devices > 0,
        "MACH_TEST_INT8_KV=1 but no HIP device is present"
    );
    let k = HipKernels::new(Arc::clone(&h)).expect("compile INT8 KV kernels");
    Some((h, k))
}

#[test]
#[ignore = "GPU parity; run explicitly with MACH_TEST_INT8_KV=1"]
fn contiguous_int8_kv_store_and_attention_match_cpu_oracle() {
    let Some((h, k)) = gpu_ctx() else { return };
    let (slots, batch, kv_heads, n_heads, dim, max_seq, pos) =
        (1usize, 1usize, 2usize, 4usize, 8usize, 6usize, 5usize);
    let mut seed = 101u64;
    let mut k_cache = random_vec(&mut seed, slots * max_seq * kv_heads * dim, 2.0);
    let mut v_cache = random_vec(&mut seed, slots * max_seq * kv_heads * dim, 2.0);
    let k_cur = random_vec(&mut seed, batch * kv_heads * dim, 2.0);
    let v_cur = random_vec(&mut seed, batch * kv_heads * dim, 2.0);
    let q = random_vec(&mut seed, batch * n_heads * dim, 1.0);
    let pos_buf = vec![pos as i32; batch];
    let slot_buf = vec![0i32; batch];
    let run_mask = vec![0i32; batch];

    let (init_kp, init_ks) = quantize_contiguous_cache(&k_cache, slots, max_seq, kv_heads, dim);
    let (init_vp, init_vs) = quantize_contiguous_cache(&v_cache, slots, max_seq, kv_heads, dim);
    let dkp = DevBuf::new(Arc::clone(&h), &init_kp);
    let dks = DevBuf::new(Arc::clone(&h), &init_ks);
    let dvp = DevBuf::new(Arc::clone(&h), &init_vp);
    let dvs = DevBuf::new(Arc::clone(&h), &init_vs);
    let dkcur = DevBuf::new(Arc::clone(&h), &k_cur);
    let dvcur = DevBuf::new(Arc::clone(&h), &v_cur);
    let dpos = DevBuf::new(Arc::clone(&h), &pos_buf);
    let dslot = DevBuf::new(Arc::clone(&h), &slot_buf);
    let drun = DevBuf::new(Arc::clone(&h), &run_mask);

    k.launch_kv_store_int8(
        dkcur.ptr as *const f32,
        dkp.ptr as *mut i8,
        dks.ptr as *mut f32,
        dpos.ptr as *const i32,
        dslot.ptr as *const i32,
        batch as i32,
        kv_heads as i32,
        dim as i32,
        max_seq as i32,
    )
    .unwrap();
    k.launch_kv_store_int8(
        dvcur.ptr as *const f32,
        dvp.ptr as *mut i8,
        dvs.ptr as *mut f32,
        dpos.ptr as *const i32,
        dslot.ptr as *const i32,
        batch as i32,
        kv_heads as i32,
        dim as i32,
        max_seq as i32,
    )
    .unwrap();
    k.sync().unwrap();

    let cur_k = Int8Kv::quantize(&k_cur, kv_heads, dim).unwrap();
    let cur_v = Int8Kv::quantize(&v_cur, kv_heads, dim).unwrap();
    let got_kp = dkp.download::<i8>(init_kp.len());
    let got_ks = dks.download::<f32>(init_ks.len());
    let got_vp = dvp.download::<i8>(init_vp.len());
    let got_vs = dvs.download::<f32>(init_vs.len());
    let row = pos * kv_heads * dim;
    let scale_row = pos * kv_heads;
    assert_eq!(
        &got_kp[row..row + cur_k.quantized().len()],
        cur_k.quantized()
    );
    assert_eq!(
        &got_vp[row..row + cur_v.quantized().len()],
        cur_v.quantized()
    );
    assert_eq!(&got_ks[scale_row..scale_row + kv_heads], cur_k.scales());
    assert_eq!(&got_vs[scale_row..scale_row + kv_heads], cur_v.scales());

    let cache_row = pos * kv_heads * dim;
    k_cache[cache_row..cache_row + kv_heads * dim].copy_from_slice(&k_cur);
    v_cache[cache_row..cache_row + kv_heads * dim].copy_from_slice(&v_cur);
    let dq = DevBuf::new(Arc::clone(&h), &q);
    let dout = DevBuf::new(Arc::clone(&h), &vec![0.0f32; batch * n_heads * dim]);
    k.launch_attn_decode_batched_int8_gqa(
        dq.ptr as *const f32,
        dkp.ptr as *const i8,
        dks.ptr as *const f32,
        dvp.ptr as *const i8,
        dvs.ptr as *const f32,
        dout.ptr as *mut f32,
        dpos.ptr as *const i32,
        dslot.ptr as *const i32,
        drun.ptr as *const i32,
        batch as i32,
        n_heads as i32,
        kv_heads as i32,
        dim as i32,
        1.0 / (dim as f32).sqrt(),
        max_seq as i32,
    )
    .unwrap();
    k.sync().unwrap();
    let got = dout.download::<f32>(batch * n_heads * dim);
    let want = attention_decode_int8(
        &q,
        n_heads,
        &Int8Kv::quantize(&k_cache[..(pos + 1) * kv_heads * dim], kv_heads, dim).unwrap(),
        &Int8Kv::quantize(&v_cache[..(pos + 1) * kv_heads * dim], kv_heads, dim).unwrap(),
        1.0 / (dim as f32).sqrt(),
    )
    .unwrap();
    let diff = max_abs_diff(&got, &want);
    assert!(diff < 2e-4, "contiguous INT8 attention diff {diff}");
}

#[test]
#[ignore = "GPU parity; run explicitly with MACH_TEST_INT8_KV=1"]
fn paged_int8_kv_store_and_attention_match_cpu_oracle() {
    let Some((h, k)) = gpu_ctx() else { return };
    let (pages, tpp, batch, kv_heads, n_heads, dim, tokens, pos) = (
        4usize, 4usize, 2usize, 2usize, 4usize, 8usize, 6usize, 5usize,
    );
    let layout = PagedInt8KvLayout::new(pages, tpp, kv_heads, dim).unwrap();
    let page_table = [0usize, 1usize, 2usize, 3usize];
    let table = page_table.iter().map(|&p| p as i32).collect::<Vec<_>>();
    let offsets = [0i32, 2i32];
    let pos_buf = vec![pos as i32; batch];
    let mut seed = 202u64;
    let k_cache0 = random_vec(&mut seed, tokens * kv_heads * dim, 2.0);
    let v_cache0 = random_vec(&mut seed, tokens * kv_heads * dim, 2.0);
    let k_cache1 = random_vec(&mut seed, tokens * kv_heads * dim, 2.0);
    let v_cache1 = random_vec(&mut seed, tokens * kv_heads * dim, 2.0);
    let kq0 = Int8Kv::quantize(&k_cache0, kv_heads, dim).unwrap();
    let vq0 = Int8Kv::quantize(&v_cache0, kv_heads, dim).unwrap();
    let kq1 = Int8Kv::quantize(&k_cache1, kv_heads, dim).unwrap();
    let vq1 = Int8Kv::quantize(&v_cache1, kv_heads, dim).unwrap();
    let mut payload_k = vec![0i8; layout.payload_len()];
    let mut scales_k = vec![0.0f32; layout.scale_len()];
    let mut payload_v = vec![0i8; layout.payload_len()];
    let mut scales_v = vec![0.0f32; layout.scale_len()];
    for s in 0..batch {
        let seq_table = &page_table[s * 2..s * 2 + 2];
        let (ks, vs) = if s == 0 { (&kq0, &vq0) } else { (&kq1, &vq1) };
        scatter_paged_int8(ks, &layout, seq_table, &mut payload_k, &mut scales_k).unwrap();
        scatter_paged_int8(vs, &layout, seq_table, &mut payload_v, &mut scales_v).unwrap();
    }
    let dkp = DevBuf::new(Arc::clone(&h), &payload_k);
    let dks = DevBuf::new(Arc::clone(&h), &scales_k);
    let dvp = DevBuf::new(Arc::clone(&h), &payload_v);
    let dvs = DevBuf::new(Arc::clone(&h), &scales_v);
    let dtable = DevBuf::new(Arc::clone(&h), &table);
    let doffs = DevBuf::new(Arc::clone(&h), &offsets);
    let dpos = DevBuf::new(Arc::clone(&h), &pos_buf);
    let q = random_vec(&mut seed, batch * n_heads * dim, 1.0);
    let dq = DevBuf::new(Arc::clone(&h), &q);
    let dout = DevBuf::new(Arc::clone(&h), &vec![0.0f32; batch * n_heads * dim]);
    k.launch_attn_decode_paged_int8_gqa(
        dq.ptr as *const f32,
        dkp.ptr as *const i8,
        dks.ptr as *const f32,
        dvp.ptr as *const i8,
        dvs.ptr as *const f32,
        dtable.ptr as *const i32,
        dout.ptr as *mut f32,
        dpos.ptr as *const i32,
        doffs.ptr as *const i32,
        batch as i32,
        n_heads as i32,
        kv_heads as i32,
        dim as i32,
        1.0 / (dim as f32).sqrt(),
        tpp as i32,
    )
    .unwrap();
    k.sync().unwrap();
    let got = dout.download::<f32>(batch * n_heads * dim);
    for s in 0..batch {
        let qs = &q[s * n_heads * dim..(s + 1) * n_heads * dim];
        let (ks, vs) = if s == 0 { (&kq0, &vq0) } else { (&kq1, &vq1) };
        let want = attention_decode_int8(qs, n_heads, ks, vs, 1.0 / (dim as f32).sqrt()).unwrap();
        let diff = max_abs_diff(&got[s * n_heads * dim..(s + 1) * n_heads * dim], &want);
        assert!(diff < 2e-4, "paged INT8 attention seq {s} diff {diff}");
    }

    // Store parity: overwrite the current token's paged row and compare to
    // the same per-row CPU quantizer.
    let k_cur = random_vec(&mut seed, batch * kv_heads * dim, 2.0);
    let v_cur = random_vec(&mut seed, batch * kv_heads * dim, 2.0);
    let dkcur = DevBuf::new(Arc::clone(&h), &k_cur);
    let dvcur = DevBuf::new(Arc::clone(&h), &v_cur);
    k.launch_kv_store_paged_int8(
        dkcur.ptr as *const f32,
        dkp.ptr as *mut i8,
        dks.ptr as *mut f32,
        dpos.ptr as *const i32,
        doffs.ptr as *const i32,
        dtable.ptr as *const i32,
        batch as i32,
        kv_heads as i32,
        dim as i32,
        tpp as i32,
    )
    .unwrap();
    k.launch_kv_store_paged_int8(
        dvcur.ptr as *const f32,
        dvp.ptr as *mut i8,
        dvs.ptr as *mut f32,
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
    let got_kp = dkp.download::<i8>(payload_k.len());
    let got_ks = dks.download::<f32>(scales_k.len());
    let got_vp = dvp.download::<i8>(payload_v.len());
    let got_vs = dvs.download::<f32>(scales_v.len());
    let logical = pos / tpp;
    let off = pos % tpp;
    for s in 0..batch {
        let page = page_table[s * 2 + logical];
        let ck = Int8Kv::quantize(
            &k_cur[s * kv_heads * dim..(s + 1) * kv_heads * dim],
            kv_heads,
            dim,
        )
        .unwrap();
        let cv = Int8Kv::quantize(
            &v_cur[s * kv_heads * dim..(s + 1) * kv_heads * dim],
            kv_heads,
            dim,
        )
        .unwrap();
        for head in 0..kv_heads {
            let pi = layout.payload_index(page, off, head, 0).unwrap();
            assert_eq!(
                &got_kp[pi..pi + dim],
                &ck.quantized()[head * dim..(head + 1) * dim]
            );
            assert_eq!(
                &got_vp[pi..pi + dim],
                &cv.quantized()[head * dim..(head + 1) * dim]
            );
            let si = layout.scale_index(page, off, head).unwrap();
            assert_eq!(got_ks[si], ck.scales()[head]);
            assert_eq!(got_vs[si], cv.scales()[head]);
        }
    }
}
