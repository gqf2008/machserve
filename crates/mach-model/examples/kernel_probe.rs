//! Isolates each custom HIP kernel at Qwen-scale shapes and compares against a
//! CPU reference, to find which kernel diverges on larger configs.

#[cfg(feature = "hip")]
fn main() {
    use mach_kernel_sys::hip::{self, Hip};
    use mach_model::kernels::{HipKernels, RopeParams};
    use std::sync::Arc;

    let h: Arc<Hip> = hip::hip().expect("hip");
    let k = HipKernels::new(Arc::clone(&h)).expect("kernels");
    let stream = k.stream;

    let d = 896usize;
    let n_heads = 14usize;
    let n_kv = 2usize;
    let hd = 64usize;
    let inter = 4864usize;
    let pos: i32 = 3;

    // helper allocations
    let dev = |n: usize| hip::malloc(&h, n * 4).unwrap() as *mut f32;
    let free_all: Vec<*mut core::ffi::c_void> = Vec::new();
    let _ = free_all;

    // ---- rms_norm (two data scales) ----
    {
        // large-ish values (the original probe)
        let x0: Vec<f32> = (0..d).map(|i| ((i % 97) as f32 - 48.0) / 100.0).collect();
        // Qwen-embedding-like small values
        let x1: Vec<f32> = (0..d).map(|i| ((i % 103) as f32 - 51.0) / 1000.0).collect();
        let w: Vec<f32> = (0..d).map(|i| ((i % 31) as f32 - 15.0) / 300.0).collect();
        // real Qwen embedding row + rms weight
        let real = load_qwen_rms_data();
        let w_real: Vec<f32> = real.1.clone();
        for (x, wt, tag) in [
            (&x0, &w, "large"),
            (&x1, &w, "small"),
            (&real.0, &w_real, "real-emb"),
        ] {
            let dx = dev(d);
            let dw = dev(d);
            let dy = dev(d);
            hip::memcpy(
                &h,
                dx as *mut _,
                x.as_ptr() as *const _,
                d * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )
            .unwrap();
            hip::memcpy(
                &h,
                dw as *mut _,
                wt.as_ptr() as *const _,
                d * 4,
                hip::HIP_MEMCPY_HOST_TO_DEVICE,
            )
            .unwrap();
            k.launch_rms_norm(dx, dw, dy, 1, d as i32, 1e-6).unwrap();
            let mut y = vec![0.0f32; d];
            hip::memcpy(
                &h,
                y.as_mut_ptr() as *mut _,
                dy as *const _,
                d * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )
            .unwrap();
            let mean: f32 = x.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let inv = 1.0 / (mean + 1e-6).sqrt();
            let mut err = 0.0f32;
            for i in 0..d {
                err = err.max((y[i] - x[i] * inv * wt[i]).abs());
            }
            println!("rms_norm d={d} [{tag}] maxerr={err:.3e}");
            hip::free(&h, dx as *mut _).unwrap();
            hip::free(&h, dw as *mut _).unwrap();
            hip::free(&h, dy as *mut _).unwrap();
        }
    }

    // ---- rope ----
    {
        let total = n_heads * hd;
        let q: Vec<f32> = (0..total)
            .map(|i| ((i % 61) as f32 - 30.0) / 100.0)
            .collect();
        let dq = dev(total);
        let dk = dev(n_kv * hd);
        let dpos = hip::malloc(&h, 4).unwrap() as *mut i32;
        let pos_host = pos;
        hip::memcpy(
            &h,
            dpos as *mut _,
            &pos_host as *const _ as *const core::ffi::c_void,
            4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        hip::memcpy(
            &h,
            dq as *mut _,
            q.as_ptr() as *const _,
            total * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        k.launch_rope(
            dq,
            dk,
            dpos,
            n_heads as i32,
            n_kv as i32,
            hd as i32,
            RopeParams {
                theta: 1_000_000.0,
                yarn: 0,
                factor: 0.0,
                beta_fast: 0.0,
                beta_slow: 0.0,
                orig_len: 0,
                attn_factor: 1.0,
                // MACH_ROPE_INTERLEAVE=1 exercises the DeepSeek-V2 adjacent-pair
                // convention; default 0 is the Qwen3/Llama split-halves one.
                interleave: i32::from(std::env::var("MACH_ROPE_INTERLEAVE").as_deref() == Ok("1")),
            },
        )
        .unwrap();
        let mut y = vec![0.0f32; total];
        hip::memcpy(
            &h,
            y.as_mut_ptr() as *mut _,
            dq as *const _,
            total * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        // cpu rope
        let mut qc = q.clone();
        for hh in 0..n_heads {
            for dd in 0..(hd / 2) {
                let freq = 1.0 / (1_000_000.0f32).powf(2.0 * dd as f32 / hd as f32);
                let ang = pos as f32 * freq;
                let (c, s) = (ang.cos(), ang.sin());
                let idx = hh * hd + 2 * dd;
                let (a, b) = (qc[idx], qc[idx + 1]);
                qc[idx] = a * c - b * s;
                qc[idx + 1] = a * s + b * c;
            }
        }
        let mut err = 0.0f32;
        for i in 0..total {
            err = err.max((y[i] - qc[i]).abs());
        }
        println!("rope total={total} maxerr={err:.3e}");
        hip::free(&h, dq as *mut _).unwrap();
        hip::free(&h, dk as *mut _).unwrap();
        hip::free(&h, dpos as *mut _).unwrap();
    }

    // ---- silu_mul + add ----
    {
        let a: Vec<f32> = (0..inter)
            .map(|i| ((i % 71) as f32 - 35.0) / 100.0)
            .collect();
        let b: Vec<f32> = (0..inter)
            .map(|i| ((i % 43) as f32 - 21.0) / 100.0)
            .collect();
        let da = dev(inter);
        let db = dev(inter);
        let dout = dev(inter);
        hip::memcpy(
            &h,
            da as *mut _,
            a.as_ptr() as *const _,
            inter * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        hip::memcpy(
            &h,
            db as *mut _,
            b.as_ptr() as *const _,
            inter * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        k.launch_silu_mul(da, db, dout, inter as i32).unwrap();
        let mut y = vec![0.0f32; inter];
        hip::memcpy(
            &h,
            y.as_mut_ptr() as *mut _,
            dout as *const _,
            inter * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        let mut err = 0.0f32;
        for i in 0..inter {
            let v = b[i];
            let want = a[i] * (v / (1.0 + (-v).exp()));
            err = err.max((y[i] - want).abs());
        }
        println!("silu_mul n={inter} maxerr={err:.3e}");
        hip::free(&h, da as *mut _).unwrap();
        hip::free(&h, db as *mut _).unwrap();
        hip::free(&h, dout as *mut _).unwrap();
    }

    // ---- attn_decode (GQA, groups=7) ----
    {
        let total_q = n_heads * hd;
        let npos = (pos + 1) as usize;
        let q: Vec<f32> = (0..total_q)
            .map(|i| ((i % 89) as f32 - 44.0) / 100.0)
            .collect();
        let kc: Vec<f32> = (0..npos * n_kv * hd)
            .map(|i| ((i % 53) as f32 - 26.0) / 100.0)
            .collect();
        let vc: Vec<f32> = (0..npos * n_kv * hd)
            .map(|i| ((i % 67) as f32 - 33.0) / 100.0)
            .collect();
        let dq = dev(total_q);
        let dkc = dev(kc.len());
        let dvc = dev(vc.len());
        let dout = dev(total_q);
        let dpos = hip::malloc(&h, 4).unwrap() as *mut i32;
        let pos_host = pos;
        hip::memcpy(
            &h,
            dpos as *mut _,
            &pos_host as *const _ as *const core::ffi::c_void,
            4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        hip::memcpy(
            &h,
            dq as *mut _,
            q.as_ptr() as *const _,
            total_q * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        hip::memcpy(
            &h,
            dkc as *mut _,
            kc.as_ptr() as *const _,
            kc.len() * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        hip::memcpy(
            &h,
            dvc as *mut _,
            vc.as_ptr() as *const _,
            vc.len() * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        let scale = 1.0 / (hd as f32).sqrt();
        k.launch_attn_decode(
            dq,
            dkc,
            dvc,
            dout,
            dpos,
            n_heads as i32,
            n_kv as i32,
            hd as i32,
            scale,
            pos + 1, // smem scores capacity must cover positions 0..=pos
        )
        .unwrap();
        let mut y = vec![0.0f32; total_q];
        hip::memcpy(
            &h,
            y.as_mut_ptr() as *mut _,
            dout as *const _,
            total_q * 4,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .unwrap();
        // cpu GQA attention
        let groups = n_heads / n_kv;
        let mut err = 0.0f32;
        for hh in 0..n_heads {
            let kv = hh / groups;
            let qh = &q[hh * hd..(hh + 1) * hd];
            let mut scores = vec![0.0f32; npos];
            let mut maxv = f32::NEG_INFINITY;
            for p in 0..npos {
                let kp = &kc[(p * n_kv + kv) * hd..(p * n_kv + kv + 1) * hd];
                let s = qh.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>() * scale;
                scores[p] = s;
                maxv = maxv.max(s);
            }
            let mut tot = 0.0f32;
            for s in &mut scores {
                *s = (*s - maxv).exp();
                tot += *s;
            }
            for dd in 0..hd {
                let acc: f32 = (0..npos)
                    .map(|p| scores[p] * vc[(p * n_kv + kv) * hd + dd])
                    .sum();
                let want = acc / tot;
                err = err.max((y[hh * hd + dd] - want).abs());
            }
        }
        println!("attn_decode heads={n_heads} kv={n_kv} hd={hd} pos={pos} maxerr={err:.3e}");
        hip::free(&h, dq as *mut _).unwrap();
        hip::free(&h, dkc as *mut _).unwrap();
        hip::free(&h, dvc as *mut _).unwrap();
        hip::free(&h, dout as *mut _).unwrap();
        hip::free(&h, dpos as *mut _).unwrap();
    }
    let _ = stream;
    println!("probe done");
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!("kernel_probe requires the `hip` feature");
}

/// Loads Qwen embedding row 1 and layer-0 input_layernorm weights for testing.
#[cfg(feature = "hip")]
fn load_qwen_rms_data() -> (Vec<f32>, Vec<f32>) {
    use mach_model::Config;
    use mach_model::loader::load_safetensors;
    let mut cfg = Config::llama(896, 24, 14, 2, 151936, 2048);
    cfg.intermediate_size = 4864;
    let w = load_safetensors(
        &std::path::Path::new(".models").join("qwen-0.5b.safetensors"),
        &cfg,
        true,
    )
    .expect("load qwen");
    let d = cfg.d_model;
    (w.tok_emb[d..2 * d].to_vec(), w.layers[0].rms_attn.clone())
}
