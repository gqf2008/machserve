//! Bandwidth-adaptive execution probe + q* placement (FreeToken-style).
//!
//! An offloaded expert costs PCIe bandwidth to fetch, but can also be computed
//! directly on the CPU. Given the measured PCIe bandwidth and the compute cost of
//! one expert, we pick the cheaper path per miss. This is the q* hook: when PCIe
//! is fast, fetch to GPU; when CPU is cheaper than the fetch pull (e.g. a
//! saturated bus), compute on CPU.

use crate::Error;
use crate::config::Config;
use crate::moe_offload::expert_mlp;
use mach_kernel_sys::hip::{self, Hip};
use std::time::Instant;

// The placement policy is hip-free; re-exported so `crate::adaptive::` keeps
// working for the hip callers (model.rs) and for `FetchChoice` users.
pub use crate::adaptive_policy::{AdaptiveProfile, BandwidthProfile, FetchChoice};

/// Measures effective PCIe throughput and estimates per-expert CPU cost.
pub struct BandwidthProbe {
    pub profile: BandwidthProfile,
}

impl BandwidthProbe {
    /// Measures PCIe bandwidth on `hip` and estimates CPU expert cost for `cfg`.
    pub fn measure(hip: &Hip, cfg: &Config) -> Result<Self, Error> {
        // PCIe: time a 1 MiB host<->device round trip.
        let bytes = 1 << 20;
        let mut host = vec![0.0f32; bytes / 4];
        let dev =
            hip::malloc(hip, bytes).map_err(|e| Error::Model(format!("probe malloc: {e}")))?;
        let start = Instant::now();
        hip::memcpy(
            hip,
            dev,
            host.as_ptr() as *const core::ffi::c_void,
            bytes,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .map_err(|e| Error::Model(format!("probe h2d: {e}")))?;
        hip::memcpy(
            hip,
            host.as_mut_ptr() as *mut core::ffi::c_void,
            dev,
            bytes,
            hip::HIP_MEMCPY_DEVICE_TO_HOST,
        )
        .map_err(|e| Error::Model(format!("probe d2h: {e}")))?;
        let elapsed = start.elapsed().as_secs_f64().max(1e-9);
        let pcie_bytes_per_sec = (2.0 * bytes as f64) / elapsed;
        hip::free(hip, dev).map_err(|e| Error::Model(format!("probe free: {e}")))?;

        // CPU: time one expert_mlp on a small deterministic expert of the model shape.
        let d = cfg.d_model;
        let inter = cfg.expert_size();
        let genv = |n: usize, seed: u64| -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (((s >> 33) as f64) / ((1u64 << 31) as f64)) as f32 - 1.0
                })
                .collect()
        };
        let xn: Vec<f32> = genv(d, 1);
        let wg: Vec<f32> = genv(inter * d, 2);
        let wu: Vec<f32> = genv(inter * d, 3);
        let wd: Vec<f32> = genv(d * inter, 4);
        let n = 100;
        let start = Instant::now();
        for _ in 0..n {
            let _ = expert_mlp(&xn, &wg, &wu, &wd, inter, d);
        }
        let cpu_expert_sec = start.elapsed().as_secs_f64() / n as f64;

        Ok(Self {
            profile: BandwidthProfile {
                pcie_bytes_per_sec,
                cpu_expert_sec,
            },
        })
    }
}
