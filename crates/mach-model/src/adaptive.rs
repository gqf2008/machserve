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

/// Measured PCIe bandwidth and per-expert CPU compute cost.
#[derive(Debug, Clone, Copy)]
pub struct BandwidthProfile {
    /// Effective PCIe bandwidth (bytes/second) measured on this machine.
    pub pcie_bytes_per_sec: f64,
    /// Wall-time to compute one expert for one token on the CPU (seconds).
    pub cpu_expert_sec: f64,
}

/// Decision for one routed expert miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FetchChoice {
    /// Fetch the expert weight to the GPU and compute there.
    FetchGpu,
    /// Compute the expert on the CPU instead of fetching.
    ComputeCpu,
}

impl BandwidthProfile {
    /// For one miss of an expert of `expert_bytes`, prefer CPU when the CPU
    /// compute time is strictly cheaper than the PCIe fetch pull.
    #[must_use]
    pub fn choose(&self, expert_bytes: usize) -> FetchChoice {
        let fetch_sec = expert_bytes as f64 / self.pcie_bytes_per_sec.max(1.0);
        if self.cpu_expert_sec < fetch_sec {
            FetchChoice::ComputeCpu
        } else {
            FetchChoice::FetchGpu
        }
    }
}

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

/// Realtime q* profile: continually folds newly-measured PCIe bandwidth samples
/// into a smoothed estimate so that, when the bus is contended (bandwidth drops),
/// the per-miss decision flips to CPU, and recovers only gradually on the CPU
/// (or as the bus frees).
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveProfile {
    current: BandwidthProfile,
    /// EMA weight for a degrading (slower) sample; recovery uses 0.25.
    alpha: f64,
}

impl AdaptiveProfile {
    #[must_use]
    pub fn new(pcie_bytes_per_sec: f64, cpu_expert_sec: f64, alpha: f64) -> Self {
        Self {
            current: BandwidthProfile {
                pcie_bytes_per_sec,
                cpu_expert_sec,
            },
            alpha,
        }
    }

    /// Folds a freshly-measured PCIe sample into the estimate. Degradation
    /// (contention) reacts at `alpha`; recovery (bandwidth back) is slower (0.25),
    /// so a burst of I/O promptly shifts work to CPU but does not thrash.
    pub fn observe(&mut self, sample_bytes_per_sec: f64) {
        let w = if sample_bytes_per_sec < self.current.pcie_bytes_per_sec {
            self.alpha
        } else {
            0.25
        };
        let cur = self.current.pcie_bytes_per_sec;
        self.current.pcie_bytes_per_sec = cur * (1.0 - w) + sample_bytes_per_sec * w;
    }

    #[must_use]
    pub fn profile(&self) -> BandwidthProfile {
        self.current
    }

    /// Per-miss decision under the current (possibly contended) estimate.
    pub fn choose(&self, expert_bytes: usize) -> FetchChoice {
        self.current.choose(expert_bytes)
    }
}
#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_bandwidth_prefers_gpu() {
        let prof = BandwidthProfile {
            pcie_bytes_per_sec: 10_000_000_000.0,
            cpu_expert_sec: 100e-6,
        };
        assert_eq!(prof.choose(96 * 1024), FetchChoice::FetchGpu);
    }

    #[test]
    fn contended_bus_prefers_cpu() {
        let prof = BandwidthProfile {
            pcie_bytes_per_sec: 100_000_000.0,
            cpu_expert_sec: 10e-6,
        };
        assert_eq!(prof.choose(96 * 1024), FetchChoice::ComputeCpu);
    }

    #[test]
    fn zero_bandwidth_is_safe() {
        let prof = BandwidthProfile {
            pcie_bytes_per_sec: 0.0,
            cpu_expert_sec: 1e-6,
        };
        assert_eq!(prof.choose(96 * 1024), FetchChoice::ComputeCpu);
    }

    #[test]
    fn realtime_contention_flips_choice_to_cpu() {
        let mut q = AdaptiveProfile::new(10_000_000_000.0, 10e-6, 0.9);
        assert_eq!(q.choose(96 * 1024), FetchChoice::FetchGpu);
        q.observe(100_000_000.0);
        assert_eq!(q.choose(96 * 1024), FetchChoice::ComputeCpu);
    }

    #[test]
    fn realtime_recovery_is_slower() {
        let mut q = AdaptiveProfile::new(100_000_000.0, 30e-6, 0.9);
        q.observe(10_000_000_000.0);
        assert_eq!(q.choose(96 * 1024), FetchChoice::ComputeCpu);
        q.observe(10_000_000_000.0);
        q.observe(10_000_000_000.0);
        assert_eq!(q.choose(96 * 1024), FetchChoice::FetchGpu);
    }

    #[test]
    fn zero_bandwidth_observe_is_safe() {
        let mut q = AdaptiveProfile::new(10e6, 1e-6, 0.5);
        q.observe(0.0);
        assert_eq!(q.choose(96 * 1024), FetchChoice::ComputeCpu);
    }
}
