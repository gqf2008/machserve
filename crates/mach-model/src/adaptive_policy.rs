//! Bandwidth-adaptive placement policy (FreeToken-style q* hook), hip-free.
//!
//! The decision logic — "is fetching this expert over PCIe cheaper than
//! computing it on the CPU, under the current (possibly contended) bandwidth
//! estimate" — is pure arithmetic and lived in the hip-gated `adaptive`
//! module, so `cargo test -p mach-model --lib` reported no tests for it
//! (`adaptive` is `#[cfg(feature = "hip")]`). The measuring probe stays in
//! `adaptive`; the policy and its unit tests live here so the default test面
//! actually executes them.

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
