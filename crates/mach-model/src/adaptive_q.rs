//! Bandwidth-adaptive q* offload-ratio kernel (FreeToken P2).
//!
//! Decides the per-token split `q*` of MoE expert work between the GPU
//! (`q*`, 1.0 = everything on GPU) and the host CPU (`1 - q*`) for a
//! host-resident expert offload. The placement machinery already exists
//! (`moe_backend::LruExpertCache` + `moe_offload`); this module supplies the
//! *ratio* those layers can use to steer work GPU-vs-CPU.
//!
//! This is the **offline, CPU-only decision kernel**: pure float math, no GPU
//! or I/O, unit-testable with plain `cargo test`. Real bandwidth / compute
//! probing on a live ROCm device (PCIe H2D/D2H memcpy timing + timed expert
//! MLP, see the HIP-gated `adaptive` module) is a later batch; this module
//! consumes the *outputs* of such probes through [`AdaptiveConfig`].
//!
//! # Model
//!
//! Per token, `E` experts are routed, each costing `F` FLOPs and `W` bytes of
//! weights (bytes that must cross the PCIe bus when the expert is computed on
//! the GPU). With fraction `q` on the GPU and `1 - q` on the CPU, the per-token
//! times are:
//!
//! ```text
//! gpu_time(q) = q*E*F/G            # GPU compute, FLOPs/s = G
//! cpu_time(q) = (1 - q)*E*F/C      # CPU compute, FLOPs/s = C
//! transfer(q) = q*E*W/BW           # host -> GPU weight fetch, bytes/s = BW
//! ```
//!
//! The GPU and CPU pipelines run concurrently per token, so latency is set by
//! the slower of the two *binding* pipelines. The GPU is the primary engine
//! (the same philosophy as the existing per-miss rule in `adaptive.rs`: GPU is
//! the default, CPU only when it is cheaper), so we pay the bus traffic only
//! when the GPU is at least as fast per expert as the CPU (`F/G <= F/C`). In
//! that feasible regime the binding resources are the bus and the CPU:
//!
//! ```text
//! q*E*W/BW = (1 - q)*E*F/C
//!   =>  q* = (F/C) / (F/C + W/BW)
//! ```
//!
//! (`E` cancels: `q*` depends on per-expert times, not on how many experts a
//! token routes.) The result is clamped to `[0, 1]`. Extremes:
//!
//! - infinite bandwidth -> `1.0` (transfer is free -> everything on the GPU);
//! - zero bandwidth -> `0.0` (nothing can be fetched -> everything on CPU);
//! - faster CPU -> lower `q*` (the CPU drains the work cheaply);
//! - GPU slower per expert than the CPU -> `0.0` (not worth the fetch).
//!
//! Zero / NaN inputs are handled defensively (see [`optimal_offload_ratio`]).

#![forbid(unsafe_code)]

/// Host-side measurements driving the per-token GPU/CPU expert split.
///
/// All values are plain `f64` so the kernel stays pure and testable without a
/// device; a later batch feeds these from real ROCm / PCIe probes.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveConfig {
    /// Effective PCIe host-to-device bandwidth in Gbit/s (bytes/s = Gbps * 1e9 / 8).
    pub pcie_bw_gbps: f64,
    /// Sustained CPU FLOPs/s available for expert (SwiGLU) math.
    pub cpu_flops: f64,
    /// Sustained GPU FLOPs/s available for expert (SwiGLU) math.
    pub gpu_flops: f64,
    /// Bytes of weights per expert that cross the bus when computed on the GPU.
    pub expert_bytes: f64,
    /// Routed experts computed per token (top-k summed over MoE layers).
    pub experts_per_token: f64,
    /// FLOPs to compute one expert for one token.
    pub flops_per_expert_token: f64,
}

/// Returns `q*` in `[0, 1]`: the fraction of per-token expert work routed to
/// the GPU (`1.0` = everything on the GPU, `0.0` = everything on the CPU).
///
/// See the module documentation for the model. Defensive behaviour:
///
/// - any `NaN` input (unmeasured value) -> neutral default `0.5`;
/// - zero bus or zero GPU compute -> `0.0` (nothing can reach the GPU);
/// - zero CPU compute or zero transfer cost -> `1.0` (GPU is the only option);
/// - zero experts per token -> `0.5` (no work to split);
/// - negative inputs are clamped to zero first.
#[must_use]
pub fn optimal_offload_ratio(cfg: AdaptiveConfig) -> f64 {
    // NaN anywhere means an unmeasured / unknown input; do not trust any part
    // of the model and return the neutral 50/50 default.
    if [
        cfg.pcie_bw_gbps,
        cfg.cpu_flops,
        cfg.gpu_flops,
        cfg.expert_bytes,
        cfg.experts_per_token,
        cfg.flops_per_expert_token,
    ]
    .into_iter()
    .any(f64::is_nan)
    {
        return 0.5;
    }

    // Treat negative "capacities" as zero (no capacity) instead of propagating
    // nonsense ratios. Convert Gbit/s to bytes/s.
    let bw = (cfg.pcie_bw_gbps * 1e9 / 8.0).max(0.0);
    let cpu = cfg.cpu_flops.max(0.0);
    let gpu = cfg.gpu_flops.max(0.0);
    let w = cfg.expert_bytes.max(0.0);
    let e = cfg.experts_per_token.max(0.0);
    let f = cfg.flops_per_expert_token.max(0.0);

    // No expert work per token -> the split ratio is meaningless; return the
    // neutral default so callers get a deterministic, in-range answer.
    if e == 0.0 {
        return 0.5;
    }
    // Nothing can be fetched (dead bus) or computed (dead GPU) -> all CPU.
    if bw == 0.0 || gpu == 0.0 {
        return 0.0;
    }
    // No CPU capacity or no transfer cost -> the GPU is the only option.
    if cpu == 0.0 || w == 0.0 {
        return 1.0;
    }

    // Per-expert times in seconds. `E` cancels out of the ratio, so it is not
    // needed below; it is kept in the derivation above for traceability.
    let gpu_expert_sec = f / gpu;
    let cpu_expert_sec = f / cpu;
    let transfer_expert_sec = w / bw;

    // Feasibility gate: if the GPU is slower per expert than the CPU, fetching
    // its weights is never worth it -> everything on the CPU.
    if gpu_expert_sec > cpu_expert_sec {
        return 0.0;
    }

    // Non-finite per-expert compute (e.g. an unmeasured +INF flops value)
    // would make the ratio NaN; treat the model as unmeasured -> neutral.
    // (+INF bandwidth/expert bytes are meaningful extremes handled above:
    // transfer 0 -> all-GPU, transfer INF -> all-CPU.)
    if !gpu_expert_sec.is_finite() || !cpu_expert_sec.is_finite() {
        return 0.5;
    }

    // Balance the bus against the CPU: q*E*W/BW == (1 - q)*E*F/C.
    let q = cpu_expert_sec / (cpu_expert_sec + transfer_expert_sec);
    q.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Realistic default: 32 GB/s PCIe (256 Gbit/s), GPU 100x the CPU, 1 GB
    /// experts, 8 routed experts per token, 1 GFLOP per expert-token.
    fn base() -> AdaptiveConfig {
        AdaptiveConfig {
            pcie_bw_gbps: 256.0,
            cpu_flops: 1e12,
            gpu_flops: 1e14,
            expert_bytes: 1e9,
            experts_per_token: 8.0,
            flops_per_expert_token: 1e9,
        }
    }

    #[test]
    fn infinite_bandwidth_goes_all_gpu() {
        let mut cfg = base();
        cfg.pcie_bw_gbps = f64::INFINITY;
        assert_eq!(optimal_offload_ratio(cfg), 1.0);
    }

    #[test]
    fn infinite_flops_returns_neutral_default() {
        // INF flops would make cpu_expert_sec = INF and the ratio NaN; the
        // guard must return the neutral default instead (regression).
        let mut cfg = base();
        cfg.flops_per_expert_token = f64::INFINITY;
        let q = optimal_offload_ratio(cfg);
        assert!(q.is_finite(), "q must stay finite, got {q}");
        assert_eq!(q, 0.5, "non-finite input -> neutral default");
    }

    #[test]
    fn zero_bandwidth_goes_all_cpu() {
        let mut cfg = base();
        cfg.pcie_bw_gbps = 0.0;
        assert_eq!(optimal_offload_ratio(cfg), 0.0);
    }

    #[test]
    fn bandwidth_increase_never_lowers_qstar() {
        let mut low = base();
        low.pcie_bw_gbps = 8.0; // 1 GB/s
        let mut mid = base();
        mid.pcie_bw_gbps = 64.0; // 8 GB/s
        let mut high = base();
        high.pcie_bw_gbps = 256.0; // 32 GB/s

        let q_low = optimal_offload_ratio(low);
        let q_mid = optimal_offload_ratio(mid);
        let q_high = optimal_offload_ratio(high);

        assert!(
            q_low <= q_mid && q_mid <= q_high,
            "q* must be non-decreasing in bandwidth: {q_low} <= {q_mid} <= {q_high}"
        );
    }

    #[test]
    fn faster_cpu_never_raises_qstar() {
        let mut slow_cpu = base();
        slow_cpu.cpu_flops = 1e11;
        let mut mid_cpu = base();
        mid_cpu.cpu_flops = 1e12;
        let mut fast_cpu = base();
        fast_cpu.cpu_flops = 1e13;

        let q_slow = optimal_offload_ratio(slow_cpu);
        let q_mid = optimal_offload_ratio(mid_cpu);
        let q_fast = optimal_offload_ratio(fast_cpu);

        assert!(
            q_slow >= q_mid && q_mid >= q_fast,
            "q* must be non-increasing as the CPU gets faster: {q_slow} >= {q_mid} >= {q_fast}"
        );
    }

    #[test]
    fn gpu_slower_than_cpu_falls_back_to_cpu() {
        let mut cfg = base();
        cfg.gpu_flops = 5e11; // GPU per-expert compute now slower than the CPU
        assert_eq!(optimal_offload_ratio(cfg), 0.0);
    }

    #[test]
    fn finite_reasonable_inputs_stay_in_unit_range() {
        let cfg = base();
        let q = optimal_offload_ratio(cfg);
        // Strictly interior: a finite bus yields a real split, not an extreme.
        assert!(
            (0.0..1.0).contains(&q),
            "expected an interior split, got {q}"
        );

        // Lock the closed form: q* = (F/C) / (F/C + W/BW).
        let f = cfg.flops_per_expert_token;
        let cpu_expert_sec = f / cfg.cpu_flops;
        let transfer_expert_sec = cfg.expert_bytes / (cfg.pcie_bw_gbps * 1e9 / 8.0);
        let expected = cpu_expert_sec / (cpu_expert_sec + transfer_expert_sec);
        assert!((q - expected).abs() < 1e-12);

        // A spread of finite configurations all stay in [0, 1].
        let variants = [
            AdaptiveConfig {
                pcie_bw_gbps: 4.0,
                ..base()
            },
            AdaptiveConfig {
                cpu_flops: 1e9,
                ..base()
            },
            AdaptiveConfig {
                gpu_flops: 1e11,
                ..base()
            },
            AdaptiveConfig {
                expert_bytes: 1e6,
                ..base()
            },
            AdaptiveConfig {
                experts_per_token: 64.0,
                ..base()
            },
            AdaptiveConfig {
                flops_per_expert_token: 1e11,
                ..base()
            },
        ];
        for v in variants {
            let qv = optimal_offload_ratio(v);
            assert!((0.0..=1.0).contains(&qv), "q* out of range for {v:?}: {qv}");
        }
    }

    #[test]
    fn nan_inputs_return_neutral_default() {
        for i in 0..6 {
            let mut cfg = base();
            match i {
                0 => cfg.pcie_bw_gbps = f64::NAN,
                1 => cfg.cpu_flops = f64::NAN,
                2 => cfg.gpu_flops = f64::NAN,
                3 => cfg.expert_bytes = f64::NAN,
                4 => cfg.experts_per_token = f64::NAN,
                _ => cfg.flops_per_expert_token = f64::NAN,
            }
            assert_eq!(
                optimal_offload_ratio(cfg),
                0.5,
                "field {i} NaN must be defended"
            );
        }
    }

    #[test]
    fn zero_capacity_inputs_return_defensive_extremes() {
        // No CPU capacity -> everything on the GPU.
        let mut cfg = base();
        cfg.cpu_flops = 0.0;
        assert_eq!(optimal_offload_ratio(cfg), 1.0);

        // No GPU capacity -> everything on the CPU.
        let mut cfg = base();
        cfg.gpu_flops = 0.0;
        assert_eq!(optimal_offload_ratio(cfg), 0.0);

        // No transfer cost -> everything on the GPU.
        let mut cfg = base();
        cfg.expert_bytes = 0.0;
        assert_eq!(optimal_offload_ratio(cfg), 1.0);

        // No expert work per token -> neutral default.
        let mut cfg = base();
        cfg.experts_per_token = 0.0;
        assert_eq!(optimal_offload_ratio(cfg), 0.5);

        // No compute work -> deterministic in-range result (no transfer is paid).
        let mut cfg = base();
        cfg.flops_per_expert_token = 0.0;
        assert_eq!(optimal_offload_ratio(cfg), 0.0);

        // Negative bandwidth is clamped to the zero-bandwidth behaviour.
        let mut cfg = base();
        cfg.pcie_bw_gbps = -1.0;
        assert_eq!(optimal_offload_ratio(cfg), 0.0);
    }
}
