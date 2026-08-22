//! The `Kernel` trait: the contract every kernel implementation satisfies.

use crate::backend::BackendCaps;

/// A registered kernel implementation.
///
/// Kernels are identified by a `(family, name)` pair, e.g.
/// `("attention", "flashinfer.decode")` or `("gemm", "cutlass.fp8")`.
/// Implementations are stateless descriptors plus a version; the actual
/// compute entry points live in the [`crate::ops`] traits.
pub trait Kernel: Send + Sync {
    /// Family, e.g. `"attention"`, `"gemm"`, `"moe"`, `"quant"`, `"sampling"`.
    fn family(&self) -> &'static str;

    /// Short implementation name, e.g. `"flashinfer.decode"`.
    fn name(&self) -> &'static str;

    /// Capabilities of this implementation.
    fn caps(&self) -> BackendCaps;

    /// Source version, e.g. the pinned thirdparty tag (`"v0.3.2"`).
    fn version(&self) -> &'static str;
}

/// Convenience macro to declare a simple kernel descriptor struct.
///
/// ```
/// use mach_kernel::{kernel, BackendCaps};
///
/// kernel!(MyKernel, "attention", "flashinfer.decode", BackendCaps::cpu(), "0.0.0");
/// ```
#[macro_export]
macro_rules! kernel {
    ($name:ident, $family:expr, $impl_name:expr, $caps:expr, $version:expr) => {
        #[derive(Debug, Clone, Copy)]
        pub struct $name;

        impl $crate::Kernel for $name {
            fn family(&self) -> &'static str {
                $family
            }
            fn name(&self) -> &'static str {
                $impl_name
            }
            fn caps(&self) -> $crate::BackendCaps {
                $caps
            }
            fn version(&self) -> &'static str {
                $version
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    kernel!(
        TestKernel,
        "gemm",
        "cpu.reference",
        BackendCaps::cpu(),
        "0.1.0"
    );

    #[test]
    fn kernel_descriptor_works() {
        let k = TestKernel;
        assert_eq!(k.family(), "gemm");
        assert_eq!(k.name(), "cpu.reference");
        assert_eq!(k.caps().backend, crate::BackendId::Cpu);
        assert_eq!(k.version(), "0.1.0");
    }
}
