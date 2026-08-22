//! Runtime devices.

use core::fmt;

/// A logical device that tensors and kernels live on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// Host CPU.
    Cpu,
    /// NVIDIA CUDA device, indexed by ordinal.
    Cuda(u32),
    /// AMD ROCm device, indexed by ordinal.
    Hip(u32),
}

impl Device {
    /// Returns `true` for host CPU devices.
    #[must_use]
    pub fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Returns the ordinal for accelerator devices, or `None` on CPU.
    #[must_use]
    pub fn ordinal(self) -> Option<u32> {
        match self {
            Self::Cpu => None,
            Self::Cuda(i) | Self::Hip(i) => Some(i),
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Cuda(i) => write!(f, "cuda:{i}"),
            Self::Hip(i) => write!(f, "hip:{i}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_and_ordinal() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
        assert_eq!(Device::Hip(3).to_string(), "hip:3");
        assert_eq!(Device::Cuda(1).ordinal(), Some(1));
        assert_eq!(Device::Cpu.ordinal(), None);
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cuda(0).is_cpu());
    }
}
