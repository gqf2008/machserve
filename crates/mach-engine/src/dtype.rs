//! Element data types.

use core::fmt;

/// Element dtype. Mirrors the set used by modern LLM inference stacks,
/// including the FP8 variants required by FP8-quantized models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    F16,
    Bf16,
    /// NVIDIA/AMD FP8 e4m3fn.
    Fp8E4m3fn,
    /// FP8 e5m2.
    Fp8E5m2,
    /// FP8 e4m3fnuz (AMD).
    Fp8E4m3fnuz,
    /// FP8 e5m2fnuz (AMD).
    Fp8E5m2fnuz,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
}

impl DType {
    /// Size of a single element in bytes, when the dtype is fixed-size.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F16 | Self::Bf16 | Self::I16 | Self::U16 => 2,
            Self::Fp8E4m3fn
            | Self::Fp8E5m2
            | Self::Fp8E4m3fnuz
            | Self::Fp8E5m2fnuz
            | Self::I8
            | Self::U8
            | Self::Bool => 1,
        }
    }

    /// Returns `true` for floating point dtypes.
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Self::F32
                | Self::F64
                | Self::F16
                | Self::Bf16
                | Self::Fp8E4m3fn
                | Self::Fp8E5m2
                | Self::Fp8E4m3fnuz
                | Self::Fp8E5m2fnuz
        )
    }

    /// Returns `true` for FP8 dtypes.
    #[must_use]
    pub const fn is_fp8(self) -> bool {
        matches!(
            self,
            Self::Fp8E4m3fn | Self::Fp8E5m2 | Self::Fp8E4m3fnuz | Self::Fp8E5m2fnuz
        )
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Fp8E4m3fn => "fp8_e4m3fn",
            Self::Fp8E5m2 => "fp8_e5m2",
            Self::Fp8E4m3fnuz => "fp8_e4m3fnuz",
            Self::Fp8E5m2fnuz => "fp8_e5m2fnuz",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::Bool => "bool",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_flags() {
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::Bf16.size(), 2);
        assert_eq!(DType::Fp8E4m3fn.size(), 1);
        assert!(DType::Fp8E4m3fn.is_fp8());
        assert!(DType::Bf16.is_float());
        assert!(!DType::I32.is_float());
        assert_eq!(DType::Fp8E4m3fnuz.to_string(), "fp8_e4m3fnuz");
    }
}
