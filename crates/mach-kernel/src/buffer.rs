//! Abstract device buffer: the unit kernels operate on.

use mach_engine::{Allocation, DType, Device, Shape};

/// A device buffer: allocation + layout metadata.
///
/// This is deliberately **not** a full tensor (no autograd, no ops): kernels
/// consume and produce buffers; the engine owns their lifetime through the
/// memory pool.
#[derive(Debug, Clone)]
pub struct Buffer {
    /// Device the buffer lives on.
    pub device: Device,
    /// Element type.
    pub dtype: DType,
    /// Logical shape.
    pub shape: Shape,
    /// Pool allocation backing the data.
    pub allocation: Allocation,
}

impl Buffer {
    /// Creates a buffer from parts.
    #[must_use]
    pub fn new(device: Device, dtype: DType, shape: Shape, allocation: Allocation) -> Self {
        Self {
            device,
            dtype,
            shape,
            allocation,
        }
    }

    /// Number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Byte size of the buffer payload.
    pub fn byte_size(&self) -> Result<usize, mach_engine::Error> {
        self.shape.byte_size(self.dtype)
    }

    /// Display key for diagnostics, e.g. `cuda:0 f32 [2, 3, 4]`.
    #[must_use]
    pub fn key(&self) -> String {
        format!("{} {} {}", self.device, self.dtype, self.shape)
    }
}
