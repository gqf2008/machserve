//! Memory pools and allocations.
//!
//! The pool interface is intentionally tiny: serving engines care about
//! deterministic allocation/reuse (zero fragmentation) and about pinning
//! buffers that a captured CUDA graph touches, so the pool never hands graph
//! memory to a later allocation. The CPU implementation here is the reference
//! and test harness; the CUDA path lives behind the `cuda` feature.

pub mod cpu;

use core::fmt;

/// A logical allocation handed out by a [`MemoryPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Allocation {
    /// Unique id of the owning pool (or allocator).
    pub pool_id: u64,
    /// Byte offset within the pool's backing store.
    pub offset: usize,
    /// Requested byte size.
    pub bytes: usize,
}

impl fmt::Display for Allocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "alloc#{}[{}..{}]",
            self.pool_id,
            self.offset,
            self.offset + self.bytes
        )
    }
}

/// A backend-agnostic memory pool contract.
///
/// Implementations must be internally synchronized; pools are shared across
/// worker threads.
pub trait MemoryPool: Send + Sync {
    /// Allocates `bytes` contiguous bytes, returning a logical [`Allocation`].
    fn allocate(&self, bytes: usize, align: usize) -> Result<Allocation, crate::Error>;

    /// Frees a previously returned allocation, making its space reusable.
    fn free(&self, alloc: Allocation) -> Result<(), crate::Error>;

    /// Pins `bytes` starting at `alloc` so the pool never reuses that slice
    /// until unpinned. Used to keep captured-graph buffers alive.
    fn pin(&self, alloc: Allocation) -> Result<(), crate::Error>;

    /// Releases a pin previously acquired with [`MemoryPool::pin`].
    fn unpin(&self, alloc: Allocation) -> Result<(), crate::Error>;

    /// High-water mark in bytes (for diagnostics).
    fn bytes_in_use(&self) -> usize;
}
