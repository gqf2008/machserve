//! Memory pools and allocations.
//!
//! The pool interface is intentionally tiny: serving engines care about
//! deterministic allocation/reuse (zero fragmentation) and about pinning
//! buffers that a captured CUDA graph touches, so the pool never hands graph
//! memory to a later allocation. The CPU implementation here is the reference
//! and test harness; the CUDA path lives behind the `cuda` feature.
//!
//! **Elastic memory (P3)**: on top of the plain [`MemoryPool`] contract, a
//! [`TaggedPool`] hands out named, resizable [`Region`]s (e.g. `expert_cache`
//! vs `kv`) that can be shrunk, grown and rebalanced at runtime without
//! restarting the process — the primitive behind elastic VRAM management.
//! Pools implement both traits; existing [`MemoryPool`] users are untouched.

pub mod cpu;

use core::fmt;
use std::sync::Arc;

/// A named memory region tag (e.g. `expert_cache`, `kv`).
///
/// Tags let a [`TaggedPool`] account for, resize and rebalance regions
/// independently — the mechanism behind elastic VRAM where the expert cache
/// and the KV cache hand memory to each other at runtime.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tag(pub Arc<str>);

impl Tag {
    /// Builds a tag from any string-like value.
    #[must_use]
    pub fn new(s: impl Into<Arc<str>>) -> Self {
        Self(s.into())
    }

    /// Tag for the MoE expert-cache region.
    #[must_use]
    pub fn expert_cache() -> Self {
        Self::new("expert_cache")
    }

    /// Tag for the KV-cache region.
    #[must_use]
    pub fn kv() -> Self {
        Self::new("kv")
    }

    /// The tag as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for Tag {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for Tag {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<Arc<str>> for Tag {
    fn from(s: Arc<str>) -> Self {
        Self(s)
    }
}

/// A logical allocation handed out by a [`MemoryPool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Allocation {
    /// Unique id of the owning pool (or allocator).
    pub pool_id: u64,
    /// Byte offset within the pool's backing store (CPU pools) or 0.
    pub offset: usize,
    /// Requested byte size.
    pub bytes: usize,
    /// Backend device address (HIP/CUDA) or 0 for CPU pools.
    pub ptr: usize,
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

/// A tagged, resizable region handed out by a [`TaggedPool`].
///
/// Unlike a plain [`Allocation`], a region keeps its tag and may be resized
/// (grown/shrunk) at runtime. Resize may **relocate** the region (offset/ptr
/// changes) — callers must always use the handle returned by
/// [`TaggedPool::resize_region`] afterwards.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Region {
    /// Unique id of the owning pool.
    pub pool_id: u64,
    /// Tag this region was allocated under.
    pub tag: Tag,
    /// Byte offset within the pool's backing store (CPU pools) or 0.
    pub offset: usize,
    /// Current byte size (may change across resizes).
    pub bytes: usize,
    /// Backend device address (HIP/CUDA) or 0 for CPU pools.
    pub ptr: usize,
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "region<{}>#{}[{}..{}]",
            self.tag,
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

/// A backend-agnostic contract for **tagged, resizable memory regions** —
/// the elastic-memory extension of [`MemoryPool`].
///
/// Regions are named (e.g. `expert_cache` / `kv`), can be resized without
/// restarting the process, and the pool can be told to shrink its committed
/// footprint when external pressure (a game, a renderer) eats VRAM.
///
/// The **smooth-degradation contract** for [`TaggedPool::shrink_to`]: when the
/// pool's *live* bytes already exceed the new budget, shrinking fails and the
/// caller is responsible for first resizing/freeing regions (e.g. evicting
/// experts to host RAM, trimming KV capacity). Once live bytes fit, the pool
/// compacts/releases its slack so the committed footprint drops — service
/// keeps running throughout, never OOM.
pub trait TaggedPool: MemoryPool + Send + Sync {
    /// Allocates a tagged region of `bytes` contiguous bytes.
    fn allocate_region(&self, tag: Tag, bytes: usize, align: usize)
    -> Result<Region, crate::Error>;

    /// Grows or shrinks a region. Data is preserved up to the smaller of the
    /// two sizes. May relocate the region; always use the returned handle.
    fn resize_region(&self, region: Region, new_bytes: usize) -> Result<Region, crate::Error>;

    /// Frees a region, making its space reusable/releasable.
    fn free_region(&self, region: Region) -> Result<(), crate::Error>;

    /// Returns the **current** handle for a live region by its tag.
    ///
    /// Region handles are not stable across pool-initiated relocation
    /// ([TaggedPool::shrink_to] may compact regions), so callers that hold
    /// handles across a shrink must re-fetch them by tag before touching the
    /// region again. Returns None when no live region carries the tag.
    fn region_by_tag(&self, tag: &Tag) -> Option<Region>;

    /// Releases pooled slack so the **committed** footprint is at most
    /// `budget` bytes. Returns the number of bytes released.
    ///
    /// Errors when live bytes still exceed `budget` (see the trait-level
    /// smooth-degradation contract); the error carries the current committed
    /// byte count so callers can size their next degradation step.
    fn shrink_to(&self, budget: usize) -> Result<usize, crate::Error>;

    /// Total committed bytes (pool footprint, including slack).
    fn committed_bytes(&self) -> usize;
}
