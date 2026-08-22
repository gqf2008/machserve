//! CPU reference memory pool.
//!
//! This is a deliberately simple slab pool: allocations are served from a
//! growing backing `Vec<u8>` and freed ranges are tracked for reuse. It exists
//! to (a) exercise the [`MemoryPool`] contract on CI without a GPU and
//! (b) act as a reference for the CUDA caching allocator.

use super::{Allocation, MemoryPool};
use crate::Error;
use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// Simple CPU slab pool. Interior-mutable and thread-safe.
#[derive(Debug, Default)]
pub struct CpuMemoryPool {
    inner: Mutex<Inner>,
    pool_id: u64,
}

#[derive(Debug, Default)]
struct Inner {
    backing: Vec<u8>,
    free: BTreeSet<(usize, usize)>, // (offset, len)
    pinned: BTreeSet<(usize, usize)>,
    in_use: usize,
}

impl CpuMemoryPool {
    /// Creates a fresh pool with a unique id.
    #[must_use]
    pub fn new() -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            inner: Mutex::new(Inner::default()),
            pool_id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl MemoryPool for CpuMemoryPool {
    fn allocate(&self, bytes: usize, align: usize) -> Result<Allocation, Error> {
        if bytes == 0 {
            return Err(Error::InvalidArgument("zero-size allocation".into()));
        }
        let align = align.max(1).next_power_of_two();
        let mut inner = self.inner.lock().unwrap();

        // 1) Best-fit from reusable free ranges, honoring alignment by splitting.
        let mut best: Option<(usize, usize, usize)> = None; // (free_off, free_len, aligned_off)
        for &(off, len) in &inner.free {
            let aligned = align_up(off, align);
            if aligned
                .checked_add(bytes)
                .is_some_and(|end| end <= off + len)
            {
                best = Some((off, len, aligned));
                break;
            }
        }
        if let Some((off, len, aligned)) = best {
            inner.free.remove(&(off, len));
            if aligned > off {
                inner.free.insert((off, aligned - off));
            }
            let end = aligned + bytes;
            if end < off + len {
                inner.free.insert((end, off + len - end));
            }
            inner.in_use += bytes;
            return Ok(Allocation {
                pool_id: self.pool_id,
                offset: aligned,
                bytes,
            });
        }

        // 2) Grow the backing store, aligning the new region.
        let raw = inner.backing.len();
        let aligned = align_up(raw, align);
        let grow_to = aligned + bytes;
        inner.backing.resize(grow_to, 0);
        if aligned > raw {
            inner.free.insert((raw, aligned - raw));
        }
        inner.in_use += bytes;
        Ok(Allocation {
            pool_id: self.pool_id,
            offset: aligned,
            bytes,
        })
    }

    fn free(&self, alloc: Allocation) -> Result<(), Error> {
        if alloc.pool_id != self.pool_id {
            return Err(Error::InvalidArgument(
                "allocation from another pool".into(),
            ));
        }
        let mut inner = self.inner.lock().unwrap();
        inner.free.insert((alloc.offset, alloc.bytes));
        inner.in_use = inner.in_use.saturating_sub(alloc.bytes);
        Ok(())
    }

    fn pin(&self, alloc: Allocation) -> Result<(), Error> {
        if alloc.pool_id != self.pool_id {
            return Err(Error::InvalidArgument(
                "allocation from another pool".into(),
            ));
        }
        let mut inner = self.inner.lock().unwrap();
        inner.free.remove(&(alloc.offset, alloc.bytes));
        inner.pinned.insert((alloc.offset, alloc.bytes));
        Ok(())
    }

    fn unpin(&self, alloc: Allocation) -> Result<(), Error> {
        if alloc.pool_id != self.pool_id {
            return Err(Error::InvalidArgument(
                "allocation from another pool".into(),
            ));
        }
        let mut inner = self.inner.lock().unwrap();
        if inner.pinned.remove(&(alloc.offset, alloc.bytes)) {
            inner.free.insert((alloc.offset, alloc.bytes));
        }
        Ok(())
    }

    fn bytes_in_use(&self) -> usize {
        self.inner.lock().unwrap().in_use
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_free_reuse() {
        let pool = CpuMemoryPool::new();
        let a = pool.allocate(100, 8).unwrap();
        let b = pool.allocate(50, 8).unwrap();
        assert_eq!(pool.bytes_in_use(), 150);
        pool.free(a).unwrap();
        // Reuse the freed 100-byte range for another 100-byte allocation.
        let c = pool.allocate(100, 8).unwrap();
        assert_eq!(c.offset, a.offset);
        assert_eq!(pool.bytes_in_use(), 150);
        let _ = (b, c);
    }

    #[test]
    fn pin_blocks_reuse() {
        let pool = CpuMemoryPool::new();
        let a = pool.allocate(64, 8).unwrap();
        pool.free(a).unwrap();
        pool.pin(a).unwrap();
        // A same-size allocation must not land on the pinned slice.
        let b = pool.allocate(64, 8).unwrap();
        assert_ne!(b.offset, a.offset);
        pool.unpin(a).unwrap();
        let c = pool.allocate(64, 8).unwrap();
        assert_eq!(c.offset, a.offset);
    }

    #[test]
    fn alignment_is_respected() {
        let pool = CpuMemoryPool::new();
        let a = pool.allocate(10, 16).unwrap();
        assert_eq!(a.offset % 16, 0);
        let b = pool.allocate(10, 16).unwrap();
        assert_eq!(b.offset % 16, 0);
        let c = pool.allocate(4, 16).unwrap();
        assert_eq!(c.offset % 16, 0);
        let _ = (a, b, c);
    }

    #[test]
    fn wrong_pool_is_rejected() {
        let pool = CpuMemoryPool::new();
        let bad = Allocation {
            pool_id: 999,
            offset: 0,
            bytes: 8,
        };
        assert!(pool.free(bad).is_err());
    }
}
