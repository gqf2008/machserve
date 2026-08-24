//! CPU reference memory pool.
//!
//! This is a deliberately simple slab pool: allocations are served from a
//! growing backing `Vec<u8>` and freed ranges are tracked for reuse. It exists
//! to (a) exercise the [`MemoryPool`] contract on CI without a GPU and
//! (b) act as a reference for the CUDA caching allocator.
//!
//! On top of the slab contract it implements the elastic-memory extension
//! ([`TaggedPool`]): tagged regions live in the same backing store and can be
//! resized (grown/shrunk, possibly relocated) at runtime; [`TaggedPool::shrink_to`]
//! compacts the store and truncates the backing `Vec` so the committed
//! footprint drops when external pressure demands it.

use super::{Allocation, MemoryPool, Region, Tag, TaggedPool};
use crate::Error;
use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

fn align_up(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

/// A live tagged region carved out of the backing store.
#[derive(Debug)]
struct RegionEntry {
    tag: Tag,
    offset: usize,
    bytes: usize,
    align: usize,
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
    free: BTreeSet<(usize, usize)>, // (offset, len), adjacent ranges merged
    /// Live slab allocations (offset, len); freed ranges move to `free`.
    alloc_live: BTreeSet<(usize, usize)>,
    pinned: BTreeSet<(usize, usize)>,
    /// Live tagged regions.
    regions: Vec<RegionEntry>,
    in_use: usize,
}

impl Inner {
    /// Records a free range, merging it with adjacent free ranges so
    /// compaction sees the largest possible contiguous holes.
    fn insert_free(&mut self, off: usize, len: usize) {
        if let Some(&(poff, plen)) = self.free.range(..(off, 0)).next_back()
            && poff + plen == off
        {
            self.free.remove(&(poff, plen));
            return self.insert_free(poff, plen + len);
        }
        if let Some(&(noff, nlen)) = self.free.range((off + len, 0)..).next()
            && noff == off + len
        {
            self.free.remove(&(noff, nlen));
            return self.insert_free(off, len + nlen);
        }
        self.free.insert((off, len));
    }

    /// End offset of the highest live byte (slab allocations, pinned ranges
    /// and regions). Trailing free slack beyond this is releasable.
    fn high_water(&self) -> usize {
        let mut h = 0usize;
        for &(off, len) in &self.alloc_live {
            h = h.max(off + len);
        }
        for &(off, len) in &self.pinned {
            h = h.max(off + len);
        }
        for r in &self.regions {
            h = h.max(r.offset + r.bytes);
        }
        h
    }

    /// Carves `bytes` (aligned) out of a best-fit free range or by growing the
    /// backing store; returns the aligned offset.
    fn carve(&mut self, bytes: usize, align: usize) -> usize {
        let mut best: Option<(usize, usize, usize)> = None; // (free_off, free_len, aligned_off)
        for &(off, len) in &self.free {
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
            self.free.remove(&(off, len));
            if aligned > off {
                self.insert_free(off, aligned - off);
            }
            let end = aligned + bytes;
            if end < off + len {
                self.insert_free(end, off + len - end);
            }
            return aligned;
        }
        // Grow the backing store, aligning the new region.
        let raw = self.backing.len();
        let aligned = align_up(raw, align);
        let grow_to = aligned + bytes;
        self.backing.resize(grow_to, 0);
        if aligned > raw {
            self.insert_free(raw, aligned - raw);
        }
        aligned
    }

    /// Length of the free range starting exactly at `at` (0 when none).
    fn adjacent_free_len(&self, at: usize) -> usize {
        self.free
            .range((at, 0)..)
            .next()
            .filter(|&&(off, _)| off == at)
            .map_or(0, |&(_, len)| len)
    }

    /// Packs live regions toward the front of the backing store so trailing
    /// free slack can be released. Region handles change; callers re-fetch by
    /// tag ([`TaggedPool::region_by_tag`]) after a compacting [`shrink_to`].
    fn compact_regions(&mut self) {
        let mut order: Vec<usize> = (0..self.regions.len()).collect();
        order.sort_by_key(|&i| self.regions[i].offset);
        for &i in &order {
            let (off, bytes, align) = {
                let r = &self.regions[i];
                (r.offset, r.bytes, r.align)
            };
            let Some(new_off) = self.find_fit_left_of(off, bytes, align) else {
                continue;
            };
            if new_off != off {
                self.backing.copy_within(off..off + bytes, new_off);
                self.insert_free(off, bytes);
                self.regions[i].offset = new_off;
            }
        }
    }

    /// Finds the lowest free range that can hold `bytes` (aligned) ending at
    /// or left of `max_off`, carving it out of the free set. Returns the
    /// aligned offset (data is *not* copied here).
    fn find_fit_left_of(&mut self, max_off: usize, bytes: usize, align: usize) -> Option<usize> {
        let candidates: Vec<(usize, usize)> = self.free.iter().copied().collect();
        let mut best: Option<(usize, usize, usize)> = None; // (free_off, free_len, aligned)
        for &(off, len) in &candidates {
            let aligned = align_up(off, align);
            if aligned
                .checked_add(bytes)
                .is_some_and(|end| end <= off + len)
                && aligned + bytes <= max_off
                && best.is_none_or(|(b_off, _, _)| off < b_off)
            {
                best = Some((off, len, aligned));
            }
        }
        let (off, len, aligned) = best?;
        self.free.remove(&(off, len));
        if aligned > off {
            self.insert_free(off, aligned - off);
        }
        let end = aligned + bytes;
        if end < off + len {
            self.insert_free(end, off + len - end);
        }
        Some(aligned)
    }
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

    /// Copies a live region's bytes to host (CPU test/verification helper).
    pub fn region_data(&self, region: Region) -> Result<Vec<u8>, Error> {
        self.ensure_live(&region)?;
        let inner = self.inner.lock().unwrap();
        Ok(inner.backing[region.offset..region.offset + region.bytes].to_vec())
    }

    /// Overwrites the first `data.len()` bytes of a live region (CPU
    /// test/verification helper).
    pub fn set_region_data(&self, region: Region, data: &[u8]) -> Result<(), Error> {
        if data.len() > region.bytes {
            return Err(Error::InvalidArgument(format!(
                "{} bytes do not fit in region of {} bytes",
                data.len(),
                region.bytes
            )));
        }
        self.ensure_live(&region)?;
        let mut inner = self.inner.lock().unwrap();
        inner.backing[region.offset..region.offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Returns the index of a live region matching `region`, or an error.
    fn ensure_live(&self, region: &Region) -> Result<(), Error> {
        if region.pool_id != self.pool_id {
            return Err(Error::InvalidArgument("region from another pool".into()));
        }
        let inner = self.inner.lock().unwrap();
        if !inner
            .regions
            .iter()
            .any(|r| r.offset == region.offset && r.bytes == region.bytes)
        {
            return Err(Error::Memory("region is not live".into()));
        }
        Ok(())
    }
}

impl MemoryPool for CpuMemoryPool {
    fn allocate(&self, bytes: usize, align: usize) -> Result<Allocation, Error> {
        if bytes == 0 {
            return Err(Error::InvalidArgument("zero-size allocation".into()));
        }
        let align = align.max(1).next_power_of_two();
        let mut inner = self.inner.lock().unwrap();
        let offset = inner.carve(bytes, align);
        inner.alloc_live.insert((offset, bytes));
        inner.in_use += bytes;
        Ok(Allocation {
            pool_id: self.pool_id,
            offset,
            bytes,
            ptr: 0,
        })
    }

    fn free(&self, alloc: Allocation) -> Result<(), Error> {
        if alloc.pool_id != self.pool_id {
            return Err(Error::InvalidArgument(
                "allocation from another pool".into(),
            ));
        }
        let mut inner = self.inner.lock().unwrap();
        inner.alloc_live.remove(&(alloc.offset, alloc.bytes));
        inner.insert_free(alloc.offset, alloc.bytes);
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
            inner.insert_free(alloc.offset, alloc.bytes);
        }
        Ok(())
    }

    fn bytes_in_use(&self) -> usize {
        self.inner.lock().unwrap().in_use
    }
}

impl TaggedPool for CpuMemoryPool {
    fn allocate_region(&self, tag: Tag, bytes: usize, align: usize) -> Result<Region, Error> {
        if bytes == 0 {
            return Err(Error::InvalidArgument("zero-size region".into()));
        }
        let align = align.max(1).next_power_of_two();
        let mut inner = self.inner.lock().unwrap();
        let offset = inner.carve(bytes, align);
        inner.in_use += bytes;
        inner.regions.push(RegionEntry {
            tag: tag.clone(),
            offset,
            bytes,
            align,
        });
        Ok(Region {
            pool_id: self.pool_id,
            tag,
            offset,
            bytes,
            ptr: 0,
        })
    }

    fn resize_region(&self, region: Region, new_bytes: usize) -> Result<Region, Error> {
        if region.pool_id != self.pool_id {
            return Err(Error::InvalidArgument("region from another pool".into()));
        }
        if new_bytes == 0 {
            return Err(Error::InvalidArgument("zero-size region".into()));
        }
        let mut inner = self.inner.lock().unwrap();
        let idx = inner
            .regions
            .iter()
            .position(|r| r.offset == region.offset && r.bytes == region.bytes)
            .ok_or_else(|| Error::Memory("region is not live".into()))?;
        let (old, align, tag) = {
            let e = &inner.regions[idx];
            (e.bytes, e.align, e.tag.clone())
        };
        if new_bytes == old {
            return Ok(region);
        }
        if new_bytes < old {
            // Shrink in place; the tail becomes reusable.
            let freed = old - new_bytes;
            inner.insert_free(region.offset + new_bytes, freed);
            inner.regions[idx].bytes = new_bytes;
            inner.in_use -= freed;
            return Ok(Region {
                pool_id: self.pool_id,
                tag,
                offset: region.offset,
                bytes: new_bytes,
                ptr: 0,
            });
        }

        // Growing.
        let need = new_bytes - old;
        // 1) Extend in place into adjacent free space (the common case after a
        //    sibling region shrank).
        let at = region.offset + old;
        let adj = inner.adjacent_free_len(at);
        if adj >= need {
            inner.free.remove(&(at, adj));
            if adj > need {
                inner.insert_free(at + need, adj - need);
            }
            inner.regions[idx].bytes = new_bytes;
            inner.in_use += need;
            return Ok(Region {
                pool_id: self.pool_id,
                tag,
                offset: region.offset,
                bytes: new_bytes,
                ptr: 0,
            });
        }

        // 2) Relocate into a free range that fits, or grow the backing tail.
        let old_off = region.offset;
        let new_off = inner.carve(new_bytes, align);
        inner.backing.copy_within(old_off..old_off + old, new_off);
        inner.insert_free(old_off, old);
        inner.regions[idx].offset = new_off;
        inner.regions[idx].bytes = new_bytes;
        inner.in_use += need;
        Ok(Region {
            pool_id: self.pool_id,
            tag,
            offset: new_off,
            bytes: new_bytes,
            ptr: 0,
        })
    }

    fn free_region(&self, region: Region) -> Result<(), Error> {
        if region.pool_id != self.pool_id {
            return Err(Error::InvalidArgument("region from another pool".into()));
        }
        let mut inner = self.inner.lock().unwrap();
        let idx = inner
            .regions
            .iter()
            .position(|r| r.offset == region.offset && r.bytes == region.bytes)
            .ok_or_else(|| Error::Memory("region is not live".into()))?;
        let entry = inner.regions.remove(idx);
        inner.insert_free(entry.offset, entry.bytes);
        inner.in_use -= entry.bytes;
        Ok(())
    }

    fn shrink_to(&self, budget: usize) -> Result<usize, Error> {
        let mut inner = self.inner.lock().unwrap();
        let committed = inner.backing.len();
        if committed <= budget {
            return Ok(0);
        }
        let live = inner.in_use;
        if live > budget {
            return Err(Error::Memory(format!(
                "pool uses {live} live bytes > budget {budget} (shrink/free regions first)"
            )));
        }
        // All live bytes fit in `budget`: compact regions toward the front so
        // the high-water mark drops, then truncate the backing store and drop
        // free ranges beyond the new length.
        inner.compact_regions();
        let h = inner.high_water();
        if h > budget {
            return Err(Error::Memory(format!(
                "pool high-water {h} bytes > budget {budget} after compaction (shrink/free regions first)"
            )));
        }
        let new_len = budget;
        inner.backing.truncate(new_len);
        inner.free.retain(|&(off, len)| off + len <= new_len);
        Ok(committed - new_len)
    }

    fn committed_bytes(&self) -> usize {
        self.inner.lock().unwrap().backing.len()
    }

    fn region_by_tag(&self, tag: &Tag) -> Option<Region> {
        let inner = self.inner.lock().unwrap();
        inner
            .regions
            .iter()
            .find(|r| r.tag == *tag)
            .map(|r| Region {
                pool_id: self.pool_id,
                tag: r.tag.clone(),
                offset: r.offset,
                bytes: r.bytes,
                ptr: 0,
            })
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
            ptr: 0,
        };
        assert!(pool.free(bad).is_err());
    }

    fn pattern(seed: u8, len: usize) -> Vec<u8> {
        (0..len).map(|i| seed.wrapping_add(i as u8)).collect()
    }

    #[test]
    fn region_alloc_grow_shrink_free_preserves_data() {
        let pool = CpuMemoryPool::new();
        let exp = pool.allocate_region(Tag::expert_cache(), 1024, 64).unwrap();
        let kv = pool.allocate_region(Tag::kv(), 2048, 64).unwrap();
        assert_eq!(pool.bytes_in_use(), 1024 + 2048);

        let p1 = pattern(1, 1024);
        let p2 = pattern(2, 2048);
        pool.set_region_data(exp.clone(), &p1).unwrap();
        pool.set_region_data(kv.clone(), &p2).unwrap();

        // Grow the expert cache (may relocate); data prefix must survive.
        let exp2 = pool.resize_region(exp.clone(), 4096).unwrap();
        assert_eq!(exp2.bytes, 4096);
        let d = pool.region_data(exp2.clone()).unwrap();
        assert_eq!(&d[..1024], &p1[..], "grow must preserve data prefix");

        // Shrink the KV cache; data prefix must survive.
        let kv2 = pool.resize_region(kv.clone(), 512).unwrap();
        assert_eq!(kv2.bytes, 512);
        let d2 = pool.region_data(kv2.clone()).unwrap();
        assert_eq!(&d2[..], &p2[..512], "shrink must preserve data prefix");
        assert_eq!(pool.bytes_in_use(), 4096 + 512);

        pool.free_region(exp2).unwrap();
        pool.free_region(kv2).unwrap();
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn region_grow_relocates_when_fragmented() {
        let pool = CpuMemoryPool::new();
        let a = pool.allocate_region(Tag::new("a"), 256, 8).unwrap();
        let b = pool.allocate_region(Tag::new("b"), 256, 8).unwrap();
        let c = pool.allocate_region(Tag::new("c"), 256, 8).unwrap();
        let pa = pattern(10, 256);
        pool.set_region_data(a.clone(), &pa).unwrap();

        // Free b so the only free space around a is 256 bytes; growing a to
        // 512 must relocate it (no adjacent room), preserving its data.
        pool.free_region(b).unwrap();
        let a2 = pool.resize_region(a, 512).unwrap();
        assert_eq!(a2.bytes, 512);
        let d = pool.region_data(a2.clone()).unwrap();
        assert_eq!(&d[..256], &pa[..], "relocation must preserve data");
        let _ = (c, a2);
    }

    /// Simulated VRAM crash (显存骤降): a background renderer/game eats memory,
    /// forcing the expert cache and KV regions to shrink. The pool must never
    /// OOM; service degrades smoothly and keeps serving after the squeeze.
    #[test]
    fn region_shrink_to_under_pressure_no_oom() {
        let pool = CpuMemoryPool::new();
        let exp = pool
            .allocate_region(Tag::expert_cache(), 1 << 20, 64)
            .unwrap();
        let kv = pool.allocate_region(Tag::kv(), 1 << 20, 64).unwrap();
        let pe = pattern(7, 1 << 20);
        let pk = pattern(9, 1 << 20);
        pool.set_region_data(exp.clone(), &pe).unwrap();
        pool.set_region_data(kv.clone(), &pk).unwrap();

        // External pressure: a hard budget below what the service currently uses.
        let budget = 512 << 10; // 512 KiB for 2 MiB of live regions
        let err = pool.shrink_to(budget).unwrap_err();
        assert!(
            err.to_string().contains("live bytes"),
            "shrink above live bytes must fail with a degradable error: {err}"
        );
        assert_eq!(pool.committed_bytes(), 2 << 20);

        // Smooth degradation: evict experts to host + trim KV, then re-shrink.
        let exp_s = pool.resize_region(exp.clone(), 256 << 10).unwrap();
        let kv_s = pool.resize_region(kv.clone(), 256 << 10).unwrap();
        assert_eq!(
            &pool.region_data(exp_s.clone()).unwrap()[..256 << 10],
            &pe[..256 << 10],
            "evicted expert region keeps its resident prefix"
        );
        assert_eq!(
            &pool.region_data(kv_s.clone()).unwrap()[..256 << 10],
            &pk[..256 << 10],
            "trimmed KV region keeps its resident prefix"
        );

        let freed = pool.shrink_to(budget).unwrap();
        assert!(freed > 0, "shrink must release pooled slack");
        assert!(
            pool.committed_bytes() <= budget,
            "committed {} must drop to budget {budget}",
            pool.committed_bytes()
        );
        assert_eq!(pool.bytes_in_use(), 512 << 10);

        // Compaction may have relocated regions: re-fetch by tag and verify
        // the resident prefixes survived the squeeze.
        let exp_r = pool
            .region_by_tag(&Tag::expert_cache())
            .expect("expert region live");
        let kv_r = pool.region_by_tag(&Tag::kv()).expect("kv region live");
        assert_eq!(
            &pool.region_data(exp_r).unwrap()[..256 << 10],
            &pe[..256 << 10],
            "evicted expert region keeps its resident prefix after compaction"
        );
        assert_eq!(
            &pool.region_data(kv_r).unwrap()[..256 << 10],
            &pk[..256 << 10],
            "trimmed KV region keeps its resident prefix after compaction"
        );

        // Service continues after the squeeze: new scratch + a new region fit.
        let scratch = pool.allocate(4096, 8).unwrap();
        let extra = pool
            .allocate_region(Tag::new("scratch"), 64 << 10, 8)
            .unwrap();
        assert_eq!(pool.bytes_in_use(), (512 << 10) + 4096 + (64 << 10));
        let _ = (scratch, extra);
    }

    /// Dynamic handover: the expert cache shrinks so the KV region can grow in
    /// the same total budget — elastic rebalancing without a restart.
    #[test]
    fn region_rebalance_hands_memory_to_kv() {
        let pool = CpuMemoryPool::new();
        let exp = pool
            .allocate_region(Tag::expert_cache(), 512 << 10, 64)
            .unwrap();
        let kv = pool.allocate_region(Tag::kv(), 256 << 10, 64).unwrap();
        let pe = pattern(3, 512 << 10);
        let pk = pattern(4, 256 << 10);
        pool.set_region_data(exp.clone(), &pe).unwrap();
        pool.set_region_data(kv.clone(), &pk).unwrap();

        // Hand 128 KiB from the expert cache to KV within the same total budget
        // (768 KiB before and after the handover).
        let budget = 768 << 10;
        let exp2 = pool.resize_region(exp.clone(), 384 << 10).unwrap();
        let kv2 = pool.resize_region(kv.clone(), 384 << 10).unwrap();
        assert_eq!(
            &pool.region_data(exp2.clone()).unwrap()[..384 << 10],
            &pe[..384 << 10]
        );
        // kv grew: only its original 256 KiB prefix is meaningful (the rest is fresh).
        assert_eq!(
            &pool.region_data(kv2.clone()).unwrap()[..256 << 10],
            &pk[..]
        );

        // Live bytes exactly match the budget; committed may still hold slack.
        assert_eq!(pool.bytes_in_use(), budget);
        pool.shrink_to(budget).unwrap();
        assert!(pool.committed_bytes() <= budget);
        let _ = (exp2, kv2);
    }

    #[test]
    fn region_wrong_pool_is_rejected() {
        let pool = CpuMemoryPool::new();
        let bad = Region {
            pool_id: 999,
            tag: Tag::kv(),
            offset: 0,
            bytes: 8,
            ptr: 0,
        };
        assert!(pool.free_region(bad.clone()).is_err());
        assert!(pool.resize_region(bad, 16).is_err());
    }
}
