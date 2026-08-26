// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Ported from LightSeek TokenSpeed ts-scheduler-core (MIT):
// `tokenspeed-scheduler-rs/crates/ts-scheduler-core/src/block_pool.rs`,
// `.../block_table.rs`, `.../cache_block_ref.rs`, plus the placement geometry
// (`cache_blocks_per_lcm_block`) from `.../cache_types.rs`.

//! Paged-KV LCM block pool and per-group logical page tables.
//!
//! [`BlockPool`] owns the physical placement of child slots inside LCM-sized
//! parent blocks. It deliberately has no cache key, LRU node, or ownership
//! count; it only tracks which group a parent is bound to and which child
//! slots are occupied. Children are handed out as [`CacheBlockRef`]s whose
//! last owner returns the slot via [`CacheBlock`]'s `Drop` (RAII), so the
//! scheduler never leaks a slot when a request dies.
//!
//! [`BlockTable`] is machserve's adaptation of the upstream block table: a
//! pure page-number table (`Vec<Vec<i32>>`) instead of a `Vec<CacheBlockRef>`
//! list, so kernel-ready rows can be exported without keeping ownership refs
//! alive. Both halves are single-threaded by design (`Rc`/`RefCell`); the GPU
//! path consumes exported page ids, not the pool.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;

/// Stable logical placement of one cache block inside an LCM-sized physical
/// block. LCM block 0 is reserved as the kernel null page, so valid
/// `lcm_block_id`s are `1..=num_lcm_blocks`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheBlockLocation {
    pub lcm_block_id: i32,
    pub slot_index: i32,
}

/// Shared handle to a [`BlockPool`]. The scheduler and every [`CacheBlock`]
/// share one instance; `Rc` guarantees the pool outlives all live blocks
/// (the C++ port required the same invariant via raw pointers).
pub type BlockPoolHandle = Rc<RefCell<BlockPool>>;

/// One LCM-sized physical parent block.
struct LcmBlock {
    /// Group that owns this parent; unset while free.
    bound_group: Option<u32>,
    /// Child-slot occupancy; empty while free, sized `slots_per_parent` while
    /// bound. `Vec<bool>` is used because the C++ `std::vector<bool>`
    /// bit-packing is an implementation detail, not a contract.
    occupancy: Vec<bool>,
    occupied_count: u32,
}

/// Physical LCM placement pool for one scheduler.
pub struct BlockPool {
    lcm_blocks: Vec<LcmBlock>,
    /// Free parents are interchangeable: release appends and allocation
    /// consumes the front. Bound parents are selected separately by
    /// `plan_locations`.
    free_parent_ids: VecDeque<i32>,
}

impl BlockPool {
    /// Build a pool with `num_lcm_blocks` physical parents (ids `1..=n`).
    pub fn new(num_lcm_blocks: i32) -> Self {
        assert!(num_lcm_blocks >= 0, "num_lcm_blocks must be >= 0");
        let mut free_parent_ids = VecDeque::with_capacity(num_lcm_blocks as usize);
        for id in 1..=num_lcm_blocks {
            free_parent_ids.push_back(id);
        }
        Self {
            lcm_blocks: (0..num_lcm_blocks as usize)
                .map(|_| LcmBlock {
                    bound_group: None,
                    occupancy: Vec::new(),
                    occupied_count: 0,
                })
                .collect(),
            free_parent_ids,
        }
    }

    /// Number of physical LCM blocks (kernel page 0 is reserved separately).
    pub fn num_lcm_blocks(&self) -> i32 {
        self.lcm_blocks.len() as i32
    }

    /// Number of completely free LCM parents.
    pub fn num_empty_lcm_blocks(&self) -> i32 {
        self.free_parent_ids.len() as i32
    }

    /// Acquire a single block for `group_id`, or `None` when out of space.
    pub fn acquire_block(
        &mut self,
        handle: &BlockPoolHandle,
        group_id: u32,
        cache_blocks_per_lcm_block: i32,
    ) -> Option<CacheBlockRef> {
        self.acquire_blocks(handle, group_id, cache_blocks_per_lcm_block, 1)
            .into_iter()
            .next()
    }

    /// Acquire `num` blocks for `group_id`, or an empty vector when the pool
    /// cannot satisfy the full request (all-or-nothing).
    pub fn acquire_blocks(
        &mut self,
        handle: &BlockPoolHandle,
        group_id: u32,
        cache_blocks_per_lcm_block: i32,
        num: usize,
    ) -> Vec<CacheBlockRef> {
        assert!(
            cache_blocks_per_lcm_block > 0,
            "cache_blocks_per_lcm_block must be > 0"
        );
        if num == 0 {
            return Vec::new();
        }
        let locations = self.plan_locations(group_id, cache_blocks_per_lcm_block, num);
        if locations.len() != num {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(num);
        for location in locations {
            // Create the owner before committing the slot so its `Drop` (which
            // releases the location) is registered even on a later panic.
            let control = Rc::new(CacheBlock::new(Rc::clone(handle), location));
            self.occupy(group_id, cache_blocks_per_lcm_block, location);
            out.push(CacheBlockRef::new(control));
        }
        out
    }

    /// Group bound to a parent, if any.
    pub fn bound_group(&self, lcm_block_id: i32) -> Option<u32> {
        self.lcm_block(lcm_block_id).bound_group
    }

    /// Number of occupied child slots in a parent.
    pub fn occupied_count(&self, lcm_block_id: i32) -> i32 {
        self.lcm_block(lcm_block_id).occupied_count as i32
    }

    /// Whether a specific child slot is occupied.
    pub fn is_occupied(&self, location: CacheBlockLocation) -> bool {
        let parent = self.lcm_block(location.lcm_block_id);
        location.slot_index >= 0
            && (location.slot_index as usize) < parent.occupancy.len()
            && parent.occupancy[location.slot_index as usize]
    }

    /// Total number of occupied child slots across all parents.
    pub fn num_occupied_slots(&self) -> i32 {
        self.lcm_blocks
            .iter()
            .map(|block| block.occupied_count as i32)
            .sum()
    }

    /// Occupied child locations of one parent.
    pub fn occupied_locations(&self, lcm_block_id: i32) -> Vec<CacheBlockLocation> {
        let parent = self.lcm_block(lcm_block_id);
        parent
            .occupancy
            .iter()
            .enumerate()
            .filter(|(_, occupied)| **occupied)
            .map(|(slot, _)| CacheBlockLocation {
                lcm_block_id,
                slot_index: slot as i32,
            })
            .collect()
    }

    /// Return a child slot to its parent (called by [`CacheBlock::drop`]).
    /// Once the last slot of a parent is released the parent is unbound and
    /// returned to the free queue.
    pub fn release(&mut self, location: CacheBlockLocation) {
        assert!(
            location.lcm_block_id > 0 && (location.lcm_block_id as usize) <= self.lcm_blocks.len(),
            "CacheBlock location has invalid LCM block id"
        );
        let parent = &mut self.lcm_blocks[location.lcm_block_id as usize - 1];
        assert!(
            location.slot_index >= 0 && (location.slot_index as usize) < parent.occupancy.len(),
            "CacheBlock location has invalid slot"
        );
        let slot = location.slot_index as usize;
        assert!(
            parent.occupancy[slot] && parent.occupied_count > 0,
            "CacheBlock location is not occupied"
        );
        parent.occupancy[slot] = false;
        parent.occupied_count -= 1;
        if parent.occupied_count == 0 {
            parent.bound_group = None;
            parent.occupancy.clear();
            assert!(
                self.free_parent_ids.len() < self.lcm_blocks.len(),
                "free LCM block queue cannot exceed the pool size"
            );
            self.free_parent_ids.push_back(location.lcm_block_id);
        }
    }

    /// Immutable access to a parent (1-based id), panicking on invalid ids.
    fn lcm_block(&self, lcm_block_id: i32) -> &LcmBlock {
        assert!(
            lcm_block_id > 0 && (lcm_block_id as usize) <= self.lcm_blocks.len(),
            "LCM block id out of range"
        );
        &self.lcm_blocks[lcm_block_id as usize - 1]
    }

    /// Commit a planned location: binds the parent on first use and marks the
    /// slot occupied.
    fn occupy(&mut self, group_id: u32, slots_per_parent: i32, location: CacheBlockLocation) {
        let parent = &mut self.lcm_blocks[location.lcm_block_id as usize - 1];
        if parent.occupied_count == 0 {
            assert!(
                !self.free_parent_ids.is_empty()
                    && self.free_parent_ids.front() == Some(&location.lcm_block_id),
                "empty LCM placement must consume the next free parent"
            );
            assert!(
                parent.occupancy.is_empty(),
                "empty LCM parent must not retain child slots"
            );
            parent.occupancy = vec![false; slots_per_parent as usize];
            self.free_parent_ids.pop_front();
            parent.bound_group = Some(group_id);
        }
        assert!(
            parent.bound_group == Some(group_id)
                && parent.occupancy.len() == slots_per_parent as usize,
            "LCM parent binding changed while occupied"
        );
        let slot = location.slot_index as usize;
        assert!(
            slot < parent.occupancy.len(),
            "LCM child slot is out of range"
        );
        assert!(!parent.occupancy[slot], "LCM child slot already occupied");
        parent.occupancy[slot] = true;
        parent.occupied_count += 1;
    }

    /// Plan `count` child locations: fill partially occupied parents of the
    /// group first (most occupied first, then lowest id), then consume free
    /// parents from the front of the queue. Returns an empty vector when the
    /// pool cannot satisfy the full request.
    fn plan_locations(
        &self,
        group_id: u32,
        slots_per_parent: i32,
        count: usize,
    ) -> Vec<CacheBlockLocation> {
        let mut partially_filled_parent_ids: Vec<i32> = Vec::new();
        for (index, parent) in self.lcm_blocks.iter().enumerate() {
            if parent.bound_group != Some(group_id) {
                continue;
            }
            assert!(
                parent.occupancy.len() == slots_per_parent as usize,
                "group packing changed while LCM block is occupied"
            );
            if (parent.occupied_count as usize) < parent.occupancy.len() {
                partially_filled_parent_ids.push(index as i32 + 1);
            }
        }
        partially_filled_parent_ids.sort_by(|lhs, rhs| {
            let lhs_occupied = self.lcm_block(*lhs).occupied_count;
            let rhs_occupied = self.lcm_block(*rhs).occupied_count;
            rhs_occupied.cmp(&lhs_occupied).then_with(|| lhs.cmp(rhs))
        });

        let mut locations = Vec::with_capacity(count);
        for lcm_block_id in partially_filled_parent_ids {
            let parent = self.lcm_block(lcm_block_id);
            for (slot, occupied) in parent.occupancy.iter().enumerate() {
                if locations.len() == count {
                    break;
                }
                if !*occupied {
                    locations.push(CacheBlockLocation {
                        lcm_block_id,
                        slot_index: slot as i32,
                    });
                }
            }
            if locations.len() == count {
                return locations;
            }
        }

        for lcm_block_id in self.free_parent_ids.iter().copied() {
            for slot in 0..slots_per_parent {
                if locations.len() == count {
                    break;
                }
                locations.push(CacheBlockLocation {
                    lcm_block_id,
                    slot_index: slot,
                });
            }
            if locations.len() == count {
                return locations;
            }
        }
        // Match the C++ contract: an unsatisfiable request yields no locations
        // (the caller checks `len == num` and fails the whole acquisition).
        Vec::new()
    }
}

/// Owning object for one block slot. Releasing the last [`CacheBlockRef`]
/// drops the block, which returns the slot to its [`BlockPool`].
pub struct CacheBlock {
    pool: Rc<RefCell<BlockPool>>,
    location: CacheBlockLocation,
}

impl CacheBlock {
    /// Create a block bound to `pool` at `location` (created by [`BlockPool`]).
    pub(crate) fn new(pool: Rc<RefCell<BlockPool>>, location: CacheBlockLocation) -> Self {
        Self { pool, location }
    }

    /// Stable placement of this block.
    pub fn location(&self) -> CacheBlockLocation {
        self.location
    }

    /// Whether this block belongs to `pool` (same shared instance).
    pub fn is_owned_by(&self, pool: &Rc<RefCell<BlockPool>>) -> bool {
        Rc::ptr_eq(&self.pool, pool)
    }
}

impl Drop for CacheBlock {
    fn drop(&mut self) {
        self.pool.borrow_mut().release(self.location);
    }
}

/// Shared, nullable handle to a [`CacheBlock`]. Mirrors the C++ `CacheBlockRef`
/// value semantics: `Eq` compares control identity, `use_count` reports the
/// number of live handles, and an empty handle is the null block.
#[derive(Clone, Default)]
pub struct CacheBlockRef(Option<Rc<CacheBlock>>);

impl CacheBlockRef {
    /// Wrap a freshly created block (called by [`BlockPool`]).
    pub(crate) fn new(block: Rc<CacheBlock>) -> Self {
        Self(Some(block))
    }

    /// Whether this handle is empty (null block).
    pub fn is_null(&self) -> bool {
        self.0.is_none()
    }

    /// Placement of the referenced block, or `None` for a null handle.
    pub fn location(&self) -> Option<CacheBlockLocation> {
        self.0.as_ref().map(|block| block.location())
    }

    /// Number of live handles sharing this block (0 for a null handle).
    pub fn use_count(&self) -> u32 {
        self.0
            .as_ref()
            .map_or(0, |block| Rc::strong_count(block) as u32)
    }

    /// Whether this handle is the only live one.
    pub fn unique(&self) -> bool {
        self.use_count() == 1
    }

    /// Whether the referenced block belongs to `pool`.
    pub fn is_owned_by(&self, pool: &Rc<RefCell<BlockPool>>) -> bool {
        self.0.as_ref().is_some_and(|block| block.is_owned_by(pool))
    }

    /// Drop this handle's reference.
    pub fn reset(&mut self) {
        self.0 = None;
    }

    /// Borrow the underlying block, or `None` for a null handle.
    pub fn as_block(&self) -> Option<&CacheBlock> {
        self.0.as_deref()
    }
}

impl PartialEq for CacheBlockRef {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (None, None) => true,
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for CacheBlockRef {}

impl std::fmt::Debug for CacheBlockRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheBlockRef")
            .field("location", &self.location())
            .field("use_count", &self.use_count())
            .finish()
    }
}

/// One group's logical page table.
///
/// Rows are `Vec<Vec<i32>>`: each row is one request, and each entry is an
/// absolute logical page number. `0` marks a null hole and `-1` is a trailing
/// padding sentinel that every row ends with, so kernel consumers can treat
/// rows as fixed-shape. The scheduler holds one table per group, mirroring a
/// `BTreeMap<group_id, Vec<Vec<i32>>>`; this type is the per-group value of
/// that map.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlockTable {
    /// One row per request; every row ends with a trailing `-1` sentinel.
    rows: Vec<Vec<i32>>,
    /// Unconsumed capacity at the logical tail. May span multiple blocks when
    /// admission preallocates a later decode step; caller-maintained.
    available_tokens: i32,
}

impl BlockTable {
    /// Read-only view of the rows (request -> absolute page numbers).
    pub fn rows(&self) -> &[Vec<i32>] {
        &self.rows
    }

    /// Number of request rows.
    pub fn num_rows(&self) -> i32 {
        self.rows.len() as i32
    }

    /// Number of materialized blocks: page entries that are neither null holes
    /// (`0`) nor the `-1` padding sentinel.
    pub fn num_blocks(&self) -> i32 {
        self.rows
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&page| page > 0)
            .count() as i32
    }

    /// Unconsumed token capacity at the logical tail.
    pub fn available_tokens(&self) -> i32 {
        self.available_tokens
    }

    /// Set the unconsumed token capacity (must be non-negative).
    pub fn set_available_tokens(&mut self, tokens: i32) {
        assert!(
            tokens >= 0,
            "BlockTable available_tokens must be non-negative"
        );
        self.available_tokens = tokens;
    }

    /// Start a new request row (an empty row holding only the padding
    /// sentinel).
    pub fn begin_request(&mut self) {
        self.rows.push(vec![-1]);
    }

    /// Append `page` to the last request row, keeping the trailing `-1`
    /// sentinel in place. The first call creates the first row. `page` must
    /// not be the `-1` sentinel; `0` is allowed and marks a null hole.
    pub fn append_page(&mut self, page: i32) {
        assert!(page != -1, "page cannot be the -1 padding sentinel");
        if self.rows.last().is_none() {
            self.rows.push(vec![-1]);
        }
        let last = self.rows.last_mut().expect("row exists");
        last.insert(last.len() - 1, page);
    }

    /// Remove and return the last page of the last request row. Trailing
    /// empty rows (holding only the sentinel) are dropped, and a row emptied
    /// by this removal is dropped too, so `remove_last_page` is the exact
    /// inverse of `append_page`. Returns `None` when the table holds no pages.
    pub fn remove_last_page(&mut self) -> Option<i32> {
        // Pop trailing empty request rows (just the padding sentinel).
        while self.rows.last().is_some_and(|row| row.len() <= 1) {
            self.rows.pop();
        }
        let last = self.rows.last_mut()?;
        let page = last.remove(last.len() - 2);
        if last.len() == 1 {
            // Row emptied by this removal; drop it so append/remove stay
            // exact inverses.
            self.rows.pop();
        }
        Some(page)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(num_lcm_blocks: i32) -> Rc<RefCell<BlockPool>> {
        Rc::new(RefCell::new(BlockPool::new(num_lcm_blocks)))
    }

    #[test]
    fn new_pool_has_all_parents_free() {
        let p = pool(4);
        assert_eq!(p.borrow().num_lcm_blocks(), 4);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 4);
        assert_eq!(p.borrow().num_occupied_slots(), 0);
    }

    #[test]
    fn acquire_single_block_consumes_first_free_parent() {
        let p = pool(3);
        let block = p.borrow_mut().acquire_block(&p, 7, 1).expect("block");
        assert_eq!(
            block.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 0
            })
        );
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 2);
        assert_eq!(p.borrow().bound_group(1), Some(7));
        assert_eq!(p.borrow().occupied_count(1), 1);
        assert_eq!(p.borrow().num_occupied_slots(), 1);
        assert!(p.borrow().is_occupied(CacheBlockLocation {
            lcm_block_id: 1,
            slot_index: 0
        }));
    }

    #[test]
    fn packing_packs_children_into_same_parent_then_next() {
        let p = pool(2);
        let a = p.borrow_mut().acquire_block(&p, 1, 2).expect("a");
        let b = p.borrow_mut().acquire_block(&p, 1, 2).expect("b");
        let c = p.borrow_mut().acquire_block(&p, 1, 2).expect("c");
        assert_eq!(
            a.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 0
            })
        );
        assert_eq!(
            b.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 1
            })
        );
        // Parent 1 is full; parent 2 is consumed next.
        assert_eq!(
            c.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 2,
                slot_index: 0
            })
        );
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 0);
        assert_eq!(p.borrow().num_occupied_slots(), 3);
    }

    #[test]
    fn releasing_last_child_frees_parent_and_reuses_it() {
        let p = pool(2);
        let a = p.borrow_mut().acquire_block(&p, 1, 2).expect("a");
        let b = p.borrow_mut().acquire_block(&p, 1, 2).expect("b");
        drop(a);
        assert_eq!(p.borrow().num_occupied_slots(), 1);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 1); // parent 2 free, parent 1 still bound
        drop(b);
        assert_eq!(p.borrow().num_occupied_slots(), 0);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 2);
        // Release appends, allocation consumes the front: after both parents
        // are free the queue is [2, 1], so parent 2 is consumed next.
        let c = p.borrow_mut().acquire_block(&p, 1, 2).expect("c");
        assert_eq!(
            c.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 2,
                slot_index: 0
            })
        );
    }

    #[test]
    fn plan_fills_most_occupied_parent_first() {
        let p = pool(3);
        // Fill parent 1 completely (2 slots) and parent 2 partially (1 slot).
        let a1 = p.borrow_mut().acquire_block(&p, 1, 2).expect("a1");
        let a2 = p.borrow_mut().acquire_block(&p, 1, 2).expect("a2");
        let b1 = p.borrow_mut().acquire_block(&p, 1, 2).expect("b1");
        drop(a1); // parent 1 now 1/2, parent 2 now 1/2
        // Next acquisition prefers the most-occupied parent; both are 1/2 so
        // the lowest id wins (parent 1).
        let c = p.borrow_mut().acquire_block(&p, 1, 2).expect("c");
        assert_eq!(
            c.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 0
            })
        );
        let _ = (a2, b1);
    }

    #[test]
    fn acquire_returns_empty_when_exhausted() {
        let p = pool(1);
        let a = p.borrow_mut().acquire_block(&p, 1, 1).expect("a");
        // Bind the result so the block drops before the next RefMut borrow.
        let none = p.borrow_mut().acquire_block(&p, 1, 1);
        assert!(none.is_none());
        drop(a);
        let some = p.borrow_mut().acquire_block(&p, 1, 1);
        assert!(some.is_some());
    }

    #[test]
    fn acquire_many_respects_partial_fill_then_free() {
        let p = pool(2);
        // Parent 1: 1 occupied, parent 2: free.
        let a = p.borrow_mut().acquire_block(&p, 5, 2).expect("a");
        let got = p.borrow_mut().acquire_blocks(&p, 5, 2, 3);
        assert_eq!(got.len(), 3);
        let locs: Vec<_> = got.iter().map(|r| r.location().unwrap()).collect();
        // parent 1 slot 1 (fill), then parent 2 slots 0,1.
        assert_eq!(
            locs,
            vec![
                CacheBlockLocation {
                    lcm_block_id: 1,
                    slot_index: 1
                },
                CacheBlockLocation {
                    lcm_block_id: 2,
                    slot_index: 0
                },
                CacheBlockLocation {
                    lcm_block_id: 2,
                    slot_index: 1
                },
            ]
        );
        let _ = a;
    }

    #[test]
    fn acquire_blocks_partial_failure_returns_empty() {
        let p = pool(1);
        let a = p.borrow_mut().acquire_block(&p, 1, 2).expect("a");
        // Only one slot left; asking for 2 must return empty (all-or-nothing).
        assert!(p.borrow_mut().acquire_blocks(&p, 1, 2, 2).is_empty());
        let _ = a;
    }

    #[test]
    fn pool_is_reusable_after_full_cycle() {
        let p = pool(3);
        let blocks: Vec<_> = p.borrow_mut().acquire_blocks(&p, 9, 1, 3);
        assert_eq!(p.borrow().num_occupied_slots(), 3);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 0);
        drop(blocks);
        assert_eq!(p.borrow().num_occupied_slots(), 0);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 3);
    }

    #[test]
    fn default_ref_is_null() {
        let r: CacheBlockRef = CacheBlockRef::default();
        assert!(r.is_null());
        assert_eq!(r.location(), None);
        assert_eq!(r.use_count(), 0);
        assert!(!r.unique());
    }

    #[test]
    fn refs_share_identity_and_release_on_drop() {
        let p = pool(2);
        let a = p.borrow_mut().acquire_block(&p, 1, 1).expect("block");
        let b = a.clone();
        assert_eq!(a.use_count(), 2);
        assert_eq!(a, b);
        assert!(!a.unique());
        assert_eq!(
            a.location(),
            Some(CacheBlockLocation {
                lcm_block_id: 1,
                slot_index: 0
            })
        );
        assert!(a.is_owned_by(&p));
        drop(b);
        assert_eq!(a.use_count(), 1);
        assert!(a.unique());
        drop(a);
        assert_eq!(p.borrow().num_occupied_slots(), 0);
        assert_eq!(p.borrow().num_empty_lcm_blocks(), 2);
    }

    #[test]
    fn null_ref_never_equals_occupied_ref() {
        let p = pool(1);
        let a = p.borrow_mut().acquire_block(&p, 1, 1).expect("block");
        let null = CacheBlockRef::default();
        assert_ne!(a, null);
        assert!(!null.is_owned_by(&p));
    }

    #[test]
    fn reset_releases_handle() {
        let p = pool(1);
        let mut a = p.borrow_mut().acquire_block(&p, 1, 1).expect("block");
        assert_eq!(p.borrow().num_occupied_slots(), 1);
        a.reset();
        assert!(a.is_null());
        assert_eq!(p.borrow().num_occupied_slots(), 0);
    }

    #[test]
    fn block_table_starts_empty() {
        let table = BlockTable::default();
        assert!(table.rows().is_empty());
        assert_eq!(table.num_rows(), 0);
        assert_eq!(table.num_blocks(), 0);
        assert_eq!(table.available_tokens(), 0);
    }

    #[test]
    fn append_page_maintains_padding_and_counts_blocks() {
        let mut table = BlockTable::default();
        table.append_page(101);
        table.append_page(0); // null hole
        table.append_page(103);
        assert_eq!(table.rows(), &[vec![101, 0, 103, -1]][..]);
        assert_eq!(table.num_rows(), 1);
        assert_eq!(table.num_blocks(), 2); // holes and padding are not blocks
    }

    #[test]
    fn begin_request_starts_a_new_row() {
        let mut table = BlockTable::default();
        table.begin_request();
        assert_eq!(table.rows(), &[vec![-1]][..]);
        table.append_page(7);
        table.begin_request();
        table.append_page(9);
        assert_eq!(table.rows(), &[vec![7, -1], vec![9, -1]][..]);
        assert_eq!(table.num_blocks(), 2);
    }

    #[test]
    fn remove_last_page_rolls_back_and_drops_empty_rows() {
        let mut table = BlockTable::default();
        table.append_page(1);
        table.append_page(2);
        assert_eq!(table.remove_last_page(), Some(2));
        assert_eq!(table.rows(), &[vec![1, -1]][..]);
        assert_eq!(table.remove_last_page(), Some(1));
        // The emptied row (just the sentinel) is dropped.
        assert!(table.rows().is_empty());
        assert_eq!(table.remove_last_page(), None);
    }

    #[test]
    fn available_tokens_round_trip() {
        let mut table = BlockTable::default();
        table.set_available_tokens(16);
        assert_eq!(table.available_tokens(), 16);
        table.set_available_tokens(0);
        assert_eq!(table.available_tokens(), 0);
    }

    #[test]
    #[should_panic(expected = "must be non-negative")]
    fn available_tokens_reject_negative() {
        let mut table = BlockTable::default();
        table.set_available_tokens(-1);
    }

    // -- deterministic stress: random acquire/release preserves invariants --

    struct Xs(u64);
    impl Xs {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_add(0x9e37_79b9_7f4a_7c15))
        }
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    fn check_pool_invariants(p: &Rc<RefCell<BlockPool>>, live: &[CacheBlockRef]) {
        use std::collections::HashSet;
        let pool = p.borrow();
        let mut seen = HashSet::new();
        for b in live {
            let loc = b.location().expect("live block is non-null");
            assert!(seen.insert(loc), "duplicate live location {loc:?}");
            assert!(
                pool.is_occupied(loc),
                "live location {loc:?} must be occupied"
            );
        }
        assert_eq!(
            pool.num_occupied_slots() as usize,
            live.len(),
            "occupied slots must equal live blocks"
        );
        let occupied: HashSet<_> = (1..=pool.num_lcm_blocks())
            .flat_map(|id| pool.occupied_locations(id))
            .collect();
        assert_eq!(
            occupied.len(),
            live.len(),
            "pool-occupied == live locations"
        );
        for loc in &occupied {
            assert!(seen.contains(loc), "pool-occupied {loc:?} not tracked live");
        }
    }

    #[test]
    fn random_acquire_release_preserves_pool_invariants() {
        let p = pool(8);
        let mut rng = Xs::new(0x5eed_1234);
        let mut live: Vec<CacheBlockRef> = Vec::new();
        for _ in 0..3000 {
            match rng.below(10) {
                0..=5 => {
                    // Acquire 0..=3 blocks for a random group (all-or-nothing).
                    let group = rng.below(3) as u32;
                    let n = rng.below(4);
                    let blocks = p.borrow_mut().acquire_blocks(&p, group, 3, n);
                    assert!(blocks.is_empty() || blocks.len() == n, "all-or-nothing");
                    live.extend(blocks);
                }
                6..=8 => {
                    // Drop a random live block (release its slot).
                    if !live.is_empty() {
                        let idx = rng.below(live.len());
                        live.swap_remove(idx);
                    }
                }
                _ => {
                    // Reset a random live block to null (also releases).
                    if !live.is_empty() {
                        let idx = rng.below(live.len());
                        let mut b = live.swap_remove(idx);
                        b.reset();
                    }
                }
            }
            check_pool_invariants(&p, &live);
        }
    }
}
