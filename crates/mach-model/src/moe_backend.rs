//! Host-side MoE offload backend: global-LRU expert residency + placement.
//!
//! This is the vendor-independent policy core behind FreeToken global LRU expert
//! cache. It tracks which experts are GPU-resident in a fixed number of slots per
//! MoE layer, and for each decode/prefill step decides whether a routed expert is
//! computed on GPU (resident, or fetched into a slot) or on the CPU. Device upload
//! and copy live in the GPU integration (a later P1 step); this module is pure
//! logic and unit-testable without a device.

use std::collections::{HashMap, VecDeque};

/// Where a routed expert should be computed for one step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// The expert is GPU-resident; compute on GPU at this slot.
    Gpu(usize),
    /// The expert is not GPU-resident; compute it on the CPU.
    Cpu,
}

/// A single expert to be uploaded into a GPU slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fetch {
    pub id: u32,
    pub slot: usize,
}

/// Result of planning one step routed experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepPlan {
    /// `placements[i]` corresponds to `routed[i]`.
    pub placements: Vec<Placement>,
    /// Experts to upload this step, in encounter order.
    pub fetches: Vec<Fetch>,
    /// Experts evicted to make room, in eviction order.
    pub evictions: Vec<u32>,
}

/// A fixed-capacity, most-recently-used cache of GPU-resident expert slots for
/// one MoE layer.
#[derive(Debug)]
pub struct LruExpertCache {
    capacity: usize,
    /// Expert id -> slot index.
    slots: HashMap<u32, usize>,
    /// Recency: front = most recently used.
    recency: VecDeque<u32>,
}

impl LruExpertCache {
    /// Creates a cache of `capacity` GPU-resident expert slots.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            slots: HashMap::new(),
            recency: VecDeque::new(),
        }
    }

    /// Number of currently resident experts.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    pub fn contains(&self, id: u32) -> bool {
        self.slots.contains_key(&id)
    }

    /// Returns the GPU slot for a resident expert, touching recency.
    pub fn get(&mut self, id: u32) -> Option<usize> {
        let slot = *self.slots.get(&id)?;
        self.touch(id);
        Some(slot)
    }

    /// Reserves a slot for `id`. If already resident, just touches recency.
    /// Otherwise assigns a slot, evicting the least-recently-used expert when at
    /// capacity. The returned slot is always a valid index into `0..capacity`.
    pub fn put(&mut self, id: u32) -> Put {
        if let Some(&slot) = self.slots.get(&id) {
            self.touch(id);
            return Put {
                slot,
                evicted: None,
                was_resident: true,
            };
        }
        let (evicted, slot) = if self.slots.len() >= self.capacity {
            let (evicted, slot) = self.evict_lru();
            (Some(evicted), slot)
        } else {
            (None, self.slots.len())
        };
        self.slots.insert(id, slot);
        self.recency.push_front(id);
        Put {
            slot,
            evicted,
            was_resident: false,
        }
    }

    /// Plans one step routed experts under a GPU fetch budget.
    ///
    /// `gpu_fetch_budget` caps how many new experts may be uploaded to GPU this
    /// step; routed experts beyond the budget are placed on the CPU. This is the
    /// policy hook that the bandwidth-adaptive q* (P2) will tune.
    pub fn plan(&mut self, routed: &[u32], gpu_fetch_budget: usize) -> StepPlan {
        let mut placements = Vec::with_capacity(routed.len());
        let mut fetches = Vec::new();
        let mut evictions = Vec::new();
        let mut budget = gpu_fetch_budget;

        for &id in routed {
            if let Some(slot) = self.get(id) {
                placements.push(Placement::Gpu(slot));
                continue;
            }
            // A miss: fetch if budget remains, otherwise compute on CPU.
            if budget > 0 {
                let put = self.put(id);
                if let Some(ev) = put.evicted {
                    evictions.push(ev);
                }
                fetches.push(Fetch { id, slot: put.slot });
                placements.push(Placement::Gpu(put.slot));
                budget -= 1;
            } else {
                placements.push(Placement::Cpu);
            }
        }
        StepPlan {
            placements,
            fetches,
            evictions,
        }
    }

    /// Plans one step routed experts for a bounded-slot offline cache without
    /// intra-step eviction: resident experts stay on GPU, misses fill free slots up
    /// to capacity, and any overflow is computed on CPU. Cross-step LRU eviction is
    /// a follow-up (retention) optimization; this is the correct-placement core.
    pub fn plan_step(&mut self, routed: &[u32]) -> StepPlan {
        let mut placements = Vec::with_capacity(routed.len());
        let mut fetches = Vec::new();
        for &id in routed {
            if let Some(slot) = self.get(id) {
                placements.push(Placement::Gpu(slot));
            } else if self.slots.len() < self.capacity {
                let put = self.put(id);
                fetches.push(Fetch { id, slot: put.slot });
                placements.push(Placement::Gpu(put.slot));
            } else {
                placements.push(Placement::Cpu);
            }
        }
        StepPlan {
            placements,
            fetches,
            evictions: Vec::new(),
        }
    }

    fn touch(&mut self, id: u32) {
        if let Some(pos) = self.recency.iter().position(|&x| x == id) {
            self.recency.remove(pos);
        }
        self.recency.push_front(id);
    }

    fn evict_lru(&mut self) -> (u32, usize) {
        let id = self.recency.pop_back().expect("evict on non-empty cache");
        let slot = self.slots.remove(&id).expect("resident expert has a slot");
        (id, slot)
    }
}

/// Result of [`LruExpertCache::put`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Put {
    pub slot: usize,
    /// The expert evicted to make room, if any.
    pub evicted: Option<u32>,
    pub was_resident: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_and_get() {
        let mut c = LruExpertCache::new(4);
        let p = c.put(7);
        assert_eq!(p.slot, 0);
        assert_eq!(c.get(7), Some(0));
    }

    #[test]
    fn evict_least_recently_used() {
        let mut c = LruExpertCache::new(2);
        c.put(1);
        c.put(2);
        // Make 1 most recently used so 2 becomes LRU.
        assert_eq!(c.get(1), Some(0));
        let p = c.put(3);
        assert_eq!(p.evicted, Some(2));
        assert!(!c.contains(2));
        assert!(c.contains(3));
    }

    #[test]
    fn resident_put_is_noop() {
        let mut c = LruExpertCache::new(2);
        c.put(1);
        c.put(2);
        let p = c.put(1);
        assert!(p.was_resident);
        assert_eq!(p.evicted, None);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn plan_all_resident() {
        let mut c = LruExpertCache::new(4);
        c.put(1);
        c.put(2);
        let plan = c.plan(&[1, 2, 1], 4);
        assert!(plan.fetches.is_empty());
        assert_eq!(
            plan.placements,
            vec![Placement::Gpu(0), Placement::Gpu(1), Placement::Gpu(0)]
        );
    }

    #[test]
    fn plan_fetch_within_budget() {
        let mut c = LruExpertCache::new(2);
        c.put(1);
        let plan = c.plan(&[1, 2], 1);
        assert_eq!(plan.placements, vec![Placement::Gpu(0), Placement::Gpu(1)]);
        assert_eq!(plan.fetches, vec![Fetch { id: 2, slot: 1 }]);
        assert!(plan.evictions.is_empty());
    }

    #[test]
    fn plan_over_budget_goes_to_cpu() {
        let mut c = LruExpertCache::new(2);
        c.put(1);
        // budget 0: no fetches, miss (2) computed on CPU.
        let plan = c.plan(&[1, 2], 0);
        assert_eq!(plan.placements, vec![Placement::Gpu(0), Placement::Cpu]);
        assert!(plan.fetches.is_empty());
        assert!(!c.contains(2));
    }

    #[test]
    fn plan_step_fills_free_slots_and_overflow_to_cpu() {
        let mut c = LruExpertCache::new(2);
        c.put(1);
        // resident 1 -> Gpu(0); miss 2 fills free slot -> Gpu(1); miss 3 overflows -> Cpu.
        let plan = c.plan_step(&[1, 2, 3]);
        assert_eq!(
            plan.placements,
            vec![Placement::Gpu(0), Placement::Gpu(1), Placement::Cpu]
        );
        assert_eq!(plan.fetches, vec![Fetch { id: 2, slot: 1 }]);
        assert!(plan.evictions.is_empty());
        assert!(c.contains(2));
        assert!(!c.contains(3));
    }

    #[test]
    fn plan_evicts_lru_when_full() {
        let mut c = LruExpertCache::new(2);
        c.put(1);
        c.put(2);
        assert_eq!(c.get(1), Some(0)); // 2 becomes LRU
        let plan = c.plan(&[3], 1);
        assert_eq!(plan.placements, vec![Placement::Gpu(1)]); // evicted 2, route to its slot
        assert_eq!(plan.evictions, vec![2]);
        assert_eq!(plan.fetches, vec![Fetch { id: 3, slot: 1 }]);
    }
}
