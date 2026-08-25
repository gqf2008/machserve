//! Cross-request prefix-reuse admission planner.
//!
//! Composes [`crate::prefix_cache`] (content-addressable page keys + matcher)
//! with [`crate::kv_block_pool`] (physical LCM placement) into the admission
//! decision a scheduler makes before a request's prefill: which prefix pages
//! are already resident in the KV cache (reused), how many fresh blocks the
//! tail needs, and an ordered per-page view of the request. This is the
//! machserve port of TokenSpeed's CacheCoordinator admission step
//! (prefix match + acquire).
//!
//! The planner is deliberately CPU-only and model-agnostic: it operates on
//! token ids and page hashes, so it is testable without a GPU or a loaded
//! model. The KV layer later maps each [`PageRef`] to a real KV buffer (the
//! reused pages need no recompute; fresh pages must be computed).
//!
//! # Lifecycle / ownership
//!
//! [`crate::prefix_cache::PrefixCacheIndex`] holds owning `CacheBlockRef`s
//! (upstream TokenSpeed semantics), so a cached page pins its block
//! independently of any request plan. Plans may be dropped as soon as their
//! prefill finishes; releasing a plan only removes the keys **it** registered
//! and frees blocks that no other plan or the index still references. A
//! "hole" (a released key whose chained successors remain cached) is handled
//! gracefully: the chained page dedup-reuses the canonical block.

use crate::kv_block_pool::{BlockPoolHandle, CacheBlockLocation, CacheBlockRef};
use crate::prefix_cache::{CacheKey, PrefixCacheIndex, PrefixMatcher, compute_prefix_hashes};

/// One page of a request's token stream: either already resident (reused) or
/// freshly acquired (must be computed and populated). A reused page carries an
/// owning [`CacheBlockRef`] clone so the block stays pinned even if the plan
/// that originally cached it is released.
#[derive(Debug, Clone)]
pub enum PageRef {
    /// The page's KV is already resident; reuse this pinned block.
    Reused(CacheBlockRef),
    /// The page was freshly acquired; compute its KV into `block`.
    Fresh(CacheBlockRef),
}

/// The result of planning one request's admission.
///
/// **Ownership**: the plan holds [`CacheBlockRef`] clones; the prefix index
/// also pins every cached page, so the plan can be dropped once its prefill
/// finishes. When a request completes, call [`ReusePlanner::release`] to
/// remove the keys **this plan registered** and release its block refs (blocks
/// still referenced by the index or other plans stay alive).
#[derive(Debug)]
pub struct PrefixReusePlan {
    /// Ordered pages of the request (`reused_pages` reused + `fresh_pages` fresh).
    pub pages: Vec<PageRef>,
    /// Number of leading pages already resident (the shared prefix).
    pub reused_pages: usize,
    /// Number of pages that must be computed.
    pub fresh_pages: usize,
    /// Prompt tokens covered by the reused pages — the delta starts here.
    pub reused_tokens: usize,
    /// Total prompt tokens.
    pub total_tokens: usize,
    /// Content keys of this plan's fresh pages (for [`ReusePlanner::release`]).
    fresh_keys: Vec<CacheKey>,
    /// Content keys of every page, in order (reused prefix then fresh tail).
    pub page_keys: Vec<CacheKey>,
}

/// Admission planner for one cache group: computes page hashes, probes the
/// prefix index for the longest contiguous shared prefix, and acquires fresh
/// blocks for the tail (all-or-nothing).
pub struct ReusePlanner {
    group_id: u32,
    namespace_id: u32,
    tokens_per_page: usize,
    cache_blocks_per_lcm_block: i32,
    index: PrefixCacheIndex,
}

impl ReusePlanner {
    /// Build a planner for one cache group.
    #[must_use]
    pub fn new(
        group_id: u32,
        namespace_id: u32,
        tokens_per_page: usize,
        cache_blocks_per_lcm_block: i32,
    ) -> Self {
        assert!(tokens_per_page > 0, "tokens_per_page must be > 0");
        Self {
            group_id,
            namespace_id,
            tokens_per_page,
            cache_blocks_per_lcm_block,
            index: PrefixCacheIndex::new(group_id),
        }
    }

    /// Number of pages currently tracked in the prefix index.
    #[must_use]
    pub fn cached_pages(&self) -> usize {
        self.index.num_entries()
    }

    /// Whether `key` is currently advertised in the prefix index.
    #[must_use]
    pub fn index_contains(&self, key: &CacheKey) -> bool {
        self.index.contains(key)
    }

    /// Plan admission for `tokens`.
    ///
    /// Returns `None` when the block pool cannot satisfy the fresh demand
    /// (all-or-nothing: nothing is inserted into the index on failure).
    pub fn plan(&mut self, pool: &BlockPoolHandle, tokens: &[i32]) -> Option<PrefixReusePlan> {
        let total_tokens = tokens.len();
        let pages: Vec<&[i32]> = tokens.chunks(self.tokens_per_page).collect();
        if pages.is_empty() {
            return Some(PrefixReusePlan {
                pages: Vec::new(),
                reused_pages: 0,
                fresh_pages: 0,
                reused_tokens: 0,
                total_tokens: 0,
                fresh_keys: Vec::new(),
                page_keys: Vec::new(),
            });
        }

        // Content-addressable keys for every page (chained hashes).
        let hashes = compute_prefix_hashes(&pages, "", &[]);
        let keys: Vec<CacheKey> = hashes
            .iter()
            .enumerate()
            .map(|(i, h)| CacheKey {
                namespace_id: self.namespace_id,
                group_id: self.group_id,
                content_hash: h.clone(),
                page_offset: (i * self.tokens_per_page) as i32,
            })
            .collect();

        // Longest contiguous cached prefix (full attention). Reused pages pin
        // their blocks via the index's owning refs, so releasing the original
        // plan never frees a block another request still reuses.
        let hits = PrefixMatcher.probe(&self.index, &keys, 0, keys.len());
        let reused_pages = hits.len();
        let reused_tokens = (reused_pages * self.tokens_per_page).min(total_tokens);

        let mut reused_blocks: Vec<CacheBlockRef> = keys[..reused_pages]
            .iter()
            .map(|k| {
                self.index
                    .block_for(k)
                    .expect("matcher hit must resolve to a pinned block")
            })
            .collect();

        // Acquire the tail fresh blocks (all-or-nothing).
        let fresh_pages = pages.len() - reused_pages;
        let fresh_blocks = pool.borrow_mut().acquire_blocks(
            pool,
            self.group_id,
            self.cache_blocks_per_lcm_block,
            fresh_pages,
        );
        if fresh_blocks.len() != fresh_pages {
            return None; // pool cannot satisfy; nothing inserted
        }

        // Record fresh blocks in the index (owning clones pin them). When the
        // key is already canonical elsewhere (e.g. a released plan left a
        // chain hole and this page's chained key is still cached), reuse the
        // canonical block and release the duplicate. `fresh_keys` only records
        // keys this plan actually registered (insert returned `None`) so
        // `release` can never remove another plan's canonical entry.
        let mut fresh_keys: Vec<CacheKey> = Vec::with_capacity(fresh_pages);
        let mut pages_out: Vec<PageRef> = Vec::with_capacity(pages.len());
        pages_out.extend(reused_blocks.drain(..).map(PageRef::Reused));
        for (i, block) in fresh_blocks.into_iter().enumerate() {
            let key = &keys[reused_pages + i];
            match self.index.insert(key.clone(), &block) {
                None => {
                    fresh_keys.push(key.clone());
                    pages_out.push(PageRef::Fresh(block));
                }
                Some(previous) => {
                    // Duplicate content: keep the canonical block, free the new one.
                    drop(block);
                    pages_out.push(PageRef::Reused(previous));
                }
            }
        }

        Some(PrefixReusePlan {
            pages: pages_out,
            reused_pages,
            fresh_pages,
            reused_tokens,
            total_tokens,
            fresh_keys,
            page_keys: keys,
        })
    }

    /// Evicts the coldest cached page (LRU), freeing its block. Returns the
    /// evicted key and location, or `None` when the index is empty.
    pub fn evict_oldest(&mut self) -> Option<(CacheKey, CacheBlockLocation)> {
        self.index.evict_oldest()
    }

    /// Release a plan: removes the keys it registered from the prefix index
    /// and drops the plan (releasing its block refs). Blocks still pinned by
    /// the index or other plans stay alive; only orphaned blocks return to
    /// the pool.
    pub fn release(&mut self, plan: PrefixReusePlan) {
        for key in &plan.fresh_keys {
            self.index.remove(key);
        }
        drop(plan); // frees the fresh CacheBlockRefs -> slots back to the pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::kv_block_pool::BlockPool;
    use crate::prefix_cache::compute_prefix_hashes;

    fn keys_for(tokens: &[i32], page: usize) -> Vec<CacheKey> {
        let pages: Vec<&[i32]> = tokens.chunks(page).collect();
        let hashes = compute_prefix_hashes(&pages, "", &[]);
        hashes
            .iter()
            .enumerate()
            .map(|(i, h)| CacheKey {
                namespace_id: 7,
                group_id: 0,
                content_hash: h.clone(),
                page_offset: (i * page) as i32,
            })
            .collect()
    }

    fn planner() -> (ReusePlanner, BlockPoolHandle) {
        let p = ReusePlanner::new(0, 7, 4, 2);
        let pool = Rc::new(RefCell::new(BlockPool::new(8)));
        (p, pool)
    }

    #[test]
    fn first_request_acquires_all_fresh() {
        let (mut p, pool) = planner();
        let plan = p.plan(&pool, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("plan");
        assert_eq!(plan.reused_pages, 0);
        assert_eq!(plan.fresh_pages, 2);
        assert_eq!(plan.reused_tokens, 0);
        assert_eq!(plan.pages.len(), 2);
        assert!(matches!(plan.pages[0], PageRef::Fresh(_)));
        assert_eq!(p.cached_pages(), 2);
        // Releasing frees the blocks and un-advertises the keys.
        p.release(plan);
        assert_eq!(p.cached_pages(), 0);
        assert_eq!(pool.borrow().num_occupied_slots(), 0);
    }

    #[test]
    fn second_request_reuses_shared_prefix() {
        let (mut p, pool) = planner();
        // Request A stays alive so B can reuse its prefix pages.
        let plan_a = p.plan(&pool, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("plan A");
        let plan_b = p.plan(&pool, &[1, 2, 3, 4, 9, 10, 11, 12]).expect("plan B");
        assert_eq!(plan_b.reused_pages, 1);
        assert_eq!(plan_b.fresh_pages, 1);
        assert_eq!(plan_b.reused_tokens, 4);
        assert!(matches!(plan_b.pages[0], PageRef::Reused(_)));
        assert!(matches!(plan_b.pages[1], PageRef::Fresh(_)));
        // B must NOT be released while A still references its page 1? No —
        // B owns its own fresh page 1; releasing B frees only that.
        p.release(plan_b);
        p.release(plan_a);
    }

    #[test]
    fn identical_request_is_fully_reused() {
        let (mut p, pool) = planner();
        let plan_a = p.plan(&pool, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("plan A");
        let plan_b = p.plan(&pool, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("plan B");
        assert_eq!(plan_b.reused_pages, 2);
        assert_eq!(plan_b.fresh_pages, 0);
        assert_eq!(plan_b.reused_tokens, 8);
        assert!(matches!(plan_b.pages[0], PageRef::Reused(_)));
        assert!(matches!(plan_b.pages[1], PageRef::Reused(_)));
        // B reused A's pages; releasing B frees nothing (no fresh keys), A still owns them.
        p.release(plan_b);
        assert_eq!(p.cached_pages(), 2);
        assert_eq!(pool.borrow().num_occupied_slots(), 2);
        p.release(plan_a);
        assert_eq!(pool.borrow().num_occupied_slots(), 0);
    }

    #[test]
    fn disjoint_request_reuses_nothing() {
        let (mut p, pool) = planner();
        let plan_a = p.plan(&pool, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("plan A");
        let plan_b = p
            .plan(&pool, &[21, 22, 23, 24, 25, 26, 27, 28])
            .expect("plan B");
        assert_eq!(plan_b.reused_pages, 0);
        assert_eq!(plan_b.fresh_pages, 2);
        p.release(plan_b);
        p.release(plan_a);
    }

    #[test]
    fn chain_hash_prevents_reuse_after_different_prefix() {
        let (mut p, pool) = planner();
        // A: page0=[1,2,3,4], page1=[5,6,7,8] (page1 hash chains on A's page0).
        let plan_a = p.plan(&pool, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("plan A");
        // C: page0=[9,10,11,12] differs, so C's page1=[5,6,7,8] gets a
        // different chained hash than A's page1 even though the page tokens
        // match — nothing is reused.
        let plan_c = p.plan(&pool, &[9, 10, 11, 12, 5, 6, 7, 8]).expect("plan C");
        assert_eq!(plan_c.reused_pages, 0, "chain hash differs -> no reuse");
        assert_eq!(plan_c.fresh_pages, 2);
        // An identical re-request fully reuses C (page0 + chained page1).
        let plan_d = p.plan(&pool, &[9, 10, 11, 12, 5, 6, 7, 8]).expect("plan D");
        assert_eq!(plan_d.reused_pages, 2, "identical to C -> full reuse");
        assert_eq!(plan_d.fresh_pages, 0);
        p.release(plan_d);
        p.release(plan_c);
        p.release(plan_a);
    }

    #[test]
    fn partial_page_share_is_not_reused() {
        let (mut p, pool) = planner();
        let plan_a = p.plan(&pool, &[1, 2, 3, 4]).expect("plan A");
        // Shares 2 tokens but not a whole page boundary -> no reuse.
        let plan_b = p.plan(&pool, &[1, 2, 9, 10]).expect("plan B");
        assert_eq!(plan_b.reused_pages, 0);
        assert_eq!(plan_b.fresh_pages, 1);
        p.release(plan_b);
        p.release(plan_a);
    }

    #[test]
    fn pool_exhaustion_returns_none_and_keeps_index_clean() {
        // 1 LCM parent with 1 slot => only 1 fresh block available.
        let mut p = ReusePlanner::new(0, 7, 4, 1);
        let pool = Rc::new(RefCell::new(BlockPool::new(1)));
        let plan_a = p.plan(&pool, &[1, 2, 3, 4]).expect("plan A (1 page)");
        assert_eq!(p.cached_pages(), 1);
        // Second request needs 1 fresh page but the pool is exhausted.
        assert!(p.plan(&pool, &[9, 10, 11, 12]).is_none());
        assert_eq!(p.cached_pages(), 1);
        // Release A, then the same request can be admitted again.
        p.release(plan_a);
        let plan_c = p.plan(&pool, &[9, 10, 11, 12]).expect("plan C");
        assert_eq!(plan_c.reused_pages, 0);
        assert_eq!(plan_c.fresh_pages, 1);
        p.release(plan_c);
    }

    #[test]
    fn releasing_reused_prefix_then_rescheduling_reuses_canonical() {
        let (mut p, pool) = planner();
        let plan_a = p.plan(&pool, &[1, 2, 3, 4]).expect("plan A");
        let plan_b = p.plan(&pool, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("plan B");
        // Releasing A removes its key from the index, but B pinned page 0 via
        // an owning ref, so the block stays valid.
        p.release(plan_a);
        // Re-planning the same tokens: page 0 misses (key gone), page 1's
        // chained key is still canonical (B cached it) -> dedup reuses it.
        let plan_c = p.plan(&pool, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("plan C");
        assert_eq!(plan_c.reused_pages, 0, "page 0 key was released");
        assert_eq!(plan_c.fresh_pages, 2);
        assert!(matches!(plan_c.pages[0], PageRef::Fresh(_)));
        assert!(
            matches!(plan_c.pages[1], PageRef::Reused(_)),
            "page 1 dedup-reuses B's canonical block"
        );
        // Release must not remove B's canonical key: only keys C registered
        // (just k0) are removed, so the index still holds B's k1.
        p.release(plan_c);
        assert_eq!(
            p.cached_pages(),
            1,
            "release must not touch B's canonical k1"
        );
        let k1 = keys_for(&[1, 2, 3, 4, 5, 6, 7, 8], 4)[1].clone();
        assert!(p.index_contains(&k1), "B's page-1 key still resolves");
        p.release(plan_b);
    }

    #[test]
    fn partial_last_page_reused_tokens_is_clamped() {
        let (mut p, pool) = planner();
        // 6 tokens = page0 (4) + partial page1 (2).
        let plan_a = p.plan(&pool, &[1, 2, 3, 4, 5, 6]).expect("plan A");
        assert_eq!(plan_a.reused_tokens, 0);
        assert_eq!(plan_a.total_tokens, 6);
        let plan_b = p.plan(&pool, &[1, 2, 3, 4, 5, 6]).expect("plan B");
        assert_eq!(plan_b.reused_pages, 2);
        assert_eq!(plan_b.reused_tokens, 6, "clamped to total_tokens, not 8");
        p.release(plan_b);
        p.release(plan_a);
    }

    #[test]
    fn empty_tokens_plan_is_vacuous() {
        let (mut p, pool) = planner();
        let plan = p.plan(&pool, &[]).expect("plan");
        assert_eq!(plan.reused_pages, 0);
        assert_eq!(plan.fresh_pages, 0);
        assert_eq!(plan.pages.len(), 0);
        p.release(plan);
    }
}
