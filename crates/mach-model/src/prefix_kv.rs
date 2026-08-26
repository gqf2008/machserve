//! Cross-request prefix reuse on the CPU reference path.
//!
//! Ties the admission planner ([`crate::reuse_planner`]) to the reference
//! model's KV: a request whose token prefix is already cached (a shared system
//! prompt / tool definitions) restores the reused pages' KV and computes only
//! the delta. This is the "cross-request prefix sharing" goal of the TokenSpeed
//! alignment — the CPU-side proof that reuse logits equal full-recompute
//! logits exactly, while the model processes `total - reused_tokens` tokens.
//!
//! # Host-page store vs physical blocks
//!
//! The page KV lives on the host in [`PrefixKvCache::pages`] (keyed by content
//! hash), so reused data stays valid regardless of block lifetimes. The block
//! pool provides capacity accounting: admission is all-or-nothing, and the
//! cache is bounded by pool capacity. The prefix index holds **owning**
//! `CacheBlockRef`s, so a request's plan can be dropped as soon as its prefill
//! finishes — reused blocks stay pinned by the index until evicted. When the
//! pool cannot satisfy fresh demand, [`PrefixKvCache`] evicts the coldest
//! cached page (LRU) and retries; only a request that needs more pages than
//! the entire pool can hold errors.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::Error;
use crate::kv_block_pool::{BlockPool, BlockPoolHandle};
use crate::prefix_cache::CacheKey;
use crate::ref_model::RefModel;
use crate::reuse_planner::{PageRef, PrefixReusePlan, ReusePlanner};
use crate::state_reuse::{Anchor, KvSnapshot};

/// KV of one token page: per-layer byte blobs (same framing as an anchor) plus
/// the hidden state at the page boundary, which is what continuation after a
/// reused prefix needs.
#[derive(Debug, Clone)]
pub struct PageKv {
    /// Per-layer `(k, v)` byte blobs for this page's positions.
    pub layers: Vec<(Vec<u8>, Vec<u8>)>,
    /// Hidden state after this page's last token (before final norm/lm_head).
    pub boundary_hidden: Vec<f32>,
    /// Number of tokens in this page.
    pub len: usize,
}

/// How much a request got from the cache vs computed fresh.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixReuseStats {
    /// Prompt tokens already resident (not recomputed).
    pub reused_tokens: usize,
    /// Prompt tokens actually computed (the delta).
    pub computed_tokens: usize,
    /// Total prompt tokens.
    pub total_tokens: usize,
    /// Pages reused from the cache.
    pub reused_pages: usize,
    /// Pages computed fresh.
    pub fresh_pages: usize,
}

/// Cross-request prefix KV cache over the CPU reference model.
pub struct PrefixKvCache {
    planner: ReusePlanner,
    pool: BlockPoolHandle,
    pages: HashMap<CacheKey, PageKv>,
    tokens_per_page: usize,
}

impl PrefixKvCache {
    /// Build a cache for one cache group (`namespace_id = 1`) with `pool`
    /// LCM blocks and `tokens_per_page` tokens per page.
    #[must_use]
    pub fn new(pool_blocks: i32, tokens_per_page: usize) -> Self {
        Self {
            planner: ReusePlanner::new(0, 1, tokens_per_page, 1),
            pool: Rc::new(RefCell::new(BlockPool::new(pool_blocks))),
            pages: HashMap::new(),
            tokens_per_page,
        }
    }

    /// Number of cached pages.
    #[must_use]
    pub fn cached_pages(&self) -> usize {
        self.pages.len()
    }

    /// Evicts the coldest cached page (LRU): drops its index entry (freeing
    /// the block) and its host KV data. Returns the evicted page key, or
    /// `None` when the cache is empty.
    pub fn evict_oldest(&mut self) -> Option<CacheKey> {
        let (key, _location) = self.planner.evict_oldest()?;
        self.pages.remove(&key);
        Some(key)
    }

    /// Serves `tokens` through `model` (which must be fresh, at position 0),
    /// reusing any cached prefix and computing only the delta. Returns the
    /// final logits and reuse stats. Errors when the pool cannot satisfy the
    /// fresh demand (cache full).
    pub fn serve(
        &mut self,
        model: &mut RefModel,
        tokens: &[u32],
    ) -> Result<(Vec<f32>, PrefixReuseStats), Error> {
        let total_tokens = tokens.len();
        // Runtime check (not debug-only): reusing a non-fresh model would
        // silently produce wrong logits in release builds.
        if model.pos() != 0 {
            return Err(Error::InvalidArgument(format!(
                "prefix_kv serve requires a fresh model at position 0, got pos {}",
                model.pos()
            )));
        }
        let itokens: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let mut plan = self.planner.plan(&self.pool, &itokens);
        if plan.is_none() {
            // Pool full: evict the coldest cached pages until the fresh demand
            // fits, or until there is nothing left to evict.
            while plan.is_none() {
                if self.evict_oldest().is_none() {
                    return Err(Error::Model(
                        "prefix KV cache too small: fresh-page demand exceeds the whole pool"
                            .into(),
                    ));
                }
                plan = self.planner.plan(&self.pool, &itokens);
            }
        }
        let plan = plan.expect("plan after eviction");
        // The contiguous prefix (0..reused_pages) is restored via the anchor;
        // every page after it must be freshly computed. A dedup-reused page
        // beyond the prefix (a chain hole left by eviction/release) would need
        // offset-KV restore this consumer cannot do — assert loudly instead of
        // silently producing wrong logits.
        //
        // Why this is unreachable today: `PrefixKvCache` never releases plans,
        // and an eviction-triggering request always ends with a full pool, so
        // any later request must evict again (removing the hole's chained
        // survivors) before it can plan. A future consumer that calls
        // `ReusePlanner::release` or reuses plans must add offset-KV restore
        // (or route such pages as fresh) before hitting this assert.
        assert!(
            plan.pages[plan.reused_pages..]
                .iter()
                .all(|p| matches!(p, PageRef::Fresh(_))),
            "non-prefix dedup-reused page reached the CPU consumer; add offset-KV restore"
        );
        let reused_tokens = plan.reused_tokens;

        if reused_tokens > 0 {
            let anchor = self.build_anchor(tokens, &plan, reused_tokens)?;
            model.restore_anchor(&anchor)?;
        }

        // Compute the fresh pages one page at a time, snapshotting each page's
        // KV + boundary hidden so future requests can reuse it.
        let mut logits = Vec::new();
        let mut computed_tokens = 0usize;
        for (i, page) in plan.pages.iter().enumerate() {
            if let PageRef::Fresh(_) = page {
                let start = i * self.tokens_per_page;
                let end = (start + self.tokens_per_page).min(total_tokens);
                for &t in &tokens[start..end] {
                    logits = model.decode_step(t);
                }
                computed_tokens += end - start;
                let key = &plan.page_keys[i];
                let page_kv = PageKv {
                    layers: model.kv_slice_bytes(start, end),
                    boundary_hidden: model.hidden().to_vec(),
                    len: end - start,
                };
                self.pages.insert(key.clone(), page_kv);
            }
        }
        let reused_pages = plan.reused_pages;
        let fresh_pages = plan.fresh_pages;
        // A fully-reused request computed nothing: the final logits are the
        // anchor's (the last reused token's hidden through final norm+lm_head).
        if computed_tokens == 0 && total_tokens > 0 {
            logits = model.logits_at_anchor();
        }

        let stats = PrefixReuseStats {
            reused_tokens,
            computed_tokens,
            total_tokens,
            reused_pages,
            fresh_pages,
        };
        Ok((logits, stats))
    }

    /// Builds the anchor for a reused prefix from cached page KV.
    fn build_anchor(
        &self,
        tokens: &[u32],
        plan: &PrefixReusePlan,
        reused_tokens: usize,
    ) -> Result<Anchor, Error> {
        let mut layers: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut hidden = Vec::new();
        for i in 0..plan.reused_pages {
            let key = &plan.page_keys[i];
            let page_kv = self
                .pages
                .get(key)
                .ok_or_else(|| Error::Model("reused page missing from cache".into()))?;
            if layers.is_empty() {
                layers = page_kv.layers.clone();
            } else {
                for (li, (k, v)) in page_kv.layers.iter().enumerate() {
                    layers[li].0.extend_from_slice(k);
                    layers[li].1.extend_from_slice(v);
                }
            }
            hidden = page_kv.boundary_hidden.clone();
        }
        Ok(Anchor {
            id: 0,
            token_idx: reused_tokens - 1,
            tokens: tokens[..reused_tokens].to_vec(),
            kv: KvSnapshot { layers },
            hidden,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Weights};

    fn model(cfg: &Config, w: &Weights) -> RefModel {
        RefModel::new(*cfg, w.clone())
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn shared_prefix_reuses_and_matches_full_recompute() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 42).expect("weights");
        // System prompt = 2 full pages; two requests share it and diverge.
        let system = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let mut a = system.to_vec();
        a.extend([100, 101]);
        let mut b = system.to_vec();
        b.extend([200, 201]);

        let mut cache = PrefixKvCache::new(16, 4);
        let (_, stats_a) = cache.serve(&mut model(&cfg, &w), &a).expect("serve A");
        assert_eq!(stats_a.reused_tokens, 0);
        assert_eq!(stats_a.computed_tokens, 10);
        assert_eq!(stats_a.fresh_pages, 3);

        // B reuses the 8-token system prefix; only the 2-token tail is computed.
        let (logits_b, stats_b) = cache.serve(&mut model(&cfg, &w), &b).expect("serve B");
        assert_eq!(stats_b.reused_tokens, 8);
        assert_eq!(stats_b.computed_tokens, 2);
        assert_eq!(stats_b.total_tokens, 10);
        assert_eq!(stats_b.reused_pages, 2);
        assert_eq!(stats_b.fresh_pages, 1);
        assert_eq!(cache.cached_pages(), 4); // 2 system + qA tail + qB tail

        // Reuse logits must equal full recompute exactly.
        let full = model(&cfg, &w);
        let mut full_m = full;
        let full_logits = full_m.forward(&b);
        let max = max_abs_diff(&logits_b, &full_logits);
        assert_eq!(
            max, 0.0,
            "prefix-reuse logits must match full recompute (max {max})"
        );
    }

    #[test]
    fn identical_request_fully_reused() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 7).expect("weights");
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut cache = PrefixKvCache::new(16, 4);
        cache.serve(&mut model(&cfg, &w), &tokens).expect("A");
        let (logits, stats) = cache.serve(&mut model(&cfg, &w), &tokens).expect("B");
        assert_eq!(stats.reused_tokens, 12);
        assert_eq!(stats.computed_tokens, 0);
        assert_eq!(stats.reused_pages, 3);
        assert_eq!(stats.fresh_pages, 0);
        assert_eq!(
            logits.len(),
            cfg.vocab_size,
            "fully-reused request must return real logits"
        );
        let mut full_m = model(&cfg, &w);
        let full = full_m.forward(&tokens);
        assert_eq!(max_abs_diff(&logits, &full), 0.0);
    }

    #[test]
    fn disjoint_request_computes_all() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 9).expect("weights");
        let a = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let c = [21u32, 22, 23, 24, 25, 26, 27, 28, 29];
        let mut cache = PrefixKvCache::new(16, 4);
        cache.serve(&mut model(&cfg, &w), &a).expect("A");
        let (_, stats_c) = cache.serve(&mut model(&cfg, &w), &c).expect("C");
        assert_eq!(stats_c.reused_tokens, 0);
        assert_eq!(stats_c.computed_tokens, 9);
        assert_eq!(stats_c.total_tokens, 9);
    }

    #[test]
    fn partial_last_page_cached_and_reused() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 3).expect("weights");
        // 6 tokens = full page (4) + partial page (2).
        let tokens = [1u32, 2, 3, 4, 5, 6];
        let mut cache = PrefixKvCache::new(16, 4);
        cache.serve(&mut model(&cfg, &w), &tokens).expect("A");
        let (logits, stats) = cache.serve(&mut model(&cfg, &w), &tokens).expect("B");
        assert_eq!(stats.reused_tokens, 6, "clamped to total");
        assert_eq!(stats.computed_tokens, 0);
        assert_eq!(
            logits.len(),
            cfg.vocab_size,
            "fully-reused request must return real logits"
        );
        let mut full_m = model(&cfg, &w);
        let full = full_m.forward(&tokens);
        assert_eq!(max_abs_diff(&logits, &full), 0.0);
    }

    #[test]
    fn mla_shared_prefix_reuses_and_matches_full_recompute() {
        // MLA (kv_lora_rank > 0) exercises the expanded per-head KV slice path.
        let cfg = Config::mla(128, 2, 4, 1024, 256, 8, 16, 64, 64, 64);
        let w = Weights::random(&cfg, 5).expect("weights");
        let system = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let mut a = system.to_vec();
        a.extend([100, 101]);
        let mut b = system.to_vec();
        b.extend([200, 201]);

        let mut cache = PrefixKvCache::new(16, 4);
        cache.serve(&mut model(&cfg, &w), &a).expect("serve A");
        let (logits_b, stats_b) = cache.serve(&mut model(&cfg, &w), &b).expect("serve B");
        assert_eq!(stats_b.reused_tokens, 8);
        assert_eq!(stats_b.computed_tokens, 2);
        assert_eq!(logits_b.len(), cfg.vocab_size);
        let mut full_m = model(&cfg, &w);
        let full = full_m.forward(&b);
        assert_eq!(max_abs_diff(&logits_b, &full), 0.0);

        // Fully-reused request on the MLA path returns the anchor logits.
        let (logits_b2, stats_b2) = cache.serve(&mut model(&cfg, &w), &b).expect("serve B2");
        assert_eq!(stats_b2.computed_tokens, 0);
        assert_eq!(logits_b2.len(), cfg.vocab_size);
        assert_eq!(max_abs_diff(&logits_b2, &full), 0.0);
    }

    #[test]
    fn pool_full_evicts_coldest_for_new_work() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 11).expect("weights");
        // 1 block only: one 4-token page fills it.
        let mut cache = PrefixKvCache::new(1, 4);
        cache.serve(&mut model(&cfg, &w), &[1, 2, 3, 4]).expect("A");
        assert_eq!(cache.cached_pages(), 1);
        // A disjoint request needs a fresh page; the pool evicts the coldest
        // (the only) cached page and serves it.
        let (_, stats) = cache
            .serve(&mut model(&cfg, &w), &[9, 10, 11, 12])
            .expect("B");
        assert_eq!(stats.reused_tokens, 0);
        assert_eq!(stats.computed_tokens, 4);
        assert_eq!(cache.cached_pages(), 1, "cache stays bounded");
        // The evicted page is recomputed, not reused.
        let (_, stats) = cache.serve(&mut model(&cfg, &w), &[1, 2, 3, 4]).expect("C");
        assert_eq!(stats.reused_tokens, 0, "page 1 was evicted");
        assert_eq!(stats.computed_tokens, 4);
    }

    #[test]
    fn request_larger_than_whole_pool_errors() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 13).expect("weights");
        // Pool holds 1 block but a request needs 2 fresh pages (8 tokens).
        let mut cache = PrefixKvCache::new(1, 4);
        cache.serve(&mut model(&cfg, &w), &[1, 2, 3, 4]).expect("A");
        assert!(
            cache
                .serve(&mut model(&cfg, &w), &[9, 10, 11, 12, 5, 6, 7, 8])
                .is_err()
        );
    }

    #[test]
    fn heavy_eviction_recompute_stays_bit_exact() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 11).expect("weights");
        // Pool = 4 pages. A and B share a 2-page system prefix; C's disjoint
        // 3-page request forces eviction that frees the prefix head but leaves
        // B's chained tail page cached (a chain hole).
        let system = [1u32, 2, 3, 4, 5, 6, 7, 8]; // 2 pages
        let mut a = system.to_vec();
        a.extend([100, 101]);
        let b = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let c: Vec<u32> = (21..=32).collect();

        let mut cache = PrefixKvCache::new(4, 4);
        // A caches 3 pages (2 system prefix + its tail); B reuses the 2-page
        // prefix and adds a fresh tail page -> pool (4) is full.
        cache.serve(&mut model(&cfg, &w), &a).expect("serve A");
        let (_, stats_b) = cache.serve(&mut model(&cfg, &w), &b).expect("serve B");
        assert_eq!(stats_b.reused_pages, 2);
        assert_eq!(stats_b.fresh_pages, 1);
        // C's disjoint 3-page request needs 3 fresh pages -> evicts the three
        // coldest pages (A's prefix + A's tail) but leaves B's tail cached:
        // a chain hole for B's prefix.
        let (_, stats_c) = cache.serve(&mut model(&cfg, &w), &c).expect("serve C");
        assert_eq!(stats_c.fresh_pages, 3);

        // Re-request B: the prefix head pages are evicted; the re-request's
        // fresh demand forces another eviction that clears the hole's chained
        // survivors, so the plan ends fully fresh. Logits must stay bit-exact
        // vs full recompute (regression: no panic, no stale reuse).
        let (logits, stats) = cache.serve(&mut model(&cfg, &w), &b).expect("serve D");
        assert_eq!(stats.reused_pages, 0);
        assert_eq!(stats.fresh_pages, 3);
        let mut full = model(&cfg, &w);
        let full_logits = full.forward(&b);
        assert_eq!(max_abs_diff(&logits, &full_logits), 0.0);
    }
}
