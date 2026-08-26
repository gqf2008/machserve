//! Paged KV: block table + paged-attention reference (GPU-wiring foundation).
//!
//! Replaces the contiguous per-slot KV layout (`[slot, max_seq, kv_heads,
//! head_dim]`) with a **page pool** (`[num_pages, tokens_per_page, kv_heads,
//! head_dim]`) referenced by per-request block tables. Paging is what enables
//! cross-request prefix sharing (multiple requests' block tables point at the
//! same physical pages) and capacity eviction (the reuse-planner / owning-ref
//! stack already plans it). This module is the hardware-agnostic side of the
//! GPU wiring: the block table type plus a CPU reference of the paged-attention
//! access pattern, verified against the contiguous reference. The HIP kernel
//! that mirrors this reference lives in [`crate::kernels`] and is covered by
//! the offline hiprtc compile gate.

/// Physical page id per logical page (`pages[logical] = physical`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlockTable {
    pages: Vec<u32>,
}

impl BlockTable {
    /// An empty block table (no pages yet).
    #[must_use]
    pub fn new() -> Self {
        Self { pages: Vec::new() }
    }

    /// Appends one physical page to the end of the logical page sequence.
    pub fn append(&mut self, physical: u32) {
        self.pages.push(physical);
    }

    /// The physical page backing `logical`, if present.
    #[must_use]
    pub fn get(&self, logical: usize) -> Option<u32> {
        self.pages.get(logical).copied()
    }

    /// Number of logical pages.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// True when no pages are mapped.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Truncates to `len` logical pages (drops the tail mapping).
    pub fn truncate(&mut self, len: usize) {
        self.pages.truncate(len);
    }

    /// Read-only view of the logical -> physical mapping.
    #[must_use]
    pub fn pages(&self) -> &[u32] {
        &self.pages
    }
}

/// Maps a token position to its (physical page, in-page offset) under a block
/// table with `tokens_per_page` tokens per page. Returns `None` when the page
/// is not mapped.
#[must_use]
pub fn page_offsets(
    table: &BlockTable,
    pos: usize,
    tokens_per_page: usize,
) -> Option<(u32, usize)> {
    let logical = pos / tokens_per_page;
    let physical = table.get(logical)?;
    Some((physical, pos % tokens_per_page))
}

/// Paged decode attention (CPU reference).
///
/// `q` is `[n_heads, head_dim]`; the KV page pool is
/// `[num_pages, tokens_per_page, n_kv_heads, head_dim]` (row-major). Reads the
/// prefix `0..=pos` through `table`, computing the same per-head softmax
/// attention the contiguous kernel produces — so the block-table access
/// pattern is verified independently of the GPU.
#[must_use]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn paged_attention_decode(
    q: &[f32],
    k_pool: &[f32],
    v_pool: &[f32],
    table: &BlockTable,
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    tokens_per_page: usize,
    scale: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * head_dim, "q shape");
    let groups = n_heads / n_kv_heads;
    let mut out = vec![0.0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kv = h / groups;
        // scores[p] for p in 0..=pos
        let n = pos + 1;
        let mut scores = vec![0.0f32; n];
        for p in 0..n {
            let (page, off) = page_offsets(table, p, tokens_per_page).expect("page mapped");
            let koff = ((page as usize * tokens_per_page + off) * n_kv_heads + kv) * head_dim;
            let mut sc = 0.0f32;
            for dd in 0..head_dim {
                sc += q[h * head_dim + dd] * k_pool[koff + dd];
            }
            scores[p] = sc * scale;
        }
        let maxv = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &scores {
            sum += (s - maxv).exp();
        }
        for dd in 0..head_dim {
            let mut acc = 0.0f32;
            for p in 0..n {
                let (page, off) = page_offsets(table, p, tokens_per_page).expect("page mapped");
                let voff =
                    (((page as usize * tokens_per_page + off) * n_kv_heads + kv) * head_dim) + dd;
                acc += (scores[p] - maxv).exp() * v_pool[voff];
            }
            out[h * head_dim + dd] = acc / sum;
        }
    }
    out
}

/// Contiguous decode attention (reference): same math, KV laid out contiguously
/// per request `[max_seq, kv_heads, head_dim]` (as the current batched.rs
/// kernels do). Used to prove the paged access pattern is equivalent.
#[must_use]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn contiguous_attention_decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let groups = n_heads / n_kv_heads;
    let mut out = vec![0.0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kv = h / groups;
        let n = pos + 1;
        let mut scores = vec![0.0f32; n];
        for p in 0..n {
            let koff = (p * n_kv_heads + kv) * head_dim;
            let mut sc = 0.0f32;
            for dd in 0..head_dim {
                sc += q[h * head_dim + dd] * k[koff + dd];
            }
            scores[p] = sc * scale;
        }
        let maxv = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &scores {
            sum += (s - maxv).exp();
        }
        for dd in 0..head_dim {
            let mut acc = 0.0f32;
            for p in 0..n {
                let voff = (p * n_kv_heads + kv) * head_dim + dd;
                acc += (scores[p] - maxv).exp() * v[voff];
            }
            out[h * head_dim + dd] = acc / sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 32) as u32 as f32) / (u32::MAX as f32) - 0.5
        }
    }

    #[test]
    fn block_table_basics() {
        let mut t = BlockTable::new();
        assert!(t.is_empty());
        t.append(7);
        t.append(3);
        assert_eq!(t.get(0), Some(7));
        assert_eq!(t.get(1), Some(3));
        assert_eq!(t.get(2), None);
        assert_eq!(t.len(), 2);
        t.truncate(1);
        assert_eq!(t.get(1), None);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn page_offsets_map_positions() {
        let mut t = BlockTable::new();
        t.append(10);
        t.append(20);
        assert_eq!(page_offsets(&t, 0, 4), Some((10, 0)));
        assert_eq!(page_offsets(&t, 3, 4), Some((10, 3)));
        assert_eq!(page_offsets(&t, 4, 4), Some((20, 0)));
        assert_eq!(page_offsets(&t, 7, 4), Some((20, 3)));
        assert_eq!(page_offsets(&t, 8, 4), None);
    }

    #[test]
    fn paged_matches_contiguous_attention() {
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let tokens_per_page = 4;
        let max_pos: usize = 11; // spans 3 pages
        let mut rng = lcg(42);

        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| rng()).collect();
        // Contiguous KV per request: [max_seq, n_kv_heads, head_dim].
        let k: Vec<f32> = (0..(max_pos + 1) * n_kv_heads * head_dim)
            .map(|_| rng())
            .collect();
        let v: Vec<f32> = (0..(max_pos + 1) * n_kv_heads * head_dim)
            .map(|_| rng())
            .collect();

        // Pack the same KV into a page pool with a sequential block table.
        let num_pages: usize = (max_pos + 1).div_ceil(tokens_per_page);
        let mut table = BlockTable::new();
        for page in 0..num_pages as u32 {
            table.append(page);
        }
        let pool_len = num_pages * tokens_per_page * n_kv_heads * head_dim;
        let mut k_pool = vec![0.0f32; pool_len];
        let mut v_pool = vec![0.0f32; pool_len];
        for p in 0..=max_pos {
            for kv in 0..n_kv_heads {
                for dd in 0..head_dim {
                    let src = (p * n_kv_heads + kv) * head_dim + dd;
                    let (page, off) = page_offsets(&table, p, tokens_per_page).unwrap();
                    let dst =
                        ((page as usize * tokens_per_page + off) * n_kv_heads + kv) * head_dim + dd;
                    k_pool[dst] = k[src];
                    v_pool[dst] = v[src];
                }
            }
        }

        for pos in [0usize, 3, 4, 7, 10, 11] {
            let scale = 1.0 / (head_dim as f32).sqrt();
            let paged = paged_attention_decode(
                &q,
                &k_pool,
                &v_pool,
                &table,
                pos,
                n_heads,
                n_kv_heads,
                head_dim,
                tokens_per_page,
                scale,
            );
            let contig =
                contiguous_attention_decode(&q, &k, &v, pos, n_heads, n_kv_heads, head_dim, scale);
            assert_eq!(paged.len(), contig.len());
            let max_diff = paged
                .iter()
                .zip(&contig)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-6,
                "pos {pos}: paged vs contiguous max diff {max_diff}"
            );
        }
    }

    #[test]
    fn shared_prefix_tables_read_same_pages() {
        // Two requests share the first page (physical 5) and diverge after.
        let mut a = BlockTable::new();
        a.append(5);
        a.append(9);
        let mut b = BlockTable::new();
        b.append(5); // shared prefix page
        b.append(11);
        assert_eq!(
            a.get(0),
            b.get(0),
            "shared prefix maps to the same physical page"
        );
        assert_ne!(a.get(1), b.get(1), "tail pages differ");
    }
}
