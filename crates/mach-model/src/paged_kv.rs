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

use crate::{Config, Error, Weights};

/// Physical page id per logical page (`pages[logical] = physical`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PagedTable {
    pages: Vec<u32>,
}

impl PagedTable {
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
    table: &PagedTable,
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
    table: &PagedTable,
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

/// Allocates physical page ids from a fixed-size pool (free-list reuse).
#[derive(Debug, Clone, Default)]
pub struct PageAllocator {
    free: Vec<u32>,
    /// One slot per page: true while the page is owned by a table or plan.
    /// Makes `free` catch every *observable* double free in O(1) — the
    /// free→realloc→free-by-first-owner residue is indistinguishable from
    /// the second owner's legitimate free (see `free`'s doc).
    allocated: Vec<bool>,
    next: u32,
    num_pages: u32,
}

impl PageAllocator {
    /// A pool with `num_pages` physical pages (ids `0..num_pages`).
    #[must_use]
    pub fn new(num_pages: u32) -> Self {
        Self {
            free: Vec::new(),
            allocated: vec![false; num_pages as usize],
            next: 0,
            num_pages,
        }
    }

    /// Allocates a physical page, or `None` when the pool is exhausted.
    pub fn alloc(&mut self) -> Option<u32> {
        let p = if let Some(p) = self.free.pop() {
            p
        } else if self.next < self.num_pages {
            let p = self.next;
            self.next += 1;
            p
        } else {
            return None;
        };
        assert!(
            !self.allocated[p as usize],
            "PageAllocator alloc of live page {p}"
        );
        self.allocated[p as usize] = true;
        Some(p)
    }

    /// Returns a page to the pool for reuse.
    ///
    /// Panics on an out-of-range id or freeing a page whose ownership was
    /// already released (double free), in O(1) via the allocated bitmap.
    /// Note the residue a bitmap cannot see: a page freed, re-allocated to
    /// a second owner, then freed *again by the first owner* is
    /// indistinguishable from the second owner's legitimate free — catching
    /// that would need per-owner tracking (the four free paths here are
    /// mutually exclusive by invariant).
    pub fn free(&mut self, page: u32) {
        assert!(
            page < self.num_pages && self.allocated[page as usize],
            "PageAllocator double-free or out-of-range page {page} (num_pages {})",
            self.num_pages
        );
        self.allocated[page as usize] = false;
        self.free.push(page);
    }
}

/// Writes one token's `[n_kv_heads, head_dim]` K/V row into a page pool at
/// `(page, off)` — the layout [`paged_attention_decode`] reads back.
fn store_row_paged(pool: &mut [f32], row: &[f32], page: u32, off: usize, cfg: Config, tpp: usize) {
    let base = (page as usize * tpp + off) * cfg.n_kv_heads * cfg.head_dim;
    pool[base..base + row.len()].copy_from_slice(row);
}

/// Writes one token's per-head `[heads, dim]` row into an MLA page pool at
/// `(page, off)` — the layout [`paged_attention_decode_mla`] reads back.
fn store_row_paged_mla(
    pool: &mut [f32],
    row: &[f32],
    page: u32,
    off: usize,
    heads: usize,
    dim: usize,
    tokens_per_page: usize,
) {
    let base = (page as usize * tokens_per_page + off) * heads * dim;
    pool[base..base + row.len()].copy_from_slice(row);
}

/// Paged MLA decode attention (CPU reference): per-head expanded q/k/v stored
/// in per-head page pools, read through the block table. Mirrors
/// `ref_model::decode_step_mla`'s attention with the paged layout.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn paged_attention_decode_mla(
    qm: &[f32],
    k_pool: &[f32],
    v_pool: &[f32],
    table: &PagedTable,
    pos: usize,
    heads: usize,
    hd: usize,
    v_hd: usize,
    tokens_per_page: usize,
    scale: f32,
) -> Vec<f32> {
    let tpp = tokens_per_page;
    let k_row = heads * hd;
    let v_row = heads * v_hd;
    let mut attn = vec![0.0f32; heads * v_hd];
    for h in 0..heads {
        let qh = &qm[h * hd..(h + 1) * hd];
        let mut scores = vec![0.0f32; pos + 1];
        let mut maxv = f32::NEG_INFINITY;
        for pp in 0..=pos {
            let (page, off) = page_offsets(table, pp, tpp).expect("page mapped");
            let base = (page as usize * tpp + off) * k_row + h * hd;
            let mut s = 0.0f32;
            for dd in 0..hd {
                s += qh[dd] * k_pool[base + dd];
            }
            s *= scale;
            scores[pp] = s;
            maxv = maxv.max(s);
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - maxv).exp();
            sum += *s;
        }
        for dd in 0..v_hd {
            let mut acc = 0.0f32;
            for pp in 0..=pos {
                let (page, off) = page_offsets(table, pp, tpp).expect("page mapped");
                let base = (page as usize * tpp + off) * v_row + h * v_hd + dd;
                acc += scores[pp] * v_pool[base];
            }
            attn[h * v_hd + dd] = acc / sum;
        }
    }
    attn
}

/// Per-slot state of a paged reference sequence.
#[derive(Debug)]
struct PagedSlot {
    table: PagedTable,
    pos: usize,
    /// Hidden state after the last token (reserved for prefix-boundary
    /// continuation / anchor-style restore in the GPU integration).
    #[allow(dead_code)]
    last_hidden: Vec<f32>,
}

/// Paged-KV reference transformer (GPU-wiring blueprint).
///
/// Mirrors [`crate::ref_model::RefModel`]'s dense forward but stores K/V into a
/// page pool referenced by per-slot [`PagedTable`]s — the exact layout
/// `attn_decode_paged` reads on the GPU. Parity with `RefModel` (exact) proves
/// the paged store + block-table attention are correct through a full
/// transformer, so the `batched.rs` GPU integration can mirror this structure
/// with kernels. Slots may share leading physical pages (a shared system
/// prefix), which the block-table layout makes free.
pub struct PagedRef {
    cfg: Config,
    w: Weights,
    tokens_per_page: usize,
    /// Per-layer K/V page pools `[num_pages, tpp, n_kv_heads, head_dim]`.
    k_pools: Vec<Vec<f32>>,
    v_pools: Vec<Vec<f32>>,
    /// MLA (kv_lora_rank > 0): per-head expanded K/V page pools
    /// `[num_pages, tpp, n_heads, qk_nope+qk_rope]` / `[num_pages, tpp, n_heads, v_head_dim]`.
    mla_k_pools: Vec<Vec<f32>>,
    mla_v_pools: Vec<Vec<f32>>,
    allocator: PageAllocator,
    slots: Vec<Option<PagedSlot>>,
}

impl PagedRef {
    /// Builds a paged reference with `capacity` slots and a `num_pages` page pool.
    #[must_use]
    pub fn new(
        cfg: Config,
        w: Weights,
        capacity: usize,
        num_pages: u32,
        tokens_per_page: usize,
    ) -> Self {
        assert!(tokens_per_page > 0);
        let pool_len = num_pages as usize * tokens_per_page * cfg.n_kv_heads * cfg.head_dim;
        let mla = cfg.kv_lora_rank > 0;
        let mla_k_len = if mla {
            num_pages as usize
                * tokens_per_page
                * cfg.n_heads
                * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim)
        } else {
            0
        };
        let mla_v_len = if mla {
            num_pages as usize * tokens_per_page * cfg.n_heads * cfg.v_head_dim
        } else {
            0
        };
        Self {
            k_pools: (0..cfg.n_layers).map(|_| vec![0.0; pool_len]).collect(),
            v_pools: (0..cfg.n_layers).map(|_| vec![0.0; pool_len]).collect(),
            mla_k_pools: (0..cfg.n_layers).map(|_| vec![0.0; mla_k_len]).collect(),
            mla_v_pools: (0..cfg.n_layers).map(|_| vec![0.0; mla_v_len]).collect(),
            allocator: PageAllocator::new(num_pages),
            slots: (0..capacity).map(|_| None).collect(),
            cfg,
            w,
            tokens_per_page,
        }
    }

    /// Prefills `tokens` into `slot`, returning the final logits.
    pub fn prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        let mut logits = Vec::new();
        for &t in tokens {
            logits = self.decode_step(slot, t)?;
        }
        Ok(logits)
    }

    /// Starts a slot from a **shared prefix**: `shared_pages` already hold the
    /// KV for the first `reused_tokens` prompt tokens, so only the delta is
    /// computed — the slot's position begins at `reused_tokens` and fresh pages
    /// are allocated beyond the shared prefix. This is the delta-only decode
    /// the GPU path mirrors (block tables + page pool).
    ///
    /// **Precondition (enforced in debug builds)**: `shared_pages` must cover
    /// exactly `reused_tokens` tokens on page boundaries
    /// (`shared_pages.len() * tokens_per_page == reused_tokens`). A misaligned
    /// start would write the delta into a shared page's middle, silently
    /// corrupting the canonical owner's KV for other slots.
    pub fn start_with_shared_prefix(
        &mut self,
        slot: usize,
        shared_pages: &[u32],
        reused_tokens: usize,
    ) {
        debug_assert_eq!(
            shared_pages.len() * self.tokens_per_page,
            reused_tokens,
            "shared prefix must cover exactly reused_tokens on page boundaries"
        );
        let mut table = PagedTable::new();
        for &p in shared_pages {
            table.append(p);
        }
        self.slots[slot] = Some(PagedSlot {
            table,
            pos: reused_tokens,
            last_hidden: Vec::new(),
        });
    }

    /// Reuses `pages` as `slot`'s leading block table (a shared prefix). The
    /// slot resumes **after** the shared pages (delta-only: position
    /// `pages.len() * tokens_per_page`) and appends fresh pages beyond them.
    /// Call this only on a slot that has not prefilled anything yet; sharing
    /// into a used slot would silently leak its allocated pages and is
    /// rejected with a panic. The shared pages must already hold KV for the
    /// exact same token sequence (written by the owner).
    pub fn share_prefix(&mut self, slot: usize, pages: &[u32]) {
        assert!(
            self.slots[slot].is_none(),
            "share_prefix on a used slot would leak its allocated pages"
        );
        let mut table = PagedTable::new();
        for &p in pages {
            table.append(p);
        }
        self.slots[slot] = Some(PagedSlot {
            table,
            // Delta-only: the shared pages already hold KV for
            // pages.len()*tokens_per_page tokens, so the slot resumes after
            // them. (Starting at 0 would recompute and overwrite the shared
            // pages, corrupting them for other slots unless the KV writes are
            // bit-identical — which is not guaranteed across configs/quant.)
            pos: pages.len() * self.tokens_per_page,
            last_hidden: Vec::new(),
        });
    }

    /// Current block table of `slot` (read-only view for tests/inspection).
    #[must_use]
    pub fn slot_table(&self, slot: usize) -> Option<&PagedTable> {
        self.slots
            .get(slot)
            .and_then(|s| s.as_ref())
            .map(|s| &s.table)
    }

    /// One decode step for `token` at `slot`'s current position, storing K/V
    /// into the page pool and attending through the block table.
    pub fn decode_step(&mut self, slot: usize, token: u32) -> Result<Vec<f32>, Error> {
        let cfg = self.cfg;
        let d = cfg.d_model;
        if self.slots[slot].is_none() {
            self.slots[slot] = Some(PagedSlot {
                table: PagedTable::new(),
                pos: 0,
                last_hidden: Vec::new(),
            });
        }
        let pos = self.slots[slot].as_ref().expect("slot").pos;
        if pos >= cfg.max_seq_len {
            return Err(Error::InvalidArgument(
                "sequence length exceeded max_seq_len".into(),
            ));
        }
        if token as usize >= cfg.vocab_size {
            return Err(Error::InvalidArgument(format!(
                "decode_step token {token} out of range (vocab {})",
                cfg.vocab_size
            )));
        }
        // Ensure the page backing `pos` exists (allocate on page boundary).
        let logical = pos / self.tokens_per_page;
        if self.slots[slot].as_ref().expect("slot").table.len() <= logical {
            let page = self
                .allocator
                .alloc()
                .ok_or_else(|| Error::Model("page pool exhausted".into()))?;
            self.slots[slot].as_mut().expect("slot").table.append(page);
        }
        let table = &self.slots[slot].as_ref().expect("slot").table;

        let x0 = &self.w.tok_emb[token as usize * d..(token as usize + 1) * d];
        let mut x = x0.to_vec();
        let tpp = self.tokens_per_page;
        for (li, lw) in self.w.layers.iter().enumerate() {
            let xn = crate::ref_model::rms_norm(&x, &lw.rms_attn, cfg.rms_eps);
            let attn_proj = if cfg.kv_lora_rank > 0 {
                // MLA (DeepSeek-V2 style): low-rank Q + compressed KV, expanded
                // per-head, stored in the per-head MLA page pools.
                let heads = cfg.n_heads;
                let nope = cfg.qk_nope_head_dim;
                let rope_hd = cfg.qk_rope_head_dim;
                let v_hd = cfg.v_head_dim;
                let hd = nope + rope_hd;
                let q_lora = crate::ref_model::matvec_t(&xn, &lw.mla_q_a, cfg.q_lora_rank);
                let q_lora = crate::ref_model::rms_norm(&q_lora, &lw.mla_q_a_norm, cfg.rms_eps);
                let q_nope = crate::ref_model::matvec_t(&q_lora, &lw.mla_q_b, heads * nope);
                let mut q_rope = crate::ref_model::matvec_t(&xn, &lw.mla_q_rope, heads * rope_hd);
                crate::ref_model::apply_rope(&mut q_rope, heads, rope_hd, pos, cfg.rope_theta);
                let kv_a =
                    crate::ref_model::matvec_t(&xn, &lw.mla_kv_a, cfg.kv_lora_rank + rope_hd);
                let kv_lora = crate::ref_model::rms_norm(
                    &kv_a[..cfg.kv_lora_rank],
                    &lw.mla_kv_a_norm,
                    cfg.rms_eps,
                );
                let mut k_rope = kv_a[cfg.kv_lora_rank..].to_vec();
                crate::ref_model::apply_rope(&mut k_rope, 1, rope_hd, pos, cfg.rope_theta);
                let kv = crate::ref_model::matvec_t(&kv_lora, &lw.mla_kv_b, heads * (nope + v_hd));
                let mut qm = vec![0.0f32; heads * hd];
                let mut km = vec![0.0f32; heads * hd];
                let mut vm = vec![0.0f32; heads * v_hd];
                for h in 0..heads {
                    let base = h * (nope + v_hd);
                    qm[h * hd..h * hd + nope].copy_from_slice(&q_nope[h * nope..(h + 1) * nope]);
                    qm[h * hd + nope..(h + 1) * hd]
                        .copy_from_slice(&q_rope[h * rope_hd..(h + 1) * rope_hd]);
                    km[h * hd..h * hd + nope].copy_from_slice(&kv[base..base + nope]);
                    km[h * hd + nope..(h + 1) * hd].copy_from_slice(&k_rope);
                    vm[h * v_hd..(h + 1) * v_hd]
                        .copy_from_slice(&kv[base + nope..base + nope + v_hd]);
                }
                let (page, off) = page_offsets(table, pos, tpp).expect("page mapped");
                store_row_paged_mla(&mut self.mla_k_pools[li], &km, page, off, heads, hd, tpp);
                store_row_paged_mla(&mut self.mla_v_pools[li], &vm, page, off, heads, v_hd, tpp);
                let scale = 1.0 / (hd as f32).sqrt();
                let attn = paged_attention_decode_mla(
                    &qm,
                    &self.mla_k_pools[li],
                    &self.mla_v_pools[li],
                    table,
                    pos,
                    heads,
                    hd,
                    v_hd,
                    tpp,
                    scale,
                );
                crate::ref_model::matvec_t(&attn, &lw.mla_o, d)
            } else {
                let mut q = crate::ref_model::matvec_t(&xn, &lw.wq, cfg.n_heads * cfg.head_dim);
                let mut k = crate::ref_model::matvec_t(&xn, &lw.wk, cfg.n_kv_heads * cfg.head_dim);
                let v = crate::ref_model::matvec_t(&xn, &lw.wv, cfg.n_kv_heads * cfg.head_dim);
                if !lw.q_norm.is_empty() {
                    crate::ref_model::qk_norm(
                        &mut q,
                        &lw.q_norm,
                        cfg.n_heads,
                        cfg.head_dim,
                        cfg.rms_eps,
                    );
                    crate::ref_model::qk_norm(
                        &mut k,
                        &lw.k_norm,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        cfg.rms_eps,
                    );
                }
                crate::ref_model::apply_rope(
                    &mut q,
                    cfg.n_heads,
                    cfg.head_dim,
                    pos,
                    cfg.rope_theta,
                );
                crate::ref_model::apply_rope(
                    &mut k,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    pos,
                    cfg.rope_theta,
                );
                let (page, off) = page_offsets(table, pos, tpp).expect("page mapped");
                store_row_paged(&mut self.k_pools[li], &k, page, off, cfg, tpp);
                store_row_paged(&mut self.v_pools[li], &v, page, off, cfg, tpp);
                let scale = 1.0 / (cfg.head_dim as f32).sqrt();
                let attn = paged_attention_decode(
                    &q,
                    &self.k_pools[li],
                    &self.v_pools[li],
                    table,
                    pos,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    tpp,
                    scale,
                );
                crate::ref_model::matvec_t(&attn, &lw.wo, d)
            };
            for i in 0..d {
                x[i] += attn_proj[i];
            }

            // MLP: routed MoE (Qwen-MoE style) when configured, else dense
            // SwiGLU. Mirrors ref_model::forward_from's MLP branch exactly;
            // MLA paged variants are follow-ups.
            let xn2 = crate::ref_model::rms_norm(&x, &lw.rms_mlp, cfg.rms_eps);
            if cfg.num_experts > 0 && !lw.moe_router.is_empty() {
                // MoE: router softmax -> top-k experts -> weighted sum of
                // per-expert SwiGLU MLPs.
                let ne = cfg.num_experts;
                let topk = cfg.num_experts_per_tok.min(ne);
                let einter = cfg.expert_size();
                let router = crate::ref_model::matvec_t(&xn2, &lw.moe_router, ne);
                let mut probs = vec![0.0; ne];
                let mut maxr = f32::NEG_INFINITY;
                for r in &router {
                    maxr = maxr.max(*r);
                }
                let mut sumr = 0.0f32;
                for i in 0..ne {
                    probs[i] = (router[i] - maxr).exp();
                    sumr += probs[i];
                }
                for p in probs.iter_mut() {
                    *p /= sumr;
                }
                // Top-k expert indices by probability (ties: lower index).
                let mut order: Vec<usize> = (0..ne).collect();
                order.sort_by(|&a, &b| {
                    probs[b]
                        .partial_cmp(&probs[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.cmp(&b))
                });
                let mut norm = 0.0f32;
                for &e in order.iter().take(topk) {
                    norm += probs[e];
                }
                let mut h = vec![0.0; d];
                for &e in order.iter().take(topk) {
                    // Expert e: gate/up [einter, d], down [d, einter].
                    let wg = &lw.moe_wg[e * einter * d..(e + 1) * einter * d];
                    let wu = &lw.moe_wu[e * einter * d..(e + 1) * einter * d];
                    let wd = &lw.moe_wd[e * d * einter..(e + 1) * d * einter];
                    let gate = crate::ref_model::matvec_t(&xn2, wg, einter);
                    let up = crate::ref_model::matvec_t(&xn2, wu, einter);
                    let mut eh = vec![0.0; einter];
                    for i in 0..einter {
                        eh[i] = crate::ref_model::silu(gate[i]) * up[i];
                    }
                    let down = crate::ref_model::matvec_t(&eh, wd, d);
                    let w = probs[e] / norm;
                    for i in 0..d {
                        h[i] += w * down[i];
                    }
                }
                for i in 0..d {
                    x[i] += h[i];
                }
            } else {
                // Dense SwiGLU.
                let gate = crate::ref_model::matvec_t(&xn2, &lw.wg, cfg.intermediate_size);
                let up = crate::ref_model::matvec_t(&xn2, &lw.wu, cfg.intermediate_size);
                let mut h = vec![0.0; cfg.intermediate_size];
                for i in 0..cfg.intermediate_size {
                    h[i] = crate::ref_model::silu(gate[i]) * up[i];
                }
                let down = crate::ref_model::matvec_t(&h, &lw.wd, d);
                for i in 0..d {
                    x[i] += down[i];
                }
            }
        }

        let xf = crate::ref_model::rms_norm(&x, &self.w.rms_final, cfg.rms_eps);
        let logits = crate::ref_model::matvec_t(&xf, &self.w.lm_head, cfg.vocab_size);
        let slot_state = self.slots[slot].as_mut().expect("slot");
        slot_state.pos = pos + 1;
        slot_state.last_hidden = x;
        Ok(logits)
    }
}

/// Admission-time resolution result: the per-request block table, the reused
/// prefix token count, and the precomputed content-hash chain (consumed by
/// [`GpuPagedTableBuilder::register_chain`] at materialization — no rehash).
pub struct PagedTablePlan {
    /// Logical → physical block table (prompt pages only; pad separately).
    pub table: PagedTable,
    /// Prompt tokens covered by materialized, aliased prefix pages.
    pub reused_tokens: usize,
    /// Aliased prefix pages (exact page count — `reused_tokens` clamps a
    /// partial final page, which `free_plan_pages` must not misread).
    pub reused_pages: usize,
    /// Per-page content hashes (chain), in page order. Consumed by the
    /// hip-gated engine (`continuous`) at materialization and eviction, so
    /// the CPU-only face sees no reader.
    #[cfg_attr(not(feature = "hip"), allow(dead_code))]
    pub(crate) chain: Vec<String>,
}

/// GPU block-table builder: allocates physical page ids in a shared page pool,
/// reusing pages already cached under the same content hash (a shared system
/// prompt), and produces per-request logical->physical block tables that
/// `kv_store_paged` / `attn_decode_paged` consume. The hash chain makes the
/// reused prefix exactly the leading run of cached pages; fresh pages after it
/// are allocated and cached for future requests.
pub struct GpuPagedTableBuilder {
    tokens_per_page: usize,
    /// Content hash -> physical page id (shared-prefix cache).
    cache: std::collections::HashMap<String, u32>,
    allocator: PageAllocator,
}

impl GpuPagedTableBuilder {
    /// A builder over a `num_pages`-page GPU pool.
    #[must_use]
    pub fn new(num_pages: u32, tokens_per_page: usize) -> Self {
        Self {
            tokens_per_page,
            cache: std::collections::HashMap::new(),
            allocator: PageAllocator::new(num_pages),
        }
    }

    /// Number of distinct physical pages currently cached.
    #[must_use]
    pub fn cached_pages(&self) -> usize {
        self.cache.len()
    }

    /// Builds the block table for one request, reusing cached prefix pages and
    /// allocating fresh pages for the tail. Returns the table and the number of
    /// prompt tokens covered by the reused prefix (the delta starts there).
    /// Errors when the page pool cannot satisfy the fresh demand.
    ///
    /// The content cache is **read-only** here: fresh tail pages are allocated
    /// but NOT registered — registration happens in [`Self::register_chain`]
    /// once the pages actually hold their KV (the engine's materialization
    /// gate). An eager insert would let a concurrent request alias pages whose
    /// content is not written yet, and both requests would then clobber each
    /// other's generated KV in the same physical page.
    pub fn build_table(&mut self, tokens: &[i32]) -> Result<(PagedTable, usize), Error> {
        let plan = self.plan(tokens, true)?;
        let reused = plan.reused_tokens;
        Ok((plan.table, reused))
    }

    /// Computes the per-page content-hash chain for `tokens` (prompt chunked
    /// into pages, in page order). Admission computes this **once** and feeds
    /// it to [`Self::plan_with_chain`] — including every eviction-retry
    /// attempt, so pool pressure never triggers a rehash.
    #[must_use]
    pub fn compute_chain(&self, tokens: &[i32]) -> Vec<String> {
        let pages: Vec<&[i32]> = tokens.chunks(self.tokens_per_page).collect();
        crate::prefix_cache::compute_prefix_hashes(&pages, "", &[])
    }

    /// One admission-time resolution: either aliases materialized cached
    /// pages (`allow_reuse`) or allocates fresh pages for every prompt page.
    /// The returned plan carries the chain so [`Self::register_chain`] can
    /// register the content at materialization **without rehashing**.
    ///
    /// Reuse covers **full prompt pages only**: a partial last page is always
    /// allocated fresh. An aliased partial page would be written by every
    /// request that reuses it — the first generated token lands in that page,
    /// and two concurrently-active identical prompts would clobber each
    /// other's generated KV at the same offsets.
    pub fn plan(&mut self, tokens: &[i32], allow_reuse: bool) -> Result<PagedTablePlan, Error> {
        let chain = self.compute_chain(tokens);
        self.plan_with_chain(tokens, &chain, allow_reuse)
    }

    /// [`Self::plan`] over a precomputed [`Self::compute_chain`] chain: same
    /// aliasing, allocation, and rollback behavior without rehashing. `chain`
    /// must be `tokens`' own chain (the engine computes both together at
    /// admission).
    pub fn plan_with_chain(
        &mut self,
        tokens: &[i32],
        chain: &[String],
        allow_reuse: bool,
    ) -> Result<PagedTablePlan, Error> {
        // Only the leading FULL pages may be aliased (see plan's doc above).
        let full_pages = tokens.len() / self.tokens_per_page;
        let mut table = PagedTable::new();
        let mut reused_pages = 0usize;
        let mut allocating = false;
        for (page_idx, h) in chain.iter().enumerate() {
            if !allocating && allow_reuse && page_idx < full_pages {
                if let Some(&page) = self.cache.get(h) {
                    table.append(page);
                    reused_pages += 1;
                    continue;
                }
                allocating = true;
            }
            let Some(page) = self.allocator.alloc() else {
                // Roll back this call's fresh allocations (the leading reused
                // pages are cache aliases and must not be freed).
                for logical in (reused_pages..table.len()).rev() {
                    if let Some(p) = table.get(logical) {
                        self.allocator.free(p);
                    }
                }
                return Err(Error::Model("page pool exhausted".into()));
            };
            table.append(page);
        }
        // Exact reused-token boundary: only full pages alias, so this equals
        // `reused_pages * tokens_per_page` (no clamp — a partial final page
        // is never in the aliased run). `free_plan_pages` must derive its
        // fresh/aliased boundary from `reused_pages`, never from this field.
        let reused_tokens = reused_pages * self.tokens_per_page;
        Ok(PagedTablePlan {
            table,
            reused_tokens,
            reused_pages,
            chain: chain.to_vec(),
        })
    }

    /// Registers `chain`'s pages — call only once the pages hold their KV
    /// (prefill materialized). Later `build_table` calls reuse the chain from
    /// this point on. First writer wins: when two requests computed the same
    /// content into *different* fresh pages (both admitted before either
    /// materialized), only the first registration maps the hash; the second
    /// request's duplicate pages stay private to its own table (its table
    /// still addresses them directly), and later reuse resolves to the
    /// registered ones.
    pub fn register_chain(&mut self, chain: &[String], table: &PagedTable) {
        for (h, &logical) in chain.iter().zip(table.pages()) {
            if self.cache.contains_key(h) {
                continue;
            }
            self.cache.insert(h.clone(), logical);
        }
    }

    /// Extends `table` with fresh pages up to `max_pages` logical entries —
    /// the generated-KV region. Decode writes past the prompt's pages (at
    /// position >= prompt_len) and attention addresses every position up to
    /// `max_seq_len`, so every request needs its full-length table backed by
    /// dedicated pages: repeating the last prompt page (or padding with a
    /// shared page) would silently overwrite prompt KV. Rolls back appended
    /// pages on pool exhaustion.
    pub fn pad_table(&mut self, table: &mut PagedTable, max_pages: usize) -> Result<(), Error> {
        let before = table.len();
        while table.len() < max_pages {
            let Some(page) = self.allocator.alloc() else {
                // Roll back: truncate the table so the freed pages are not
                // freed again by the caller's plan cleanup.
                let appended: Vec<u32> = table.pages()[before..].to_vec();
                table.truncate(before);
                for p in appended {
                    self.allocator.free(p);
                }
                return Err(Error::Model("page pool exhausted (pad)".into()));
            };
            table.append(page);
        }
        Ok(())
    }

    /// Physical page registered for `hash`, if any (eviction support: lets the
    /// engine decide whether a content page is still referenced by live
    /// tables before freeing it).
    #[must_use]
    pub fn page_of(&self, hash: &str) -> Option<u32> {
        self.cache.get(hash).copied()
    }

    /// Unregisters `hash` and returns its page to the pool. Call only when no
    /// live table references the page (the engine checks before evicting).
    /// Returns false when the hash was not registered.
    pub fn evict_page(&mut self, hash: &str) -> bool {
        if let Some(page) = self.cache.remove(hash) {
            self.allocator.free(page);
            true
        } else {
            false
        }
    }

    /// Frees the freshly-allocated (non-reused) pages of a plan — used when a
    /// later step (e.g. padding) fails after the plan itself succeeded, so
    /// the partially-built table never leaks its pages. The reused boundary
    /// is the plan's exact page count (aliased pages are never freed).
    pub fn free_plan_pages(&mut self, plan: &PagedTablePlan, table: &PagedTable) {
        for page in table.pages().iter().skip(plan.reused_pages) {
            self.allocator.free(*page);
        }
    }

    /// Returns a raw page to the pool (pads and other unregistered pages).
    pub fn free_page(&mut self, page: u32) {
        self.allocator.free(page);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ref_model::RefModel;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

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
        let mut t = PagedTable::new();
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
        let mut t = PagedTable::new();
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
        let mut table = PagedTable::new();
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
    fn page_allocator_reuses_freed_pages() {
        let mut a = PageAllocator::new(3);
        assert_eq!(a.alloc(), Some(0));
        assert_eq!(a.alloc(), Some(1));
        assert_eq!(a.alloc(), Some(2));
        assert_eq!(a.alloc(), None, "pool exhausted");
        a.free(1);
        assert_eq!(a.alloc(), Some(1), "freed page reused");
        assert_eq!(a.alloc(), None);
    }

    #[test]
    fn paged_ref_matches_ref_model_exact() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 51).expect("weights");
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // spans 3 pages of 4
        let mut pr = PagedRef::new(cfg, w.clone(), 2, 8, 4);
        let paged_logits = pr.prefill(0, &tokens).expect("prefill");
        let mut ref_m = RefModel::new(cfg, w);
        let ref_logits = ref_m.forward(&tokens);
        let max = max_abs_diff(&paged_logits, &ref_logits);
        assert_eq!(
            max, 0.0,
            "paged ref must equal ref_model exactly (max {max})"
        );
        assert_eq!(
            pr.slot_table(0).expect("table").len(),
            3,
            "10 tokens over 4/page -> 3 pages"
        );
    }

    #[test]
    fn paged_ref_shared_prefix_pages_keep_correctness() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 52).expect("weights");
        let system = [1u32, 2, 3, 4, 5, 6, 7, 8]; // 2 pages
        let mut a = system.to_vec();
        a.push(100);
        let mut b = system.to_vec();
        b.push(200);

        let mut pr = PagedRef::new(cfg, w.clone(), 2, 16, 4);
        let logits_a = pr.prefill(0, &a).expect("prefill A");
        let table_a = pr.slot_table(0).expect("table A").pages().to_vec();
        assert_eq!(table_a.len(), 3);

        // B shares A's two system pages (8 tokens), then appends its own tail
        // page delta-only (only the token past the shared prefix).
        pr.share_prefix(1, &table_a[..2]);
        let logits_b = pr.prefill(1, &b[8..]).expect("prefill B");

        // Correctness: B's logits equal a full recompute.
        let mut ref_b = RefModel::new(cfg, w.clone());
        let ref_logits_b = ref_b.forward(&b);
        assert_eq!(
            max_abs_diff(&logits_b, &ref_logits_b),
            0.0,
            "shared-prefix paged decode must equal full recompute"
        );
        // A is unaffected by B's writes to the shared pages: continuing A with
        // a fresh token after B's prefill must equal a full-recompute model
        // continuing the same way (A's KV pages are intact).
        let a_continue = pr.decode_step(0, 999).expect("continue A");
        let mut ref_a = RefModel::new(cfg, w.clone());
        ref_a.forward(&a);
        let ref_continue = ref_a.decode_step(999);
        assert_eq!(
            max_abs_diff(&a_continue, &ref_continue),
            0.0,
            "A's KV pages unchanged after B shared them"
        );
        // A's first-pass logits still equal full recompute.
        let mut ref_a1 = RefModel::new(cfg, w);
        let ref_logits_a = ref_a1.forward(&a);
        assert_eq!(max_abs_diff(&logits_a, &ref_logits_a), 0.0, "A first pass");
        // The shared physical page holds identical KV after both prefills.
        let (shared_page_a, off_a) = page_offsets(pr.slot_table(0).unwrap(), 0, 4).unwrap();
        let (shared_page_b, _) = page_offsets(pr.slot_table(1).unwrap(), 0, 4).unwrap();
        assert_eq!(shared_page_a, shared_page_b, "both slots share page 0");
        let _ = off_a;
        let n = cfg.n_kv_heads * cfg.head_dim;
        let ka = &pr.k_pools[0][shared_page_a as usize * 4 * n..shared_page_a as usize * 4 * n + n];
        let kb = &pr.k_pools[0][shared_page_b as usize * 4 * n..shared_page_b as usize * 4 * n + n];
        assert_eq!(ka, kb, "shared page KV identical after both writes");
    }

    #[test]
    fn paged_ref_mla_matches_ref_model_exact() {
        let cfg = Config::mla(128, 2, 4, 1024, 256, 8, 16, 64, 64, 64);
        let w = Weights::random(&cfg, 91).expect("weights");
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // spans 3 pages of 4
        let mut pr = PagedRef::new(cfg, w.clone(), 2, 8, 4);
        let paged = pr.prefill(0, &tokens).expect("prefill");
        let mut ref_m = RefModel::new(cfg, w);
        let ref_logits = ref_m.forward(&tokens);
        assert_eq!(
            max_abs_diff(&paged, &ref_logits),
            0.0,
            "paged MLA ref must equal RefModel exactly"
        );
    }

    #[test]
    fn paged_ref_mla_shared_prefix_pages_keep_correctness() {
        let cfg = Config::mla(128, 2, 4, 1024, 256, 8, 16, 64, 64, 64);
        let w = Weights::random(&cfg, 92).expect("weights");
        let system = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let mut a = system.to_vec();
        a.push(100);
        let mut b = system.to_vec();
        b.push(200);
        let mut pr = PagedRef::new(cfg, w.clone(), 2, 16, 4);
        let logits_a = pr.prefill(0, &a).expect("A");
        let pages = pr.slot_table(0).expect("A table").pages().to_vec();
        pr.share_prefix(1, &pages[..2]);
        let logits_b = pr.prefill(1, &b[8..]).expect("B");
        let mut ref_b = RefModel::new(cfg, w.clone());
        let ref_logits_b = ref_b.forward(&b);
        assert_eq!(
            max_abs_diff(&logits_b, &ref_logits_b),
            0.0,
            "MLA shared-prefix pages must equal full recompute"
        );
        let mut ref_a = RefModel::new(cfg, w);
        assert_eq!(
            max_abs_diff(&logits_a, &ref_a.forward(&a)),
            0.0,
            "A unchanged"
        );
    }

    #[test]
    fn paged_ref_moe_matches_ref_model_exact() {
        // MoE config (mirrors tests/state_reuse.rs::cfg_moe): routed MLP with
        // per-expert gate/up/down replaces the dense SwiGLU.
        let mut cfg = Config::tiny();
        cfg.intermediate_size = 64;
        cfg.moe_intermediate_size = 48;
        cfg.num_experts = 4;
        cfg.num_experts_per_tok = 2;
        let w = Weights::random(&cfg, 61).expect("weights");
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // spans 3 pages of 4
        let mut pr = PagedRef::new(cfg, w.clone(), 2, 8, 4);
        let paged_logits = pr.prefill(0, &tokens).expect("prefill");
        let mut ref_m = RefModel::new(cfg, w);
        let ref_logits = ref_m.forward(&tokens);
        let max = max_abs_diff(&paged_logits, &ref_logits);
        assert_eq!(
            max, 0.0,
            "paged MoE ref must equal ref_model exactly (max {max})"
        );
    }

    #[test]
    fn paged_ref_moe_shared_prefix_pages_keep_correctness() {
        // Same sharing scenario as the dense test, but with routed MoE MLPs so
        // the shared physical pages are read back through top-k experts.
        let mut cfg = Config::tiny();
        cfg.intermediate_size = 64;
        cfg.moe_intermediate_size = 48;
        cfg.num_experts = 4;
        cfg.num_experts_per_tok = 2;
        let w = Weights::random(&cfg, 62).expect("weights");
        let system = [1u32, 2, 3, 4, 5, 6, 7, 8]; // 2 pages
        let mut a = system.to_vec();
        a.push(100);
        let mut b = system.to_vec();
        b.push(200);

        let mut pr = PagedRef::new(cfg, w.clone(), 2, 16, 4);
        let logits_a = pr.prefill(0, &a).expect("prefill A");
        let table_a = pr.slot_table(0).expect("table A").pages().to_vec();
        assert_eq!(table_a.len(), 3);

        // B shares A's two system pages (8 tokens), then appends its own tail
        // page delta-only (only the token past the shared prefix).
        pr.share_prefix(1, &table_a[..2]);
        let logits_b = pr.prefill(1, &b[8..]).expect("prefill B");

        // Correctness: B's logits equal a full recompute.
        let mut ref_b = RefModel::new(cfg, w.clone());
        let ref_logits_b = ref_b.forward(&b);
        assert_eq!(
            max_abs_diff(&logits_b, &ref_logits_b),
            0.0,
            "shared-prefix MoE decode must equal full recompute"
        );
        // A is unaffected by B's writes to the shared pages: continuing A with
        // a fresh token after B's prefill must equal a full-recompute model
        // continuing the same way (A's KV pages are intact).
        let a_continue = pr.decode_step(0, 999).expect("continue A");
        let mut ref_a = RefModel::new(cfg, w.clone());
        ref_a.forward(&a);
        let ref_continue = ref_a.decode_step(999);
        assert_eq!(
            max_abs_diff(&a_continue, &ref_continue),
            0.0,
            "A's MoE KV pages unchanged after B shared them"
        );
        // A's first-pass logits still equal full recompute.
        let mut ref_a1 = RefModel::new(cfg, w);
        let ref_logits_a = ref_a1.forward(&a);
        assert_eq!(max_abs_diff(&logits_a, &ref_logits_a), 0.0, "A first pass");
        // The shared physical page holds identical KV after both prefills.
        let (shared_page_a, _) = page_offsets(pr.slot_table(0).unwrap(), 0, 4).unwrap();
        let (shared_page_b, _) = page_offsets(pr.slot_table(1).unwrap(), 0, 4).unwrap();
        assert_eq!(shared_page_a, shared_page_b, "both slots share page 0");
        let n = cfg.n_kv_heads * cfg.head_dim;
        let ka = &pr.k_pools[0][shared_page_a as usize * 4 * n..shared_page_a as usize * 4 * n + n];
        let kb = &pr.k_pools[0][shared_page_b as usize * 4 * n..shared_page_b as usize * 4 * n + n];
        assert_eq!(ka, kb, "shared page KV identical after both writes");
    }

    #[test]
    fn paged_ref_reports_pool_exhaustion_and_seq_overflow() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 53).expect("weights");
        // 1 page only: the 2nd token (crossing the page) errors.
        let mut pr = PagedRef::new(cfg, w, 1, 1, 4);
        pr.prefill(0, &[1, 2, 3, 4]).expect("fits one page");
        assert!(pr.prefill(0, &[5]).is_err(), "page pool exhausted");
    }

    #[test]
    fn share_prefix_on_used_slot_panics_instead_of_leaking() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 54).expect("weights");
        // share_prefix contract: unused slots only. Replacing a used slot's
        // table would silently leak its allocated pages, so it must panic.
        let mut pr = PagedRef::new(cfg, w, 2, 16, 4);
        pr.prefill(0, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("A");
        let pages = pr.slot_table(0).expect("A table").pages().to_vec();
        pr.prefill(1, &[9, 10, 11, 12]).expect("B occupies slot 1");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pr.share_prefix(1, &pages[..1]);
        }));
        assert!(result.is_err(), "used-slot share_prefix must panic");
    }

    #[test]
    fn share_prefix_fresh_slot_resumes_delta_only() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 56).expect("weights");
        // Fresh slot shares one page (4 tokens) and decodes only the delta;
        // the result must equal a full recompute (regression: pos was 0, which
        // recomputed and overwrote the shared pages instead of resuming).
        let mut pr = PagedRef::new(cfg, w.clone(), 2, 16, 4);
        pr.prefill(0, &[1, 2, 3, 4, 5, 6, 7, 8]).expect("A");
        let pages = pr.slot_table(0).expect("A table").pages().to_vec();
        pr.share_prefix(1, &pages[..1]);
        let b = pr.prefill(1, &[9]).expect("B shares A page0 delta-only");
        let mut ref_b = RefModel::new(cfg, w);
        let want = ref_b.forward(&[1, 2, 3, 4, 9]);
        assert_eq!(
            max_abs_diff(&b, &want),
            0.0,
            "delta-only must match recompute"
        );
    }

    #[test]
    fn pagedref_shared_prefix_e2e_via_table_builder() {
        // End-to-end shared-prefix flow on CPU: GpuPagedTableBuilder allocates
        // the shared prefix pages, PagedRef consumes them delta-only, and the
        // result equals a full recompute.
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 81).expect("weights");
        let system = [1u32, 2, 3, 4, 5, 6, 7, 8]; // 2 pages of 4
        let mut a = system.to_vec();
        a.push(100);
        let mut b = system.to_vec();
        b.push(200);
        let a_i32: Vec<i32> = a.iter().map(|&t| t as i32).collect();
        let b_i32: Vec<i32> = b.iter().map(|&t| t as i32).collect();

        // 1) Table builder: B reuses A's two system pages (A's content is
        //    registered first — materialized pages are the only reusable ones).
        let mut bld = GpuPagedTableBuilder::new(16, 4);
        let chain_a = bld.compute_chain(&a_i32);
        let (table_a, r_a) = bld.build_table(&a_i32).expect("A");
        bld.register_chain(&chain_a, &table_a);
        let (table_b, r_b) = bld.build_table(&b_i32).expect("B");
        assert_eq!(r_a, 0);
        assert_eq!(r_b, 8, "B reuses the 8-token system prefix");
        let shared: Vec<u32> = (0..2).map(|i| table_b.get(i).unwrap()).collect();
        assert_eq!(
            shared,
            vec![table_a.get(0).unwrap(), table_a.get(1).unwrap()]
        );

        // 2) PagedRef: A full prefill; B starts from the shared prefix (the
        //    builder's physical page ids coincide with PagedRef's for the first
        //    request — the canonical cache owner) and computes only the delta.
        let mut pr = PagedRef::new(cfg, w.clone(), 2, 16, 4);
        let logits_a = pr.prefill(0, &a).expect("prefill A");
        pr.start_with_shared_prefix(1, &shared, 8);
        let logits_b = pr.prefill(1, &b[8..]).expect("prefill B delta");

        // 3) Correctness: B (delta-only from shared pages) == full recompute.
        let mut ref_b = RefModel::new(cfg, w.clone());
        let ref_logits_b = ref_b.forward(&b);
        assert_eq!(
            max_abs_diff(&logits_b, &ref_logits_b),
            0.0,
            "delta-only shared-prefix decode must equal full recompute"
        );
        // A is unchanged (its system pages were read, not rewritten by B).
        let mut ref_a = RefModel::new(cfg, w);
        let ref_logits_a = ref_a.forward(&a);
        assert_eq!(max_abs_diff(&logits_a, &ref_logits_a), 0.0, "A unchanged");
    }

    #[test]
    #[should_panic(expected = "must cover exactly reused_tokens")]
    fn start_with_shared_prefix_rejects_misaligned_reused_tokens() {
        let cfg = Config::tiny();
        let w = Weights::random(&cfg, 82).expect("weights");
        let mut pr = PagedRef::new(cfg, w, 2, 16, 4);
        // 6 tokens with 2 shared pages (8 token capacity) is misaligned.
        pr.start_with_shared_prefix(0, &[0, 1], 6);
    }

    #[test]
    fn pagedref_mla_shared_prefix_delta_only_e2e() {
        // MLA delta-only flow: B starts from A's shared prefix pages via
        // start_with_shared_prefix and computes only its tail.
        let cfg = Config::mla(128, 2, 4, 1024, 256, 8, 16, 64, 64, 64);
        let w = Weights::random(&cfg, 93).expect("weights");
        let system = [1u32, 2, 3, 4, 5, 6, 7, 8]; // 2 pages of 4
        let mut a = system.to_vec();
        a.push(100);
        let mut b = system.to_vec();
        b.push(200);

        let mut pr = PagedRef::new(cfg, w.clone(), 2, 16, 4);
        let logits_a = pr.prefill(0, &a).expect("A");
        let shared: Vec<u32> = (0..2)
            .map(|i| pr.slot_table(0).expect("A table").get(i).unwrap())
            .collect();
        pr.start_with_shared_prefix(1, &shared, 8);
        let logits_b = pr.prefill(1, &b[8..]).expect("B delta");

        let mut ref_b = RefModel::new(cfg, w.clone());
        assert_eq!(
            max_abs_diff(&logits_b, &ref_b.forward(&b)),
            0.0,
            "MLA delta-only shared-prefix decode must equal full recompute"
        );
        let mut ref_a = RefModel::new(cfg, w);
        assert_eq!(
            max_abs_diff(&logits_a, &ref_a.forward(&a)),
            0.0,
            "A unchanged"
        );
    }

    #[test]
    fn gpu_table_builder_shares_prefix_pages() {
        let system = [1i32, 2, 3, 4, 5, 6, 7, 8]; // 2 pages of 4
        let mut b = GpuPagedTableBuilder::new(16, 4);
        let mk = |tail: i32| {
            let mut t = system.to_vec();
            t.push(tail);
            t
        };
        let c_a = b.compute_chain(&mk(100));
        let (t_a, r_a) = b.build_table(&mk(100)).expect("A");
        b.register_chain(&c_a, &t_a); // A materialized
        let (t_b, r_b) = b.build_table(&mk(200)).expect("B");
        let (t_c, r_c) = b.build_table(&mk(300)).expect("C");
        // First request: everything fresh.
        assert_eq!(r_a, 0);
        assert_eq!(t_a.len(), 3);
        // B and C share A's two system pages, then append their own tails.
        assert_eq!(r_b, 8, "B reuses the 8-token system prefix");
        assert_eq!(r_c, 8, "C reuses the 8-token system prefix");
        assert_eq!(t_b.get(0), t_a.get(0), "B shares page 0");
        assert_eq!(t_b.get(1), t_a.get(1), "B shares page 1");
        assert_eq!(t_c.get(0), t_a.get(0), "C shares page 0");
        assert_eq!(t_c.get(1), t_a.get(1), "C shares page 1");
        assert_ne!(t_a.get(2), t_b.get(2), "tails differ");
        assert_ne!(t_b.get(2), t_c.get(2), "tails differ");
        assert_eq!(
            b.cached_pages(),
            3,
            "only A's materialized pages are cached (2 system + 1 tail)"
        );
    }

    #[test]
    fn gpu_table_builder_pool_exhaustion_rolls_back() {
        // Pool holds 2 pages; the first request needs 3 (system 2 + tail 1).
        let mut b = GpuPagedTableBuilder::new(2, 4);
        assert!(b.build_table(&[1, 2, 3, 4, 5, 6, 7, 8, 9]).is_err());
        // Nothing is cached by builds alone (registration is the materialized
        // marker), so a failed build leaves the cache empty by construction.
        assert_eq!(b.cached_pages(), 0, "failed build must not cache pages");
        // The freed pages serve a 2-page request...
        let (t, r) = b
            .build_table(&[1, 2, 3, 4, 5, 6, 7, 8])
            .expect("2 pages fit after rollback");
        assert_eq!(r, 0, "no unwritten page reused");
        assert_eq!(t.len(), 2);
        assert_eq!(b.cached_pages(), 0, "unmaterialized pages are not reusable");
        // ...and enter the cache only once materialized (register_chain).
        let c = b.compute_chain(&[1, 2, 3, 4, 5, 6, 7, 8]);
        b.register_chain(&c, &t);
        assert_eq!(b.cached_pages(), 2, "registered pages");
        // A later identical request now resolves the full chain.
        let (t2, r2) = b
            .build_table(&[1, 2, 3, 4, 5, 6, 7, 8])
            .expect("cached rebuild");
        assert_eq!(r2, 8, "materialized pages are reusable");
        assert_eq!(t2.pages(), t.pages(), "identical content, identical pages");
    }

    #[test]
    fn gpu_table_builder_disjoint_request_reuses_nothing() {
        let mut b = GpuPagedTableBuilder::new(16, 4);
        let (t_a, _) = b.build_table(&[1, 2, 3, 4]).expect("A");
        let (t_b, r_b) = b.build_table(&[21, 22, 23, 24]).expect("B");
        assert_eq!(r_b, 0);
        assert_ne!(t_a.get(0), t_b.get(0), "disjoint pages");
    }

    #[test]
    fn shared_prefix_tables_read_same_pages() {
        // Two requests share the first page (physical 5) and diverge after.
        let mut a = PagedTable::new();
        a.append(5);
        a.append(9);
        let mut b = PagedTable::new();
        b.append(5); // shared prefix page
        b.append(11);
        assert_eq!(
            a.get(0),
            b.get(0),
            "shared prefix maps to the same physical page"
        );
        assert_ne!(a.get(1), b.get(1), "tail pages differ");
    }

    // -- deterministic stress: random build_table preserves builder invariants --

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

    #[test]
    fn random_build_table_preserves_builder_invariants() {
        const NUM_PAGES: u32 = 32;
        const TPP: usize = 4;
        const VOCAB: usize = 8;
        const MAX_TOKENS: usize = 40;
        const ITERS: usize = 3000;
        // Fixed shared prefixes (mutually distinct page contents) are mixed
        // into the random stream so shared-prefix reuse is exercised
        // structurally instead of only by chance.
        let prefixes: Vec<Vec<i32>> = vec![
            vec![1, 2, 3, 4],
            vec![7, 6, 5, 4, 3, 2, 1],
            vec![0, 2, 4, 6, 1, 3, 5, 7, 2, 4],
            vec![3, 1, 4, 1, 5, 9],
        ];
        let mut b = GpuPagedTableBuilder::new(NUM_PAGES, TPP);
        let mut rng = Xs::new(0x1234_5678);
        let mut successes = 0usize;
        let mut failures = 0usize;
        for _ in 0..ITERS {
            // Random prompt of 0..=40 tokens from the 8-id vocab; a quarter of
            // the time it is a fixed shared prefix + random tail.
            let mut tokens: Vec<i32> = Vec::new();
            let prefix = if rng.below(4) == 0 {
                let p = prefixes[rng.below(prefixes.len())].clone();
                tokens.extend_from_slice(&p);
                Some(p)
            } else {
                None
            };
            let tail_len = rng.below(MAX_TOKENS + 1 - tokens.len());
            for _ in 0..tail_len {
                tokens.push(rng.below(VOCAB) as i32);
            }

            let cached_before = b.cached_pages();
            match b.build_table(&tokens) {
                Ok((table, reused)) => {
                    successes += 1;
                    // Every logical page maps to a physical page inside the pool.
                    assert_eq!(table.len(), tokens.len().div_ceil(TPP), "table length");
                    for logical in 0..table.len() {
                        let physical = table.get(logical).expect("logical page mapped");
                        assert!(
                            physical < NUM_PAGES,
                            "physical page {physical} inside pool of {NUM_PAGES}"
                        );
                    }
                    // Reuse covers a leading page run: page-aligned unless the
                    // whole prompt (possibly ending in a partial page) is reused.
                    assert!(reused <= tokens.len(), "reused tokens within prompt");
                    assert!(
                        reused % TPP == 0 || reused == tokens.len(),
                        "reuse is a leading page run"
                    );
                    // Core invariant: the shared pool is never oversold.
                    assert!(
                        b.cached_pages() <= NUM_PAGES as usize,
                        "pool not oversold (cached {} > {NUM_PAGES})",
                        b.cached_pages()
                    );
                    // Materialize: register the built pages (the engine does
                    // this when prefill completes), so later builds can reuse
                    // them — the production flow under test.
                    let c = b.compute_chain(&tokens);
                    b.register_chain(&c, &table);

                    // Shared consistency: rebuilding the same prompt right away
                    // must reproduce the identical FULL-page prefix — full
                    // pages are cached after the success; a partial last page
                    // is per-request fresh by design (reuse covers full pages
                    // only, so generated KV never shares a partial page).
                    if rng.below(5) == 0 {
                        let (again, r_again) = b.build_table(&tokens).expect("cached rebuild");
                        let full = tokens.len() / TPP;
                        assert_eq!(
                            again.pages()[..full],
                            table.pages()[..full],
                            "identical prompt => identical full pages"
                        );
                        assert_eq!(r_again, full * TPP, "reuse covers the full pages only");
                    }

                    // Shared consistency: same shared prefix, different tail;
                    // when both builds succeed, the leading pages must match.
                    if let Some(p) = prefix
                        && rng.below(3) == 0
                    {
                        let tail2 = rng.below(MAX_TOKENS + 1 - p.len());
                        let mut t2 = p.clone();
                        for _ in 0..tail2 {
                            t2.push(rng.below(VOCAB) as i32);
                        }
                        if let Ok((t2_table, _)) = b.build_table(&t2) {
                            // Full pages of the shared prefix are aliased; a
                            // partial last prefix page is per-request fresh.
                            let shared_pages = p.len() / TPP;
                            for logical in 0..shared_pages {
                                assert_eq!(
                                    table.get(logical),
                                    t2_table.get(logical),
                                    "same prefix {p:?} shares leading page {logical}"
                                );
                            }
                        }
                    }
                }
                Err(_) => {
                    failures += 1;
                    // Rollback: a failed build never leaves new cache entries
                    // behind, and the pool is never oversold. (A failure can
                    // only shrink the cache: when a request overwrites an
                    // already-cached page after its first miss, the documented
                    // rollback removes that overwritten entry too, so the exact
                    // pre-call count is not restorable in general.)
                    assert!(
                        b.cached_pages() <= cached_before,
                        "failed build must not grow the cache ({} -> {})",
                        cached_before,
                        b.cached_pages()
                    );
                    // When the pool was already full, the failure happens at
                    // the first uncached page with no fresh allocation this
                    // call, so rollback restores the exact cache count.
                    if cached_before == NUM_PAGES as usize {
                        assert_eq!(
                            b.cached_pages(),
                            cached_before,
                            "full-pool failure rolls back exactly"
                        );
                    }
                    assert!(b.cached_pages() <= NUM_PAGES as usize, "pool not oversold");
                }
            }
        }
        // Both the success and the failure paths must have been exercised.
        assert!(successes > 0, "stress must exercise the success path");
        assert!(failures > 0, "stress must exercise the failure path");
        assert!(
            b.cached_pages() <= NUM_PAGES as usize,
            "final pool not oversold"
        );
    }

    // ---- plan/register/evict invariants (CPU parity for the engine's
    // admission and eviction paths; the GPU integration tests cover the
    // wiring, these pin the builder contract itself) ----

    /// Reuse covers materialized whole pages only: a partial final page is
    /// never aliased, and only registered content resolves.
    #[test]
    fn plan_reuses_only_materialized_whole_pages() {
        let mut b = GpuPagedTableBuilder::new(16, 4);
        let tokens = [1, 2, 3, 4, 5, 6]; // 1 full page + 2-token partial
        let chain = b.compute_chain(&tokens);

        // Nothing materialized: full fresh compute.
        let p0 = b.plan_with_chain(&tokens, &chain, true).unwrap();
        assert_eq!(p0.reused_pages, 0);
        assert_eq!(p0.reused_tokens, 0);
        assert_eq!(p0.table.len(), 2, "full page + partial page");

        // Register page 0's content only (prefill materialized).
        b.register_chain(&chain[..1], &p0.table);

        // Re-plan: exactly the one materialized full page aliases; the
        // partial page stays per-request fresh.
        let p1 = b.plan_with_chain(&tokens, &chain, true).unwrap();
        assert_eq!(p1.reused_pages, 1, "only the materialized full page");
        assert_eq!(p1.reused_tokens, 4, "one full page worth of tokens");
        assert_eq!(p1.table.get(0), p0.table.get(0), "page 0 aliased");
        assert_ne!(
            p1.table.get(1),
            p0.table.get(1),
            "partial page is per-request fresh"
        );
    }

    /// First writer wins: two materializations of the same content register
    /// once; the loser's duplicate pages stay private (its table still
    /// addresses them directly).
    #[test]
    fn register_chain_first_writer_wins() {
        let mut b = GpuPagedTableBuilder::new(16, 4);
        let tokens = [7i32; 8];
        let chain = b.compute_chain(&tokens);

        let pa = b.plan_with_chain(&tokens, &chain, false).unwrap();
        let pb = b.plan_with_chain(&tokens, &chain, false).unwrap();
        assert_ne!(
            pa.table.get(0),
            pb.table.get(0),
            "fresh plans get distinct pages"
        );

        b.register_chain(&chain, &pa.table);
        b.register_chain(&chain, &pb.table);
        assert_eq!(
            b.page_of(&chain[0]),
            pa.table.get(0),
            "the first registration owns the mapping"
        );
    }

    /// `free_plan_pages` releases exactly the fresh pages and never an
    /// aliased one (the aliased page is shared with its first owner).
    #[test]
    fn free_plan_pages_spares_aliased_pages() {
        let mut b = GpuPagedTableBuilder::new(16, 4);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let chain = b.compute_chain(&tokens);

        let pa = b.plan_with_chain(&tokens, &chain, false).unwrap();
        b.register_chain(&chain, &pa.table);
        let page0 = pa.table.get(0).unwrap();

        let pb = b.plan_with_chain(&tokens, &chain, true).unwrap();
        assert_eq!(pb.reused_pages, 2, "both materialized pages alias");
        b.free_plan_pages(&pb, &pb.table);

        assert_eq!(b.page_of(&chain[0]), Some(page0), "alias survives");
        assert_eq!(b.page_of(&chain[1]), pa.table.get(1), "alias survives");
        // The freed fresh page comes back; the aliased pages do not.
        let recycled = b.plan_with_chain(&tokens, &chain, false).unwrap();
        assert_eq!(recycled.reused_pages, 0);
        for p in recycled.table.pages() {
            assert!(
                !pa.table.pages().contains(p),
                "a freed pool must not hand out aliased pages"
            );
        }
    }

    /// `pad_table` rolls back its appended pages on exhaustion, leaving the
    /// table (and the pool) exactly as before the call.
    #[test]
    fn pad_table_rollback_frees_on_exhaustion() {
        let mut b = GpuPagedTableBuilder::new(6, 4);
        let mut t = {
            let c = b.compute_chain(&[1, 2, 3, 4]);
            let p = b.plan_with_chain(&[1, 2, 3, 4], &c, false).unwrap();
            p.table
        };
        let before = t.len();
        assert!(b.pad_table(&mut t, 10).is_err(), "6-page pool < 10 pages");
        assert_eq!(t.len(), before, "table truncated to pre-pad length");
        // All 5 remaining pages are allocatable again.
        for _ in 0..5 {
            assert!(b.allocator.alloc().is_some(), "rollback freed the pads");
        }
        assert!(b.allocator.alloc().is_none(), "pool is exactly drained");
    }

    /// Eviction unregisters and frees; a plan afterwards full-computes (the
    /// reuse boundary tracks what the cache actually resolves — the
    /// engine derives `r` from this, so a stale registry entry can never
    /// skip unwritten positions).
    #[test]
    fn evicted_content_is_not_reused() {
        let mut b = GpuPagedTableBuilder::new(16, 4);
        let tokens = [1, 2, 3, 4, 5, 6, 7, 8];
        let chain = b.compute_chain(&tokens);
        let p = b.plan_with_chain(&tokens, &chain, false).unwrap();
        b.register_chain(&chain, &p.table);
        assert_eq!(b.plan(&tokens, true).unwrap().reused_pages, 2);

        for h in &chain {
            assert!(b.evict_page(h), "registered hash must evict");
        }
        assert!(b.page_of(&chain[0]).is_none());
        let after = b.plan(&tokens, true).unwrap();
        assert_eq!(after.reused_pages, 0, "evicted content must not be reused");
        assert_eq!(after.reused_tokens, 0);
        // Freed pages are recycled, not double-counted.
        assert_eq!(b.cached_pages(), 0);
    }

    #[test]
    #[should_panic(expected = "double-free")]
    fn page_allocator_rejects_double_free() {
        let mut a = PageAllocator::new(2);
        let p = a.alloc().unwrap();
        a.free(p);
        a.free(p); // ownership already released — must panic
    }

    #[test]
    #[should_panic(expected = "out-of-range")]
    fn page_allocator_rejects_out_of_range() {
        let mut a = PageAllocator::new(2);
        a.free(99);
    }
}
