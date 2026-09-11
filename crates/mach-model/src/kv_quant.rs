//! KV-cache quantization primitives.
//!
//! Stage 1 implements only the full-attention INT8 layout and its CPU oracle.
//! Quantization is symmetric per `(token, kv_head, head_dim block)`: the
//! payload is one `i8` per K/V element and one `f32` scale per head row. This
//! keeps the layout identical for the future contiguous and paged GPU stores;
//! it is intentionally not wired into the f16/f32 runtime yet.

use crate::Error;

/// One-byte INT8 KV storage format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuantFormat {
    F16,
    Int8,
}

impl KvQuantFormat {
    /// Bytes per K/V scalar, excluding scale metadata.
    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::Int8 => 1,
        }
    }
}

/// Dequantize one symmetric INT8 value without overflowing at finite f32
/// extremes. The f64 product is saturated back to the finite f32 range.
fn dequant_value(q: i8, scale: f32) -> f32 {
    let v = q as f64 * scale as f64;
    if v > f32::MAX as f64 {
        f32::MAX
    } else if v < f32::MIN as f64 {
        f32::MIN
    } else {
        v as f32
    }
}

/// Symmetric INT8 KV payload plus one f32 scale per token/head row.
#[derive(Debug, Clone, PartialEq)]
pub struct Int8Kv {
    q: Vec<i8>,
    scales: Vec<f32>,
    heads: usize,
    head_dim: usize,
}

impl Int8Kv {
    /// Create an empty cache for later [`Self::append_rows`] calls.
    pub fn empty(heads: usize, head_dim: usize) -> Result<Self, Error> {
        if heads == 0 || head_dim == 0 {
            return Err(Error::InvalidArgument(
                "INT8 KV requires non-zero heads and head_dim".into(),
            ));
        }
        heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::InvalidArgument("INT8 KV head block overflow".into()))?;
        Ok(Self {
            q: Vec::new(),
            scales: Vec::new(),
            heads,
            head_dim,
        })
    }

    /// Append one or more `[tokens, heads, head_dim]` rows, preserving the
    /// existing per-token/head scales.
    pub fn append_rows(&mut self, values: &[f32]) -> Result<(), Error> {
        if values.is_empty() {
            return Ok(());
        }
        let add = Self::quantize(values, self.heads, self.head_dim)?;
        self.q.extend_from_slice(&add.q);
        self.scales.extend_from_slice(&add.scales);
        Ok(())
    }

    /// Quantize `values` shaped `[tokens, heads, head_dim]` with one symmetric
    /// scale per `(token, head)`. The quantized range is `[-127, 127]`; `-128`
    /// is deliberately unused so dequantization stays exactly symmetric.
    pub fn quantize(values: &[f32], heads: usize, head_dim: usize) -> Result<Self, Error> {
        if heads == 0 || head_dim == 0 {
            return Err(Error::InvalidArgument(
                "INT8 KV requires non-zero heads and head_dim".into(),
            ));
        }
        let block = heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::InvalidArgument("INT8 KV head block overflow".into()))?;
        if !values.len().is_multiple_of(block) {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV values length {} is not a multiple of heads*head_dim {block}",
                values.len()
            )));
        }
        if let Some((i, v)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, v)| !v.is_finite())
        {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV input has non-finite value at index {i}: {v}"
            )));
        }

        let rows = values.len() / block;
        let mut q = Vec::with_capacity(values.len());
        let mut scales = Vec::with_capacity(rows * heads);
        for row in 0..rows {
            for head in 0..heads {
                let start = row * block + head * head_dim;
                let end = start + head_dim;
                let max_abs = values[start..end]
                    .iter()
                    .fold(0.0f32, |m, &v| m.max(v.abs()));
                let scale = if max_abs == 0.0 {
                    1.0
                } else {
                    let s = max_abs / 127.0;
                    if s == 0.0 { max_abs } else { s }
                };
                scales.push(scale);
                for &v in &values[start..end] {
                    let r = (v / scale).round();
                    let clamped = if r >= 127.0 {
                        127i8
                    } else if r <= -127.0 {
                        -127i8
                    } else {
                        r as i8
                    };
                    q.push(clamped);
                }
            }
        }
        Ok(Self {
            q,
            scales,
            heads,
            head_dim,
        })
    }

    pub fn tokens(&self) -> usize {
        if self.heads == 0 || self.head_dim == 0 {
            0
        } else {
            self.q.len() / (self.heads * self.head_dim)
        }
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Packed payload + scale bytes for this K or V tensor.
    pub fn bytes(&self) -> usize {
        self.q.len() + self.scales.len() * std::mem::size_of::<f32>()
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn quantized(&self) -> &[i8] {
        &self.q
    }

    pub fn scale(&self, token: usize, head: usize) -> Option<f32> {
        if token >= self.tokens() || head >= self.heads {
            return None;
        }
        Some(self.scales[token * self.heads + head])
    }

    pub fn head_row(&self, token: usize, head: usize) -> Option<&[i8]> {
        if token >= self.tokens() || head >= self.heads {
            return None;
        }
        let block = self.heads * self.head_dim;
        let start = token * block + head * self.head_dim;
        Some(&self.q[start..start + self.head_dim])
    }

    /// Quantized K dot an f32 query row: `sum(q_i * k_i) * scale`. Q is
    /// intentionally not quantized; attention dequantization multiplies the
    /// completed dot by the single per-head scale.
    pub fn dot_q(&self, q: &[f32], token: usize, head: usize) -> Result<f32, Error> {
        if q.len() != self.head_dim {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV dot query length {} != head_dim {}",
                q.len(),
                self.head_dim
            )));
        }
        if let Some((i, v)) = q.iter().copied().enumerate().find(|(_, v)| !v.is_finite()) {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV dot query has non-finite value at index {i}: {v}"
            )));
        }
        let row = self.head_row(token, head).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "INT8 KV dot index out of range: token={token} head={head}"
            ))
        })?;
        let mut acc = 0.0f64;
        for (&qi, &ki) in q.iter().zip(row) {
            acc += qi as f64 * ki as f64;
        }
        let out = acc * self.scale(token, head).unwrap() as f64;
        if !out.is_finite() {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV dot produced a non-finite value: token={token} head={head} value={out}"
            )));
        }
        let scale = self.scale(token, head).unwrap() as f64;
        // The scale itself is rounded to f32, so a mathematically finite dot
        // exactly at f32::MAX can land a fraction of one quantization step
        // outside the f32 range. Accept that rounding margin; larger overflow
        // is a real out-of-range dot.
        let tol = scale * 0.5;
        if out > f32::MAX as f64 {
            if out - f32::MAX as f64 <= tol {
                return Ok(f32::MAX);
            }
            return Err(Error::InvalidArgument(format!(
                "INT8 KV dot overflow: token={token} head={head} value={out}"
            )));
        }
        if out < f32::MIN as f64 {
            if f32::MIN as f64 - out <= tol {
                return Ok(f32::MIN);
            }
            return Err(Error::InvalidArgument(format!(
                "INT8 KV dot underflow: token={token} head={head} value={out}"
            )));
        }
        Ok(out as f32)
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.q.len());
        let block = self.heads * self.head_dim;
        for row in 0..self.tokens() {
            for head in 0..self.heads {
                let scale = self.scales[row * self.heads + head];
                let start = row * block + head * self.head_dim;
                let end = start + self.head_dim;
                out.extend(self.q[start..end].iter().map(|&v| dequant_value(v, scale)));
            }
        }
        out
    }

    pub fn dequantize_into(&self, out: &mut [f32]) -> Result<(), Error> {
        if out.len() != self.q.len() {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV dequantize output length {} != {}",
                out.len(),
                self.q.len()
            )));
        }
        let block = self.heads * self.head_dim;
        for row in 0..self.tokens() {
            for head in 0..self.heads {
                let scale = self.scales[row * self.heads + head];
                let start = row * block + head * self.head_dim;
                let end = start + self.head_dim;
                for (dst, &v) in out[start..end].iter_mut().zip(&self.q[start..end]) {
                    *dst = dequant_value(v, scale);
                }
            }
        }
        Ok(())
    }
}

/// Single-token full-attention decode oracle for INT8 K/V.
///
/// `q` is `[n_heads, head_dim]`; `k`/`v` are `[tokens, n_kv_heads, head_dim]`
/// and must share heads/head_dim/token count. GQA maps query head `h` to
/// `kv_head = h / (n_heads / n_kv_heads)`, matching the existing GPU kernels.
pub fn attention_decode_int8(
    q: &[f32],
    n_heads: usize,
    k: &Int8Kv,
    v: &Int8Kv,
    softmax_scale: f32,
) -> Result<Vec<f32>, Error> {
    if n_heads == 0 || k.heads == 0 || k.head_dim == 0 {
        return Err(Error::InvalidArgument(
            "INT8 attention requires non-zero n_heads/kv_heads/head_dim".into(),
        ));
    }
    if k.heads != v.heads || k.head_dim != v.head_dim || k.tokens() != v.tokens() {
        return Err(Error::InvalidArgument(format!(
            "INT8 attention K/V shape mismatch: k={}/{}/{}, v={}/{}/{}",
            k.tokens(),
            k.heads,
            k.head_dim,
            v.tokens(),
            v.heads,
            v.head_dim
        )));
    }
    let tokens = k.tokens();
    if tokens == 0 {
        return Err(Error::InvalidArgument(
            "INT8 attention requires at least one token".into(),
        ));
    }
    if !n_heads.is_multiple_of(k.heads) {
        return Err(Error::InvalidArgument(format!(
            "INT8 attention n_heads={n_heads} is not a multiple of kv_heads={}",
            k.heads
        )));
    }
    let q_len = n_heads
        .checked_mul(k.head_dim)
        .ok_or_else(|| Error::InvalidArgument("INT8 attention q size overflow".into()))?;
    if q.len() != q_len {
        return Err(Error::InvalidArgument(format!(
            "INT8 attention q length {} != n_heads*head_dim {q_len}",
            q.len()
        )));
    }
    if let Some((i, value)) = q
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(Error::InvalidArgument(format!(
            "INT8 attention q has non-finite value at index {i}: {value}"
        )));
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(Error::InvalidArgument(format!(
            "INT8 attention softmax_scale must be finite and positive, got {softmax_scale}"
        )));
    }

    let groups = n_heads / k.heads;
    let mut out = vec![0.0f32; q_len];
    for h in 0..n_heads {
        let kv = h / groups;
        let qh = &q[h * k.head_dim..(h + 1) * k.head_dim];
        let mut scores = Vec::with_capacity(tokens);
        for token in 0..tokens {
            scores.push(k.dot_q(qh, token, kv)? * softmax_scale);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = Vec::with_capacity(tokens);
        let mut denom = 0.0f32;
        for &score in &scores {
            let p = (score - max_score).exp();
            if !p.is_finite() {
                return Err(Error::InvalidArgument(format!(
                    "INT8 attention softmax produced non-finite probability for head {h}"
                )));
            }
            probs.push(p);
            denom += p;
        }
        if !denom.is_finite() || denom <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "INT8 attention softmax denominator invalid for head {h}: {denom}"
            )));
        }
        let inv = 1.0f64 / denom as f64;
        for dim in 0..k.head_dim {
            let mut acc = 0.0f64;
            for (token, &p) in probs.iter().enumerate() {
                let vs = v.scale(token, kv).unwrap();
                if !vs.is_finite() {
                    return Err(Error::InvalidArgument(format!(
                        "INT8 attention V scale non-finite for token {token} head {kv}: {vs}"
                    )));
                }
                let value = dequant_value(v.head_row(token, kv).unwrap()[dim], vs) as f64;
                acc += p as f64 * inv * value;
            }
            if !acc.is_finite() {
                return Err(Error::InvalidArgument(format!(
                    "INT8 attention V accumulation non-finite for head {h} dim {dim}: {acc}"
                )));
            }
            out[h * k.head_dim + dim] = if acc > f32::MAX as f64 {
                f32::MAX
            } else if acc < f32::MIN as f64 {
                f32::MIN
            } else {
                acc as f32
            };
        }
    }
    Ok(out)
}

/// Contiguous INT8 KV cache layout:
/// payload `[slots, max_seq, kv_heads, head_dim]`, scales
/// `[slots, max_seq, kv_heads]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContiguousInt8KvLayout {
    slots: usize,
    max_seq: usize,
    heads: usize,
    head_dim: usize,
    payload_len: usize,
    scale_len: usize,
    scale_bytes: usize,
    total_bytes: usize,
}

impl ContiguousInt8KvLayout {
    pub fn new(slots: usize, max_seq: usize, heads: usize, head_dim: usize) -> Result<Self, Error> {
        if slots == 0 || max_seq == 0 || heads == 0 || head_dim == 0 {
            return Err(Error::InvalidArgument(
                "INT8 contiguous KV layout requires non-zero slots/max_seq/heads/head_dim".into(),
            ));
        }
        let head_block = heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::InvalidArgument("INT8 KV head block overflow".into()))?;
        let per_slot = max_seq
            .checked_mul(head_block)
            .ok_or_else(|| Error::InvalidArgument("INT8 KV per-slot payload overflow".into()))?;
        let payload_len = slots
            .checked_mul(per_slot)
            .ok_or_else(|| Error::InvalidArgument("INT8 KV payload overflow".into()))?;
        let scale_len = slots
            .checked_mul(max_seq)
            .and_then(|v| v.checked_mul(heads))
            .ok_or_else(|| Error::InvalidArgument("INT8 KV scale overflow".into()))?;
        let scale_bytes = scale_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::InvalidArgument("INT8 KV scale bytes overflow".into()))?;
        let total_bytes = payload_len
            .checked_add(scale_bytes)
            .ok_or_else(|| Error::InvalidArgument("INT8 KV total bytes overflow".into()))?;
        Ok(Self {
            slots,
            max_seq,
            heads,
            head_dim,
            payload_len,
            scale_len,
            scale_bytes,
            total_bytes,
        })
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn payload_len(&self) -> usize {
        self.payload_len
    }

    pub fn scale_len(&self) -> usize {
        self.scale_len
    }

    pub fn payload_bytes(&self) -> usize {
        self.payload_len
    }

    pub fn scale_bytes(&self) -> usize {
        self.scale_bytes
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn payload_index(&self, slot: usize, pos: usize, head: usize, dim: usize) -> Option<usize> {
        if slot >= self.slots || pos >= self.max_seq || head >= self.heads || dim >= self.head_dim {
            return None;
        }
        ((slot * self.max_seq + pos) * self.heads + head)
            .checked_mul(self.head_dim)
            .and_then(|v| v.checked_add(dim))
    }

    pub fn scale_index(&self, slot: usize, pos: usize, head: usize) -> Option<usize> {
        if slot >= self.slots || pos >= self.max_seq || head >= self.heads {
            return None;
        }
        (slot * self.max_seq + pos)
            .checked_mul(self.heads)
            .and_then(|v| v.checked_add(head))
    }
}

/// Paged INT8 KV pool layout:
/// payload `[pages, tokens_per_page, kv_heads, head_dim]`, scales
/// `[pages, tokens_per_page, kv_heads]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedInt8KvLayout {
    pages: usize,
    tokens_per_page: usize,
    heads: usize,
    head_dim: usize,
    payload_len: usize,
    scale_len: usize,
    scale_bytes: usize,
    total_bytes: usize,
}

impl PagedInt8KvLayout {
    pub fn new(
        pages: usize,
        tokens_per_page: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<Self, Error> {
        if pages == 0 || tokens_per_page == 0 || heads == 0 || head_dim == 0 {
            return Err(Error::InvalidArgument(
                "INT8 paged KV layout requires non-zero pages/tokens_per_page/heads/head_dim"
                    .into(),
            ));
        }
        let head_block = heads
            .checked_mul(head_dim)
            .ok_or_else(|| Error::InvalidArgument("INT8 KV head block overflow".into()))?;
        let per_page = tokens_per_page
            .checked_mul(head_block)
            .ok_or_else(|| Error::InvalidArgument("INT8 KV per-page payload overflow".into()))?;
        let payload_len = pages
            .checked_mul(per_page)
            .ok_or_else(|| Error::InvalidArgument("INT8 KV paged payload overflow".into()))?;
        let scale_len = pages
            .checked_mul(tokens_per_page)
            .and_then(|v| v.checked_mul(heads))
            .ok_or_else(|| Error::InvalidArgument("INT8 KV paged scale overflow".into()))?;
        let scale_bytes = scale_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::InvalidArgument("INT8 KV paged scale bytes overflow".into()))?;
        let total_bytes = payload_len
            .checked_add(scale_bytes)
            .ok_or_else(|| Error::InvalidArgument("INT8 KV paged total bytes overflow".into()))?;
        Ok(Self {
            pages,
            tokens_per_page,
            heads,
            head_dim,
            payload_len,
            scale_len,
            scale_bytes,
            total_bytes,
        })
    }

    pub fn pages(&self) -> usize {
        self.pages
    }

    pub fn tokens_per_page(&self) -> usize {
        self.tokens_per_page
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn payload_len(&self) -> usize {
        self.payload_len
    }

    pub fn scale_len(&self) -> usize {
        self.scale_len
    }

    pub fn payload_bytes(&self) -> usize {
        self.payload_len
    }

    pub fn scale_bytes(&self) -> usize {
        self.scale_bytes
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn payload_index(
        &self,
        page: usize,
        offset: usize,
        head: usize,
        dim: usize,
    ) -> Option<usize> {
        if page >= self.pages
            || offset >= self.tokens_per_page
            || head >= self.heads
            || dim >= self.head_dim
        {
            return None;
        }
        ((page * self.tokens_per_page + offset) * self.heads + head)
            .checked_mul(self.head_dim)
            .and_then(|v| v.checked_add(dim))
    }

    pub fn scale_index(&self, page: usize, offset: usize, head: usize) -> Option<usize> {
        if page >= self.pages || offset >= self.tokens_per_page || head >= self.heads {
            return None;
        }
        (page * self.tokens_per_page + offset)
            .checked_mul(self.heads)
            .and_then(|v| v.checked_add(head))
    }
}

fn checked_layout_len(values: usize, expected: usize, what: &str) -> Result<(), Error> {
    if values != expected {
        return Err(Error::InvalidArgument(format!(
            "INT8 KV {what} length {values} != expected {expected}"
        )));
    }
    Ok(())
}

fn checked_min_len(values: usize, needed: usize, what: &str) -> Result<(), Error> {
    if values < needed {
        return Err(Error::InvalidArgument(format!(
            "INT8 KV {what} length {values} < needed {needed}"
        )));
    }
    Ok(())
}

/// Scatter one contiguous `Int8Kv` into a paged payload+scale pool using
/// `page_table[logical_page] = physical_page`. Unused page slots are left
/// untouched.
pub fn scatter_paged_int8(
    src: &Int8Kv,
    layout: &PagedInt8KvLayout,
    page_table: &[usize],
    payload: &mut [i8],
    scales: &mut [f32],
) -> Result<(), Error> {
    if src.heads != layout.heads || src.head_dim != layout.head_dim {
        return Err(Error::InvalidArgument(format!(
            "INT8 KV scatter shape mismatch: src heads/head_dim={}/{} layout={}/{}",
            src.heads, src.head_dim, layout.heads, layout.head_dim
        )));
    }
    checked_layout_len(payload.len(), layout.payload_len, "payload")?;
    checked_layout_len(scales.len(), layout.scale_len, "scale")?;
    let tokens = src.tokens();
    let logical_pages = tokens.div_ceil(layout.tokens_per_page);
    checked_min_len(page_table.len(), logical_pages, "page table")?;
    for &page in &page_table[..logical_pages] {
        if page >= layout.pages {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV page table entry {page} >= pages {}",
                layout.pages
            )));
        }
    }

    for token in 0..tokens {
        let logical = token / layout.tokens_per_page;
        let off = token % layout.tokens_per_page;
        let page = page_table[logical];
        for head in 0..layout.heads {
            if let Some(idx) = layout.payload_index(page, off, head, 0) {
                let src_row = src.head_row(token, head).unwrap();
                payload[idx..idx + layout.head_dim].copy_from_slice(src_row);
            }
            if let Some(idx) = layout.scale_index(page, off, head) {
                scales[idx] = src.scale(token, head).unwrap();
            }
        }
    }
    Ok(())
}

/// Gather `tokens` from a paged INT8 pool back into the contiguous oracle
/// representation. This is the CPU reference for paged decode/store parity.
pub fn gather_paged_int8(
    layout: &PagedInt8KvLayout,
    page_table: &[usize],
    payload: &[i8],
    scales: &[f32],
    tokens: usize,
) -> Result<Int8Kv, Error> {
    checked_layout_len(payload.len(), layout.payload_len, "payload")?;
    checked_layout_len(scales.len(), layout.scale_len, "scale")?;
    let logical_pages = tokens.div_ceil(layout.tokens_per_page);
    checked_min_len(page_table.len(), logical_pages, "page table")?;
    for &page in &page_table[..logical_pages] {
        if page >= layout.pages {
            return Err(Error::InvalidArgument(format!(
                "INT8 KV page table entry {page} >= pages {}",
                layout.pages
            )));
        }
    }

    let mut q = Vec::with_capacity(tokens * layout.heads * layout.head_dim);
    let mut out_scales = Vec::with_capacity(tokens * layout.heads);
    for token in 0..tokens {
        let logical = token / layout.tokens_per_page;
        let off = token % layout.tokens_per_page;
        let page = page_table[logical];
        for head in 0..layout.heads {
            let idx = layout.payload_index(page, off, head, 0).unwrap();
            q.extend_from_slice(&payload[idx..idx + layout.head_dim]);
            let scale = scales[layout.scale_index(page, off, head).unwrap()];
            if !scale.is_finite() {
                return Err(Error::InvalidArgument(format!(
                    "INT8 paged KV has non-finite scale at token {token} head {head}: {scale}"
                )));
            }
            out_scales.push(scale);
        }
    }
    Ok(Int8Kv {
        q,
        scales: out_scales,
        heads: layout.heads,
        head_dim: layout.head_dim,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sat_f64_to_f32(v: f64) -> f32 {
        if v > f32::MAX as f64 {
            f32::MAX
        } else if v < f32::MIN as f64 {
            f32::MIN
        } else {
            v as f32
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut max = 0.0f32;
        for (x, y) in a.iter().zip(b) {
            let d = (x - y).abs();
            assert!(d.is_finite(), "non-finite diff: {x} vs {y}");
            max = max.max(d);
        }
        max
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u32 << 31) as f32) * 2.0 - 1.0
    }

    fn attention_f32_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        kv_heads: usize,
        head_dim: usize,
        softmax_scale: f32,
        head_map: &[usize],
    ) -> Vec<f32> {
        let tokens = k.len() / (kv_heads * head_dim);
        let n_heads = head_map.len();
        let mut out = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let kv = head_map[h];
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::with_capacity(tokens);
            for token in 0..tokens {
                let krow =
                    &k[(token * kv_heads + kv) * head_dim..(token * kv_heads + kv + 1) * head_dim];
                scores.push(qh.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * softmax_scale);
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let probs: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let denom: f32 = probs.iter().sum();
            for dim in 0..head_dim {
                let mut acc = 0.0f32;
                for token in 0..tokens {
                    let vrow = &v[(token * kv_heads + kv) * head_dim
                        ..(token * kv_heads + kv + 1) * head_dim];
                    acc += probs[token] / denom * vrow[dim];
                }
                out[h * head_dim + dim] = acc;
            }
        }
        out
    }

    /// CPU transcription of the contiguous INT8 kernel's double online-softmax
    /// algorithm. It intentionally does not call `attention_decode_int8`.
    fn simulated_online_int8(
        q: &[f32],
        n_heads: usize,
        k: &Int8Kv,
        v: &Int8Kv,
        softmax_scale: f32,
    ) -> Vec<f32> {
        let tokens = k.tokens();
        let head_dim = k.head_dim();
        let groups = n_heads / k.heads();
        let lanes = 256 / head_dim;
        let mut out = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            let kv = h / groups;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            for d in 0..head_dim {
                let mut partials = Vec::with_capacity(lanes);
                for c in 0..lanes {
                    let mut m = -1.0e300f64;
                    let mut l = 0.0f64;
                    let mut acc = 0.0f64;
                    for p in (c..tokens).step_by(lanes) {
                        let krow = k.head_row(p, kv).unwrap();
                        let mut dot = 0.0f64;
                        for j in 0..head_dim {
                            dot += qh[j] as f64 * krow[j] as f64;
                        }
                        let sc = dot * k.scale(p, kv).unwrap() as f64 * softmax_scale as f64;
                        let vv =
                            v.head_row(p, kv).unwrap()[d] as f64 * v.scale(p, kv).unwrap() as f64;
                        let mnew = m.max(sc);
                        let alpha = (m - mnew).exp();
                        let beta = (sc - mnew).exp();
                        l = l * alpha + beta;
                        acc = acc * alpha + beta * vv;
                        m = mnew;
                    }
                    partials.push((m, l, acc));
                }
                let mut m = -1.0e300f64;
                let mut l = 0.0f64;
                let mut acc = 0.0f64;
                for (mi, li, ai) in partials {
                    let mnew = m.max(mi);
                    let alpha = (m - mnew).exp();
                    let beta = (mi - mnew).exp();
                    l = l * alpha + li * beta;
                    acc = acc * alpha + ai * beta;
                    m = mnew;
                }
                out[h * head_dim + d] = sat_f64_to_f32(acc / l);
            }
        }
        out
    }

    /// CPU transcription of the paged INT8 kernel with explicit
    /// `table_offsets[sequence]` addressing.
    #[allow(clippy::too_many_arguments)] // Mirrors the kernel signature.
    fn simulated_paged_online_int8(
        q: &[f32],
        n_heads: usize,
        kv_heads: usize,
        dim: usize,
        tokens: usize,
        payload_k: &[i8],
        scales_k: &[f32],
        payload_v: &[i8],
        scales_v: &[f32],
        layout: &PagedInt8KvLayout,
        page_table: &[usize],
        table_offsets: &[usize],
        seq: usize,
        softmax_scale: f32,
    ) -> Vec<f32> {
        let groups = n_heads / kv_heads;
        let lanes = 256 / dim;
        let mut out = vec![0.0f32; n_heads * dim];
        for h in 0..n_heads {
            let kv = h / groups;
            let qh = &q[h * dim..(h + 1) * dim];
            for d in 0..dim {
                let mut partials = Vec::with_capacity(lanes);
                for c in 0..lanes {
                    let mut m = -1.0e300f64;
                    let mut l = 0.0f64;
                    let mut acc = 0.0f64;
                    for p in (c..tokens).step_by(lanes) {
                        let logical = p / layout.tokens_per_page();
                        let off = p % layout.tokens_per_page();
                        let page = page_table[table_offsets[seq] + logical];
                        let pi = layout.payload_index(page, off, kv, 0).unwrap();
                        let si = layout.scale_index(page, off, kv).unwrap();
                        let mut dot = 0.0f64;
                        for j in 0..dim {
                            dot += qh[j] as f64 * payload_k[pi + j] as f64;
                        }
                        let sc = dot * scales_k[si] as f64 * softmax_scale as f64;
                        let vv = payload_v[pi + d] as f64 * scales_v[si] as f64;
                        let mnew = m.max(sc);
                        let alpha = (m - mnew).exp();
                        let beta = (sc - mnew).exp();
                        l = l * alpha + beta;
                        acc = acc * alpha + beta * vv;
                        m = mnew;
                    }
                    partials.push((m, l, acc));
                }
                let mut m = -1.0e300f64;
                let mut l = 0.0f64;
                let mut acc = 0.0f64;
                for (mi, li, ai) in partials {
                    let mnew = m.max(mi);
                    let alpha = (m - mnew).exp();
                    let beta = (mi - mnew).exp();
                    l = l * alpha + li * beta;
                    acc = acc * alpha + ai * beta;
                    m = mnew;
                }
                out[h * dim + d] = sat_f64_to_f32(acc / l);
            }
        }
        out
    }

    #[test]
    fn simulated_online_attention_matches_full_oracle() {
        for dim in [8usize, 16] {
            let (n_heads, kv_heads, tokens) = (6usize, 2usize, 260usize);
            let mut seed = 41u64 + dim as u64;
            let q: Vec<f32> = (0..n_heads * dim).map(|_| lcg(&mut seed)).collect();
            let k32: Vec<f32> = (0..tokens * kv_heads * dim)
                .map(|_| lcg(&mut seed) * 2.0)
                .collect();
            let v32: Vec<f32> = (0..tokens * kv_heads * dim)
                .map(|_| lcg(&mut seed) * 2.0)
                .collect();
            let k = Int8Kv::quantize(&k32, kv_heads, dim).unwrap();
            let v = Int8Kv::quantize(&v32, kv_heads, dim).unwrap();
            let scale = 1.0 / (dim as f32).sqrt();
            let got = simulated_online_int8(&q, n_heads, &k, &v, scale);
            let want = attention_decode_int8(&q, n_heads, &k, &v, scale).unwrap();
            let diff = max_abs_diff(&got, &want);
            assert!(diff < 1e-5, "dim {dim}: online-vs-full diff {diff}");
        }
    }

    #[test]
    fn simulated_online_attention_saturates_finite_extremes() {
        let q = vec![0.0f32; 8];
        let k = Int8Kv::quantize(&[0.0f32; 8], 1, 8).unwrap();
        let v = Int8Kv::quantize(&[f32::MAX; 8], 1, 8).unwrap();
        let got = simulated_online_int8(&q, 1, &k, &v, 1.0);
        assert!(got.iter().all(|x| x.is_finite()));
        assert!(got.iter().all(|x| *x == f32::MAX));
    }

    #[test]
    fn paged_gather_rejects_non_finite_scale() {
        let layout = PagedInt8KvLayout::new(2, 2, 1, 2).unwrap();
        let payload = vec![0i8; layout.payload_len()];
        let mut scales = vec![1.0f32; layout.scale_len()];
        scales[layout.scale_index(1, 0, 0).unwrap()] = f32::INFINITY;
        assert!(
            gather_paged_int8(&layout, &[1], &payload, &scales, 1).is_err(),
            "non-finite paged scale must be rejected"
        );
        scales[layout.scale_index(1, 0, 0).unwrap()] = f32::NAN;
        assert!(
            gather_paged_int8(&layout, &[1], &payload, &scales, 1).is_err(),
            "NaN paged scale must be rejected"
        );
    }

    #[test]
    fn simulated_paged_online_attention_matches_full_oracle() {
        let (pages, tpp, kv_heads, n_heads, dim, tokens) =
            (10usize, 64usize, 2usize, 4usize, 8usize, 260usize);
        let layout = PagedInt8KvLayout::new(pages, tpp, kv_heads, dim).unwrap();
        let page_table = [0usize, 1, 2, 3, 4, 5, 1, 2, 3, 4];
        let table_offsets = [0usize, 5];
        let mut seed = 43u64;
        let q: Vec<f32> = (0..n_heads * dim).map(|_| lcg(&mut seed)).collect();
        let k0: Vec<f32> = (0..tokens * kv_heads * dim)
            .map(|_| lcg(&mut seed) * 2.0)
            .collect();
        let v0: Vec<f32> = (0..tokens * kv_heads * dim)
            .map(|_| lcg(&mut seed) * 2.0)
            .collect();
        let k1: Vec<f32> = (0..tokens * kv_heads * dim)
            .map(|_| lcg(&mut seed) * 2.0)
            .collect();
        let v1: Vec<f32> = (0..tokens * kv_heads * dim)
            .map(|_| lcg(&mut seed) * 2.0)
            .collect();
        let kq0 = Int8Kv::quantize(&k0, kv_heads, dim).unwrap();
        let vq0 = Int8Kv::quantize(&v0, kv_heads, dim).unwrap();
        let kq1 = Int8Kv::quantize(&k1, kv_heads, dim).unwrap();
        let vq1 = Int8Kv::quantize(&v1, kv_heads, dim).unwrap();
        let mut payload_k = vec![0i8; layout.payload_len()];
        let mut scales_k = vec![0.0f32; layout.scale_len()];
        let mut payload_v = vec![0i8; layout.payload_len()];
        let mut scales_v = vec![0.0f32; layout.scale_len()];
        scatter_paged_int8(
            &kq0,
            &layout,
            &page_table[..table_offsets[1]],
            &mut payload_k,
            &mut scales_k,
        )
        .unwrap();
        scatter_paged_int8(
            &vq0,
            &layout,
            &page_table[..table_offsets[1]],
            &mut payload_v,
            &mut scales_v,
        )
        .unwrap();
        scatter_paged_int8(
            &kq1,
            &layout,
            &page_table[table_offsets[1]..],
            &mut payload_k,
            &mut scales_k,
        )
        .unwrap();
        scatter_paged_int8(
            &vq1,
            &layout,
            &page_table[table_offsets[1]..],
            &mut payload_v,
            &mut scales_v,
        )
        .unwrap();
        let scale = 1.0 / (dim as f32).sqrt();
        let got = simulated_paged_online_int8(
            &q,
            n_heads,
            kv_heads,
            dim,
            tokens,
            &payload_k,
            &scales_k,
            &payload_v,
            &scales_v,
            &layout,
            &page_table,
            &table_offsets,
            1,
            scale,
        );
        let want = attention_decode_int8(&q, n_heads, &kq1, &vq1, scale).unwrap();
        let diff = max_abs_diff(&got, &want);
        assert!(diff < 1e-5, "paged online-vs-full diff {diff}");
    }

    #[test]
    fn attention_decode_matches_f32_reference() {
        let (n_heads, kv_heads, dim, tokens) = (6usize, 2usize, 32usize, 17usize);
        let mut seed = 29u64;
        let q: Vec<f32> = (0..n_heads * dim).map(|_| lcg(&mut seed)).collect();
        let k: Vec<f32> = (0..tokens * kv_heads * dim)
            .map(|_| lcg(&mut seed) * 2.0)
            .collect();
        let v: Vec<f32> = (0..tokens * kv_heads * dim)
            .map(|_| lcg(&mut seed) * 2.0)
            .collect();
        let kq = Int8Kv::quantize(&k, kv_heads, dim).unwrap();
        let vq = Int8Kv::quantize(&v, kv_heads, dim).unwrap();
        let scale = 1.0 / (dim as f32).sqrt();
        let got = attention_decode_int8(&q, n_heads, &kq, &vq, scale).unwrap();
        let want = attention_f32_reference(&q, &k, &v, kv_heads, dim, scale, &[0, 0, 0, 1, 1, 1]);
        assert_eq!(got.len(), want.len());
        let mut max_err = 0.0f32;
        for (a, b) in got.iter().zip(&want) {
            assert!(
                a.is_finite() && b.is_finite(),
                "non-finite attention output"
            );
            let diff = (a - b).abs();
            assert!(diff.is_finite(), "non-finite attention diff");
            max_err = max_err.max(diff);
        }
        assert!(max_err < 0.02, "INT8 attention max error {max_err}");
    }

    #[test]
    fn attention_decode_rejects_bad_shapes_and_non_finite_query() {
        let k = Int8Kv::quantize(&[0.0f32; 8], 2, 4).unwrap();
        let v = Int8Kv::quantize(&[0.0f32; 8], 2, 4).unwrap();
        assert!(attention_decode_int8(&[0.0f32; 8], 3, &k, &v, 0.5).is_err());
        let mut bad_q = [0.0f32; 8];
        bad_q[1] = f32::NAN;
        assert!(attention_decode_int8(&bad_q, 2, &k, &v, 0.5).is_err());
        assert!(attention_decode_int8(&[0.0f32; 8], 2, &k, &v, 0.0).is_err());
    }

    #[test]
    fn attention_v_extreme_scale_stays_finite() {
        let k = Int8Kv::quantize(&[0.0f32], 1, 1).unwrap();
        let v = Int8Kv::quantize(&[f32::MAX], 1, 1).unwrap();
        let out = attention_decode_int8(&[0.0f32], 1, &k, &v, 1.0).unwrap();
        assert!(out[0].is_finite());
        assert_eq!(out[0], f32::MAX);
    }

    #[test]
    fn attention_v_scale_is_applied_per_token() {
        let k = Int8Kv::quantize(&[0.0f32; 3], 1, 1).unwrap();
        let v = Int8Kv::quantize(&[1.0f32, 2.0, 3.0], 1, 1).unwrap();
        let out = attention_decode_int8(&[0.0f32], 1, &k, &v, 1.0).unwrap();
        assert!(
            (out[0] - 2.0).abs() < 1e-6,
            "uniform V average got {}",
            out[0]
        );
    }

    #[test]
    fn layout_offsets_and_bytes_match_contract() {
        let contiguous = ContiguousInt8KvLayout::new(3, 5, 2, 4).unwrap();
        assert_eq!(contiguous.slots(), 3);
        assert_eq!(contiguous.max_seq(), 5);
        assert_eq!(contiguous.heads(), 2);
        assert_eq!(contiguous.head_dim(), 4);
        assert_eq!(contiguous.payload_len(), 3 * 5 * 2 * 4);
        assert_eq!(contiguous.scale_len(), 3 * 5 * 2);
        assert_eq!(contiguous.payload_bytes(), 120);
        assert_eq!(contiguous.scale_bytes(), 30 * 4);
        assert_eq!(contiguous.total_bytes(), 120 + 30 * 4);
        assert_eq!(contiguous.payload_index(1, 2, 0, 3), Some(59));
        assert_eq!(contiguous.payload_index(1, 2, 1, 0), Some(60));
        assert_eq!(contiguous.payload_index(2, 4, 1, 3), Some(119));
        assert_eq!(contiguous.scale_index(1, 2, 1), Some(15));
        assert_eq!(contiguous.scale_index(2, 4, 1), Some(29));
        assert_eq!(contiguous.payload_index(3, 0, 0, 0), None);
        assert_eq!(contiguous.scale_index(0, 5, 0), None);

        let paged = PagedInt8KvLayout::new(3, 5, 2, 4).unwrap();
        assert_eq!(paged.pages(), 3);
        assert_eq!(paged.tokens_per_page(), 5);
        assert_eq!(paged.heads(), 2);
        assert_eq!(paged.head_dim(), 4);
        assert_eq!(paged.payload_len(), 120);
        assert_eq!(paged.scale_len(), 30);
        assert_eq!(paged.total_bytes(), 120 + 30 * 4);
        assert_eq!(paged.payload_index(2, 1, 0, 3), Some(91));
        assert_eq!(paged.payload_index(2, 1, 1, 0), Some(92));
        assert_eq!(paged.payload_index(2, 4, 1, 3), Some(119));
        assert_eq!(paged.scale_index(2, 1, 1), Some(23));
        assert_eq!(paged.scale_index(2, 4, 1), Some(29));
        assert_eq!(paged.payload_index(0, 5, 0, 0), None);
    }

    #[test]
    fn layout_rejects_scale_byte_overflow() {
        assert!(
            ContiguousInt8KvLayout::new(usize::MAX / 2, 2, 1, 1).is_err(),
            "scale byte overflow must be rejected at construction"
        );
        assert!(
            PagedInt8KvLayout::new(usize::MAX / 2, 2, 1, 1).is_err(),
            "paged scale byte overflow must be rejected at construction"
        );
    }

    #[test]
    fn paged_scatter_gather_roundtrips_payload_and_scales() {
        let mut seed = 23u64;
        let values: Vec<f32> = (0..7 * 2 * 4).map(|_| lcg(&mut seed) * 4.0).collect();
        let src = Int8Kv::quantize(&values, 2, 4).unwrap();
        let layout = PagedInt8KvLayout::new(3, 4, 2, 4).unwrap();
        // Existing runtime tables are padded with unused tail entries.
        let page_table = [2usize, 0usize, 99usize];
        let mut payload = vec![0i8; layout.payload_len()];
        let mut scales = vec![0.0f32; layout.scale_len()];

        scatter_paged_int8(&src, &layout, &page_table, &mut payload, &mut scales).unwrap();
        let gathered = gather_paged_int8(&layout, &page_table, &payload, &scales, 7).unwrap();
        assert_eq!(gathered.quantized(), src.quantized());
        assert_eq!(gathered.scales(), src.scales());
        assert_eq!(gathered.tokens(), 7);
    }

    #[test]
    fn paged_scatter_rejects_bad_page_table() {
        let src = Int8Kv::quantize(&[0.0f32; 4], 1, 4).unwrap();
        let layout = PagedInt8KvLayout::new(1, 4, 1, 4).unwrap();
        let mut payload = vec![0i8; layout.payload_len()];
        let mut scales = vec![0.0f32; layout.scale_len()];
        assert!(
            scatter_paged_int8(&src, &layout, &[1], &mut payload, &mut scales).is_err(),
            "physical page must be in range"
        );
        assert!(
            scatter_paged_int8(&src, &layout, &[], &mut payload, &mut scales).is_err(),
            "one logical page is required"
        );
    }

    #[test]
    fn empty_rejects_head_block_overflow() {
        assert!(Int8Kv::empty(usize::MAX, 2).is_err());
    }

    #[test]
    fn append_rows_matches_one_shot_quantization() {
        let mut seed = 31u64;
        let first: Vec<f32> = (0..3 * 2 * 4).map(|_| lcg(&mut seed) * 2.0).collect();
        let second: Vec<f32> = (0..2 * 2 * 4).map(|_| lcg(&mut seed) * 2.0).collect();
        let mut appended = Int8Kv::empty(2, 4).unwrap();
        appended.append_rows(&first).unwrap();
        appended.append_rows(&second).unwrap();
        let mut all = first;
        all.extend_from_slice(&second);
        let one_shot = Int8Kv::quantize(&all, 2, 4).unwrap();
        assert_eq!(appended.quantized(), one_shot.quantized());
        assert_eq!(appended.scales(), one_shot.scales());
    }

    #[test]
    fn int8_roundtrip_error_is_bounded_by_half_scale() {
        let mut seed = 7u64;
        let values: Vec<f32> = (0..3 * 4 * 256).map(|_| lcg(&mut seed) * 8.0).collect();
        let kv = Int8Kv::quantize(&values, 4, 256).unwrap();
        let out = kv.dequantize();
        assert_eq!(out.len(), values.len());
        for token in 0..3 {
            for head in 0..4 {
                let scale = kv.scale(token, head).unwrap();
                let block = 4 * 256;
                let start = token * block + head * 256;
                for i in start..start + 256 {
                    assert!(
                        (out[i] - values[i]).abs() <= scale * 0.500_01,
                        "token={token} head={head} i={i} err={} scale={scale}",
                        (out[i] - values[i]).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn zero_head_roundtrips_exactly_with_finite_scale() {
        let mut values = vec![0.0f32; 2 * 4 * 256];
        values[256] = 1.0;
        let kv = Int8Kv::quantize(&values, 4, 256).unwrap();
        assert!(kv.scales().iter().all(|x| x.is_finite()));
        assert_eq!(kv.scale(0, 0), Some(1.0));
        let out = kv.dequantize();
        assert_eq!(out[0..256], values[0..256]);
    }

    #[test]
    fn subnormal_input_does_not_poison_scale() {
        let tiny = f32::from_bits(1);
        let values = [tiny, tiny * 2.0, -tiny, 0.0];
        let kv = Int8Kv::quantize(&values, 1, 4).unwrap();
        let out = kv.dequantize();
        assert!(kv.scales().iter().all(|x| x.is_finite()));
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(kv.quantized().iter().any(|&q| q != 0));
        for (a, b) in values.iter().zip(&out) {
            assert!((a - b).abs() <= tiny);
        }
    }

    #[test]
    fn dot_rejects_non_finite_query_and_extreme_overflow() {
        let kv = Int8Kv::quantize(&[f32::MAX, 1.0], 1, 2).unwrap();
        assert!(kv.dot_q(&[1.0, f32::NAN], 0, 0).is_err());
        assert!(kv.dot_q(&[2.0, 0.0], 0, 0).is_err());
        assert!(kv.dot_q(&[1.0, 0.0], 0, 0).is_ok());
    }

    #[test]
    fn finite_extreme_roundtrip_does_not_overflow() {
        let values = [f32::MAX, f32::MIN, 0.0, 1.0];
        let kv = Int8Kv::quantize(&values, 1, 4).unwrap();
        let out = kv.dequantize();
        assert!(out.iter().all(|x| x.is_finite()));
        assert_eq!(out[0], f32::MAX);
        assert_eq!(out[1], f32::MIN);
    }

    #[test]
    fn non_finite_input_is_rejected() {
        let values = [0.0f32, f32::NAN];
        assert!(Int8Kv::quantize(&values, 1, 2).is_err());
        let values = [0.0f32, f32::INFINITY];
        assert!(Int8Kv::quantize(&values, 1, 2).is_err());
    }

    #[test]
    fn layout_counts_and_bytes_match_contract() {
        let values = vec![0.25f32; 5 * 4 * 256];
        let kv = Int8Kv::quantize(&values, 4, 256).unwrap();
        assert_eq!(kv.tokens(), 5);
        assert_eq!(kv.heads(), 4);
        assert_eq!(kv.head_dim(), 256);
        assert_eq!(kv.scales().len(), 5 * 4);
        assert_eq!(kv.quantized().len(), values.len());
        assert_eq!(kv.bytes(), values.len() + 5 * 4 * 4);
        assert!(kv.bytes() < values.len() * std::mem::size_of::<f32>() / 2);
    }

    #[test]
    fn qk_dot_error_tracks_quantization_error() {
        let mut seed = 11u64;
        let head_dim = 256usize;
        let k: Vec<f32> = (0..head_dim).map(|_| lcg(&mut seed) * 3.0).collect();
        let q: Vec<f32> = (0..head_dim).map(|_| lcg(&mut seed)).collect();
        let kv = Int8Kv::quantize(&k, 1, head_dim).unwrap();
        let exact: f32 = q.iter().zip(&k).map(|(a, b)| a * b).sum();
        let got = kv.dot_q(&q, 0, 0).unwrap();
        let scale = kv.scale(0, 0).unwrap();
        let bound = q.iter().map(|x| x.abs()).sum::<f32>() * scale * 0.500_01;
        assert!(
            (got - exact).abs() <= bound,
            "dot err {} bound {bound}",
            (got - exact).abs()
        );
    }

    #[test]
    fn qk_softmax_stays_close_with_int8_k() {
        let mut seed = 19u64;
        let tokens = 64usize;
        let head_dim = 256usize;
        let q: Vec<f32> = (0..head_dim).map(|_| lcg(&mut seed)).collect();
        let k: Vec<f32> = (0..tokens * head_dim)
            .map(|_| lcg(&mut seed) * 2.0)
            .collect();
        let kv = Int8Kv::quantize(&k, 1, head_dim).unwrap();

        let mut exact = vec![0.0f32; tokens];
        let mut quant = vec![0.0f32; tokens];
        for t in 0..tokens {
            exact[t] = q
                .iter()
                .zip(&k[t * head_dim..(t + 1) * head_dim])
                .map(|(a, b)| a * b)
                .sum();
            quant[t] = kv.dot_q(&q, t, 0).unwrap();
        }
        let max = exact.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exact_sum: f32 = exact.iter().map(|x| (x - max).exp()).sum();
        let quant_max = quant.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let quant_sum: f32 = quant.iter().map(|x| (x - quant_max).exp()).sum();
        let mut max_prob_diff = 0.0f32;
        for t in 0..tokens {
            let pe = (exact[t] - max).exp() / exact_sum;
            let pq = (quant[t] - quant_max).exp() / quant_sum;
            assert!(pe.is_finite() && pq.is_finite());
            max_prob_diff = max_prob_diff.max((pe - pq).abs());
        }
        assert!(
            max_prob_diff < 0.02,
            "max softmax prob diff {max_prob_diff}"
        );
    }
}
