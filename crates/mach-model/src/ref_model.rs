//! CPU f32 reference implementation of the slice transformer.
//!
//! Used as the golden reference for the GPU path: identical math, naive
//! loops, no SIMD. The GPU tests compare against this within tolerance.

use crate::{Config, LayerWeights, Weights};

/// CPU reference model with an explicit KV cache.
#[derive(Debug)]
pub struct RefModel {
    cfg: Config,
    w: Weights,
    /// Per layer: (k, v), each `[n_kv_heads, max_seq_len, head_dim]`.
    kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// Per layer MLA (kv_lora_rank > 0): expanded per-head k `[n_heads,
    /// max_seq_len, qk_nope+qk_rope]` and v `[n_heads, max_seq_len, v_head_dim]`.
    mla_kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// Number of tokens stored so far.
    pos: usize,
    /// Hidden state of the most recently processed token (after the last
    /// layer, before the final norm + lm_head). The anchor checkpoint data for
    /// agentic state reuse ([`Self::save_anchor`]).
    last_hidden: Vec<f32>,
}

impl RefModel {
    /// Number of tokens processed so far (next position).
    #[must_use]
    pub const fn pos(&self) -> usize {
        self.pos
    }

    /// Builds a reference model from weights.
    #[must_use]
    pub fn new(cfg: Config, w: Weights) -> Self {
        let kv_slots = cfg.n_kv_heads * cfg.max_seq_len * cfg.head_dim;
        let kv = (0..cfg.n_layers)
            .map(|_| (vec![0.0; kv_slots], vec![0.0; kv_slots]))
            .collect();
        let mla_kv = if cfg.kv_lora_rank > 0 {
            let hd = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
            let k_slots = cfg.n_heads * cfg.max_seq_len * hd;
            let v_slots = cfg.n_heads * cfg.max_seq_len * cfg.v_head_dim;
            (0..cfg.n_layers)
                .map(|_| (vec![0.0; k_slots], vec![0.0; v_slots]))
                .collect()
        } else {
            Vec::new()
        };
        Self {
            cfg,
            w,
            kv,
            mla_kv,
            pos: 0,
            last_hidden: Vec::new(),
        }
    }

    /// Processes `tokens` one by one and returns logits of the final token.
    pub fn forward(&mut self, tokens: &[u32]) -> Vec<f32> {
        let mut logits = Vec::new();
        for &t in tokens {
            logits = self.decode_step(t);
        }
        logits
    }

    /// One decode step: `token` at position `self.pos`, returns `[vocab]` logits.
    pub fn decode_step(&mut self, token: u32) -> Vec<f32> {
        let d = self.cfg.d_model;
        let x0 = self.w.tok_emb[token as usize * d..(token as usize + 1) * d].to_vec();
        self.forward_from(x0)
    }

    /// Runs one transformer position starting from an input hidden state
    /// (a token embedding, or a restored anchor's hidden), storing KV at the
    /// current position and returning `[vocab]` logits.
    fn forward_from(&mut self, mut x: Vec<f32>) -> Vec<f32> {
        let cfg = self.cfg;
        let d = cfg.d_model;
        let pos = self.pos;
        assert!(
            pos < cfg.max_seq_len,
            "sequence length exceeded max_seq_len"
        );

        for (li, lw) in self.w.layers.iter().enumerate() {
            let xn = rms_norm(&x, &lw.rms_attn, cfg.rms_eps);
            if cfg.kv_lora_rank > 0 {
                // MLA (DeepSeek-V2 style): low-rank Q + compressed KV.
                let (kc, vc) = &mut self.mla_kv[li];
                decode_step_mla(&mut x, &xn, lw, kc, vc, pos, cfg);
            } else {
                let mut q = matvec_t(&xn, &lw.wq, cfg.n_heads * cfg.head_dim);
                let mut k = matvec_t(&xn, &lw.wk, cfg.n_kv_heads * cfg.head_dim);
                let v = matvec_t(&xn, &lw.wv, cfg.n_kv_heads * cfg.head_dim);
                // Qwen3 QK-norm: per-head RMSNorm after projection, before RoPE.
                if !lw.q_norm.is_empty() {
                    qk_norm(&mut q, &lw.q_norm, cfg.n_heads, cfg.head_dim, cfg.rms_eps);
                    qk_norm(
                        &mut k,
                        &lw.k_norm,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        cfg.rms_eps,
                    );
                }
                apply_rope(&mut q, cfg.n_heads, cfg.head_dim, pos, cfg.rope_theta);
                apply_rope(&mut k, cfg.n_kv_heads, cfg.head_dim, pos, cfg.rope_theta);

                // Store into KV cache.
                store_row(&mut self.kv[li].0, &k, pos, cfg);
                store_row(&mut self.kv[li].1, &v, pos, cfg);

                // Attention over positions 0..=pos.
                let attn = attention_decode(&q, &self.kv[li].0, &self.kv[li].1, pos, cfg);
                let attn_proj = matvec_t(&attn, &lw.wo, d);
                for i in 0..d {
                    x[i] += attn_proj[i];
                }
            }

            let inter = cfg.intermediate_size;
            let einter = cfg.expert_size();
            let xn2 = rms_norm(&x, &lw.rms_mlp, cfg.rms_eps);
            let moe = cfg.num_experts > 0 && !lw.moe_router.is_empty();
            if moe {
                // MoE: router softmax -> top-k experts -> weighted sum of
                // per-expert SwiGLU MLPs.
                let ne = cfg.num_experts;
                let topk = cfg.num_experts_per_tok.min(ne);
                let router = matvec_t(&xn2, &lw.moe_router, ne);
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
                    // Expert e: gate/up [inter, d], down [d, inter].
                    let wg = &lw.moe_wg[e * einter * d..(e + 1) * einter * d];
                    let wu = &lw.moe_wu[e * einter * d..(e + 1) * einter * d];
                    let wd = &lw.moe_wd[e * d * einter..(e + 1) * d * einter];
                    let gate = matvec_t(&xn2, wg, einter);
                    let up = matvec_t(&xn2, wu, einter);
                    let mut eh = vec![0.0; einter];
                    for i in 0..einter {
                        eh[i] = silu(gate[i]) * up[i];
                    }
                    let down = matvec_t(&eh, wd, d);
                    let w = probs[e] / norm;
                    for i in 0..d {
                        h[i] += w * down[i];
                    }
                }
                for i in 0..d {
                    x[i] += h[i];
                }
            } else {
                let gate = matvec_t(&xn2, &lw.wg, inter);
                let up = matvec_t(&xn2, &lw.wu, inter);
                let mut h = vec![0.0; inter];
                for i in 0..inter {
                    // SwiGLU: h = silu(gate) * up.
                    h[i] = silu(gate[i]) * up[i];
                }
                let down = matvec_t(&h, &lw.wd, d);
                for i in 0..d {
                    x[i] += down[i];
                }
            }
        }

        self.last_hidden = x.clone();
        let xf = rms_norm(&x, &self.w.rms_final, cfg.rms_eps);
        self.pos += 1;
        matvec_t(&xf, &self.w.lm_head, cfg.vocab_size)
    }

    /// Saves a lightweight token-boundary anchor at `token_idx`: the per-layer
    /// KV prefix `[0..=token_idx]` plus the final hidden state at that
    /// position. The model must have processed exactly `token_idx + 1` tokens
    /// (the anchor lives at the current sequence end).
    pub fn save_anchor(
        &self,
        tokens: &[u32],
        token_idx: usize,
    ) -> Result<crate::state_reuse::Anchor, crate::Error> {
        use crate::state_reuse::{Anchor, KvSnapshot};
        if tokens.len() != token_idx + 1 {
            return Err(crate::Error::InvalidArgument(format!(
                "anchor token_idx {token_idx} does not match {} prefix tokens",
                tokens.len()
            )));
        }
        if self.pos != token_idx + 1 {
            return Err(crate::Error::InvalidArgument(format!(
                "anchor at token_idx {token_idx} requires {} processed positions, model has {}",
                token_idx + 1,
                self.pos
            )));
        }
        if self.last_hidden.len() != self.cfg.d_model {
            return Err(crate::Error::Model(
                "no hidden state at anchor position (nothing processed yet)".into(),
            ));
        }
        let cfg = self.cfg;
        let row = cfg.n_kv_heads * cfg.head_dim;
        let n = (token_idx + 1) * row;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        if cfg.kv_lora_rank == 0 {
            for (k, v) in &self.kv {
                layers.push((f32s_to_bytes(&k[..n]), f32s_to_bytes(&v[..n])));
            }
        } else {
            let heads = cfg.n_heads;
            let hd = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
            let kn = (token_idx + 1) * heads * hd;
            let vn = (token_idx + 1) * heads * cfg.v_head_dim;
            for (k, v) in &self.mla_kv {
                layers.push((f32s_to_bytes(&k[..kn]), f32s_to_bytes(&v[..vn])));
            }
        }
        Ok(Anchor {
            id: 0,
            token_idx,
            tokens: tokens.to_vec(),
            kv: KvSnapshot { layers },
            hidden: self.last_hidden.clone(),
        })
    }

    /// Restores an anchor: copies the per-layer KV prefix into the (empty)
    /// caches and resumes at `token_idx + 1` with the saved hidden state.
    /// The next [`Self::decode_step`] processes the first delta token.
    pub fn restore_anchor(
        &mut self,
        anchor: &crate::state_reuse::Anchor,
    ) -> Result<(), crate::Error> {
        let cfg = self.cfg;
        if anchor.kv.layers.len() != cfg.n_layers {
            return Err(crate::Error::InvalidArgument(format!(
                "anchor layer count {} != model {}",
                anchor.kv.layers.len(),
                cfg.n_layers
            )));
        }
        let prefix = anchor.token_idx + 1;
        if prefix > cfg.max_seq_len {
            return Err(crate::Error::InvalidArgument(format!(
                "anchor prefix {prefix} exceeds max_seq_len {}",
                cfg.max_seq_len
            )));
        }
        if anchor.hidden.len() != cfg.d_model {
            return Err(crate::Error::InvalidArgument(
                "anchor hidden size does not match d_model".into(),
            ));
        }
        if cfg.kv_lora_rank == 0 {
            let row = cfg.n_kv_heads * cfg.head_dim;
            let n = prefix * row;
            for (li, (k, v)) in self.kv.iter_mut().enumerate() {
                let (kb, vb) = &anchor.kv.layers[li];
                if kb.len() != n * 4 || vb.len() != n * 4 {
                    return Err(crate::Error::InvalidArgument(format!(
                        "anchor kv size mismatch at layer {li} (expected {} bytes, got {} / {})",
                        n * 4,
                        kb.len(),
                        vb.len()
                    )));
                }
                k[..n].copy_from_slice(&bytes_to_f32s(kb));
                v[..n].copy_from_slice(&bytes_to_f32s(vb));
            }
        } else {
            let heads = cfg.n_heads;
            let hd = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
            let kn = prefix * heads * hd;
            let vn = prefix * heads * cfg.v_head_dim;
            for (li, (k, v)) in self.mla_kv.iter_mut().enumerate() {
                let (kb, vb) = &anchor.kv.layers[li];
                if kb.len() != kn * 4 || vb.len() != vn * 4 {
                    return Err(crate::Error::InvalidArgument(format!(
                        "anchor MLA kv size mismatch at layer {li}"
                    )));
                }
                k[..kn].copy_from_slice(&bytes_to_f32s(kb));
                v[..vn].copy_from_slice(&bytes_to_f32s(vb));
            }
        }
        self.pos = prefix;
        self.last_hidden = anchor.hidden.clone();
        Ok(())
    }

    /// Logits at the position right after the anchor, computed directly from
    /// the saved hidden state (final norm + lm_head) — no forward pass. Equal
    /// to what a full recompute produced at that position.
    #[must_use]
    pub fn logits_at_anchor(&self) -> Vec<f32> {
        let cfg = self.cfg;
        let xf = rms_norm(&self.last_hidden, &self.w.rms_final, cfg.rms_eps);
        matvec_t(&xf, &self.w.lm_head, cfg.vocab_size)
    }
}

/// Serializes f32 values to little-endian bytes (host anchor snapshot).
fn f32s_to_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Deserializes little-endian bytes back to f32 values (anchor restore).
fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

/// Per-head RMSNorm (Qwen3 QK-norm): each head's `head_dim` slice is
/// normalized independently, scaled by that head's weight vector.
fn qk_norm(x: &mut [f32], w: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
    for h in 0..n_heads {
        let s = h * head_dim;
        let mut ss = 0.0;
        for &v in &x[s..s + head_dim] {
            ss += v * v;
        }
        let inv = 1.0 / (ss / head_dim as f32 + eps).sqrt();
        for i in 0..head_dim {
            x[s + i] = x[s + i] * inv * w[s + i];
        }
    }
}

/// MLA attention for one decode step (expanded per-head KV form, matching
/// the transformers DeepseekV2 reference math).
fn decode_step_mla(
    x: &mut [f32],
    xn: &[f32],
    lw: &LayerWeights,
    kc: &mut [f32],
    vc: &mut [f32],
    pos: usize,
    cfg: Config,
) {
    let d = cfg.d_model;
    let heads = cfg.n_heads;
    let nope = cfg.qk_nope_head_dim;
    let rope_hd = cfg.qk_rope_head_dim;
    let v_hd = cfg.v_head_dim;
    let hd = nope + rope_hd;
    let scale = 1.0 / (hd as f32).sqrt();

    // q_nope = q_b(rms(q_a(x))), q_rope = q_rope(x) + RoPE.
    let q_lora = matvec_t(xn, &lw.mla_q_a, cfg.q_lora_rank);
    let q_lora = rms_norm(&q_lora, &lw.mla_q_a_norm, cfg.rms_eps);
    let q_nope = matvec_t(&q_lora, &lw.mla_q_b, heads * nope);
    let mut q_rope = matvec_t(xn, &lw.mla_q_rope, heads * rope_hd);
    apply_rope(&mut q_rope, heads, rope_hd, pos, cfg.rope_theta);

    // compressed_kv = kv_a(x); kv_lora (rms) feeds kv_b; k_rope is shared
    // across heads and rotated like a single-head rope.
    let kv_a = matvec_t(xn, &lw.mla_kv_a, cfg.kv_lora_rank + rope_hd);
    let kv_lora = rms_norm(&kv_a[..cfg.kv_lora_rank], &lw.mla_kv_a_norm, cfg.rms_eps);
    let mut k_rope = kv_a[cfg.kv_lora_rank..].to_vec();
    apply_rope(&mut k_rope, 1, rope_hd, pos, cfg.rope_theta);

    // kv_b expands the latent into per-head k_nope + v.
    let kv = matvec_t(&kv_lora, &lw.mla_kv_b, heads * (nope + v_hd));

    // Assemble per-head q (nope || rope), k (nope || rope broadcast), v.
    let mut qm = vec![0.0; heads * hd];
    let mut km = vec![0.0; heads * hd];
    let mut vm = vec![0.0; heads * v_hd];
    for h in 0..heads {
        let base = h * (nope + v_hd);
        qm[h * hd..h * hd + nope].copy_from_slice(&q_nope[h * nope..(h + 1) * nope]);
        qm[h * hd + nope..(h + 1) * hd].copy_from_slice(&q_rope[h * rope_hd..(h + 1) * rope_hd]);
        km[h * hd..h * hd + nope].copy_from_slice(&kv[base..base + nope]);
        km[h * hd + nope..(h + 1) * hd].copy_from_slice(&k_rope);
        vm[h * v_hd..(h + 1) * v_hd].copy_from_slice(&kv[base + nope..base + nope + v_hd]);
    }

    let ko = pos * heads * hd;
    kc[ko..ko + heads * hd].copy_from_slice(&km);
    let vo = pos * heads * v_hd;
    vc[vo..vo + heads * v_hd].copy_from_slice(&vm);

    // Per-head attention over positions 0..=pos.
    let mut attn = vec![0.0; heads * v_hd];
    for h in 0..heads {
        let qh = &qm[h * hd..(h + 1) * hd];
        let mut scores = vec![0.0; pos + 1];
        let mut maxv = f32::NEG_INFINITY;
        for pp in 0..=pos {
            let kp = &kc[pp * heads * hd + h * hd..pp * heads * hd + (h + 1) * hd];
            let mut s = 0.0;
            for dd in 0..hd {
                s += qh[dd] * kp[dd];
            }
            s *= scale;
            scores[pp] = s;
            maxv = maxv.max(s);
        }
        let mut sum = 0.0;
        for item in scores.iter_mut().take(pos + 1) {
            *item = (*item - maxv).exp();
            sum += *item;
        }
        for dd in 0..v_hd {
            let mut acc = 0.0;
            for pp in 0..=pos {
                acc += scores[pp] * vc[pp * heads * v_hd + h * v_hd + dd];
            }
            attn[h * v_hd + dd] = acc / sum;
        }
    }

    let attn_proj = matvec_t(&attn, &lw.mla_o, d);
    for i in 0..d {
        x[i] += attn_proj[i];
    }
}

fn apply_rope(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        for d in 0..half {
            let freq = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
            let ang = pos as f32 * freq;
            let c = ang.cos();
            let sn = ang.sin();
            // GPT-NeoX rotary: pairs (d, d + half), matching HF rotate_half.
            let idx = h * head_dim + d;
            let a = x[idx];
            let b = x[idx + half];
            x[idx] = a * c - b * sn;
            x[idx + half] = a * sn + b * c;
        }
    }
}

fn store_row(cache: &mut [f32], row: &[f32], pos: usize, cfg: Config) {
    let off = pos * cfg.n_kv_heads * cfg.head_dim;
    cache[off..off + row.len()].copy_from_slice(row);
}

#[allow(clippy::needless_range_loop)]
fn attention_decode(q: &[f32], kc: &[f32], vc: &[f32], pos: usize, cfg: Config) -> Vec<f32> {
    let hd = cfg.head_dim;
    let groups = cfg.n_heads / cfg.n_kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0; cfg.n_heads * hd];
    for h in 0..cfg.n_heads {
        let kv = h / groups;
        let qh = &q[h * hd..(h + 1) * hd];
        let mut scores = vec![0.0; pos + 1];
        let mut maxv = f32::NEG_INFINITY;
        for p in 0..=pos {
            let kp = &kc[(p * cfg.n_kv_heads + kv) * hd..(p * cfg.n_kv_heads + kv + 1) * hd];
            let mut s = 0.0;
            for dd in 0..hd {
                s += qh[dd] * kp[dd];
            }
            s *= scale;
            scores[p] = s;
            maxv = maxv.max(s);
        }
        let mut sum = 0.0;
        for p in 0..=pos {
            scores[p] = (scores[p] - maxv).exp();
            sum += scores[p];
        }
        for dd in 0..hd {
            let mut acc = 0.0;
            for p in 0..=pos {
                let vp = &vc[(p * cfg.n_kv_heads + kv) * hd + dd];
                acc += scores[p] * vp;
            }
            out[h * hd + dd] = acc / sum;
        }
    }
    out
}

fn matvec_t(x: &[f32], w: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    let mut out = vec![0.0; out_dim];
    for o in 0..out_dim {
        let mut s = 0.0;
        for i in 0..in_dim {
            s += w[o * in_dim + i] * x[i];
        }
        out[o] = s;
    }
    out
}

fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0.0;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    (0..n).map(|i| x[i] * inv * w[i]).collect()
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}
