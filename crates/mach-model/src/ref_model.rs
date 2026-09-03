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
                apply_rope(&mut q, cfg.n_heads, cfg.head_dim, pos, cfg);
                apply_rope(&mut k, cfg.n_kv_heads, cfg.head_dim, pos, cfg);

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
                    // HF `MoEGate.forward`: renormalize only when
                    // `top_k > 1 && norm_topk_prob` (`p / sum(p)`, no
                    // `routed_scaling_factor`); otherwise the softmax score
                    // is kept and scaled by `routed_scaling_factor`.
                    let w = if cfg.moe_norm_topk && topk > 1 {
                        probs[e] / norm
                    } else {
                        probs[e] * cfg.moe_routed_scale
                    };
                    for i in 0..d {
                        h[i] += w * down[i];
                    }
                }
                // Shared experts (DeepSeek-V2): a dense SwiGLU MLP of width
                // `n_shared_experts * expert_size` on the same normalized
                // input, ADDED to the routed experts' weighted sum.
                if !lw.shared_wg.is_empty() {
                    let shinter = cfg.shared_size();
                    let gate = matvec_t(&xn2, &lw.shared_wg, shinter);
                    let up = matvec_t(&xn2, &lw.shared_wu, shinter);
                    let mut eh = vec![0.0; shinter];
                    for i in 0..shinter {
                        eh[i] = silu(gate[i]) * up[i];
                    }
                    let down = matvec_t(&eh, &lw.shared_wd, d);
                    for i in 0..d {
                        h[i] += down[i];
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

    /// Hidden state of the most recently processed token (after the last
    /// layer, before the final norm + lm_head); empty until something runs.
    #[must_use]
    pub fn hidden(&self) -> &[f32] {
        &self.last_hidden
    }

    /// Extracts per-layer KV bytes for positions `[start, end)` (same host
    /// framing as [`Self::save_anchor`]). Used by the page-prefix cache to
    /// store one page's KV independently of the rest of the prefix.
    #[must_use]
    pub fn kv_slice_bytes(&self, start: usize, end: usize) -> Vec<(Vec<u8>, Vec<u8>)> {
        assert!(
            start < end && end <= self.pos,
            "kv slice must be within processed positions"
        );
        let cfg = self.cfg;
        if cfg.kv_lora_rank == 0 {
            let row = cfg.n_kv_heads * cfg.head_dim;
            let n0 = start * row;
            let n1 = end * row;
            self.kv
                .iter()
                .map(|(k, v)| (f32s_to_bytes(&k[n0..n1]), f32s_to_bytes(&v[n0..n1])))
                .collect()
        } else {
            let heads = cfg.n_heads;
            let hd = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
            let kn0 = start * heads * hd;
            let kn1 = end * heads * hd;
            let vn0 = start * heads * cfg.v_head_dim;
            let vn1 = end * heads * cfg.v_head_dim;
            self.mla_kv
                .iter()
                .map(|(k, v)| (f32s_to_bytes(&k[kn0..kn1]), f32s_to_bytes(&v[vn0..vn1])))
                .collect()
        }
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
pub(crate) fn qk_norm(x: &mut [f32], w: &[f32], n_heads: usize, head_dim: usize, eps: f32) {
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
    let scale = cfg.attn_scale(hd);

    // q_nope = q_b(rms(q_a(x))), q_rope = q_rope(x) + RoPE. With
    // `q_lora_rank == 0` (DeepSeek-V2-Lite) there is no low-rank q: both halves
    // of the fused projection read the layer input directly.
    let q_lora = if cfg.q_lora_rank > 0 {
        let ql = matvec_t(xn, &lw.mla_q_a, cfg.q_lora_rank);
        rms_norm(&ql, &lw.mla_q_a_norm, cfg.rms_eps)
    } else {
        xn.to_vec()
    };
    let q_nope = matvec_t(&q_lora, &lw.mla_q_b, heads * nope);
    let mut q_rope = matvec_t(&q_lora, &lw.mla_q_rope, heads * rope_hd);
    apply_rope(&mut q_rope, heads, rope_hd, pos, cfg);

    // compressed_kv = kv_a(x); kv_lora (rms) feeds kv_b; k_rope is shared
    // across heads and rotated like a single-head rope.
    let kv_a = matvec_t(xn, &lw.mla_kv_a, cfg.kv_lora_rank + rope_hd);
    let kv_lora = rms_norm(&kv_a[..cfg.kv_lora_rank], &lw.mla_kv_a_norm, cfg.rms_eps);
    let mut k_rope = kv_a[cfg.kv_lora_rank..].to_vec();
    apply_rope(&mut k_rope, 1, rope_hd, pos, cfg);

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

/// YaRN inverse frequency for pair index `d`, mirroring the `yarn_freq` device
/// helper in `kernels.rs` (HF `_compute_yarn_parameters`): plain RoPE when YaRN
/// is off, else an interpolation of `freq_extra` and `freq_inter` over the
/// `beta_fast`/`beta_slow` correction range.
fn rope_freq(d: usize, head_dim: usize, cfg: Config) -> f32 {
    let theta = cfg.rope_theta;
    let freq_extra = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
    if !cfg.yarn() {
        return freq_extra;
    }
    let freq_inter = 1.0 / (cfg.rope_yarn_factor * theta.powf(2.0 * d as f32 / head_dim as f32));
    let low = (head_dim as f32
        * (cfg.rope_yarn_orig_len as f32
            / (cfg.rope_yarn_beta_fast * 2.0 * core::f32::consts::PI))
            .ln()
        / (2.0 * theta.ln()))
    .floor();
    let high = (head_dim as f32
        * (cfg.rope_yarn_orig_len as f32
            / (cfg.rope_yarn_beta_slow * 2.0 * core::f32::consts::PI))
            .ln()
        / (2.0 * theta.ln()))
    .ceil();
    let low = low.max(0.0);
    let mut high = high.min(head_dim as f32 - 1.0);
    if high == low {
        high += 0.001;
    }
    let ramp = ((d as f32 - low) / (high - low)).clamp(0.0, 1.0);
    freq_inter * ramp + freq_extra * (1.0 - ramp)
}

pub(crate) fn apply_rope(x: &mut [f32], n_heads: usize, head_dim: usize, pos: usize, cfg: Config) {
    let half = head_dim / 2;
    let attn_factor = cfg.yarn_attention_factor();
    for h in 0..n_heads {
        for d in 0..half {
            let freq = rope_freq(d, head_dim, cfg);
            let ang = pos as f32 * freq;
            let c = ang.cos() * attn_factor;
            let sn = ang.sin() * attn_factor;
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
    let scale = cfg.attn_scale(hd);
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

pub(crate) fn matvec_t(x: &[f32], w: &[f32], out_dim: usize) -> Vec<f32> {
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

pub(crate) fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0.0;
    for &v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    (0..n).map(|i| x[i] * inv * w[i]).collect()
}

pub(crate) fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent transcription of HF's YaRN rotary
    /// (`modeling_deepseek.DeepseekV2YarnRotaryEmbedding` plus the three
    /// `yarn_*` helpers), used to pin `rope_freq` / `apply_rope`. Deliberately
    /// written in the source's own shape — `arange(0, dim, 2)` pair indices,
    /// `floor`/`ceil` on the correction range, the `1 - ramp` mask, and the
    /// `_mscale` ratio on cos/sin — rather than reusing the implementation it
    /// checks, so a shared misreading would not pass.
    struct HfYarn {
        cos: Vec<f32>,
        sin: Vec<f32>,
    }

    fn yarn_get_mscale(scale: f64, mscale: f64) -> f64 {
        if scale <= 1.0 {
            return 1.0;
        }
        0.1 * mscale * scale.ln() + 1.0
    }

    /// cos/sin per pair index at `pos`, plus the `inv_freq` HF would cache.
    fn hf_yarn_cos_sin(cfg: Config, pos: usize) -> (HfYarn, Vec<f64>) {
        let dim = cfg.qk_rope_head_dim.max(cfg.head_dim);
        let base = f64::from(cfg.rope_theta);
        let factor = f64::from(cfg.rope_yarn_factor);
        let orig = cfg.rope_yarn_orig_len as f64;
        let pairs = dim / 2;

        // `arange(0, dim, 2) / dim`, then `base ** that`.
        let pow: Vec<f64> = (0..pairs)
            .map(|d| base.powf((2 * d) as f64 / dim as f64))
            .collect();
        let freq_extra: Vec<f64> = pow.iter().map(|p| 1.0 / p).collect();
        let freq_inter: Vec<f64> = pow.iter().map(|p| 1.0 / (factor * p)).collect();

        // yarn_find_correction_dim(num_rotations, dim, base, max_pos)
        let correction_dim = |num_rotations: f64| -> f64 {
            (dim as f64 * (orig / (num_rotations * 2.0 * std::f64::consts::PI)).ln())
                / (2.0 * base.ln())
        };
        // yarn_find_correction_range(beta_fast, beta_slow, dim, ...)
        let low = correction_dim(f64::from(cfg.rope_yarn_beta_fast)).floor();
        let high = correction_dim(f64::from(cfg.rope_yarn_beta_slow)).ceil();
        let low = low.max(0.0);
        let mut high = high.min(dim as f64 - 1.0);
        if high == low {
            high += 0.001;
        }
        // yarn_linear_ramp_mask(low, high, dim // 2)
        let mask: Vec<f64> = (0..pairs)
            .map(|d| 1.0 - (((d as f64 - low) / (high - low)).clamp(0.0, 1.0)))
            .collect();
        let inv_freq: Vec<f64> = (0..pairs)
            .map(|d| freq_inter[d] * (1.0 - mask[d]) + freq_extra[d] * mask[d])
            .collect();

        let mscale = yarn_get_mscale(factor, f64::from(cfg.rope_yarn_mscale));
        let mscale_all_dim = yarn_get_mscale(factor, f64::from(cfg.rope_yarn_mscale_all_dim));
        let attn_factor = mscale / mscale_all_dim;
        let t = pos as f64;
        (
            HfYarn {
                cos: inv_freq
                    .iter()
                    .map(|f| ((t * f).cos() * attn_factor) as f32)
                    .collect(),
                sin: inv_freq
                    .iter()
                    .map(|f| ((t * f).sin() * attn_factor) as f32)
                    .collect(),
            },
            inv_freq,
        )
    }

    fn yarn_cfg(rope_dim: usize) -> Config {
        let mut cfg = Config::tiny();
        cfg.qk_rope_head_dim = rope_dim;
        cfg.head_dim = rope_dim;
        cfg.rope_theta = 10_000.0;
        cfg.rope_yarn_factor = 40.0;
        cfg.rope_yarn_orig_len = 4096;
        cfg.rope_yarn_beta_fast = 32.0;
        cfg.rope_yarn_beta_slow = 1.0;
        // Asymmetric so the cos/sin `_mscale` ratio is not trivially 1.0
        // (DeepSeek-V2-Lite ships 0.707/0.707, which would hide a swapped or
        // dropped term).
        cfg.rope_yarn_mscale = 1.0;
        cfg.rope_yarn_mscale_all_dim = 0.707;
        cfg
    }

    #[test]
    fn yarn_inv_freq_matches_hf() {
        let cfg = yarn_cfg(64);
        for d in 0..32 {
            let want = hf_yarn_cos_sin(cfg, 0).1[d];
            let got = f64::from(rope_freq(d, cfg.qk_rope_head_dim, cfg));
            assert!(
                (got - want).abs() <= 1e-5 * want.abs(),
                "d={d}: got {got} want {want}"
            );
        }
    }

    /// With YaRN off, `rope_freq` must be plain `theta^(-2d/hd)`.
    #[test]
    fn rope_freq_without_yarn_is_plain() {
        let mut cfg = yarn_cfg(64);
        cfg.rope_yarn_factor = 0.0;
        cfg.rope_yarn_orig_len = 0;
        assert!(!cfg.yarn());
        for d in 0..32 {
            let want = 10_000.0f32.powf(-(2.0 * d as f32) / 64.0);
            assert!((rope_freq(d, 64, cfg) - want).abs() <= 1e-6 * want, "d={d}");
        }
    }

    /// `apply_rope` must reproduce HF's `rotate_half` pair rotation with the
    /// YaRN cos/sin, including the `_mscale` ratio folded into both.
    #[test]
    fn apply_rope_matches_hf_rotate_half_with_yarn() {
        let cfg = yarn_cfg(64);
        let pos = 7usize;
        let (hf, _) = hf_yarn_cos_sin(cfg, pos);
        let heads = 2usize;
        let half = 32usize;
        let mut x: Vec<f32> = (0..heads * 64)
            .map(|i| ((i as f32) * 0.37).sin() * 3.0)
            .collect();
        let orig = x.clone();
        apply_rope(&mut x, heads, 64, pos, cfg);
        for h in 0..heads {
            for d in 0..half {
                let a = orig[h * 64 + d];
                let b = orig[h * 64 + d + half];
                let c = hf.cos[d];
                let s = hf.sin[d];
                // HF: q * cos + rotate_half(q) * sin, with rotate_half
                // mapping (a, b) -> (-b, a).
                let want_a = a * c - b * s;
                let want_b = b * c + a * s;
                let got_a = x[h * 64 + d];
                let got_b = x[h * 64 + d + half];
                let tol = 1e-3 + 1e-3 * want_a.abs().max(want_b.abs());
                assert!(
                    (got_a - want_a).abs() <= tol && (got_b - want_b).abs() <= tol,
                    "h={h} d={d}: got ({got_a}, {got_b}) want ({want_a}, {want_b})"
                );
            }
        }
    }

    /// The cos/sin factor is `mscale(factor) / mscale_all_dim(factor)`, so it
    /// must move away from 1.0 when the two differ — and be exactly 1.0 for
    /// DeepSeek-V2-Lite, which ships 0.707 for both.
    #[test]
    fn yarn_attention_factor_is_mscale_ratio() {
        let cfg = yarn_cfg(64);
        let ln = 40.0f64.ln();
        let want = (0.1 * 1.0 * ln + 1.0) / (0.1 * 0.707 * ln + 1.0);
        assert!(
            (f64::from(cfg.yarn_attention_factor()) - want).abs() < 1e-6,
            "got {} want {want}",
            cfg.yarn_attention_factor()
        );
        let mut lite = cfg;
        lite.rope_yarn_mscale = 0.707;
        assert!((lite.yarn_attention_factor() - 1.0).abs() < 1e-6);
        // YaRN off: 1.0 regardless of the mscale fields.
        let mut off = cfg;
        off.rope_yarn_factor = 0.0;
        off.rope_yarn_orig_len = 0;
        assert_eq!(off.yarn_attention_factor(), 1.0);
    }

    /// The `mscale^2` logit correction uses `mscale_all_dim`, not `mscale`
    /// (HF computes `mscale = yarn_get_mscale(factor, mscale_all_dim)` and
    /// multiplies `softmax_scale` by it twice).
    #[test]
    fn attn_scale_applies_mscale_all_dim_squared() {
        let cfg = yarn_cfg(64);
        let m = 0.1 * 0.707 * 40.0f64.ln() + 1.0;
        let want = m * m / 192.0f64.sqrt();
        assert!(
            (f64::from(cfg.attn_scale(192)) - want).abs() < 1e-6,
            "got {} want {want}",
            cfg.attn_scale(192)
        );
        // `mscale_all_dim = 0` (the HF default) disables the correction.
        let mut no_dim = cfg;
        no_dim.rope_yarn_mscale_all_dim = 0.0;
        assert!((no_dim.attn_scale(192) - 1.0 / 192.0f32.sqrt()).abs() < 1e-6);
    }

    /// HF builds the ramp mask over `dim // 2` entries but clamps `high` to
    /// `dim - 1`, so `high` can exceed the last pair index; clamping to the
    /// pair count instead would silently shrink the ramp.
    #[test]
    fn yarn_ramp_uses_head_dim_clamp_not_pair_clamp() {
        let cfg = yarn_cfg(64);
        // beta_fast=32, beta_slow=1, orig=4096, base=10000, dim=64.
        let low = (64.0 * (4096.0 / (32.0 * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * 10_000f64.ln()))
        .floor();
        let high = (64.0 * (4096.0 / (1.0 * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * 10_000f64.ln()))
        .ceil();
        assert_eq!(low, 10.0, "low");
        assert_eq!(high, 23.0, "high");
        // d=0 -> ramp 0 (pure freq_extra); d=31 -> ramp 1 (pure freq_inter).
        let f0 = f64::from(rope_freq(0, 64, cfg));
        let extra0 = 1.0 / 10_000f64.powf(0.0);
        assert!((f0 - extra0).abs() < 1e-6, "d=0 must be pure freq_extra");
        let f31 = f64::from(rope_freq(31, 64, cfg));
        let inter31 = 1.0 / (40.0 * 10_000f64.powf((2 * 31) as f64 / 64.0));
        assert!((f31 - inter31).abs() < 1e-9, "d=31 must be pure freq_inter");
    }
}
