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
    /// Per GDN layer (Qwen3.5 linear attention): recurrent state
    /// `[gdn_v_heads, k_dim, v_dim]`, v-head-major S matrices.
    gdn_state: Vec<Vec<f32>>,
    /// Per GDN layer: causal conv state — the last `conv_kernel - 1` RAW
    /// pre-conv fused-qkv inputs `[2*k_dim + v_dim]` (what the depthwise
    /// conv1d consumes before silu).
    gdn_conv: Vec<Vec<f32>>,
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
        // GDN recurrent + conv state for every layer when the config is
        // hybrid (full-attention layers simply never touch theirs), keeping
        // per-layer indexing uniform.
        let gdn_state = if cfg.gdn_enabled() {
            let kd = cfg.gdn_key_dim();
            let vd = cfg.gdn_value_dim();
            (0..cfg.n_layers)
                .map(|_| vec![0.0; cfg.gdn_v_heads * kd * vd])
                .collect()
        } else {
            Vec::new()
        };
        let gdn_conv = if cfg.gdn_enabled() {
            let conv_dim = 2 * cfg.gdn_key_dim() + cfg.gdn_value_dim();
            let keep = cfg.gdn_conv_kernel - 1;
            (0..cfg.n_layers)
                .map(|_| vec![0.0; conv_dim * keep])
                .collect()
        } else {
            Vec::new()
        };
        Self {
            cfg,
            w,
            kv,
            mla_kv,
            gdn_state,
            gdn_conv,
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
            if cfg.gdn_enabled() && !lw.gdn_in_qkv.is_empty() {
                // Qwen3.5 hybrid: linear-attention layer (gated DeltaNet).
                let st = &mut self.gdn_state[li];
                let cv = &mut self.gdn_conv[li];
                decode_step_gdn(&mut x, &xn, lw, st, cv, cfg);
            } else if cfg.kv_lora_rank > 0 {
                // MLA (DeepSeek-V2 style): low-rank Q + compressed KV.
                let (kc, vc) = &mut self.mla_kv[li];
                decode_step_mla(&mut x, &xn, lw, kc, vc, pos, cfg);
            } else {
                // Qwen3.5 `attn_output_gate`: q_proj carries
                // `n_heads * head_dim * 2` rows; each head's block is
                // `[query | gate]` (HF `chunk`s per head after the
                // projection). The gate skips QK-norm and RoPE and scales the
                // attention output elementwise (sigmoid) before o_proj.
                let hd = cfg.head_dim;
                let (mut q, gate) = if cfg.attn_output_gate {
                    let qg = matvec_t(&xn, &lw.wq, cfg.n_heads * hd * 2);
                    let mut q = vec![0.0f32; cfg.n_heads * hd];
                    let mut gate = vec![0.0f32; cfg.n_heads * hd];
                    for h in 0..cfg.n_heads {
                        for j in 0..hd {
                            q[h * hd + j] = qg[h * 2 * hd + j];
                            gate[h * hd + j] = qg[h * 2 * hd + hd + j];
                        }
                    }
                    (q, gate)
                } else {
                    (matvec_t(&xn, &lw.wq, cfg.n_heads * hd), Vec::new())
                };
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
                // Partial rotary (Qwen3.5 full-attn layers): only the first
                // `attn_rotary_dim` coordinates rotate; the tail passes
                // through unchanged.
                let rot = cfg.attn_rotary_dim();
                apply_rope(&mut q, cfg.n_heads, cfg.head_dim, rot, pos, cfg);
                apply_rope(&mut k, cfg.n_kv_heads, cfg.head_dim, rot, pos, cfg);

                // Store into KV cache.
                store_row(&mut self.kv[li].0, &k, pos, cfg);
                store_row(&mut self.kv[li].1, &v, pos, cfg);

                // Attention over positions 0..=pos.
                let mut attn = attention_decode(&q, &self.kv[li].0, &self.kv[li].1, pos, cfg);
                if cfg.attn_output_gate {
                    for (a, g) in attn.iter_mut().zip(&gate) {
                        *a *= 1.0 / (1.0 + (-g).exp());
                    }
                }
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
    apply_rope(&mut q_rope, heads, rope_hd, rope_hd, pos, cfg);

    // compressed_kv = kv_a(x); kv_lora (rms) feeds kv_b; k_rope is shared
    // across heads and rotated like a single-head rope.
    let kv_a = matvec_t(xn, &lw.mla_kv_a, cfg.kv_lora_rank + rope_hd);
    let kv_lora = rms_norm(&kv_a[..cfg.kv_lora_rank], &lw.mla_kv_a_norm, cfg.rms_eps);
    let mut k_rope = kv_a[cfg.kv_lora_rank..].to_vec();
    apply_rope(&mut k_rope, 1, rope_hd, rope_hd, pos, cfg);

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

/// Applies rotary position embeddings to the first `rotary_dim` coordinates
/// of each `head_dim` slice; coordinates `[rotary_dim, head_dim)` pass through
/// unchanged (partial rotary, Qwen3.5 full-attn layers). The frequency basis
/// is `rotary_dim` — HF computes `inv_freq` over the rotating width, not the
/// full head dim.
pub(crate) fn apply_rope(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    pos: usize,
    cfg: Config,
) {
    let half = rotary_dim / 2;
    let attn_factor = cfg.yarn_attention_factor();
    for h in 0..n_heads {
        for d in 0..half {
            let freq = rope_freq(d, rotary_dim, cfg);
            let ang = pos as f32 * freq;
            let c = ang.cos() * attn_factor;
            let sn = ang.sin() * attn_factor;
            // `d` is the PAIR index and the frequency index; which two
            // coordinates it joins depends on the checkpoint's convention:
            //   interleave=false -> GPT-NeoX/HF rotate_half: (d, d + half)
            //   interleave=true  -> DeepSeek-V2: (2d, 2d + 1)
            // Identical at pos 0 (RoPE is the identity there), so a pos-0-only
            // comparison cannot tell them apart.
            let (d0, d1) = if cfg.rope_interleave {
                (2 * d, 2 * d + 1)
            } else {
                (d, d + half)
            };
            let idx = h * head_dim + d0;
            let jdx = h * head_dim + d1;
            let a = x[idx];
            let b = x[jdx];
            x[idx] = a * c - b * sn;
            x[jdx] = a * sn + b * c;
        }
    }
}

fn store_row(cache: &mut [f32], row: &[f32], pos: usize, cfg: Config) {
    let off = pos * cfg.n_kv_heads * cfg.head_dim;
    cache[off..off + row.len()].copy_from_slice(row);
}

/// Numerically stable softplus (`log(1 + e^x)`), linear above the standard
/// threshold 20 (where `log1p(exp(x))` would overflow in f32).
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// L2-normalizes a vector in place (the reference runs it in fp32; host math
/// here is f32 throughout). The eps lives INSIDE the square root, matching
/// the reference `l2norm` (eps=1e-6, aligned with the FLA kernels).
fn l2norm(x: &mut [f32]) {
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let inv = 1.0 / (ss + 1e-6).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Gated DeltaNet decode step for one position (Qwen3.5 linear-attention
/// layers), transcribing the transformers reference math one token at a
/// time. State lives with the caller:
/// - `state` `[gdn_v_heads, k_dim, v_dim]`: per-v-head recurrent matrix S
/// - `conv` `[2*k_dim + v_dim, conv_kernel - 1]`: RAW pre-conv inputs of the
///   fused qkv stream (the conv state stores what conv1d consumed, not its
///   post-silu output).
///
/// Per v-head i (k head `i / (v_heads/k_heads)` — k heads repeat across v
/// heads):
///   g = -exp(A_log) * softplus(a + dt_bias); decay = exp(g); beta = sigmoid(b)
///   S <- S * decay; kv_mem = S^T k; delta = (v - kv_mem) * beta
///   S <- S + k (x) delta; o = S^T q   (reading the UPDATED S)
/// Then per-head gated RMSNorm over the v dim: fp32 RMS -> * w -> * silu(z),
/// weight shared across v-heads.
fn decode_step_gdn(
    x: &mut [f32],
    xn: &[f32],
    lw: &LayerWeights,
    state: &mut [f32],
    conv: &mut [f32],
    cfg: Config,
) {
    let d = cfg.d_model;
    let kh = cfg.gdn_k_heads;
    let vh = cfg.gdn_v_heads;
    let hd = cfg.gdn_head_dim;
    let kd = kh * hd;
    let vd = vh * hd;
    let conv_dim = 2 * kd + vd;
    let ck = cfg.gdn_conv_kernel;
    let keep = ck - 1;

    // 1) projections. The fused qkv stream goes through conv+silu; z/a/b do
    //    not.
    let qkv_pre = matvec_t(xn, &lw.gdn_in_qkv, conv_dim);
    let z = matvec_t(xn, &lw.gdn_in_z, vd);
    let a = matvec_t(xn, &lw.gdn_in_a, vh);
    let b = matvec_t(xn, &lw.gdn_in_b, vh);

    // 2) depthwise causal conv1d over [state || new input], then silu. The
    //    conv weight is `[conv_dim, ck]` (PyTorch depthwise `[out, 1, k]`
    //    flattened); the newest tap is the LAST column.
    let mut qkv = vec![0.0f32; conv_dim];
    for c in 0..conv_dim {
        let w = &lw.gdn_conv_w[c * ck..c * ck + ck];
        let mut s = 0.0;
        for j in 0..keep {
            s += conv[c * keep + j] * w[j];
        }
        s += qkv_pre[c] * w[keep];
        qkv[c] = silu(s);
    }
    // Shift the conv state left, appending the raw new input.
    for c in 0..conv_dim {
        for j in 0..keep - 1 {
            conv[c * keep + j] = conv[c * keep + j + 1];
        }
        conv[c * keep + keep - 1] = qkv_pre[c];
    }

    // 3) split the fused stream, per-head l2norm on q/k, q scale.
    let mut q = vec![0.0f32; kd];
    let mut k = vec![0.0f32; kd];
    let v = qkv[2 * kd..].to_vec();
    q.copy_from_slice(&qkv[..kd]);
    k.copy_from_slice(&qkv[kd..2 * kd]);
    for h in 0..kh {
        l2norm(&mut q[h * hd..(h + 1) * hd]);
        l2norm(&mut k[h * hd..(h + 1) * hd]);
    }
    // q <- q * head_k_dim^-0.5 (after l2norm, matching the reference order).
    let qs = 1.0 / (hd as f32).sqrt();
    for val in q.iter_mut() {
        *val *= qs;
    }

    // 4-5) per-v-head delta rule on the recurrent state.
    let rep = vh / kh; // v-heads per k-head (48/16 = 3)
    let mut o = vec![0.0f32; vd];
    for i in 0..vh {
        let g = -lw.gdn_a_log[i].exp() * softplus(a[i] + lw.gdn_dt_bias[i]);
        let decay = g.exp();
        let beta = 1.0 / (1.0 + (-b[i]).exp());
        let so = i * hd * hd;
        let kvec = &k[(i / rep) * hd..(i / rep) * hd + hd];
        let vvec = &v[i * hd..(i + 1) * hd];
        let qvec = &q[(i / rep) * hd..(i / rep) * hd + hd];
        // S <- S * decay
        for s in state[so..so + hd * hd].iter_mut() {
            *s *= decay;
        }
        // kv_mem = S^T k
        let mut kv_mem = vec![0.0f32; hd];
        for kk in 0..hd {
            let kw = kvec[kk];
            if kw == 0.0 {
                continue;
            }
            let row = &state[so + kk * hd..so + (kk + 1) * hd];
            for (m, r) in kv_mem.iter_mut().zip(row) {
                *m += kw * r;
            }
        }
        // S <- S + k (x) delta, delta = (v - kv_mem) * beta
        for kk in 0..hd {
            let kw = kvec[kk];
            let row = &mut state[so + kk * hd..so + (kk + 1) * hd];
            for (vv, r) in row.iter_mut().enumerate() {
                *r += kw * (vvec[vv] - kv_mem[vv]) * beta;
            }
        }
        // o = S^T q, reading the UPDATED S.
        for kk in 0..hd {
            let qw = qvec[kk];
            if qw == 0.0 {
                continue;
            }
            let row = &state[so + kk * hd..so + (kk + 1) * hd];
            for (m, r) in o[i * hd..i * hd + hd].iter_mut().zip(row) {
                *m += qw * r;
            }
        }
    }

    // 6) per-head gated RMSNorm over the v dim, weight shared across heads,
    //    gated by silu(z).
    for i in 0..vh {
        let oh = &mut o[i * hd..(i + 1) * hd];
        let mut ss = 0.0;
        for &t in oh.iter() {
            ss += t * t;
        }
        let inv = 1.0 / (ss / hd as f32 + cfg.rms_eps).sqrt();
        let zh = &z[i * hd..(i + 1) * hd];
        for (j, t) in oh.iter_mut().enumerate() {
            *t = *t * inv * lw.gdn_norm[j] * silu(zh[j]);
        }
    }

    // 7) output projection + residual.
    let out = matvec_t(&o, &lw.gdn_out, d);
    for i in 0..d {
        x[i] += out[i];
    }
}

/// Chunked GDN layer forward (`#112` Stage B oracle): `xs` holds `C` tokens'
/// residual-stream rows, processed through one gated-DeltaNet layer in place
/// (state/conv continue from the previous step). Token-local stages reuse the
/// sequential path's math; the conv window and the delta-rule recurrence are
/// the sequential parts, the latter in WY-style chunked form:
///
/// Per v-head, with per-token log-decay `g_t` (so `λ_t = e^{g_t}`, `Π_t =
/// e^{Σ_{s≤t} g_s}`):
///   `u_t + Σ_{s<t} A[t,s]·u_s = β_t·(v_t − Π_t·S₀ᵀk_t)`,  `A[t,s] =
///   β_t·(Π_t/Π_s)·(k_s·k_t)` (strictly lower; the query-side `β_t` folds
///   into the row — it scales the whole delta, retro-retrieval included) —
///   forward substitution over `t`.
///   `S_C = Π_C·S₀ + Σ_s (Π_C/Π_s)·k_s⊗u_s`
///   `o_t = Π_t·S₀ᵀq_t + Σ_{s≤t} (Π_t/Π_s)·(k_s·q_t)·u_s`  (own write included,
///   matching the decode kernel's read-after-write `o = Sᵀq`).
///
/// Reassociates the recurrence's sums, so it matches [`decode_step_gdn`]
/// applied `C` times only to f32 reassociation tolerance (pinned by test).
// Whole-model chunked prefill wiring lands with Stage B batch 2; until then
// this is exercised by the layer-level chunk-vs-sequential test below.
#[allow(dead_code, clippy::needless_range_loop)]
pub(crate) fn decode_chunk_gdn(
    xs: &mut [Vec<f32>],
    lw: &LayerWeights,
    state: &mut [f32],
    conv: &mut [f32],
    cfg: Config,
) {
    let d = cfg.d_model;
    let kh = cfg.gdn_k_heads;
    let vh = cfg.gdn_v_heads;
    let hd = cfg.gdn_head_dim;
    let kd = kh * hd;
    let vd = vh * hd;
    let conv_dim = 2 * kd + vd;
    let ck = cfg.gdn_conv_kernel;
    let keep = ck - 1;
    let c = xs.len();

    // 1-2) per-token projections + conv + silu. The conv window is a fixed
    // `ck`-tap causal dot with no cross-token carry beyond the window: run
    // the same state machine as the sequential path (bit-identical values;
    // a GPU kernel computes the same taps per token in parallel).
    let mut ks = vec![0.0f32; c * kd];
    let mut qs = vec![0.0f32; c * kd];
    let mut vs = vec![0.0f32; c * vd];
    let mut zs = vec![0.0f32; c * vd];
    let mut gl = vec![0.0f32; c * vh]; // per-(token, v-head) log decay
    let mut betas = vec![0.0f32; c * vh];
    let qs_scale = 1.0 / (hd as f32).sqrt();
    for (t, x) in xs.iter_mut().enumerate() {
        let xn = rms_norm(x, &lw.rms_attn, cfg.rms_eps);
        let qkv_pre = matvec_t(&xn, &lw.gdn_in_qkv, conv_dim);
        let z = matvec_t(&xn, &lw.gdn_in_z, vd);
        let a = matvec_t(&xn, &lw.gdn_in_a, vh);
        let b = matvec_t(&xn, &lw.gdn_in_b, vh);
        let mut qkv = vec![0.0f32; conv_dim];
        for ci in 0..conv_dim {
            let w = &lw.gdn_conv_w[ci * ck..ci * ck + ck];
            let mut s = 0.0;
            for j in 0..keep {
                s += conv[ci * keep + j] * w[j];
            }
            s += qkv_pre[ci] * w[keep];
            qkv[ci] = silu(s);
        }
        for ci in 0..conv_dim {
            for j in 0..keep - 1 {
                conv[ci * keep + j] = conv[ci * keep + j + 1];
            }
            conv[ci * keep + keep - 1] = qkv_pre[ci];
        }
        // split + per-head l2norm + q scale (same as sequential stages 3).
        for h in 0..kh {
            let q_dst = &mut qs[t * kd + h * hd..t * kd + (h + 1) * hd];
            q_dst.copy_from_slice(&qkv[h * hd..(h + 1) * hd]);
            let k_dst = &mut ks[t * kd + h * hd..t * kd + (h + 1) * hd];
            k_dst.copy_from_slice(&qkv[kd + h * hd..kd + (h + 1) * hd]);
            l2norm(q_dst);
            l2norm(k_dst);
            for v in q_dst.iter_mut() {
                *v *= qs_scale;
            }
        }
        vs[t * vd..(t + 1) * vd].copy_from_slice(&qkv[2 * kd..]);
        zs[t * vd..(t + 1) * vd].copy_from_slice(&z);
        for i in 0..vh {
            gl[t * vh + i] = -lw.gdn_a_log[i].exp() * softplus(a[i] + lw.gdn_dt_bias[i]);
            betas[t * vh + i] = 1.0 / (1.0 + (-b[i]).exp());
        }
    }

    // 3) chunked recurrence per v-head, then per-token gated norm.
    let rep = vh / kh;
    let mut os = vec![0.0f32; c * vd];
    let mut kh_buf = vec![0.0f32; c * hd];
    let mut qh_buf = vec![0.0f32; c * hd];
    let mut vh_buf = vec![0.0f32; c * hd];
    let mut oh_buf = vec![0.0f32; c * hd];
    let mut gl_buf = vec![0.0f32; c];
    let mut beta_buf = vec![0.0f32; c];
    for i in 0..vh {
        let kh_off = (i / rep) * hd;
        for t in 0..c {
            kh_buf[t * hd..(t + 1) * hd]
                .copy_from_slice(&ks[t * kd + kh_off..t * kd + kh_off + hd]);
            qh_buf[t * hd..(t + 1) * hd]
                .copy_from_slice(&qs[t * kd + kh_off..t * kd + kh_off + hd]);
            vh_buf[t * hd..(t + 1) * hd]
                .copy_from_slice(&vs[t * vd + i * hd..t * vd + (i + 1) * hd]);
            gl_buf[t] = gl[t * vh + i];
            beta_buf[t] = betas[t * vh + i];
        }
        gdn_core_chunk(
            &mut state[i * hd * hd..(i + 1) * hd * hd],
            &kh_buf,
            &qh_buf,
            &vh_buf,
            &gl_buf,
            &beta_buf,
            c,
            hd,
            &mut oh_buf,
        );
        for t in 0..c {
            os[t * vd + i * hd..t * vd + (i + 1) * hd]
                .copy_from_slice(&oh_buf[t * hd..(t + 1) * hd]);
        }
    }

    // 4) per-token gated RMSNorm + out projection + residual.
    for (t, x) in xs.iter_mut().enumerate() {
        let o = &mut os[t * vd..(t + 1) * vd];
        for i in 0..vh {
            let oh = &mut o[i * hd..(i + 1) * hd];
            let mut ss = 0.0;
            for &v in oh.iter() {
                ss += v * v;
            }
            let inv = 1.0 / (ss / hd as f32 + cfg.rms_eps).sqrt();
            let zh = &zs[t * vd + i * hd..t * vd + (i + 1) * hd];
            for (j, val) in oh.iter_mut().enumerate() {
                *val = *val * inv * lw.gdn_norm[j] * silu(zh[j]);
            }
        }
        let out = matvec_t(o, &lw.gdn_out, d);
        for i in 0..d {
            x[i] += out[i];
        }
    }
}

/// Per-v-head chunked delta-rule core — the `#112` Stage B recurrence and
/// the shared numeric oracle for the GPU `gdn_chunk` kernel. `s` is the
/// k-major `[hd, hd]` recurrent state (updated in place to `S_C`), `k`/`q`/
/// `v` are `[c, hd]` per-head rows (already l2-normed, q pre-scaled), `gl`
/// the per-token log-decay and `beta` the per-token gate. Writes the `[c,
/// hd]` pre-norm outputs `o`. Derivation in [`decode_chunk_gdn`].
#[allow(dead_code, clippy::needless_range_loop, clippy::too_many_arguments)]
pub(crate) fn gdn_core_chunk(
    s: &mut [f32],
    k: &[f32],
    q: &[f32],
    v: &[f32],
    gl: &[f32],
    beta: &[f32],
    c: usize,
    hd: usize,
    o: &mut [f32],
) {
    // Cumulative log decay within the chunk (Π_t = exp(lpi[t])).
    let mut lpi = vec![0.0f32; c];
    let mut acc = 0.0;
    for t in 0..c {
        acc += gl[t];
        lpi[t] = acc;
    }
    // K0[t] = S₀ᵀk_t, Q0[t] = S₀ᵀq_t (S is k-major [hd, hd]).
    let mut k0 = vec![0.0f32; c * hd];
    let mut q0 = vec![0.0f32; c * hd];
    for t in 0..c {
        for m in 0..hd {
            let mut sk = 0.0;
            let mut sq = 0.0;
            for kk in 0..hd {
                let s0 = s[kk * hd + m];
                sk += k[t * hd + kk] * s0;
                sq += q[t * hd + kk] * s0;
            }
            k0[t * hd + m] = sk;
            q0[t * hd + m] = sq;
        }
    }
    // W[t] = β_t (v_t − Π_t K0[t]);  A[t,s] = β_t (Π_t/Π_s)(k_s·k_t), s<t.
    let mut w = vec![0.0f32; c * hd];
    let mut a_mat = vec![0.0f32; c * c];
    for t in 0..c {
        let pi_t = lpi[t].exp();
        for m in 0..hd {
            w[t * hd + m] = beta[t] * (v[t * hd + m] - pi_t * k0[t * hd + m]);
        }
        for s in 0..t {
            let mut dot = 0.0;
            for kk in 0..hd {
                dot += k[s * hd + kk] * k[t * hd + kk];
            }
            a_mat[t * c + s] = beta[t] * (lpi[t] - lpi[s]).exp() * dot;
        }
    }
    // Forward substitution: u_t = W_t − Σ_{s<t} A[t,s] u_s.
    let mut u = vec![0.0f32; c * hd];
    for t in 0..c {
        for m in 0..hd {
            let mut acc = w[t * hd + m];
            for s in 0..t {
                acc -= a_mat[t * c + s] * u[s * hd + m];
            }
            u[t * hd + m] = acc;
        }
    }
    // S ← Π_C S₀ + Σ_s (Π_C/Π_s) k_s ⊗ u_s.
    let pi_c = lpi[c - 1].exp();
    for kk in 0..hd {
        for m in 0..hd {
            let mut acc = pi_c * s[kk * hd + m];
            for s2 in 0..c {
                acc += (lpi[c - 1] - lpi[s2]).exp() * k[s2 * hd + kk] * u[s2 * hd + m];
            }
            s[kk * hd + m] = acc;
        }
    }
    // o_t = Π_t Q0[t] + Σ_{s≤t} (Π_t/Π_s)(k_s·q_t) u_s  (inclusive).
    let mut b_mat = vec![0.0f32; c * c];
    for t in 0..c {
        for s in 0..=t {
            let mut dot = 0.0;
            for kk in 0..hd {
                dot += k[s * hd + kk] * q[t * hd + kk];
            }
            b_mat[t * c + s] = (lpi[t] - lpi[s]).exp() * dot;
        }
        for m in 0..hd {
            let mut acc = lpi[t].exp() * q0[t * hd + m];
            for s in 0..=t {
                acc += b_mat[t * c + s] * u[s * hd + m];
            }
            o[t * hd + m] = acc;
        }
    }
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
        apply_rope(&mut x, heads, 64, 64, pos, cfg);
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

    /// DeepSeek-V2 rotates ADJACENT coordinates, not split halves: its
    /// `apply_rotary_pos_emb` permutes `view(d//2, 2).transpose(4, 3)` before
    /// `rotate_half`, and transformers' current builtin rotates
    /// `view_as_complex(x.reshape(-1, 2))`. Both pair `2d` with `2d+1`.
    #[test]
    fn apply_rope_interleave_pairs_adjacent_coordinates() {
        let mut cfg = yarn_cfg(64);
        cfg.rope_interleave = true;
        let pos = 7usize;
        let (hf, _) = hf_yarn_cos_sin(cfg, pos);
        let heads = 2usize;
        let half = 32usize;
        let mut x: Vec<f32> = (0..heads * 64)
            .map(|i| ((i as f32) * 0.37).sin() * 3.0)
            .collect();
        let orig = x.clone();
        apply_rope(&mut x, heads, 64, 64, pos, cfg);
        for h in 0..heads {
            for d in 0..half {
                let (a, b) = (orig[h * 64 + 2 * d], orig[h * 64 + 2 * d + 1]);
                let (c, s) = (hf.cos[d], hf.sin[d]);
                let (want_a, want_b) = (a * c - b * s, b * c + a * s);
                let (got_a, got_b) = (x[h * 64 + 2 * d], x[h * 64 + 2 * d + 1]);
                let tol = 1e-3 + 1e-3 * want_a.abs().max(want_b.abs());
                assert!(
                    (got_a - want_a).abs() <= tol && (got_b - want_b).abs() <= tol,
                    "h={h} d={d}: got ({got_a}, {got_b}) want ({want_a}, {want_b})"
                );
            }
        }
    }

    /// The two conventions must be INDISTINGUISHABLE at `pos == 0` — there
    /// cos=1 and sin=0 make RoPE the identity, so a first-token-only diff
    /// silently passes while every later position is rotated wrongly. This is
    /// the trap that let the DeepSeek pairing bug through: pin it, and assert
    /// the opposite direction too, that pos > 0 really does separate them.
    #[test]
    fn rope_conventions_agree_at_pos_zero_and_differ_after() {
        let mut interleave = yarn_cfg(64);
        interleave.rope_interleave = true;
        let halves = yarn_cfg(64);
        let x0: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.41).sin() * 2.0).collect();

        let a0 = {
            let mut v = x0.clone();
            apply_rope(&mut v, 1, 64, 64, 0, interleave);
            v
        };
        let b0 = {
            let mut v = x0.clone();
            apply_rope(&mut v, 1, 64, 64, 0, halves);
            v
        };
        let d0: f32 = a0
            .iter()
            .zip(&b0)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(d0 < 1e-6, "pos 0 must agree, got max diff {d0:.3e}");

        let a1 = {
            let mut v = x0.clone();
            apply_rope(&mut v, 1, 64, 64, 5, interleave);
            v
        };
        let b1 = {
            let mut v = x0.clone();
            apply_rope(&mut v, 1, 64, 64, 5, halves);
            v
        };
        let d1: f32 = a1
            .iter()
            .zip(&b1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            d1 > 1e-2,
            "pos 5 must differ (pairing is only observable at pos > 0), got {d1:.3e}"
        );
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

    // ---- Qwen3.5 partial rotary + gated DeltaNet ----

    /// `attn_rotary_dim` derives the rotating width from `rope_rotary_pct`:
    /// 0.25 of 256 -> 64, full -> head_dim, odd products rounded down to a
    /// pair-friendly even width.
    #[test]
    fn attn_rotary_dim_from_pct() {
        let mut cfg = Config::tiny();
        cfg.head_dim = 256;
        cfg.rope_rotary_pct = 0.25;
        assert_eq!(cfg.attn_rotary_dim(), 64);
        cfg.rope_rotary_pct = 1.0;
        assert_eq!(cfg.attn_rotary_dim(), 256);
        // 0.5 * 31 = 15.5 -> 15 & !1 = 14.
        cfg.head_dim = 31;
        cfg.rope_rotary_pct = 0.5;
        assert_eq!(cfg.attn_rotary_dim(), 14);
    }

    /// Partial rotary (Qwen3.5 full-attn layers): only the first
    /// `rotary_dim` coordinates rotate, with the frequency basis taken over
    /// `rotary_dim` — NOT `head_dim`. A basis bug is invisible at pos 0 and
    /// at full width, so both the tail pass-through and the basis are pinned
    /// at pos > 0 with `rotary_dim < head_dim`.
    #[test]
    fn partial_rope_rotates_prefix_and_passes_tail() {
        let mut cfg = Config::tiny();
        cfg.rope_theta = 10_000.0;
        let head_dim = 8usize;
        let rotary = 4usize;
        let pos = 3usize;
        let mut x: Vec<f32> = (0..head_dim).map(|i| 0.3 + 0.11 * i as f32).collect();
        let orig = x.clone();
        apply_rope(&mut x, 1, head_dim, rotary, pos, cfg);

        let basis = |d: usize| -> f64 { 10_000f64.powf(-(2.0 * d as f64) / rotary as f64) };
        for d in 0..rotary / 2 {
            let ang = pos as f64 * basis(d);
            let (c, s) = (ang.cos(), ang.sin());
            // Half-split pairing INSIDE the rotating prefix: (d, d+rotary/2).
            let (a, b) = (orig[d] as f64, orig[d + rotary / 2] as f64);
            let want_a = a * c - b * s;
            let want_b = b * c + a * s;
            assert!(
                ((x[d] as f64 - want_a).abs() < 1e-5)
                    && ((x[d + rotary / 2] as f64 - want_b).abs() < 1e-5),
                "pair {d}: got ({}, {}) want ({want_a}, {want_b})",
                x[d],
                x[d + rotary / 2]
            );
        }
        // Tail passes through untouched.
        for i in rotary..head_dim {
            assert_eq!(x[i], orig[i], "tail dim {i} must pass through");
        }
    }

    /// `[n, n]` identity matrix, row-major.
    fn identity_mat(n: usize) -> Vec<f32> {
        let mut m = vec![0.0f32; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    }

    /// Builds the minimal 1x1-head GDN layer for the hand-computed recurrence
    /// test: identity qkv projection, asymmetric conv taps `[1, 0, 2]`,
    /// A_log = 0 (gate = -softplus(dt_bias)), dt_bias = 1, b-rows zero
    /// (beta = 0.5), unit norm weight, out rows dropping o into x[0], x[1].
    fn gdn_single_head_case() -> (Config, LayerWeights) {
        let mut cfg = Config::tiny();
        cfg.d_model = 6;
        cfg.gdn_k_heads = 1;
        cfg.gdn_v_heads = 1;
        cfg.gdn_head_dim = 2;
        cfg.gdn_conv_kernel = 4;
        let lw = LayerWeights {
            wq: Vec::new(),
            wk: Vec::new(),
            wv: Vec::new(),
            wo: Vec::new(),
            rms_attn: vec![1.0; 6],
            wg: Vec::new(),
            wu: Vec::new(),
            wd: Vec::new(),
            rms_mlp: vec![1.0; 6],
            bq: Vec::new(),
            bk: Vec::new(),
            bv: Vec::new(),
            q_norm: Vec::new(),
            k_norm: Vec::new(),
            mla_q_a: Vec::new(),
            mla_q_a_norm: Vec::new(),
            mla_q_b: Vec::new(),
            mla_q_rope: Vec::new(),
            mla_kv_a: Vec::new(),
            mla_kv_a_norm: Vec::new(),
            mla_kv_b: Vec::new(),
            mla_o: Vec::new(),
            moe_router: Vec::new(),
            moe_wg: Vec::new(),
            moe_wu: Vec::new(),
            moe_wd: Vec::new(),
            shared_wg: Vec::new(),
            shared_wu: Vec::new(),
            shared_wd: Vec::new(),
            gdn_in_qkv: identity_mat(6),
            gdn_in_z: {
                let mut m = vec![0.0f32; 2 * 6];
                m[0] = 1.0; // z[0] = x[0]
                m[6 + 1] = 1.0; // z[1] = x[1]
                m
            },
            gdn_in_a: vec![0.0; 6],
            gdn_in_b: vec![0.0; 6],
            gdn_conv_w: {
                // Per-channel taps [oldest .. newest] = [0.5, 0.25, 1, 2].
                // Step 1 (zero state) sees only the newest tap; step 2 sees
                // the RAW pre-conv step-1 input through tap 1 — a wrong tap
                // order, a post-silu state, or a missing window shift all
                // change the output.
                let mut w = vec![0.0f32; 6 * 4];
                for c in 0..6 {
                    w[c * 4] = 0.5;
                    w[c * 4 + 1] = 0.25;
                    w[c * 4 + 2] = 1.0;
                    w[c * 4 + 3] = 2.0;
                }
                w
            },
            gdn_a_log: vec![0.0], // exp(A_log) = 1
            gdn_dt_bias: vec![1.0],
            gdn_norm: vec![1.0; 2],
            gdn_out: {
                // `[6, 2]` row-major: o lands in x[0], x[1].
                let mut m = vec![0.0f32; 6 * 2];
                m[0] = 1.0; // row 0, col 0
                m[3] = 1.0; // row 1, col 1
                m
            },
        };
        (cfg, lw)
    }

    /// Hand-computed two-token GDN recurrence in f64, structurally separate
    /// from `decode_step_gdn` (per-scalar math instead of row loops): conv
    /// tap order + RAW pre-conv state, decay, the delta rule with kv_mem, o
    /// reading the UPDATED S, and the gated norm. `rms_attn` is unit weights
    /// but still divides by the RMS, so the expected side works on the
    /// normalized vector `u = x / rms(x)` throughout.
    #[test]
    fn gdn_recurrence_matches_hand_computation() {
        let (cfg, lw) = gdn_single_head_case();
        let eps = cfg.rms_eps as f64;

        let softplus64 = |v: f64| (1.0 + v.exp()).ln();
        let silu64 = |v: f64| v / (1.0 + (-v).exp());
        // Unit-weight RMS norm in f64 (mirrors what the impl feeds the layer).
        let unit_rms = |x: &[f64]| -> Vec<f64> {
            let ss: f64 = x.iter().map(|v| v * v).sum();
            let inv = 1.0 / (ss / x.len() as f64 + eps).sqrt();
            x.iter().map(|v| v * inv).collect()
        };

        let x1: Vec<f64> = vec![0.6, -0.4, 0.9, 0.2, -0.7, 0.5];
        let x2: Vec<f64> = vec![-0.3, 0.8, -0.1, 0.4, 0.55, -0.25];
        let u1 = unit_rms(&x1);
        let u2 = unit_rms(&x2);

        // Gate values shared by both steps: g = -exp(0) * softplus(0 + 1).
        let g = -softplus64(1.0);
        let decay = g.exp();
        let beta = 0.5; // sigmoid(0)

        // ---- step 1 (expected values) ----
        // Conv taps [1, 0, 2] over [0, 0, u1]: out = silu(2 * u1).
        let qkv1: Vec<f64> = u1.iter().map(|&v| silu64(2.0 * v)).collect();
        let norm2 = |v: &[f64]| -> Vec<f64> {
            let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
            vec![v[0] / n, v[1] / n]
        };
        let q1 = norm2(&qkv1[0..2]);
        let k1 = norm2(&qkv1[2..4]);
        let v1: Vec<f64> = qkv1[4..6].to_vec();
        let qs = 1.0 / 2f64.sqrt();
        let q1s = [q1[0] * qs, q1[1] * qs];
        // S1 = k1 (x) (v1 * beta)   (kv_mem = 0 on a zero state).
        let s1 = |kk: usize, vv: usize| k1[kk] * v1[vv] * beta;
        // o1[vv] = sum_kk S1[kk, vv] * q = (k.q) * v1[vv] * beta.
        let kq1 = k1[0] * q1s[0] + k1[1] * q1s[1];
        let o1 = [kq1 * v1[0] * beta, kq1 * v1[1] * beta];
        // Gated norm (w = 1): o / rms(o) * silu(z), z = (u1[0], u1[1]).
        let rms1 = ((o1[0] * o1[0] + o1[1] * o1[1]) / 2.0 + eps).sqrt();
        let on1 = [o1[0] / rms1 * silu64(u1[0]), o1[1] / rms1 * silu64(u1[1])];

        // ---- step 2 (expected values) ----
        // Conv state holds RAW u1 (NOT silu(2*u1)) — pins what the state
        // stores. out = silu(1*u1 + 0 + 2*u2).
        let qkv2: Vec<f64> = (0..6).map(|c| silu64(u1[c] * 1.0 + u2[c] * 2.0)).collect();
        let q2 = norm2(&qkv2[0..2]);
        let k2 = norm2(&qkv2[2..4]);
        let v2: Vec<f64> = qkv2[4..6].to_vec();
        let q2s = [q2[0] * qs, q2[1] * qs];
        // S' = decay * S1; kv_mem = S'^T k2; delta = (v2 - kv_mem) * beta.
        let kmem = [
            s1(0, 0) * decay * k2[0] + s1(1, 0) * decay * k2[1],
            s1(0, 1) * decay * k2[0] + s1(1, 1) * decay * k2[1],
        ];
        let delta2 = [(v2[0] - kmem[0]) * beta, (v2[1] - kmem[1]) * beta];
        // S2 = S' + k2 (x) delta2; o2 = S2^T q2 (UPDATED S).
        let s2 = |kk: usize, vv: usize| s1(kk, vv) * decay + k2[kk] * delta2[vv];
        let o2 = [
            s2(0, 0) * q2s[0] + s2(1, 0) * q2s[1],
            s2(0, 1) * q2s[0] + s2(1, 1) * q2s[1],
        ];
        let rms2 = ((o2[0] * o2[0] + o2[1] * o2[1]) / 2.0 + eps).sqrt();
        let on2 = [o2[0] / rms2 * silu64(u2[0]), o2[1] / rms2 * silu64(u2[1])];

        // ---- run the implementation over both steps ----
        let mut state = vec![0.0f32; 4];
        let mut conv = vec![0.0f32; 6 * 3];
        let mut xr = vec![0.0f32; 6];
        let mut run_step = |x_in: &[f64], state: &mut Vec<f32>, conv: &mut Vec<f32>| {
            for i in 0..6 {
                xr[i] = x_in[i] as f32;
            }
            // Same unit-weight RMS norm the layer sees (see `unit_rms`).
            let xn = rms_norm(&xr, &lw.rms_attn, cfg.rms_eps);
            decode_step_gdn(&mut xr, &xn, &lw, state, conv, cfg);
            (xr[0] as f64, xr[1] as f64)
        };
        let (got1a, got1b) = run_step(&x1, &mut state, &mut conv);
        let (got2a, got2b) = run_step(&x2, &mut state, &mut conv);

        // The layer output is ADDED to the residual: xr[i] = x[i] + o[i].
        for (i, (got, want)) in [
            (0usize, (got1a, x1[0] + on1[0])),
            (1, (got1b, x1[1] + on1[1])),
        ] {
            assert!(
                (got - want).abs() < 1e-5 * (1.0 + want.abs()),
                "step1 o[{i}]: got {got} want {want}"
            );
        }
        for (i, (got, want)) in [
            (0usize, (got2a, x2[0] + on2[0])),
            (1, (got2b, x2[1] + on2[1])),
        ] {
            assert!(
                (got - want).abs() < 1e-5 * (1.0 + want.abs()),
                "step2 o[{i}]: got {got} want {want}"
            );
        }
    }

    /// softplus is linear above the standard threshold 20 and exact
    /// log1p(exp(x)) below it.
    #[test]
    fn softplus_threshold_behavior() {
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-6);
        assert!((softplus(1.0) - 1.3132616).abs() < 1e-6);
        assert!((softplus(25.0) - 25.0).abs() < 1e-5, "linear above 20");
        assert!((softplus(-30.0) - 0.0).abs() < 1e-6);
    }

    /// Chunked GDN layer (`decode_chunk_gdn`, the #112 Stage B oracle) must
    /// reproduce the sequential `decode_step_gdn` over the same token rows:
    /// after a warmup (nonzero recurrent/conv state), C rows through both
    /// paths must agree on every output row and the final recurrent state
    /// (f32 reassociation tolerance — the chunk form reorders the
    /// recurrence's sums); the conv window is a fixed per-token tap dot and
    /// must match bit-exactly.
    #[test]
    fn gdn_chunk_layer_matches_sequential() {
        let cfg = Config::qwen3_5(64, 5, 4, 2, 16, 176, 97, 64, 2, 4, 8, 4);
        let w = Weights::random(&cfg, 61).unwrap();
        let lw = &w.layers[0]; // GDN layer (layer 3 is the full-attention one)
        let d = cfg.d_model;
        let kd = cfg.gdn_key_dim();
        let vd = cfg.gdn_value_dim();
        let conv_dim = 2 * kd + vd;
        let keep = cfg.gdn_conv_kernel - 1;
        let state_len = cfg.gdn_v_heads * cfg.gdn_head_dim * cfg.gdn_head_dim;

        // Deterministic varied rows: x[t][i] in [-1, 1).
        let row = |t: usize| -> Vec<f32> {
            (0..d)
                .map(|i| (((t * 7 + i * 13 + 5) % 21) as f32) / 7.0 - 1.0)
                .collect()
        };
        let warmup: Vec<Vec<f32>> = (0..3).map(row).collect();
        let chunk_rows: Vec<Vec<f32>> = (3..11).map(row).collect();

        // Shared warmup leaves both paths at the same (nonzero) state.
        let mut state_w = vec![0.0f32; state_len];
        let mut conv_w = vec![0.0f32; conv_dim * keep];
        for x in &warmup {
            let mut xr = x.clone();
            let xn = rms_norm(&xr, &lw.rms_attn, cfg.rms_eps);
            decode_step_gdn(&mut xr, &xn, lw, &mut state_w, &mut conv_w, cfg);
        }

        // Path A: the same rows one at a time.
        let mut state_a = state_w.clone();
        let mut conv_a = conv_w.clone();
        let mut outs_a = Vec::new();
        for x in &chunk_rows {
            let mut xr = x.clone();
            let xn = rms_norm(&xr, &lw.rms_attn, cfg.rms_eps);
            decode_step_gdn(&mut xr, &xn, lw, &mut state_a, &mut conv_a, cfg);
            outs_a.push(xr);
        }

        // Path B: the same rows as ONE chunk.
        let mut state_b = state_w.clone();
        let mut conv_b = conv_w.clone();
        let mut chunk_b = chunk_rows.clone();
        decode_chunk_gdn(&mut chunk_b, lw, &mut state_b, &mut conv_b, cfg);

        assert_eq!(conv_a, conv_b, "conv window must match bit-exactly");
        for (t, (a, b)) in outs_a.iter().zip(&chunk_b).enumerate() {
            for i in 0..d {
                let (av, bv) = (a[i], b[i]);
                assert!(
                    (av - bv).abs() <= 1e-4 * (1.0 + av.abs()),
                    "row {t} dim {i}: seq {av:?} chunk {bv:?}"
                );
            }
        }
        for (ia, ib) in state_a.iter().zip(&state_b) {
            assert!(
                (ia - ib).abs() <= 1e-4 * (1.0 + ia.abs()),
                "state: seq {ia:?} chunk {ib:?}"
            );
        }
    }

    /// Hand-computed f64 transcription of the Qwen3.5 gated full-attention
    /// forward at position 0 — where softmax over the single stored position
    /// is exactly 1 and RoPE is the identity, so the comparison isolates the
    /// new machinery: the doubled q_proj split `[query | gate]` per head and
    /// the sigmoid scaling between attention and o_proj. One layer, one
    /// token; QK-norm/RoPE affect only q/k, which the position-0 output does
    /// not read (they are pinned by the rope/qk-norm tests).
    #[test]
    fn gated_attention_step_matches_hand_computation() {
        // gdn_v_heads = 0 -> full_attention_interval = 0 -> every layer full
        // attention; the family flags (attn_output_gate, zero_centered_norm,
        // qk_norm) stay on.
        let cfg = Config::qwen3_5(16, 1, 2, 1, 8, 12, 11, 8, 0, 0, 4, 4);
        let w = Weights::random(&cfg, 3).unwrap();
        let mut m = RefModel::new(cfg, w.clone());
        let got = m.decode_step(7);

        let d = cfg.d_model;
        let hd = cfg.head_dim;
        let heads = cfg.n_heads;
        let kvh = cfg.n_kv_heads;
        let lw = &w.layers[0];

        // mat @ x in f64, mat `[rows, len(x)]` row-major.
        let matvec64 = |mat: &[f32], x: &[f64], rows: usize| -> Vec<f64> {
            let n = x.len();
            (0..rows)
                .map(|r| (0..n).map(|i| mat[r * n + i] as f64 * x[i]).sum())
                .collect()
        };
        let rms64 = |x: &[f64], wgt: &[f32]| -> Vec<f64> {
            let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
            let inv = 1.0 / (ms + cfg.rms_eps as f64).sqrt();
            x.iter()
                .zip(wgt)
                .map(|(&v, &g)| v * inv * g as f64)
                .collect()
        };

        // Layer 0, position 0.
        let x0: Vec<f64> = w.tok_emb[7 * d..8 * d].iter().map(|&v| v as f64).collect();
        let xn = rms64(&x0, &lw.rms_attn);
        let qg = matvec64(&lw.wq, &xn, heads * hd * 2);
        let mut gate = vec![0.0f64; heads * hd];
        for h in 0..heads {
            for j in 0..hd {
                // Per-head chunk: query = first half of each head's 2*hd
                // block, gate = second half.
                gate[h * hd + j] = qg[h * 2 * hd + hd + j];
            }
        }
        let v = matvec64(&lw.wv, &xn, kvh * hd);
        // Attention over one stored position: softmax([s]) = [1], so the
        // output is the (group-broadcast) value — then the gate scales it.
        let groups = heads / kvh;
        let mut out = vec![0.0f64; heads * hd];
        for h in 0..heads {
            let kv = h / groups;
            for j in 0..hd {
                let s = 1.0 / (1.0 + (-gate[h * hd + j]).exp());
                out[h * hd + j] = v[kv * hd + j] * s;
            }
        }
        let mut x = x0;
        let proj = matvec64(&lw.wo, &out, d);
        for i in 0..d {
            x[i] += proj[i];
        }
        // Dense SwiGLU MLP.
        let silu64 = |v: f64| v / (1.0 + (-v).exp());
        let xn2 = rms64(&x, &lw.rms_mlp);
        let gi = matvec64(&lw.wg, &xn2, cfg.intermediate_size);
        let ui = matvec64(&lw.wu, &xn2, cfg.intermediate_size);
        let hi: Vec<f64> = (0..cfg.intermediate_size)
            .map(|i| silu64(gi[i]) * ui[i])
            .collect();
        let down = matvec64(&lw.wd, &hi, d);
        for i in 0..d {
            x[i] += down[i];
        }
        let xf = rms64(&x, &w.rms_final);
        let want = matvec64(&w.lm_head, &xf, cfg.vocab_size);

        for (i, (g, wn)) in got.iter().zip(&want).enumerate() {
            assert!(
                (*g as f64 - wn).abs() < 1e-3 * (1.0 + wn.abs()),
                "logit {i}: got {g} want {wn}"
            );
        }
    }
}
