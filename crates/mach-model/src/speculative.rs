//! Speculative decoding for a single sequence.
//!
//! A small draft model proposes `k` tokens; the target model verifies them in
//! ONE batched forward (`k+1` rows: the last accepted token at position
//! `len-1`, then the drafts at `len..len+k-1`). Argmax acceptance keeps the
//! output **identical to plain greedy** on the target: the longest prefix of
//! drafts matching the target's own argmax is accepted, and the next token is
//! the target's argmax at the rejection point (or after the last draft).
//!
//! The algorithm is validated by simulation (`docs/roadmap.md` P3al) and by
//! the tiny-model parity test in `tests/spec_decode.rs`.

use crate::Error;
use crate::batched::BatchedModel;
use crate::sampling::SamplingParams;

/// Prefills `prompt` into slot `slot` of `m`, chunked to the model's row
/// capacity.
fn prefill_chunked(m: &mut BatchedModel, slot: u32, prompt: &[u32]) -> Result<(), Error> {
    let rows = m.row_capacity();
    let mut pos = 0usize;
    while pos < prompt.len() {
        let take = (prompt.len() - pos).min(rows);
        let lens: Vec<u32> = (0..take as u32)
            .map(|i| (pos + i as usize) as u32)
            .collect();
        let slots = vec![slot; take];
        let mut p = vec![SamplingParams::greedy(0); take];
        let ec: Vec<Vec<(u32, u32)>> = vec![Vec::new(); take];
        let eb: Vec<Vec<(u32, f32)>> = vec![Vec::new(); take];
        m.decode_step_explicit(&prompt[pos..pos + take], &lens, &slots, &mut p, &ec, &eb)?;
        pos += take;
    }
    Ok(())
}

/// Single-sequence speculative decoder (draft + target sharing one vocab).
#[allow(clippy::len_without_is_empty)]
pub struct SpeculativeDecoder {
    draft: BatchedModel,
    target: BatchedModel,
    /// Draft tokens proposed per verify round.
    k: usize,
    /// Accepted context length (prompt + accepted tokens).
    len: usize,
    /// Token at position `len - 1` (the last accepted token).
    draft_last: u32,
}

impl SpeculativeDecoder {
    /// Prefills both models with `prompt`; `k` draft tokens per round.
    pub fn new(
        mut draft: BatchedModel,
        mut target: BatchedModel,
        k: usize,
        prompt: &[u32],
    ) -> Result<Self, Error> {
        assert!(!prompt.is_empty(), "prompt must not be empty");
        assert!(k >= 1, "k must be >= 1");
        prefill_chunked(&mut target, 0, prompt)?;
        prefill_chunked(&mut draft, 0, prompt)?;
        Ok(Self {
            draft,
            target,
            k,
            len: prompt.len(),
            draft_last: *prompt.last().expect("non-empty"),
        })
    }

    /// One speculative round: draft `k` tokens, verify on the target, accept
    /// the longest matching prefix. Returns the accepted tokens (>= 1), so the
    /// caller accumulates until it has enough.
    pub fn step(&mut self) -> Result<Vec<u32>, Error> {
        let l = self.len;
        // 1. Draft k tokens: c[i] is the draft's guess for position l+i.
        let mut c: Vec<u32> = Vec::with_capacity(self.k);
        for i in 0..self.k {
            let input = if i == 0 { self.draft_last } else { c[i - 1] };
            let pos = (l - 1 + i) as u32;
            let mut p = [SamplingParams::greedy(0)];
            let out = self.draft.decode_step_explicit(
                &[input],
                &[pos],
                &[0],
                &mut p,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
            )?;
            c.push(out.0[0]);
        }
        // 2. Verify k+1 rows on the target: [draft_last, c[0..k-1]] at
        //    positions [l-1 .. l+k-1]; pred[i] = target's guess for position
        //    l+i (pred[0] = fresh next-token prediction).
        let mut rows = Vec::with_capacity(self.k + 1);
        rows.push(self.draft_last);
        rows.extend_from_slice(&c);
        let rlens: Vec<u32> = (0..=self.k as u32).map(|i| l as u32 - 1 + i).collect();
        let mut vp = vec![SamplingParams::greedy(0); self.k + 1];
        let vout = self.target.decode_step_explicit(
            &rows,
            &rlens,
            &vec![0u32; self.k + 1],
            &mut vp,
            &vec![Vec::new(); self.k + 1],
            &vec![Vec::new(); self.k + 1],
        )?;
        let pred = vout.0;
        // 3. Accept the longest prefix c[0..a-1] matching the target argmax.
        let mut a = 0usize;
        while a < self.k && pred[a] == c[a] {
            a += 1;
        }
        let next = pred[a];
        // 4. Advance the draft context with c[0..a] + next.
        let mut advance = c[..a].to_vec();
        advance.push(next);
        for (j, &t) in advance.iter().enumerate() {
            let mut p = [SamplingParams::greedy(0)];
            self.draft.decode_step_explicit(
                &[t],
                &[(l + j) as u32],
                &[0],
                &mut p,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
            )?;
        }
        self.len += advance.len();
        self.draft_last = *advance.last().expect("non-empty");
        Ok(advance)
    }

    /// Current accepted context length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }
}

/// Batched (multi-sequence) speculative decoding.
///
/// One shared draft model + one shared target model; each sequence keeps its
/// own context (draft_last, len). A spec round:
///   1. draft: k batched forwards on the draft, one row per active sequence;
///   2. verify: one batched forward on the target with `active*(k+1)` rows
///      ([draft_last, c[0..k-1]] per sequence at its positions);
///   3. accept per sequence (argmax, longest matching prefix) — output stays
///      identical to plain greedy.
///
/// The target model must be built with `with_rows(capacity, capacity*(k+1))`
/// so the verify rows fit; the draft needs `rows >= capacity`.
pub struct SpeculativeBatch {
    draft: BatchedModel,
    target: BatchedModel,
    k: usize,
    /// Slots / capacity (one per sequence).
    capacity: usize,
    /// Per-sequence accepted context length.
    lens: Vec<usize>,
    /// Per-sequence token at position `len - 1`.
    draft_last: Vec<u32>,
    /// Whether each sequence is still decoding (finished ones are skipped).
    active: Vec<bool>,
}

impl SpeculativeBatch {
    /// Builds a batched speculative decoder for up to `capacity` sequences.
    pub fn new(draft: BatchedModel, target: BatchedModel, k: usize, capacity: usize) -> Self {
        assert!(k >= 1, "k must be >= 1");
        assert!(capacity >= 1, "capacity must be >= 1");
        Self {
            draft,
            target,
            k,
            capacity,
            lens: Vec::new(),
            draft_last: Vec::new(),
            active: Vec::new(),
        }
    }

    /// Prefills a new sequence (slot = current active count) in both models.
    pub fn add(&mut self, prompt: &[u32]) -> Result<(), Error> {
        assert!(!prompt.is_empty(), "prompt must not be empty");
        assert!(self.lens.len() < self.capacity, "capacity reached");
        let slot = self.lens.len();
        prefill_chunked(&mut self.target, slot as u32, prompt)?;
        prefill_chunked(&mut self.draft, slot as u32, prompt)?;
        self.lens.push(prompt.len());
        self.draft_last.push(*prompt.last().expect("non-empty"));
        self.active.push(true);
        Ok(())
    }

    /// Number of sequences still decoding.
    #[must_use]
    pub fn active(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }

    /// Marks sequence `s` as finished; it is skipped in later rounds.
    pub fn finish(&mut self, s: usize) {
        self.active[s] = false;
    }

    /// Whether sequence `s` is still decoding.
    #[must_use]
    pub fn is_active(&self, s: usize) -> bool {
        self.active[s]
    }

    /// One speculative round for the sequences still decoding; returns the
    /// accepted tokens per sequence, index-aligned (`None` = finished).
    #[allow(clippy::needless_range_loop)] // parallel per-sequence arrays
    pub fn step(&mut self) -> Result<Vec<Option<Vec<u32>>>, Error> {
        let n = self.lens.len();
        let active_idx: Vec<usize> = (0..n).filter(|&s| self.active[s]).collect();
        let m = active_idx.len();
        if m == 0 {
            return Ok(vec![None; n]);
        }
        // 1. Draft k tokens per active sequence.
        let mut c: Vec<Vec<u32>> = vec![Vec::with_capacity(self.k); n];
        for i in 0..self.k {
            let mut toks = Vec::with_capacity(m);
            let mut lens = Vec::with_capacity(m);
            let mut slots = Vec::with_capacity(m);
            let mut p = Vec::with_capacity(m);
            for &s in &active_idx {
                let input = if i == 0 {
                    self.draft_last[s]
                } else {
                    c[s][i - 1]
                };
                toks.push(input);
                lens.push((self.lens[s] - 1 + i) as u32);
                slots.push(s as u32);
                p.push(SamplingParams::greedy(0));
            }
            let ec: Vec<Vec<(u32, u32)>> = vec![Vec::new(); m];
            let eb: Vec<Vec<(u32, f32)>> = vec![Vec::new(); m];
            let out = self
                .draft
                .decode_step_explicit(&toks, &lens, &slots, &mut p, &ec, &eb)?;
            for (si, &s) in active_idx.iter().enumerate() {
                c[s].push(out.0[si]);
            }
        }
        // 2. Verify: `m * (k+1)` rows on the target.
        let mut toks = Vec::with_capacity(m * (self.k + 1));
        let mut lens = Vec::with_capacity(m * (self.k + 1));
        let mut slots = Vec::with_capacity(m * (self.k + 1));
        let mut p = Vec::with_capacity(m * (self.k + 1));
        for &s in &active_idx {
            for j in 0..=self.k {
                let input = if j == 0 {
                    self.draft_last[s]
                } else {
                    c[s][j - 1]
                };
                toks.push(input);
                lens.push((self.lens[s] - 1 + j) as u32);
                slots.push(s as u32);
                p.push(SamplingParams::greedy(0));
            }
        }
        let ec: Vec<Vec<(u32, u32)>> = vec![Vec::new(); m * (self.k + 1)];
        let eb: Vec<Vec<(u32, f32)>> = vec![Vec::new(); m * (self.k + 1)];
        let out = self
            .target
            .decode_step_explicit(&toks, &lens, &slots, &mut p, &ec, &eb)?;
        // pred per active sequence: pred[si][j] = guess for position
        // len[s] + j.
        let mut accepted: Vec<Option<Vec<u32>>> = vec![None; n];
        for (si, &s) in active_idx.iter().enumerate() {
            let base = si * (self.k + 1);
            let mut a = 0usize;
            while a < self.k && out.0[base + a] == c[s][a] {
                a += 1;
            }
            let next = out.0[base + a];
            let mut seq = c[s][..a].to_vec();
            seq.push(next);
            accepted[s] = Some(seq);
        }
        // 3. Advance the draft context with the accepted tokens, position by
        //    position (<= active rows per call, fitting the draft capacity).
        for j in 0..=self.k {
            let mut toks = Vec::new();
            let mut lens = Vec::new();
            let mut slots = Vec::new();
            for &s in &active_idx {
                if let Some(seq) = accepted[s].as_ref()
                    && let Some(&t) = seq.get(j)
                {
                    toks.push(t);
                    lens.push((self.lens[s] + j) as u32);
                    slots.push(s as u32);
                }
            }
            if toks.is_empty() {
                continue;
            }
            let mut p = vec![SamplingParams::greedy(0); toks.len()];
            let ec: Vec<Vec<(u32, u32)>> = vec![Vec::new(); toks.len()];
            let eb: Vec<Vec<(u32, f32)>> = vec![Vec::new(); toks.len()];
            self.draft
                .decode_step_explicit(&toks, &lens, &slots, &mut p, &ec, &eb)?;
        }
        for &s in &active_idx {
            let seq = accepted[s].as_ref().expect("active produced tokens");
            self.lens[s] += seq.len();
            self.draft_last[s] = *seq.last().expect("non-empty");
        }
        Ok(accepted)
    }
}
