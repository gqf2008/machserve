//! GPU-side sampling.
//!
//! Two samplers:
//! - [`HipSampler`]: single-sequence greedy argmax (P1 path). Reduces on-device
//!   so only a 4-byte token id is read back instead of all `vocab` logits.
//! - [`BatchedSampler`]: batched top-k / top-p / temperature sampling with a
//!   deterministic per-sequence RNG (SplitMix64), used by the batched and
//!   continuous engines. One HIP block per sequence row; the RNG seed is
//!   advanced exactly once per sampled step, so a CPU reference using the same
//!   seeds reproduces the same draws (see [`sample_cpu`]).
//!
//! Sampling semantics follow HuggingFace `LogitsProcessor`: temperature is
//! applied to the logits, then top-k keeps the k-th largest *logit* threshold
//! (ties included), then top-p keeps the cumulative threshold (boundary tier
//! included), then a single uniform draw walks the allowed set.

use crate::Error;
use mach_engine::hip::hip_arch;
use mach_kernel_sys::hip::{self, Hip, HipKernelModule, HipStream};
use std::sync::Arc;

/// SplitMix64 golden-ratio increment (state advance per sampled step).
const RNG_GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 finalizer (without the state increment; see [`advance_seed`]).
#[must_use]
pub fn mix64(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Advances an RNG state by one step (mirrors the kernel write-back).
#[must_use]
pub const fn advance_seed(state: u64) -> u64 {
    state.wrapping_add(RNG_GOLDEN)
}

/// Draws `u ~ U[0, 1)` from `state` and advances it by one step.
pub fn next_f32(state: &mut u64) -> f32 {
    let z = mix64(*state);
    *state = advance_seed(*state);
    ((z >> 40) as f32) / 16_777_216.0
}

/// Per-sequence sampling configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// Temperature; `<= 0` selects greedy (argmax).
    pub temperature: f32,
    /// Top-k; `0` disables the filter.
    pub top_k: usize,
    /// Top-p (nucleus); `>= 1.0` disables the filter.
    pub top_p: f32,
    /// RNG seed; advanced once per sampled step.
    pub seed: u64,
    /// Presence penalty: subtract `presence_penalty` from a token's logit
    /// whenever it has appeared (OpenAI `presence_penalty`).
    pub presence_penalty: f32,
    /// Frequency penalty: subtract `frequency_penalty * count` from a token's
    /// logit for each prior occurrence (OpenAI `frequency_penalty`).
    pub frequency_penalty: f32,
    /// Report top-`top_logprobs` tokens + log-probs per sampled token (OpenAI
    /// `logprobs.top_logprobs`); `0` disables.
    pub top_logprobs: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        // Greedy, preserving the pre-P3 engine behavior.
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            top_logprobs: 0,
        }
    }
}

impl SamplingParams {
    /// Greedy sampling. The seed is kept so the draw schedule stays identical
    /// to a sampling run (useful for engine-level reference comparisons).
    #[must_use]
    pub const fn greedy(seed: u64) -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            top_logprobs: 0,
        }
    }
}

/// Batched sampling result: per-row sampled token, its log-probability, and
/// the per-row top-`k` (token, logprob) lists (empty when `top_logprobs == 0`).
pub type SampleOutput = (Vec<u32>, Vec<f32>, Vec<Vec<(u32, f32)>>);

/// Batched sampling kernel: one block per sequence row (256 threads).
///
/// Steps per row:
/// 1. advance the RNG seed once and draw `u`;
/// 2. greedy shortcut when `temperature <= 0` (argmax, smallest index on tie);
/// 3. row max/min; top-k threshold = k-th largest raw logit (binary search);
/// 4. softmax total `S`; top-p cutoff = largest prob with cumulative `>= p*S`;
/// 5. allowed set = `logit >= topk_thr` and `prob >= cut`; sample by walking
///    the cumulative allowed mass with `target = u * total_allowed`.
const SAMPLE_BATCHED: &str = r#"
typedef unsigned long long ull;

__device__ ull mix64(ull z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

extern "C" __global__ void sample_batched(
    float* __restrict__ logits,
    const float* __restrict__ temp,
    const int* __restrict__ top_k,
    const float* __restrict__ top_p,
    ull* __restrict__ seed,
    int* __restrict__ out_tok,
    float* __restrict__ logprobs,
    const float* __restrict__ presence_pen,
    const float* __restrict__ freq_pen,
    const int* __restrict__ pen_tokens,
    const int* __restrict__ pen_counts,
    const int* __restrict__ pen_count,
    const int* __restrict__ bias_tokens,
    const float* __restrict__ bias_vals,
    const int* __restrict__ bias_count,
    int max_pen, int max_bias, int vocab, int batch)
{
    const int s = blockIdx.x;
    float* row = logits + (long long)s * vocab; // penalties modify in place
    const int T = blockDim.x;
    const int tid = threadIdx.x;

    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    __shared__ float s_red[256];
    __shared__ float s_u;
    __shared__ float s_lo;
    __shared__ float s_hi;
    __shared__ float s_maxv;
    __shared__ float s_minv;
    __shared__ float s_total;
    __shared__ float s_cut;
    __shared__ float seg[256];
    __shared__ float seg_pref[256];

    const float t = temp[s];
    const int k = top_k[s];
    const float p = top_p[s];

    // Apply presence/frequency penalties in place (before softmax).
    {
        const float pp = presence_pen[s];
        const float fp = freq_pen[s];
        if (pp != 0.0f || fp != 0.0f) {
            for (int j = tid; j < pen_count[s]; j += T) {
                int tok = pen_tokens[(long long)s * max_pen + j];
                int cnt = pen_counts[(long long)s * max_pen + j];
                if (cnt > 0 && tok < vocab) {
                    row[tok] -= pp + fp * (float)cnt;
                }
            }
            __syncthreads();
        }
    }

    // Apply logit_bias in place (OpenAI logit_bias adds to logits).
    {
        for (int j = tid; j < bias_count[s]; j += T) {
            int tok = bias_tokens[(long long)s * max_bias + j];
            if (tok < vocab) {
                row[tok] += bias_vals[(long long)s * max_bias + j];
            }
        }
        __syncthreads();
    }

    // One RNG draw + advance per row per call (even for greedy, so the draw
    // schedule stays identical between greedy and sampling runs).
    if (tid == 0) {
        ull st = seed[s];
        ull z = mix64(st);
        st += 0x9E3779B97F4A7C15ULL;
        seed[s] = st;
        s_u = (float)(z >> 40) * (1.0f / 16777216.0f);
    }
    __syncthreads();
    const float u = s_u;

    if (t <= 0.0f) {
        float v = -1e30f;
        int idx = -1;
        for (int i = tid; i < vocab; i += T) {
            if (row[i] > v) { v = row[i]; idx = i; }
        }
        s_val[tid] = v;
        s_idx[tid] = idx;
        __syncthreads();
        for (int st = T / 2; st > 0; st >>= 1) {
            if (tid < st) {
                float a = s_val[tid + st];
                float b = s_val[tid];
                int ai = s_idx[tid + st];
                int bi = s_idx[tid];
                if (a > b || (a == b && ai < bi)) { s_val[tid] = a; s_idx[tid] = ai; }
            }
            __syncthreads();
        }
        if (tid == 0) {
            out_tok[s] = s_idx[0];
            logprobs[s] = 0.0f; // greedy: probability 1 -> log 0
        }
        return;
    }

    // Row max / min.
    float mx = -1e30f, mn = 1e30f;
    for (int i = tid; i < vocab; i += T) {
        float x = row[i];
        mx = fmaxf(mx, x);
        mn = fminf(mn, x);
    }
    s_val[tid] = mx;
    __syncthreads();
    for (int st = T / 2; st > 0; st >>= 1) {
        if (tid < st) s_val[tid] = fmaxf(s_val[tid], s_val[tid + st]);
        __syncthreads();
    }
    if (tid == 0) s_maxv = s_val[0];
    __syncthreads();
    s_val[tid] = mn;
    __syncthreads();
    for (int st = T / 2; st > 0; st >>= 1) {
        if (tid < st) s_val[tid] = fminf(s_val[tid], s_val[tid + st]);
        __syncthreads();
    }
    if (tid == 0) s_minv = s_val[0];
    __syncthreads();
    const float maxv = s_maxv;
    const float minv = s_minv;
    const float inv_t = 1.0f / t;
    const int k_eff = (k > 0 && k < vocab) ? k : 0;

    // Top-k threshold: k-th largest raw logit (ties at the boundary included).
    float topk_thr = -1e30f;
    if (k_eff > 0) {
        float lo = minv, hi = maxv;
        for (int it = 0; it < 30; it++) {
            float mid = (lo + hi) * 0.5f;
            int cnt = 0;
            for (int i = tid; i < vocab; i += T) if (row[i] > mid) cnt++;
            s_red[tid] = (float)cnt;
            __syncthreads();
            for (int st = T / 2; st > 0; st >>= 1) {
                if (tid < st) s_red[tid] += s_red[tid + st];
                __syncthreads();
            }
            if (tid == 0) {
                if (s_red[0] >= (float)k_eff) lo = mid; else hi = mid;
                s_lo = lo; s_hi = hi;
            }
            __syncthreads();
            lo = s_lo; hi = s_hi;
        }
        topk_thr = hi;
    }

    // Softmax total S.
    float acc_sum = 0.0f;
    for (int i = tid; i < vocab; i += T) acc_sum += __expf((row[i] - maxv) * inv_t);
    s_red[tid] = acc_sum;
    __syncthreads();
    for (int st = T / 2; st > 0; st >>= 1) {
        if (tid < st) s_red[tid] += s_red[tid + st];
        __syncthreads();
    }
    if (tid == 0) s_total = s_red[0];
    __syncthreads();
    const float s_tot = s_total;

    // Top-p cutoff: largest prob whose cumulative (unnormalized) mass is
    // still >= top_p * S. Boundary tier (prob == cut) is fully included.
    const bool use_topp = (p < 1.0f);
    float cut = 0.0f;
    if (use_topp) {
        float lo = 0.0f, hi = 1.0f;
        for (int it = 0; it < 30; it++) {
            float mid = (lo + hi) * 0.5f;
            float acc = 0.0f;
            for (int i = tid; i < vocab; i += T) {
                float pr = __expf((row[i] - maxv) * inv_t);
                if (pr >= mid) acc += pr;
            }
            s_red[tid] = acc;
            __syncthreads();
            for (int st = T / 2; st > 0; st >>= 1) {
                if (tid < st) s_red[tid] += s_red[tid + st];
                __syncthreads();
            }
            if (tid == 0) {
                if (s_red[0] >= p * s_tot) lo = mid; else hi = mid;
                s_lo = lo; s_hi = hi;
            }
            __syncthreads();
            lo = s_lo; hi = s_hi;
        }
        cut = lo;
        if (tid == 0) s_cut = cut;
        __syncthreads();
        cut = s_cut;
    }

    // Total allowed mass.
    float allowed_sum = 0.0f;
    for (int i = tid; i < vocab; i += T) {
        float pr = __expf((row[i] - maxv) * inv_t);
        bool ok = true;
        if (k_eff > 0 && !(row[i] >= topk_thr)) ok = false;
        if (use_topp && !(pr >= cut)) ok = false;
        if (ok) allowed_sum += pr;
    }
    s_red[tid] = allowed_sum;
    __syncthreads();
    for (int st = T / 2; st > 0; st >>= 1) {
        if (tid < st) s_red[tid] += s_red[tid + st];
        __syncthreads();
    }
    if (tid == 0) s_total = s_red[0];
    __syncthreads();
    const float tot = s_total;

    const float target = u * tot;

    // Per-thread segment mass over the allowed set, then an exclusive prefix
    // scan; the thread whose segment contains `target` walks its elements.
    seg[tid] = 0.0f;
    for (int i = tid; i < vocab; i += T) {
        float pr = __expf((row[i] - maxv) * inv_t);
        bool ok = true;
        if (k_eff > 0 && !(row[i] >= topk_thr)) ok = false;
        if (use_topp && !(pr >= cut)) ok = false;
        if (ok) seg[tid] += pr;
    }
    __syncthreads();
    if (tid == 0) {
        float acc = 0.0f;
        for (int j = 0; j < T; j++) { seg_pref[j] = acc; acc += seg[j]; }
    }
    __syncthreads();
    if (tid == 0) out_tok[s] = -1;
    __syncthreads();
    if (seg_pref[tid] <= target && target < seg_pref[tid] + seg[tid]) {
        float acc = seg_pref[tid];
        for (int i = tid; i < vocab; i += T) {
            float pr = __expf((row[i] - maxv) * inv_t);
            bool ok = true;
            if (k_eff > 0 && !(row[i] >= topk_thr)) ok = false;
            if (use_topp && !(pr >= cut)) ok = false;
            if (ok) {
                acc += pr;
                if (acc > target) {
                    out_tok[s] = i;
                    logprobs[s] = (row[i] - maxv) * inv_t - logf(tot);
                    break;
                }
            }
        }
    }
    __syncthreads();
    // Fallback (float rounding near the top of the mass): first allowed, else
    // argmax. The allowed set is never empty (the max element always passes).
    if (tid == 0 && out_tok[s] < 0) {
        int found = -1;
        for (int i = 0; i < vocab; i++) {
            float pr = __expf((row[i] - maxv) * inv_t);
            bool ok = true;
            if (k_eff > 0 && !(row[i] >= topk_thr)) ok = false;
            if (use_topp && !(pr >= cut)) ok = false;
            if (ok) { found = i; break; }
        }
        if (found < 0) {
            float v = -1e30f;
            int idx = -1;
            for (int i = 0; i < vocab; i++) if (row[i] > v) { v = row[i]; idx = i; }
            found = idx;
        }
        out_tok[s] = found;
        logprobs[s] = (row[found] - maxv) * inv_t - logf(tot);
    }
}
"#;
/// Batched top-`k` log-prob kernel: one block per row (256 threads).
///
/// Ranks tokens by post-penalty/bias softmax probability (temperature per row,
/// `inv_t = 1/t` with `t <= 0` treated as `1.0`), reporting
/// `logprob = (logit - max) * inv_t - log(total)`. Ties break on the smaller
/// token id (deterministic). Each thread keeps a local top-k over its strided
/// slice; lists are merged through dynamic shared memory and thread 0 scans
/// `T * TOPK_MAX` entries for the global top-k.
const TOPK_BATCHED: &str = r#"
#define TOPK_MAX 20

extern "C" __global__ void topk_batched(
    const float* __restrict__ logits,
    const float* __restrict__ inv_t,
    int* __restrict__ out_tok,
    float* __restrict__ out_lp,
    int vocab, int batch, int k)
{
    const int s = blockIdx.x;
    const float* row = logits + (long long)s * vocab;
    const int T = blockDim.x;
    const int tid = threadIdx.x;
    const float it = inv_t[s];
    const int kk = (k < 1) ? 1 : (k > TOPK_MAX ? TOPK_MAX : k);

    __shared__ float s_red[256];
    __shared__ float s_maxv;
    __shared__ float s_tot;

    // Row max (max-subtracted softmax, matching sample_batched).
    float mx = -1e30f;
    for (int i = tid; i < vocab; i += T) mx = fmaxf(mx, row[i]);
    s_red[tid] = mx;
    __syncthreads();
    for (int st = T / 2; st > 0; st >>= 1) {
        if (tid < st) s_red[tid] = fmaxf(s_red[tid], s_red[tid + st]);
        __syncthreads();
    }
    if (tid == 0) s_maxv = s_red[0];
    __syncthreads();
    const float maxv = s_maxv;

    // Softmax total.
    float tot = 0.0f;
    for (int i = tid; i < vocab; i += T) tot += __expf((row[i] - maxv) * it);
    s_red[tid] = tot;
    __syncthreads();
    for (int st = T / 2; st > 0; st >>= 1) {
        if (tid < st) s_red[tid] += s_red[tid + st];
        __syncthreads();
    }
    if (tid == 0) s_tot = s_red[0];
    __syncthreads();
    const float logZ = logf(s_tot);

    // Per-thread local top-k over its strided slice, sorted by prob desc
    // (ties: token id asc). `lp = (logit - max) * it` avoids underflow.
    float lt_p[TOPK_MAX];
    int lt_t[TOPK_MAX];
    int lt_n = 0;
    for (int i = tid; i < vocab; i += T) {
        float lp = (row[i] - maxv) * it;
        float p = __expf(lp);
        if (lt_n < kk) {
            int pos = lt_n;
            while (pos > 0 && (lt_p[pos - 1] < p || (lt_p[pos - 1] == p && lt_t[pos - 1] > i))) {
                lt_p[pos] = lt_p[pos - 1];
                lt_t[pos] = lt_t[pos - 1];
                pos--;
            }
            lt_p[pos] = p;
            lt_t[pos] = i;
            lt_n++;
        } else if (p > lt_p[kk - 1] || (p == lt_p[kk - 1] && i < lt_t[kk - 1])) {
            int pos = kk - 1;
            while (pos > 0 && (lt_p[pos - 1] < p || (lt_p[pos - 1] == p && lt_t[pos - 1] > i))) {
                lt_p[pos] = lt_p[pos - 1];
                lt_t[pos] = lt_t[pos - 1];
                pos--;
            }
            lt_p[pos] = p;
            lt_t[pos] = i;
        }
    }

    // Merge local lists through dynamic shared memory; thread 0 scans all
    // entries for the global top-k.
    extern __shared__ float sm[];
    float* sm_p = (float*)sm;
    int* sm_t = (int*)(sm + (long long)T * TOPK_MAX);
    for (int j = 0; j < TOPK_MAX; j++) {
        if (j < lt_n) {
            sm_p[tid * TOPK_MAX + j] = lt_p[j];
            sm_t[tid * TOPK_MAX + j] = lt_t[j];
        } else {
            sm_p[tid * TOPK_MAX + j] = -1e30f;
            sm_t[tid * TOPK_MAX + j] = -1;
        }
    }
    __syncthreads();

    if (tid == 0) {
        float g_p[TOPK_MAX];
        int g_t[TOPK_MAX];
        int g_n = 0;
        for (int e = 0; e < T * TOPK_MAX; e++) {
            float p = sm_p[e];
            int t = sm_t[e];
            if (t < 0) continue;
            if (g_n < kk) {
                int pos = g_n;
                while (pos > 0 && (g_p[pos - 1] < p || (g_p[pos - 1] == p && g_t[pos - 1] > t))) {
                    g_p[pos] = g_p[pos - 1];
                    g_t[pos] = g_t[pos - 1];
                    pos--;
                }
                g_p[pos] = p;
                g_t[pos] = t;
                g_n++;
            } else if (p > g_p[kk - 1] || (p == g_p[kk - 1] && t < g_t[kk - 1])) {
                int pos = kk - 1;
                while (pos > 0 && (g_p[pos - 1] < p || (g_p[pos - 1] == p && g_t[pos - 1] > t))) {
                    g_p[pos] = g_p[pos - 1];
                    g_t[pos] = g_t[pos - 1];
                    pos--;
                }
                g_p[pos] = p;
                g_t[pos] = t;
            }
        }
        for (int j = 0; j < kk; j++) {
            out_tok[(long long)s * TOPK_MAX + j] = g_t[j];
            out_lp[(long long)s * TOPK_MAX + j] = (row[g_t[j]] - maxv) * it - logZ;
        }
    }
}
"#;

/// Batched top-k/top-p/temperature sampler bound to the model stream.
pub struct BatchedSampler {
    hip: Arc<Hip>,
    stream: HipStream,
    kernel: HipKernelModule,
    topk_kernel: HipKernelModule,
    temp_dev: *mut f32,
    topk_dev: *mut i32,
    topp_dev: *mut f32,
    seed_dev: *mut u64,
    out_dev: *mut i32,
    logprobs_dev: *mut f32,
    presence_dev: *mut f32,
    freq_dev: *mut f32,
    pen_tokens_dev: *mut i32,
    pen_counts_dev: *mut i32,
    pen_count_dev: *mut i32,
    bias_tokens_dev: *mut i32,
    bias_vals_dev: *mut f32,
    bias_count_dev: *mut i32,
    topk_inv_t_dev: *mut f32,
    topk_tok_dev: *mut i32,
    topk_lp_dev: *mut f32,
    temp_host: *mut f32,
    topk_host: *mut i32,
    topp_host: *mut f32,
    seed_host: *mut u64,
    out_host: *mut i32,
    logprobs_host: *mut f32,
    presence_host: *mut f32,
    freq_host: *mut f32,
    pen_tokens_host: *mut i32,
    pen_counts_host: *mut i32,
    pen_count_host: *mut i32,
    bias_tokens_host: *mut i32,
    bias_vals_host: *mut f32,
    bias_count_host: *mut i32,
    topk_inv_t_host: *mut f32,
    topk_tok_host: *mut i32,
    topk_lp_host: *mut f32,
    capacity: usize,
    allocs: Vec<*mut core::ffi::c_void>,
    pins: Vec<*mut core::ffi::c_void>,
}

// SAFETY: a BatchedSampler is confined to one engine thread; raw device
// pointers are only touched there, and the loaded HIP runtime is Send.
unsafe impl Send for BatchedSampler {}

impl BatchedSampler {
    /// Compiles the kernel and allocates per-slot buffers. `stream` must be the
    /// stream the model launches on (shared with the decode kernels).
    pub fn new(hip: Arc<Hip>, stream: HipStream, capacity: usize) -> Result<Self, Error> {
        let arch = hip_arch();
        let kernel = HipKernelModule::compile(&arch, SAMPLE_BATCHED, "sample_batched")?;
        let topk_kernel = HipKernelModule::compile(&arch, TOPK_BATCHED, "topk_batched")?;
        let mut allocs = Vec::new();
        let mut pins = Vec::new();
        let mut dalloc = |bytes: usize| -> Result<*mut core::ffi::c_void, Error> {
            let p = hip::malloc(&hip, bytes)?;
            allocs.push(p);
            Ok(p)
        };
        let mut pall = |bytes: usize| -> Result<*mut core::ffi::c_void, Error> {
            let p = hip::host_malloc(&hip, bytes)?;
            pins.push(p);
            Ok(p)
        };
        let temp_dev = dalloc(capacity * 4)? as *mut f32;
        let topk_dev = dalloc(capacity * 4)? as *mut i32;
        let topp_dev = dalloc(capacity * 4)? as *mut f32;
        let seed_dev = dalloc(capacity * 8)? as *mut u64;
        let out_dev = dalloc(capacity * 4)? as *mut i32;
        let logprobs_dev = dalloc(capacity * 4)? as *mut f32;
        let presence_dev = dalloc(capacity * 4)? as *mut f32;
        let freq_dev = dalloc(capacity * 4)? as *mut f32;
        let pen_tokens_dev = dalloc(capacity * MAX_PEN * 4)? as *mut i32;
        let pen_counts_dev = dalloc(capacity * MAX_PEN * 4)? as *mut i32;
        let pen_count_dev = dalloc(capacity * 4)? as *mut i32;
        let bias_tokens_dev = dalloc(capacity * MAX_BIAS * 4)? as *mut i32;
        let bias_vals_dev = dalloc(capacity * MAX_BIAS * 4)? as *mut f32;
        let bias_count_dev = dalloc(capacity * 4)? as *mut i32;
        let topk_inv_t_dev = dalloc(capacity * 4)? as *mut f32;
        let topk_tok_dev = dalloc(capacity * MAX_TOPK * 4)? as *mut i32;
        let topk_lp_dev = dalloc(capacity * MAX_TOPK * 4)? as *mut f32;
        let temp_host = pall(capacity * 4)? as *mut f32;
        let topk_host = pall(capacity * 4)? as *mut i32;
        let topp_host = pall(capacity * 4)? as *mut f32;
        let seed_host = pall(capacity * 8)? as *mut u64;
        let out_host = pall(capacity * 4)? as *mut i32;
        let logprobs_host = pall(capacity * 4)? as *mut f32;
        let presence_host = pall(capacity * 4)? as *mut f32;
        let freq_host = pall(capacity * 4)? as *mut f32;
        let pen_tokens_host = pall(capacity * MAX_PEN * 4)? as *mut i32;
        let pen_counts_host = pall(capacity * MAX_PEN * 4)? as *mut i32;
        let pen_count_host = pall(capacity * 4)? as *mut i32;
        let bias_tokens_host = pall(capacity * MAX_BIAS * 4)? as *mut i32;
        let bias_vals_host = pall(capacity * MAX_BIAS * 4)? as *mut f32;
        let bias_count_host = pall(capacity * 4)? as *mut i32;
        let topk_inv_t_host = pall(capacity * 4)? as *mut f32;
        let topk_tok_host = pall(capacity * MAX_TOPK * 4)? as *mut i32;
        let topk_lp_host = pall(capacity * MAX_TOPK * 4)? as *mut f32;
        Ok(Self {
            hip,
            stream,
            kernel,
            topk_kernel,
            temp_dev,
            topk_dev,
            topp_dev,
            seed_dev,
            out_dev,
            logprobs_dev,
            presence_dev,
            freq_dev,
            pen_tokens_dev,
            pen_counts_dev,
            pen_count_dev,
            bias_tokens_dev,
            bias_vals_dev,
            bias_count_dev,
            topk_inv_t_dev,
            topk_tok_dev,
            topk_lp_dev,
            temp_host,
            topk_host,
            topp_host,
            seed_host,
            out_host,
            logprobs_host,
            presence_host,
            freq_host,
            pen_tokens_host,
            pen_counts_host,
            pen_count_host,
            bias_tokens_host,
            bias_vals_host,
            bias_count_host,
            topk_inv_t_host,
            topk_tok_host,
            topk_lp_host,
            capacity,
            allocs,
            pins,
        })
    }

    /// Samples one token per sequence from a `[n, vocab]` row-major logits
    /// matrix. `params` must have length `n <= capacity`; each `params[i].seed`
    /// is advanced by one RNG step (mirroring the kernel), so callers keep the
    /// authoritative host-side seed and a CPU reference can reproduce the draw.
    pub fn sample_batched(
        &self,
        logits: *const f32,
        params: &mut [SamplingParams],
        counts: &[Vec<(u32, u32)>],
        bias: &[Vec<(u32, f32)>],
        vocab: usize,
    ) -> Result<SampleOutput, Error> {
        let n = params.len();
        assert!(n <= self.capacity, "sampler capacity exceeded");
        assert_eq!(counts.len(), n, "counts must be per-row");
        assert_eq!(bias.len(), n, "bias must be per-row");
        #[allow(clippy::needless_range_loop)] // raw pinned-host writes by slot
        for i in 0..n {
            assert!(
                counts[i].len() <= MAX_PEN,
                "too many distinct tokens for penalty scratch"
            );
            assert!(bias[i].len() <= MAX_BIAS, "too many logit_bias entries");
            unsafe {
                *self.temp_host.add(i) = params[i].temperature;
                *self.topk_host.add(i) = params[i].top_k as i32;
                *self.topp_host.add(i) = params[i].top_p;
                *self.seed_host.add(i) = params[i].seed;
                *self.presence_host.add(i) = params[i].presence_penalty;
                *self.freq_host.add(i) = params[i].frequency_penalty;
                *self.pen_count_host.add(i) = counts[i].len() as i32;
                for (j, &(t, c)) in counts[i].iter().enumerate() {
                    *self.pen_tokens_host.add(i * MAX_PEN + j) = t as i32;
                    *self.pen_counts_host.add(i * MAX_PEN + j) = c as i32;
                }
                *self.bias_count_host.add(i) = bias[i].len() as i32;
                for (j, &(t, b)) in bias[i].iter().enumerate() {
                    *self.bias_tokens_host.add(i * MAX_BIAS + j) = t as i32;
                    *self.bias_vals_host.add(i * MAX_BIAS + j) = b;
                }
            }
        }
        hip::memcpy_async(
            &self.hip,
            self.temp_dev as *mut core::ffi::c_void,
            self.temp_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;
        hip::memcpy_async(
            &self.hip,
            self.topk_dev as *mut core::ffi::c_void,
            self.topk_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;
        hip::memcpy_async(
            &self.hip,
            self.topp_dev as *mut core::ffi::c_void,
            self.topp_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;
        hip::memcpy_async(
            &self.hip,
            self.seed_dev as *mut core::ffi::c_void,
            self.seed_host as *const core::ffi::c_void,
            n * 8,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;
        hip::memcpy_async(
            &self.hip,
            self.presence_dev as *mut core::ffi::c_void,
            self.presence_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;
        hip::memcpy_async(
            &self.hip,
            self.freq_dev as *mut core::ffi::c_void,
            self.freq_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;
        hip::memcpy_async(
            &self.hip,
            self.pen_tokens_dev as *mut core::ffi::c_void,
            self.pen_tokens_host as *const core::ffi::c_void,
            n * MAX_PEN * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;
        hip::memcpy_async(
            &self.hip,
            self.pen_counts_dev as *mut core::ffi::c_void,
            self.pen_counts_host as *const core::ffi::c_void,
            n * MAX_PEN * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;
        hip::memcpy_async(
            &self.hip,
            self.pen_count_dev as *mut core::ffi::c_void,
            self.pen_count_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;
        hip::memcpy_async(
            &self.hip,
            self.bias_tokens_dev as *mut core::ffi::c_void,
            self.bias_tokens_host as *const core::ffi::c_void,
            n * MAX_BIAS * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;
        hip::memcpy_async(
            &self.hip,
            self.bias_vals_dev as *mut core::ffi::c_void,
            self.bias_vals_host as *const core::ffi::c_void,
            n * MAX_BIAS * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;
        hip::memcpy_async(
            &self.hip,
            self.bias_count_dev as *mut core::ffi::c_void,
            self.bias_count_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;

        let lp = logits as *mut f32; // kernel applies penalties in place
        let tp = self.temp_dev;
        let kp = self.topk_dev;
        let pp = self.topp_dev;
        let sp = self.seed_dev;
        let op = self.out_dev;
        let lgp = self.logprobs_dev;
        let psp = self.presence_dev;
        let fsp = self.freq_dev;
        let ptp = self.pen_tokens_dev;
        let pcp = self.pen_counts_dev;
        let pnp = self.pen_count_dev;
        let btp = self.bias_tokens_dev;
        let bvp = self.bias_vals_dev;
        let bcp = self.bias_count_dev;
        let vocab_i = vocab as i32;
        let n_i = n as i32;
        let max_pen = MAX_PEN as i32;
        let max_bias = MAX_BIAS as i32;
        let mut args: Vec<*mut core::ffi::c_void> = vec![
            &lp as *const *mut f32 as *mut core::ffi::c_void,
            &tp as *const *mut f32 as *mut core::ffi::c_void,
            &kp as *const *mut i32 as *mut core::ffi::c_void,
            &pp as *const *mut f32 as *mut core::ffi::c_void,
            &sp as *const *mut u64 as *mut core::ffi::c_void,
            &op as *const *mut i32 as *mut core::ffi::c_void,
            &lgp as *const *mut f32 as *mut core::ffi::c_void,
            &psp as *const *mut f32 as *mut core::ffi::c_void,
            &fsp as *const *mut f32 as *mut core::ffi::c_void,
            &ptp as *const *mut i32 as *mut core::ffi::c_void,
            &pcp as *const *mut i32 as *mut core::ffi::c_void,
            &pnp as *const *mut i32 as *mut core::ffi::c_void,
            &btp as *const *mut i32 as *mut core::ffi::c_void,
            &bvp as *const *mut f32 as *mut core::ffi::c_void,
            &bcp as *const *mut i32 as *mut core::ffi::c_void,
            &max_pen as *const i32 as *mut core::ffi::c_void,
            &max_bias as *const i32 as *mut core::ffi::c_void,
            &vocab_i as *const i32 as *mut core::ffi::c_void,
            &n_i as *const i32 as *mut core::ffi::c_void,
        ];
        self.kernel
            .launch([n as u32, 1, 1], [256, 1, 1], &mut args, self.stream)?;

        unsafe {
            hip::check(
                &self.hip,
                (self.hip.api.hip_stream_synchronize)(self.stream),
            )?;
            hip::memcpy(
                &self.hip,
                self.out_host as *mut core::ffi::c_void,
                self.out_dev as *const core::ffi::c_void,
                n * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )?;
            hip::memcpy(
                &self.hip,
                self.logprobs_host as *mut core::ffi::c_void,
                self.logprobs_dev as *const core::ffi::c_void,
                n * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )?;
        }
        // Advance the authoritative host-side seeds one step (kernel does the
        // same on its device copy, which is re-uploaded next call anyway).
        for p in params.iter_mut() {
            p.seed = advance_seed(p.seed);
        }
        let mut out = Vec::with_capacity(n);
        let mut lps = Vec::with_capacity(n);
        unsafe {
            for i in 0..n {
                out.push(*self.out_host.add(i) as u32);
                lps.push(*self.logprobs_host.add(i));
            }
        }
        let topk = self.sample_topk(logits, params, n, vocab)?;
        Ok((out, lps, topk))
    }

    /// Optional per-row top-`k` log-probs (OpenAI `top_logprobs`), computed on
    /// device from the post-penalty/bias logits. Rows with
    /// `params[i].top_logprobs == 0` get an empty list. `k` is clamped to
    /// [`MAX_TOPK`].
    fn sample_topk(
        &self,
        logits: *const f32,
        params: &[SamplingParams],
        n: usize,
        vocab: usize,
    ) -> Result<Vec<Vec<(u32, f32)>>, Error> {
        let k_global = params.iter().map(|p| p.top_logprobs).max().unwrap_or(0);
        if k_global == 0 || n == 0 {
            return Ok(vec![Vec::new(); n]);
        }
        let k_global = k_global.min(MAX_TOPK);
        for (i, p) in params.iter().enumerate() {
            let it = if p.temperature > 0.0 {
                1.0 / p.temperature
            } else {
                1.0
            };
            unsafe {
                *self.topk_inv_t_host.add(i) = it;
            }
        }
        hip::memcpy_async(
            &self.hip,
            self.topk_inv_t_dev as *mut core::ffi::c_void,
            self.topk_inv_t_host as *const core::ffi::c_void,
            n * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
            self.stream,
        )?;
        let vocab_i = vocab as i32;
        let n_i = n as i32;
        let k_i = k_global as i32;
        let lp = logits as *mut f32; // kernel arg convention (reads const)
        let itp = self.topk_inv_t_dev;
        let otp = self.topk_tok_dev;
        let olp = self.topk_lp_dev;
        let mut args: Vec<*mut core::ffi::c_void> = vec![
            &lp as *const *mut f32 as *mut core::ffi::c_void,
            &itp as *const *mut f32 as *mut core::ffi::c_void,
            &otp as *const *mut i32 as *mut core::ffi::c_void,
            &olp as *const *mut f32 as *mut core::ffi::c_void,
            &vocab_i as *const i32 as *mut core::ffi::c_void,
            &n_i as *const i32 as *mut core::ffi::c_void,
            &k_i as *const i32 as *mut core::ffi::c_void,
        ];
        // Dynamic shared: T * TOPK_MAX probs + T * TOPK_MAX tokens.
        let shared = 2 * 256 * MAX_TOPK * 4;
        self.topk_kernel.launch_shmem(
            [n as u32, 1, 1],
            [256, 1, 1],
            &mut args,
            self.stream,
            shared as u32,
        )?;
        unsafe {
            hip::check(
                &self.hip,
                (self.hip.api.hip_stream_synchronize)(self.stream),
            )?;
            hip::memcpy(
                &self.hip,
                self.topk_tok_host as *mut core::ffi::c_void,
                self.topk_tok_dev as *const core::ffi::c_void,
                n * MAX_TOPK * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )?;
            hip::memcpy(
                &self.hip,
                self.topk_lp_host as *mut core::ffi::c_void,
                self.topk_lp_dev as *const core::ffi::c_void,
                n * MAX_TOPK * 4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )?;
        }
        let mut rows = Vec::with_capacity(n);
        unsafe {
            for (i, p) in params.iter().enumerate() {
                let want = p.top_logprobs.min(MAX_TOPK);
                let mut row = Vec::with_capacity(want);
                for j in 0..want {
                    let tok = *self.topk_tok_host.add(i * MAX_TOPK + j) as u32;
                    let lpv = *self.topk_lp_host.add(i * MAX_TOPK + j);
                    row.push((tok, lpv));
                }
                rows.push(row);
            }
        }
        Ok(rows)
    }
}

impl Drop for BatchedSampler {
    fn drop(&mut self) {
        for &p in &self.allocs {
            let _ = hip::free(&self.hip, p);
        }
        for &p in &self.pins {
            let _ = hip::host_free(&self.hip, p);
        }
    }
}

/// Smallest index among the maximum elements (matches the GPU greedy branch).
#[must_use]
fn argmax_first(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// CPU reference for the batched sampler (used by tests and engine-level
/// comparisons). Mirrors the GPU algorithm: temperature softmax with max
/// subtraction, top-k as the k-th largest logit threshold, top-p as a
/// cumulative threshold, then one SplitMix64 draw and a cumulative walk over
/// the allowed set. `state` is advanced by exactly one step per call.
#[must_use]
pub fn sample_cpu(
    logits: &[f32],
    p: &SamplingParams,
    counts: &[(u32, u32)],
    bias: &[(u32, f32)],
    state: &mut u64,
) -> u32 {
    let vocab = logits.len();
    let u = next_f32(state);
    // Apply presence/frequency penalties and logit_bias to a copy of logits.
    let need_edit = p.presence_penalty != 0.0 || p.frequency_penalty != 0.0 || !bias.is_empty();
    let penalized: Vec<f32> = if need_edit {
        let mut v = logits.to_vec();
        for &(t, c) in counts {
            if t < vocab as u32 && c > 0 {
                v[t as usize] -= p.presence_penalty + p.frequency_penalty * c as f32;
            }
        }
        for &(t, b) in bias {
            if t < vocab as u32 {
                v[t as usize] += b;
            }
        }
        v
    } else {
        logits.to_vec()
    };
    let logits = &penalized;
    let vocab = logits.len();
    if p.temperature <= 0.0 {
        return argmax_first(logits);
    }
    let maxv = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let minv = logits.iter().copied().fold(f32::INFINITY, f32::min);
    let inv_t = 1.0 / p.temperature;
    let prob = |x: f32| ((x - maxv) * inv_t).exp();
    let k_eff = if p.top_k > 0 && p.top_k < vocab {
        p.top_k
    } else {
        0
    };

    let topk_thr = if k_eff > 0 {
        let mut lo = minv;
        let mut hi = maxv;
        for _ in 0..30 {
            let mid = (lo + hi) * 0.5;
            let gt = logits.iter().filter(|&&x| x > mid).count();
            if gt >= k_eff {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi
    } else {
        f32::NEG_INFINITY
    };

    let s_tot: f32 = logits.iter().map(|&x| prob(x)).sum();
    let use_topp = p.top_p < 1.0;
    let cut = if use_topp {
        let mut lo = 0.0f32;
        let mut hi = 1.0f32;
        for _ in 0..30 {
            let mid = (lo + hi) * 0.5;
            let acc: f32 = logits
                .iter()
                .map(|&x| prob(x))
                .filter(|&pr| pr >= mid)
                .sum();
            if acc >= p.top_p * s_tot {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    } else {
        0.0
    };

    let allowed = |x: f32, i: usize| -> bool {
        let pr = prob(x);
        (k_eff == 0 || logits[i] >= topk_thr) && (!use_topp || pr >= cut)
    };

    let tot_allowed: f32 = (0..vocab)
        .filter(|&i| allowed(logits[i], i))
        .map(|i| prob(logits[i]))
        .sum();
    let target = u * tot_allowed;

    let mut acc = 0.0f32;
    let mut chosen = None;
    for (i, &x) in logits.iter().enumerate() {
        if allowed(x, i) {
            acc += prob(x);
            if acc > target {
                chosen = Some(i as u32);
                break;
            }
        }
    }
    chosen.unwrap_or_else(|| argmax_first(logits))
}

/// CPU reference for top-`k` log-probs (OpenAI `top_logprobs`), mirroring the
/// `topk_batched` kernel: post-penalty/bias softmax at the row temperature
/// (`t <= 0` treated as `1.0`), ranked by probability descending (token id
/// ascending on ties), `logprob = (logit - max) * inv_t - log(total)`.
#[must_use]
pub fn topk_cpu(
    logits: &[f32],
    p: &SamplingParams,
    counts: &[(u32, u32)],
    bias: &[(u32, f32)],
    k: usize,
) -> Vec<(u32, f32)> {
    let vocab = logits.len();
    let need_edit = p.presence_penalty != 0.0 || p.frequency_penalty != 0.0 || !bias.is_empty();
    let mut v = logits.to_vec();
    if need_edit {
        for &(t, c) in counts {
            if (t as usize) < vocab && c > 0 {
                v[t as usize] -= p.presence_penalty + p.frequency_penalty * c as f32;
            }
        }
        for &(t, b) in bias {
            if (t as usize) < vocab {
                v[t as usize] += b;
            }
        }
    }
    let inv_t = if p.temperature > 0.0 {
        1.0 / p.temperature
    } else {
        1.0
    };
    let maxv = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = v.iter().map(|&x| ((x - maxv) * inv_t).exp()).sum();
    let log_z = total.ln();
    let mut ranked: Vec<(u32, f32)> = v
        .iter()
        .enumerate()
        .map(|(i, &x)| (i as u32, (x - maxv) * inv_t - log_z))
        .collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(k.min(MAX_TOPK));
    ranked
}

/// Maximum penalty (token, count) pairs per row (OpenAI frequency/presence).
const MAX_PEN: usize = 4096;
/// Maximum logit_bias (token, value) pairs per row.
const MAX_BIAS: usize = 4096;
/// Maximum reported top log-probs per sampled token (OpenAI `top_logprobs`
/// upper bound).
const MAX_TOPK: usize = 20;

/// Block-local argmax: each block reduces a contiguous 256-element slice.
const ARGMAX_SLICE: &str = r#"
extern "C" __global__ void argmax_slice(const float* logits, float* out_val,
                                        int* out_idx, int vocab) {
    int base = blockIdx.x * blockDim.x;
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    float v = -1e30f;
    int idx = -1;
    int i = base + threadIdx.x;
    if (i < vocab) {
        v = logits[i];
        idx = i;
    }
    s_val[threadIdx.x] = v;
    s_idx[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float a = s_val[threadIdx.x + s];
            float b = s_val[threadIdx.x];
            int ai = s_idx[threadIdx.x + s];
            int bi = s_idx[threadIdx.x];
            if (a > b || (a == b && ai < bi)) {
                s_val[threadIdx.x] = a;
                s_idx[threadIdx.x] = ai;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_val[blockIdx.x] = s_val[0];
        out_idx[blockIdx.x] = s_idx[0];
    }
}
"#;

/// Final reduce over block maxima (n may exceed blockDim; threads fold).
const ARGMAX_REDUCE: &str = r#"
extern "C" __global__ void argmax_reduce(const float* in_val, const int* in_idx,
                                         float* out_val, int* out_idx, int n) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    float v = -1e30f;
    int idx = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (in_val[j] > v || (in_val[j] == v && in_idx[j] < idx)) {
            v = in_val[j];
            idx = in_idx[j];
        }
    }
    s_val[threadIdx.x] = v;
    s_idx[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float a = s_val[threadIdx.x + s];
            float b = s_val[threadIdx.x];
            int ai = s_idx[threadIdx.x + s];
            int bi = s_idx[threadIdx.x];
            if (a > b || (a == b && ai < bi)) {
                s_val[threadIdx.x] = a;
                s_idx[threadIdx.x] = ai;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_val[0] = s_val[0];
        out_idx[0] = s_idx[0];
    }
}
"#;

/// On-device greedy sampler bound to a stream.
pub struct HipSampler {
    hip: Arc<Hip>,
    stream: HipStream,
    slice: HipKernelModule,
    reduce: HipKernelModule,
    block_val: *mut f32,
    block_idx: *mut i32,
    out_val: *mut f32,
    out_idx: *mut i32,
    host_idx: *mut i32,
}

impl HipSampler {
    /// Compiles the kernels and allocates scratch. `stream` must be the stream
    /// the model launches on.
    pub fn new(hip: Arc<Hip>, stream: HipStream) -> Result<Self, Error> {
        let arch = hip_arch();
        let max_blocks = 4096usize; // covers vocab up to ~1M
        let block_val = hip::malloc(&hip, max_blocks * 4)? as *mut f32;
        let block_idx = hip::malloc(&hip, max_blocks * 4)? as *mut i32;
        let out_val = hip::malloc(&hip, 4)? as *mut f32;
        let out_idx = hip::malloc(&hip, 4)? as *mut i32;
        let host_idx = hip::host_malloc(&hip, 4)? as *mut i32;
        Ok(Self {
            hip,
            stream,
            slice: HipKernelModule::compile(&arch, ARGMAX_SLICE, "argmax_slice")?,
            reduce: HipKernelModule::compile(&arch, ARGMAX_REDUCE, "argmax_reduce")?,
            block_val,
            block_idx,
            out_val,
            out_idx,
            host_idx,
        })
    }

    /// Returns the index of the maximum element of `logits` (greedy sample).
    /// Deterministic tie-break: smallest index wins.
    pub fn argmax(&self, logits: *const f32, vocab: usize) -> Result<u32, Error> {
        let nblocks = vocab.div_ceil(256);
        assert!(nblocks <= 4096, "vocab too large for sampler scratch");

        let lp = logits;
        let bv = self.block_val;
        let bi = self.block_idx;
        let vocab_i = vocab as i32;
        let mut p1 = vec![
            &lp as *const *const f32 as *mut core::ffi::c_void,
            &bv as *const *mut f32 as *mut core::ffi::c_void,
            &bi as *const *mut i32 as *mut core::ffi::c_void,
            &vocab_i as *const i32 as *mut core::ffi::c_void,
        ];
        self.slice
            .launch([nblocks as u32, 1, 1], [256, 1, 1], &mut p1, self.stream)?;

        let iv = self.block_val;
        let ii = self.block_idx;
        let ov = self.out_val;
        let oi = self.out_idx;
        let n = nblocks as i32;
        let mut p2 = vec![
            &iv as *const *mut f32 as *mut core::ffi::c_void,
            &ii as *const *mut i32 as *mut core::ffi::c_void,
            &ov as *const *mut f32 as *mut core::ffi::c_void,
            &oi as *const *mut i32 as *mut core::ffi::c_void,
            &n as *const i32 as *mut core::ffi::c_void,
        ];
        self.reduce
            .launch([1, 1, 1], [256, 1, 1], &mut p2, self.stream)?;

        unsafe {
            hip::check(
                &self.hip,
                (self.hip.api.hip_stream_synchronize)(self.stream),
            )?;
            hip::memcpy(
                &self.hip,
                self.host_idx as *mut core::ffi::c_void,
                self.out_idx as *const core::ffi::c_void,
                4,
                hip::HIP_MEMCPY_DEVICE_TO_HOST,
            )?;
        }
        Ok(unsafe { *self.host_idx } as u32)
    }
}

impl Drop for HipSampler {
    fn drop(&mut self) {
        let _ = hip::free(&self.hip, self.block_val as *mut _);
        let _ = hip::free(&self.hip, self.block_idx as *mut _);
        let _ = hip::free(&self.hip, self.out_val as *mut _);
        let _ = hip::free(&self.hip, self.out_idx as *mut _);
        let _ = hip::host_free(&self.hip, self.host_idx as *mut _);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn have_gpu() -> Option<Arc<Hip>> {
        match hip::hip() {
            Ok(h) => match hip::device_count() {
                Ok(n) if n > 0 => Some(h),
                _ => {
                    eprintln!("skipping HIP test: no device");
                    None
                }
            },
            Err(e) => {
                eprintln!("skipping HIP test: {e}");
                None
            }
        }
    }

    fn cpu_argmax_first(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        best as u32
    }

    fn run(vocab: usize, seed: u64) {
        let h = have_gpu().expect("gpu");
        let mut rng = seed;
        let logits: Vec<f32> = (0..vocab)
            .map(|_| {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rng >> 40) as f32) / (1u64 << 24) as f32
            })
            .collect();
        let dlogits = hip::malloc(&h, vocab * 4).unwrap() as *mut f32;
        hip::memcpy(
            &h,
            dlogits as *mut _,
            logits.as_ptr() as *const _,
            vocab * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        let mut stream = std::ptr::null_mut();
        unsafe { hip::check(&h, (h.api.hip_stream_create)(&mut stream)).unwrap() };
        let s = HipSampler::new(Arc::clone(&h), stream).unwrap();
        let got = s.argmax(dlogits, vocab).unwrap();
        let want = cpu_argmax_first(&logits);
        assert_eq!(got, want, "vocab={vocab} seed={seed}");
        hip::free(&h, dlogits as *mut _).unwrap();
        unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };
    }

    #[test]
    fn argmax_matches_cpu() {
        run(1000, 1);
        run(256, 2);
        run(257, 3);
        run(151936, 4);
        run(4096, 5);
    }

    #[test]
    fn argmax_tie_breaks_to_first_index() {
        let h = have_gpu().expect("gpu");
        let vocab = 1024usize;
        let mut logits = vec![0.1f32; vocab];
        logits[700] = 0.5; // unique max
        let dlogits = hip::malloc(&h, vocab * 4).unwrap() as *mut f32;
        hip::memcpy(
            &h,
            dlogits as *mut _,
            logits.as_ptr() as *const _,
            vocab * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        let mut stream = std::ptr::null_mut();
        unsafe { hip::check(&h, (h.api.hip_stream_create)(&mut stream)).unwrap() };
        let s = HipSampler::new(Arc::clone(&h), stream).unwrap();
        assert_eq!(s.argmax(dlogits, vocab).unwrap(), 700);
        // All-equal: must return the smallest index (0).
        let eq = vec![0.5f32; vocab];
        hip::memcpy(
            &h,
            dlogits as *mut _,
            eq.as_ptr() as *const _,
            vocab * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        assert_eq!(s.argmax(dlogits, vocab).unwrap(), 0);
        hip::free(&h, dlogits as *mut _).unwrap();
        unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };
    }

    // ---- Batched sampler tests ----

    fn upload_logits(h: &Hip, logits: &[f32]) -> *mut f32 {
        let d = hip::malloc(h, logits.len() * 4).unwrap() as *mut f32;
        hip::memcpy(
            h,
            d as *mut _,
            logits.as_ptr() as *const _,
            logits.len() * 4,
            hip::HIP_MEMCPY_HOST_TO_DEVICE,
        )
        .unwrap();
        d
    }

    fn make_stream(h: &Hip) -> HipStream {
        let mut stream = std::ptr::null_mut();
        unsafe { hip::check(h, (h.api.hip_stream_create)(&mut stream)).unwrap() };
        stream
    }

    /// Deterministic pseudo-random logits (same generator as the old argmax
    /// tests).
    fn lcg_logits(vocab: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed;
        (0..vocab)
            .map(|_| {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((rng >> 40) as f32) / (1u64 << 24) as f32
            })
            .collect()
    }

    fn gpu_sample(
        h: &Arc<Hip>,
        logits: &[f32],
        params: &mut [SamplingParams],
        vocab: usize,
    ) -> Vec<u32> {
        let stream = make_stream(h);
        let dlogits = upload_logits(h, logits);
        let s = BatchedSampler::new(Arc::clone(h), stream, params.len()).unwrap();
        let empty: Vec<Vec<(u32, u32)>> = (0..params.len()).map(|_| Vec::new()).collect();
        let ebias: Vec<Vec<(u32, f32)>> = (0..params.len()).map(|_| Vec::new()).collect();
        let (got, _, _) = s
            .sample_batched(dlogits, params, &empty, &ebias, vocab)
            .unwrap();
        hip::free(h, dlogits as *mut _).unwrap();
        unsafe { hip::check(h, (h.api.hip_stream_destroy)(stream)).unwrap() };
        got
    }

    #[test]
    fn greedy_matches_argmax() {
        let Some(h) = have_gpu() else { return };
        let vocab = 4096usize;
        let n = 8usize;
        let logits = lcg_logits(vocab * n, 7);
        let mut params: Vec<SamplingParams> = (0..n)
            .map(|i| SamplingParams::greedy(100 + i as u64))
            .collect();
        let got = gpu_sample(&h, &logits, &mut params, vocab);
        for i in 0..n {
            let want = cpu_argmax_first(&logits[i * vocab..(i + 1) * vocab]);
            assert_eq!(got[i], want, "seq {i} greedy must equal argmax");
        }
    }

    #[test]
    fn batched_sampler_is_deterministic() {
        let Some(h) = have_gpu() else { return };
        let vocab = 8192usize;
        let logits = lcg_logits(vocab * 4, 11);
        let mut a: Vec<SamplingParams> = (0..4)
            .map(|i| SamplingParams {
                temperature: 0.9,
                top_k: 0,
                top_p: 0.95,
                seed: 500 + i as u64,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                top_logprobs: 0,
            })
            .collect();
        let mut b = a.clone();
        let got_a = gpu_sample(&h, &logits, &mut a, vocab);
        let got_b = gpu_sample(&h, &logits, &mut b, vocab);
        assert_eq!(got_a, got_b, "same seed must sample the same tokens");
    }

    #[test]
    fn batched_sampler_matches_cpu_reference() {
        let Some(h) = have_gpu() else { return };
        // Peaked, well-separated logits: the top-p boundary and the RNG draw
        // land far from any bin edge, so the GPU/CPU float differences cannot
        // change the outcome.
        let vocab = 2048usize;
        let n = 8usize;
        let mut logits = vec![0.0f32; vocab * n];
        for s in 0..n {
            let row = &mut logits[s * vocab..(s + 1) * vocab];
            // Dominant geometric tail: probs fall ~ e^-1.2 per step.
            for (i, v) in row.iter_mut().enumerate() {
                *v = 0.5 - 1.2 * (i as f32) * 0.25;
            }
            // Perturb moderately so top-k matters but stays well separated.
            for i in (0..vocab).step_by(37) {
                row[i] += 0.05;
            }
        }
        let cases: Vec<SamplingParams> = (0..n)
            .map(|i| SamplingParams {
                temperature: [0.8, 1.0, 1.2][i % 3],
                top_k: [0usize, 0, 40, 10][i % 4],
                top_p: [0.95f32, 1.0, 0.9, 0.8][i % 4],
                seed: 7000 + i as u64,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                top_logprobs: 0,
            })
            .collect();
        let mut params = cases.clone();
        let got = gpu_sample(&h, &logits, &mut params, vocab);
        for (i, p) in cases.iter().enumerate() {
            let mut state = p.seed;
            let want = sample_cpu(&logits[i * vocab..(i + 1) * vocab], p, &[], &[], &mut state);
            assert_eq!(
                got[i], want,
                "seq {i} params={p:?}: gpu={} cpu={}",
                got[i], want
            );
        }
    }

    #[test]
    fn sampler_covers_multiple_tokens_over_seeds() {
        // Sanity: with top_p=0.9 and a spread distribution, different seeds
        // should pick different tokens (the sampler is not stuck on argmax).
        let vocab = 1024usize;
        let mut logits = vec![0.0f32; vocab];
        for (i, v) in logits.iter_mut().enumerate() {
            *v = 0.5 - 0.9 * (i as f32) * 0.15;
        }
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.9,
            seed: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            top_logprobs: 0,
        };
        let mut seen = std::collections::BTreeSet::new();
        for k in 0..200u64 {
            let mut state = p.seed.wrapping_add(k * 7919);
            seen.insert(sample_cpu(&logits, &p, &[], &[], &mut state));
        }
        assert!(
            seen.len() >= 8,
            "expected a varied distribution, got {} distinct tokens",
            seen.len()
        );
    }
    #[test]
    fn sample_returns_logprobs() {
        let Some(h) = have_gpu() else { return };
        let vocab = 4096usize;
        let logits = lcg_logits(vocab * 4, 17);
        let n = 4usize;
        // Greedy -> logprob 0 (probability 1).
        let mut gp: Vec<SamplingParams> = (0..n)
            .map(|i| SamplingParams::greedy(10 + i as u64))
            .collect();
        let (toks, lps, _) = {
            let stream = make_stream(&h);
            let d = upload_logits(&h, &logits);
            let s = BatchedSampler::new(Arc::clone(&h), stream, n).unwrap();
            let empty: Vec<Vec<(u32, u32)>> = (0..gp.len()).map(|_| Vec::new()).collect();
            let ebias: Vec<Vec<(u32, f32)>> = (0..gp.len()).map(|_| Vec::new()).collect();
            let r = s.sample_batched(d, &mut gp, &empty, &ebias, vocab).unwrap();
            hip::free(&h, d as *mut _).unwrap();
            unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };
            r
        };
        assert_eq!(toks.len(), n);
        assert_eq!(lps.len(), n);
        assert!(
            lps.iter().all(|&v| v == 0.0),
            "greedy logprobs must be 0, got {lps:?}"
        );

        // Sampling -> logprob must be finite, <= 0 (log of a probability).
        let mut sp: Vec<SamplingParams> = (0..n)
            .map(|i| SamplingParams {
                temperature: 0.9,
                top_k: 0,
                top_p: 0.95,
                seed: 500 + i as u64,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                top_logprobs: 0,
            })
            .collect();
        let (_, lps2, _) = {
            let stream = make_stream(&h);
            let d = upload_logits(&h, &logits);
            let s = BatchedSampler::new(Arc::clone(&h), stream, n).unwrap();
            let empty: Vec<Vec<(u32, u32)>> = (0..sp.len()).map(|_| Vec::new()).collect();
            let ebias: Vec<Vec<(u32, f32)>> = (0..sp.len()).map(|_| Vec::new()).collect();
            let r = s.sample_batched(d, &mut sp, &empty, &ebias, vocab).unwrap();
            hip::free(&h, d as *mut _).unwrap();
            unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };
            r
        };
        for &v in &lps2 {
            assert!(
                v.is_finite() && v <= 0.0,
                "sampled logprob must be <= 0, got {v}"
            );
        }
    }
    #[test]
    fn penalties_match_cpu_reference() {
        let Some(h) = have_gpu() else { return };
        let vocab = 2048usize;
        let n = 6usize;
        // Peaked logits; penalties target some top tokens so the distribution
        // shifts but stays well-separated (exact GPU==CPU match expected).
        let mut logits = vec![0.0f32; vocab * n];
        for s in 0..n {
            let row = &mut logits[s * vocab..(s + 1) * vocab];
            for (i, v) in row.iter_mut().enumerate() {
                *v = 0.5 - 1.2 * (i as f32) * 0.25;
            }
        }
        let mut params: Vec<SamplingParams> = (0..n)
            .map(|i| SamplingParams {
                temperature: if i == 0 { 0.0 } else { 0.9 },
                top_k: 0,
                top_p: 0.95,
                seed: 9000 + i as u64,
                presence_penalty: 0.5,
                frequency_penalty: 0.3,
                top_logprobs: 0,
            })
            .collect();
        // Counts: token 0 has appeared 3x, token 1 x1, token 10 x2.
        let counts: Vec<Vec<(u32, u32)>> = (0..n)
            .map(|_| vec![(0u32, 3u32), (1, 1), (10, 2)])
            .collect();
        let cases = params.clone();
        let (got, _, _) = {
            let stream = make_stream(&h);
            let d = upload_logits(&h, &logits);
            let s = BatchedSampler::new(Arc::clone(&h), stream, n).unwrap();
            let ebias: Vec<Vec<(u32, f32)>> = (0..params.len()).map(|_| Vec::new()).collect();
            let r = s
                .sample_batched(d, &mut params, &counts, &ebias, vocab)
                .unwrap();
            hip::free(&h, d as *mut _).unwrap();
            unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };
            r
        };
        for (i, p) in cases.iter().enumerate() {
            let mut state = p.seed;
            let want = sample_cpu(
                &logits[i * vocab..(i + 1) * vocab],
                p,
                &counts[i],
                &[],
                &mut state,
            );
            assert_eq!(
                got[i], want,
                "seq {i} penalty sample mismatch: gpu={} cpu={}",
                got[i], want
            );
        }
    }
    #[test]
    fn logit_bias_matches_cpu_reference() {
        let Some(h) = have_gpu() else { return };
        let vocab = 2048usize;
        let n = 4usize;
        let mut logits = vec![0.0f32; vocab * n];
        for s in 0..n {
            let row = &mut logits[s * vocab..(s + 1) * vocab];
            for (i, v) in row.iter_mut().enumerate() {
                *v = 0.5 - 1.2 * (i as f32) * 0.25;
            }
        }
        let mut params: Vec<SamplingParams> = (0..n)
            .map(|i| SamplingParams {
                temperature: if i == 0 { 0.0 } else { 0.9 },
                top_k: 0,
                top_p: 0.95,
                seed: 4000 + i as u64,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                top_logprobs: 0,
            })
            .collect();
        // Bias strongly toward token 200 and away from token 0.
        let bias: Vec<Vec<(u32, f32)>> = (0..n).map(|_| vec![(200u32, 12.0), (0, -12.0)]).collect();
        let cases = params.clone();
        let (got, _, _) = {
            let stream = make_stream(&h);
            let d = upload_logits(&h, &logits);
            let s = BatchedSampler::new(Arc::clone(&h), stream, n).unwrap();
            let empty: Vec<Vec<(u32, u32)>> = (0..n).map(|_| Vec::new()).collect();
            let r = s
                .sample_batched(d, &mut params, &empty, &bias, vocab)
                .unwrap();
            hip::free(&h, d as *mut _).unwrap();
            unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };
            r
        };
        for (i, p) in cases.iter().enumerate() {
            let mut state = p.seed;
            let want = sample_cpu(
                &logits[i * vocab..(i + 1) * vocab],
                p,
                &[],
                &bias[i],
                &mut state,
            );
            assert_eq!(
                got[i], want,
                "seq {i} logit_bias sample mismatch: gpu={} cpu={}",
                got[i], want
            );
        }
    }

    #[test]
    fn topk_matches_cpu_reference() {
        let Some(h) = have_gpu() else { return };
        let vocab = 4096usize;
        let n = 4usize;
        // Monotonic, well-separated logits: top-k tokens are unambiguous and
        // the GPU/CPU float sums cannot flip the ordering.
        let mut logits = vec![0.0f32; vocab * n];
        for s in 0..n {
            let row = &mut logits[s * vocab..(s + 1) * vocab];
            for (i, v) in row.iter_mut().enumerate() {
                *v = 0.5 - 1.1 * (i as f32) * 0.2;
            }
        }
        let mut params: Vec<SamplingParams> = (0..n)
            .map(|i| SamplingParams {
                temperature: [0.0, 0.9, 1.0, 1.2][i],
                top_k: 0,
                top_p: 1.0,
                seed: 11_000 + i as u64,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                top_logprobs: [0, 3, 5, 8][i],
            })
            .collect();
        let (_, _, got) = {
            let stream = make_stream(&h);
            let d = upload_logits(&h, &logits);
            let s = BatchedSampler::new(Arc::clone(&h), stream, n).unwrap();
            let empty: Vec<Vec<(u32, u32)>> = (0..n).map(|_| Vec::new()).collect();
            let ebias: Vec<Vec<(u32, f32)>> = (0..n).map(|_| Vec::new()).collect();
            let r = s
                .sample_batched(d, &mut params, &empty, &ebias, vocab)
                .unwrap();
            hip::free(&h, d as *mut _).unwrap();
            unsafe { hip::check(&h, (h.api.hip_stream_destroy)(stream)).unwrap() };
            r
        };
        for (i, p) in params.iter().enumerate() {
            assert_eq!(
                got[i].len(),
                p.top_logprobs,
                "seq {i} top_logprobs length mismatch"
            );
            let want = topk_cpu(
                &logits[i * vocab..(i + 1) * vocab],
                p,
                &[],
                &[],
                p.top_logprobs,
            );
            for (j, ((g_tok, g_lp), (c_tok, c_lp))) in got[i].iter().zip(want.iter()).enumerate() {
                assert_eq!(g_tok, c_tok, "seq {i} topk token {j} mismatch");
                assert!(
                    (g_lp - c_lp).abs() < 1e-3,
                    "seq {i} topk logprob {j}: gpu={g_lp} cpu={c_lp}"
                );
            }
        }
    }
}
