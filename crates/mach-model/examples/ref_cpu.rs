//! CPU reference dump for cross-validation with `tools/ref_llama.py`.
//! Computes the forward pass in both f32 (production path) and f64
//! (precision reference), so fp32-vs-fp64 noise can be separated from bugs.

use mach_model::loader::load_safetensors;
use mach_model::ref_model::RefModel;
use mach_model::{Config, Weights};
use std::path::PathBuf;

fn main() {
    let root = std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into());
    let model = std::env::var("MACH_MODEL").unwrap_or_else(|_| "tiny-llama.safetensors".into());
    let model_path = PathBuf::from(&root).join(&model);
    let cfg = if model.starts_with("qwen") {
        let mut c = Config::llama(896, 24, 14, 2, 151936, 2048);
        c.intermediate_size = 4864;
        c.rope_theta = 1_000_000.0;
        c
    } else {
        Config::llama(16, 2, 4, 4, 32000, 2048)
    };
    let tie = model.starts_with("qwen");
    let w: Weights = load_safetensors(&model_path, &cfg, tie).expect("load weights");

    let tokens: Vec<u32> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let tokens = if tokens.is_empty() {
        vec![1, 2, 3]
    } else {
        tokens
    };

    // f32 production path
    let mut m32 = RefModel::new(cfg, w.clone());
    let l32 = m32.forward(&tokens);
    std::fs::write(
        PathBuf::from(&root).join("rust_cpu_logits.json"),
        serde_json::to_string(&l32).unwrap(),
    )
    .unwrap();

    // f64 reference path (independent implementation)
    let l64 = forward_f64(&w, &cfg, &tokens);
    std::fs::write(
        PathBuf::from(&root).join("rust_cpu_f64_logits.json"),
        serde_json::to_string(&l64).unwrap(),
    )
    .unwrap();
    if model.starts_with("qwen") {
        std::fs::write(
            PathBuf::from(&root).join("qwen_rust_cpu_f64_logits.json"),
            serde_json::to_string(&l64).unwrap(),
        )
        .unwrap();
    }

    let d32 = max_diff(&l32, &l64);
    println!("f32 vs f64 max abs diff: {d32:.3e}");

    if std::env::var("MACH_LAYER0").is_ok() {
        let layer0 = forward_f64_layer0(&w, &cfg, tokens[0]);
        std::fs::write(
            PathBuf::from(&root).join("qwen_f64_layer0.json"),
            serde_json::to_string(&layer0).unwrap(),
        )
        .unwrap();
        println!("f64 layer0 dumped (xn/q/k/x)");
        return;
    }
    println!(
        "f64[0..8] = {:?}",
        l64[..8]
            .iter()
            .map(|v| (v * 1e6).round() / 1e6)
            .collect::<Vec<_>>()
    );
}

fn max_diff(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - y).abs())
        .fold(0.0f64, f64::max)
}

fn rms_norm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().map(|v| v * v).sum::<f64>() / n;
    let inv = 1.0 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(v, wi)| v * inv * wi).collect()
}

fn matvec_t(x: &[f64], w: &[f64], out_dim: usize) -> Vec<f64> {
    let k = x.len();
    (0..out_dim)
        .map(|o| (0..k).map(|i| w[o * k + i] * x[i]).sum())
        .collect()
}

/// RoPE, with the pairing convention chosen by `interleave` so this oracle
/// agrees with `tools/ref_llama.py` (which pairs `d` with `d + half`,
/// "matching HF rotate_half") on the Llama/Qwen configs it is used with.
/// Hardcoding adjacent pairs here silently disagreed with that Python
/// reference at every `pos > 0` — invisible at `pos == 0`, where cos=1 and
/// sin=0 make RoPE the identity.
fn rope(x: &mut [f64], n_heads: usize, head_dim: usize, pos: usize, theta: f64, interleave: bool) {
    let half = head_dim / 2;
    for h in 0..n_heads {
        for d in 0..half {
            let freq = 1.0 / theta.powf(2.0 * d as f64 / head_dim as f64);
            let ang = pos as f64 * freq;
            let c = ang.cos();
            let sn = ang.sin();
            let i0 = h * head_dim + if interleave { 2 * d } else { d };
            let i1 = h * head_dim + if interleave { 2 * d + 1 } else { d + half };
            let (a, b) = (x[i0], x[i1]);
            x[i0] = a * c - b * sn;
            x[i1] = a * sn + b * c;
        }
    }
}

fn attention(
    q: &[f64],
    kc: &[f64],
    vc: &[f64],
    pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f64> {
    let scale = 1.0 / (head_dim as f64).sqrt();
    let groups = n_heads / n_kv_heads;
    let mut out = vec![0.0; n_heads * head_dim];
    for h in 0..n_heads {
        let kv = h / groups;
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = vec![0.0; pos + 1];
        let mut maxv = f64::NEG_INFINITY;
        for p in 0..=pos {
            let kp = &kc[(p * n_kv_heads + kv) * head_dim..(p * n_kv_heads + kv + 1) * head_dim];
            let s = qh.iter().zip(kp).map(|(a, b)| a * b).sum::<f64>() * scale;
            scores[p] = s;
            maxv = maxv.max(s);
        }
        let mut tot = 0.0;
        for s in &mut scores {
            *s = (*s - maxv).exp();
            tot += *s;
        }
        for dd in 0..head_dim {
            let acc = (0..=pos)
                .map(|p| scores[p] * vc[(p * n_kv_heads + kv) * head_dim + dd])
                .sum::<f64>();
            out[h * head_dim + dd] = acc / tot;
        }
    }
    out
}

fn forward_f64(w: &Weights, cfg: &Config, tokens: &[u32]) -> Vec<f64> {
    let d = cfg.d_model;
    let nq = cfg.n_heads * cfg.head_dim;
    let nkv = cfg.n_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let mut logits = vec![0.0; cfg.vocab_size];
    // Persistent per-layer KV caches.
    let mut kcaches: Vec<Vec<f64>> = (0..cfg.n_layers)
        .map(|_| vec![0.0; cfg.max_seq_len * nkv])
        .collect();
    let mut vcaches: Vec<Vec<f64>> = (0..cfg.n_layers)
        .map(|_| vec![0.0; cfg.max_seq_len * nkv])
        .collect();
    for (pos, &tok) in tokens.iter().enumerate() {
        let emb: Vec<f64> = w.tok_emb[tok as usize * d..(tok as usize + 1) * d]
            .iter()
            .map(|v| *v as f64)
            .collect();
        let mut x = emb;
        for (li, lw) in w.layers.iter().enumerate() {
            let xn = rms_norm(
                &x,
                &lw.rms_attn.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                cfg.rms_eps as f64,
            );
            let mut q = matvec_t(
                &xn,
                &lw.wq.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                nq,
            );
            let mut k = matvec_t(
                &xn,
                &lw.wk.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                nkv,
            );
            let v = matvec_t(
                &xn,
                &lw.wv.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                nkv,
            );
            rope(
                &mut q,
                cfg.n_heads,
                cfg.head_dim,
                pos,
                cfg.rope_theta as f64,
                cfg.rope_interleave,
            );
            rope(
                &mut k,
                cfg.n_kv_heads,
                cfg.head_dim,
                pos,
                cfg.rope_theta as f64,
                cfg.rope_interleave,
            );
            let _ = li;
            kcaches[li][pos * nkv..(pos + 1) * nkv].copy_from_slice(&k);
            vcaches[li][pos * nkv..(pos + 1) * nkv].copy_from_slice(&v);
            let attn = attention(
                &q,
                &kcaches[li],
                &vcaches[li],
                pos,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
            );
            let proj = matvec_t(
                &attn,
                &lw.wo.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                d,
            );
            for i in 0..d {
                x[i] += proj[i];
            }
            let xn2 = rms_norm(
                &x,
                &lw.rms_mlp.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                cfg.rms_eps as f64,
            );
            let gate = matvec_t(
                &xn2,
                &lw.wg.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                inter,
            );
            let up = matvec_t(
                &xn2,
                &lw.wu.iter().map(|v| *v as f64).collect::<Vec<_>>(),
                inter,
            );
            let h: Vec<f64> = (0..inter)
                .map(|i| gate[i] * (up[i] / (1.0 + (-up[i]).exp())))
                .collect();
            let down = matvec_t(&h, &lw.wd.iter().map(|v| *v as f64).collect::<Vec<_>>(), d);
            for i in 0..d {
                x[i] += down[i];
            }
        }
        let xf = rms_norm(
            &x,
            &w.rms_final.iter().map(|v| *v as f64).collect::<Vec<_>>(),
            cfg.rms_eps as f64,
        );
        logits = matvec_t(
            &xf,
            &w.lm_head.iter().map(|v| *v as f64).collect::<Vec<_>>(),
            cfg.vocab_size,
        );
    }
    logits
}

/// Single-layer f64 forward for one token, returning (xn, q, k, x after layer 0).
fn forward_f64_layer0(w: &Weights, cfg: &Config, token: u32) -> Vec<Vec<f64>> {
    let d = cfg.d_model;
    let nq = cfg.n_heads * cfg.head_dim;
    let nkv = cfg.n_kv_heads * cfg.head_dim;
    let inter = cfg.intermediate_size;
    let lw = &w.layers[0];
    let emb: Vec<f64> = w.tok_emb[token as usize * d..(token as usize + 1) * d]
        .iter()
        .map(|v| *v as f64)
        .collect();
    let mut x = emb;
    let xn = rms_norm(
        &x,
        &lw.rms_attn.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        cfg.rms_eps as f64,
    );
    let mut q = matvec_t(
        &xn,
        &lw.wq.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        nq,
    );
    let mut k = matvec_t(
        &xn,
        &lw.wk.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        nkv,
    );
    let v = matvec_t(
        &xn,
        &lw.wv.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        nkv,
    );
    rope(
        &mut q,
        cfg.n_heads,
        cfg.head_dim,
        0,
        cfg.rope_theta as f64,
        cfg.rope_interleave,
    );
    rope(
        &mut k,
        cfg.n_kv_heads,
        cfg.head_dim,
        0,
        cfg.rope_theta as f64,
        cfg.rope_interleave,
    );
    let mut kc = vec![0.0; cfg.max_seq_len * nkv];
    let mut vc = vec![0.0; cfg.max_seq_len * nkv];
    kc[..nkv].copy_from_slice(&k);
    vc[..nkv].copy_from_slice(&v);
    let attn = attention(&q, &kc, &vc, 0, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    let proj = matvec_t(
        &attn,
        &lw.wo.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        d,
    );
    for i in 0..d {
        x[i] += proj[i];
    }
    let xn2 = rms_norm(
        &x,
        &lw.rms_mlp.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        cfg.rms_eps as f64,
    );
    let gate = matvec_t(
        &xn2,
        &lw.wg.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        inter,
    );
    let up = matvec_t(
        &xn2,
        &lw.wu.iter().map(|v| *v as f64).collect::<Vec<_>>(),
        inter,
    );
    let h: Vec<f64> = (0..inter)
        .map(|i| gate[i] * (up[i] / (1.0 + (-up[i]).exp())))
        .collect();
    let down = matvec_t(&h, &lw.wd.iter().map(|v| *v as f64).collect::<Vec<_>>(), d);
    for i in 0..d {
        x[i] += down[i];
    }
    vec![xn, q, k, x]
}
