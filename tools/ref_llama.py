#!/usr/bin/env python3
"""Independent fp64 (numpy) reference for Llama/Qwen decode, used to validate
the Rust GPU path numerically. Mirrors mach_model's forward (rmsnorm, rope,
GQA attention, swiglu MLP, lm_head) reading a real safetensors checkpoint.

Usage:
  python tools/ref_llama.py <model.safetensors> [config.json] [rust_logits.json] [tokens...]
"""
import json
import struct
import sys

import numpy as np


def load_safetensors(path):
    with open(path, "rb") as f:
        head_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(head_len))
        data = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dtype = meta["dtype"]
        shape = meta["shape"]
        start, end = meta["data_offsets"]
        raw = data[start:end]
        n = int(np.prod(shape))
        if dtype == "F32":
            vals = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        elif dtype == "F16":
            vals = np.frombuffer(raw, dtype="<f2").astype(np.float64)
        elif dtype == "BF16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            vals = (u << 16).view(np.float32).astype(np.float64)
        else:
            raise ValueError("unsupported dtype " + dtype)
        out[name] = vals.reshape(shape)
    return out


def rms_norm(x, w, eps):
    mean = np.mean(x * x)
    inv = 1.0 / np.sqrt(mean + eps)
    return x * inv * w


def matvec_t(x, w, out_dim):
    # w is [out, in] row-major; out = w @ x
    return w @ x


def rope(x, n_heads, head_dim, pos, theta):
    half = head_dim // 2
    d = np.arange(half, dtype=np.float64)
    freq = 1.0 / (theta ** (2.0 * d / head_dim))
    ang = pos * freq
    c = np.cos(ang)
    s = np.sin(ang)
    x = x.reshape(n_heads, head_dim)
    # GPT-NeoX rotary: pairs (d, d + half), matching HF rotate_half.
    a = x[:, :half].copy()
    b = x[:, half:].copy()
    x[:, :half] = a * c - b * s
    x[:, half:] = a * s + b * c
    return x.reshape(-1)


def attention_decode(q, kc, vc, pos, n_heads, n_kv_heads, head_dim):
    # kc/vc are [max_seq, n_kv_heads, head_dim]; only rows 0..=pos are valid.
    scale = head_dim ** -0.5
    groups = n_heads // n_kv_heads
    out = np.zeros(n_heads * head_dim, dtype=np.float64)
    q = q.reshape(n_heads, head_dim)
    for h in range(n_heads):
        kv = h // groups
        kp = kc[: pos + 1, kv, :]  # [pos+1, head_dim]
        vp = vc[: pos + 1, kv, :]
        scores = (kp @ q[h]) * scale
        m = scores.max()
        ex = np.exp(scores - m)
        out[h * head_dim : (h + 1) * head_dim] = (ex @ vp) / ex.sum()
    return out


def forward(w, cfg, tokens):
    d = cfg["hidden"]
    nq = cfg["heads"] * cfg["head_dim"]
    nkv = cfg["kv_heads"] * cfg["head_dim"]
    lm = w.get("lm_head.weight", w["model.embed_tokens.weight"])
    # Persistent per-layer KV caches (rows 0..=pos valid).
    kcache = [np.zeros((cfg["max_seq"], cfg["kv_heads"], cfg["head_dim"]), dtype=np.float64) for _ in range(cfg["layers"])]
    vcache = [np.zeros((cfg["max_seq"], cfg["kv_heads"], cfg["head_dim"]), dtype=np.float64) for _ in range(cfg["layers"])]
    logits = None
    for pos, tok in enumerate(tokens):
        x = w["model.embed_tokens.weight"][tok].copy()
        for li in range(cfg["layers"]):
            p = lambda s: "model.layers.%d.%s" % (li, s)
            xn = rms_norm(x, w[p("input_layernorm.weight")], cfg["eps"])
            q = matvec_t(xn, w[p("self_attn.q_proj.weight")], nq)
            k = matvec_t(xn, w[p("self_attn.k_proj.weight")], nkv)
            v = matvec_t(xn, w[p("self_attn.v_proj.weight")], nkv)
            # Qwen2 checkpoints ship q/k/v biases even when the config says
            # `attention_bias: false`; add them when present.
            bq = w.get(p("self_attn.q_proj.bias"), None)
            bk = w.get(p("self_attn.k_proj.bias"), None)
            bv = w.get(p("self_attn.v_proj.bias"), None)
            if bq is not None:
                q = q + bq
            if bk is not None:
                k = k + bk
            if bv is not None:
                v = v + bv
            q = rope(q, cfg["heads"], cfg["head_dim"], pos, cfg["theta"])
            k = rope(k, cfg["kv_heads"], cfg["head_dim"], pos, cfg["theta"])
            kcache[li][pos] = k.reshape(cfg["kv_heads"], cfg["head_dim"])
            vcache[li][pos] = v.reshape(cfg["kv_heads"], cfg["head_dim"])
            attn = attention_decode(q, kcache[li], vcache[li], pos, cfg["heads"], cfg["kv_heads"], cfg["head_dim"])
            x = x + matvec_t(attn, w[p("self_attn.o_proj.weight")], d)
            xn = rms_norm(x, w[p("post_attention_layernorm.weight")], cfg["eps"])
            gate = matvec_t(xn, w[p("mlp.gate_proj.weight")], cfg["intermediate"])
            up = matvec_t(xn, w[p("mlp.up_proj.weight")], cfg["intermediate"])
            h = (gate / (1.0 + np.exp(-gate))) * up
            x = x + matvec_t(h, w[p("mlp.down_proj.weight")], d)
        xf = rms_norm(x, w["model.norm.weight"], cfg["eps"])
        logits = matvec_t(xf, lm, cfg["vocab"])
    return logits


def main():
    path = sys.argv[1]
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else None
    rust_json = sys.argv[3] if len(sys.argv) > 3 else None
    tokens = [int(t) for t in sys.argv[4:]] or [1, 2, 3, 4, 5]

    w = load_safetensors(path)
    if cfg_path:
        c = json.load(open(cfg_path))
        hidden = c.get("hidden_size", 16)
        n_heads = c.get("num_attention_heads", 4)
        kv = c.get("num_key_value_heads", n_heads)
        head_dim = c.get("head_dim", hidden // n_heads)
        cfg = {
            "hidden": hidden,
            "layers": c.get("num_hidden_layers", 2),
            "heads": n_heads,
            "kv_heads": kv,
            "head_dim": head_dim,
            "intermediate": c.get("intermediate_size", 4 * hidden),
            "vocab": c.get("vocab_size", 32000),
            "eps": c.get("rms_norm_eps", 1e-6),
            "theta": c.get("rope_theta", 10000.0),
            "max_seq": min(c.get("max_position_embeddings", 2048), 2048),
        }
    else:
        cfg = {
            "hidden": 16, "layers": 2, "heads": 4, "kv_heads": 4, "head_dim": 4,
            "intermediate": 64, "vocab": 32000, "eps": 1e-6, "theta": 10000.0,
            "max_seq": 2048,
        }
    ref = forward(w, cfg, tokens)
    print("ref logits[0..8] =", [round(float(v), 6) for v in ref[:8]])
    print("ref max |x| =", float(np.max(np.abs(ref))))

    if rust_json:
        rust = json.load(open(rust_json))
        diff = float(np.max(np.abs(np.asarray(rust) - ref)))
        scale = float(np.max(np.abs(ref)))
        print("RUST vs REF: max abs diff = %.6e  (scale %.3f, rel %.3e)" % (diff, scale, diff / scale))


if __name__ == "__main__":
    main()
