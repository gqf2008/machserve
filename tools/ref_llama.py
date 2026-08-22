#!/usr/bin/env python3
"""Independent fp64 reference for the tiny Llama decode, used to validate the
Rust GPU path numerically. Mirrors mach_model::ref_model (rmsnorm, rope,
GQA attention, swiglu MLP, lm_head) reading a real safetensors checkpoint.

Usage:
  python tools/ref_llama.py <model.safetensors> [rust_logits.json] [tokens...]
Prints the reference logits (and, when rust_logits.json is given, the max
absolute difference vs the Rust output).
"""
import json
import struct
import sys


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
        n = 1
        for d in shape:
            n *= d
        if dtype == "F32":
            vals = struct.unpack("<%df" % n, raw)
        elif dtype == "F16":
            vals = [struct.unpack("<e", raw[i * 2 : i * 2 + 2])[0] for i in range(n)]
        elif dtype == "BF16":
            vals = [
                struct.unpack("<f", struct.pack("<I", struct.unpack("<H", raw[i * 2 : i * 2 + 2])[0] << 16))[0]
                for i in range(n)
            ]
        else:
            raise ValueError("unsupported dtype " + dtype)
        out[name] = list(vals)
    return out


def rms_norm(x, w, eps):
    n = len(x)
    mean = sum(v * v for v in x) / n
    inv = 1.0 / (mean + eps) ** 0.5
    return [x[i] * inv * w[i] for i in range(n)]


def matvec_t(x, w, out_dim):
    k = len(x)
    return [sum(w[o * k + i] * x[i] for i in range(k)) for o in range(out_dim)]


def rope(x, n_heads, head_dim, pos, theta):
    half = head_dim // 2
    for h in range(n_heads):
        for d in range(half):
            freq = 1.0 / theta ** (2.0 * d / head_dim)
            ang = pos * freq
            c, sn = math.cos(ang), math.sin(ang)
            idx = h * head_dim + 2 * d
            a, b = x[idx], x[idx + 1]
            x[idx] = a * c - b * sn
            x[idx + 1] = a * sn + b * c


def attention_decode(q, kc, vc, pos, n_heads, n_kv_heads, head_dim):
    scale = head_dim ** -0.5
    groups = n_heads // n_kv_heads
    out = [0.0] * (n_heads * head_dim)
    for h in range(n_heads):
        kv = h // groups
        qh = q[h * head_dim : (h + 1) * head_dim]
        scores = []
        for p in range(pos + 1):
            kp = kc[(p * n_kv_heads + kv) * head_dim : (p * n_kv_heads + kv + 1) * head_dim]
            scores.append(sum(qh[d] * kp[d] for d in range(head_dim)) * scale)
        m = max(scores)
        ex = [math.exp(s - m) for s in scores]
        tot = sum(ex)
        for d in range(head_dim):
            acc = sum(ex[p] * vc[(p * n_kv_heads + kv) * head_dim + d] for p in range(pos + 1))
            out[h * head_dim + d] = acc / tot
    return out


def forward(w, cfg, tokens):
    d = cfg["hidden"]
    nq = cfg["heads"] * cfg["head_dim"]
    nkv = cfg["kv_heads"] * cfg["head_dim"]
    kv = [([0.0] * (cfg["max_seq"] * nkv), [0.0] * (cfg["max_seq"] * nkv)) for _ in range(cfg["layers"])]
    logits = None
    for pos, tok in enumerate(tokens):
        x = w["model.embed_tokens.weight"][tok * d : (tok + 1) * d]
        for li in range(cfg["layers"]):
            p = lambda s: "model.layers.%d.%s" % (li, s)
            xn = rms_norm(x, w[p("input_layernorm.weight")], cfg["eps"])
            q = matvec_t(xn, w[p("self_attn.q_proj.weight")], nq)
            k = matvec_t(xn, w[p("self_attn.k_proj.weight")], nkv)
            v = matvec_t(xn, w[p("self_attn.v_proj.weight")], nkv)
            rope(q, cfg["heads"], cfg["head_dim"], pos, cfg["theta"])
            rope(k, cfg["kv_heads"], cfg["head_dim"], pos, cfg["theta"])
            kv[li][0][pos * nkv : (pos + 1) * nkv] = k
            kv[li][1][pos * nkv : (pos + 1) * nkv] = v
            attn = attention_decode(q, kv[li][0], kv[li][1], pos, cfg["heads"], cfg["kv_heads"], cfg["head_dim"])
            proj = matvec_t(attn, w[p("self_attn.o_proj.weight")], d)
            x = [x[i] + proj[i] for i in range(d)]
            xn = rms_norm(x, w[p("post_attention_layernorm.weight")], cfg["eps"])
            gate = matvec_t(xn, w[p("mlp.gate_proj.weight")], cfg["intermediate"])
            up = matvec_t(xn, w[p("mlp.up_proj.weight")], cfg["intermediate"])
            h = [gate[i] * (up[i] / (1.0 + math.exp(-up[i]))) for i in range(cfg["intermediate"])]
            down = matvec_t(h, w[p("mlp.down_proj.weight")], d)
            x = [x[i] + down[i] for i in range(d)]
        xf = rms_norm(x, w["model.norm.weight"], cfg["eps"])
        logits = matvec_t(xf, w["lm_head.weight"], cfg["vocab"])
    return logits


def main():
    import math  # noqa: F401 (used in closures via module globals)

    global math
    import math

    path = sys.argv[1]
    rust_json = sys.argv[2] if len(sys.argv) > 2 else None
    tokens = [int(t) for t in sys.argv[3:]] or [1, 2, 3, 4, 5]

    w = load_safetensors(path)
    cfg = {
        "hidden": 16,
        "layers": 2,
        "heads": 4,
        "kv_heads": 4,
        "head_dim": 4,
        "intermediate": 64,
        "vocab": 32000,
        "max_seq": 2048,
        "eps": 1e-6,
        "theta": 10000.0,
    }
    ref = forward(w, cfg, tokens)
    print("ref logits[0..8] =", [round(v, 6) for v in ref[:8]])
    print("ref max |x| =", max(abs(v) for v in ref))

    if rust_json:
        rust = json.load(open(rust_json))
        diff = max(abs(a - b) for a, b in zip(rust, ref))
        scale = max(abs(v) for v in ref)
        print("RUST vs REF: max abs diff = %.6e  (scale %.3f, rel %.3e)" % (diff, scale, diff / scale))


if __name__ == "__main__":
    main()
