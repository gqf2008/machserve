#!/usr/bin/env python3
"""Compare HF merged vision features (.npy) with a mach-server dump.

mach-server dump: MACH_VISION_DUMP=<prefix> writes <prefix>.bin (raw
little-endian f32 merged features) and <prefix>.json (grids/values/sum).
"""

import argparse
import json
import sys

import numpy as np


def hash_decision(hf_hash, ms_hash, require_hash):
    """Return True/False for the image SHA-256 binding, or None when optional."""
    if hf_hash is None or ms_hash is None:
        return False if require_hash else None
    return hf_hash == ms_hash


def selftest():
    assert hash_decision("a", "a", True) is True
    assert hash_decision("a", "b", True) is False
    assert hash_decision("a", None, True) is False
    assert hash_decision("a", None, False) is None
    print("vision_c4_compare selftest ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-npy")
    parser.add_argument("--ms-prefix")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--hf-json", default="")
    parser.add_argument("--e2e-summary", default="")
    parser.add_argument("--require-hash", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not args.hf_npy or not args.ms_prefix:
        parser.error("--hf-npy/--ms-prefix are required unless --selftest")

    hf = np.load(args.hf_npy).astype(np.float64).reshape(-1)
    raw = np.fromfile(args.ms_prefix + ".bin", dtype=np.float32)
    ms = raw.astype(np.float64)
    with open(args.ms_prefix + ".json", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("values") != int(raw.size):
        print("bin length mismatch: meta values", meta.get("values"), "bin", raw.size, file=sys.stderr)
        return 2
    if hf.shape != ms.shape:
        print("shape mismatch: hf", hf.shape, "ms", ms.shape, file=sys.stderr)
        return 2

    grid_match = None
    hf_hash = None
    if args.hf_json:
        with open(args.hf_json, encoding="utf-8") as handle:
            hf_meta = json.load(handle)
        grid_match = hf_meta.get("grid") == meta.get("grids")
        hf_hash = hf_meta.get("image_sha256")
    ms_hash = meta.get("image_sha256")
    if args.e2e_summary:
        with open(args.e2e_summary, encoding="utf-8") as handle:
            summary = json.load(handle)
        ms_hash = summary.get("image_sha256", ms_hash)
    image_hash_match = hash_decision(hf_hash, ms_hash, args.require_hash)

    finite = np.isfinite(hf) & np.isfinite(ms)
    nonfinite = int(np.sum(~finite))
    if nonfinite == 0:
        diff = np.abs(hf - ms)
        rel = diff / np.maximum(np.abs(hf), 1e-12)
        max_abs = float(diff.max()) if diff.size else 0.0
        max_rel = float(rel.max()) if rel.size else 0.0
        mean_abs = float(diff.mean()) if diff.size else 0.0
    else:
        max_abs = None
        max_rel = None
        mean_abs = None
    values_match = nonfinite == 0 and bool(
        np.allclose(hf[finite], ms[finite], atol=args.atol, rtol=args.rtol)
    )
    hash_ok = image_hash_match is not False
    if args.require_hash and image_hash_match is not True:
        hash_ok = False
    ok = values_match and grid_match is not False and hash_ok
    result = {
        "pass": ok,
        "hf_shape": list(hf.shape),
        "ms_values": meta.get("values"),
        "ms_grids": meta.get("grids"),
        "grid_match": grid_match,
        "image_hash_match": image_hash_match,
        "require_hash": args.require_hash,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "mean_abs_diff": mean_abs,
        "nonfinite": nonfinite,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
