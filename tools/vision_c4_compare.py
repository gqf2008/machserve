#!/usr/bin/env python3
"""Compare HF merged vision features (.npy) with a mach-server dump.

mach-server dump: MACH_VISION_DUMP=<prefix> writes <prefix>.bin (raw
little-endian f32 merged features) and <prefix>.json (grids/values/sum).
"""

import argparse
import json
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-npy", required=True)
    parser.add_argument("--ms-prefix", required=True)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    args = parser.parse_args()

    hf = np.load(args.hf_npy).astype(np.float64).reshape(-1)
    ms = np.fromfile(args.ms_prefix + ".bin", dtype=np.float32).astype(np.float64)
    with open(args.ms_prefix + ".json", encoding="utf-8") as handle:
        meta = json.load(handle)
    if hf.shape != ms.shape:
        print("shape mismatch: hf", hf.shape, "ms", ms.shape, file=sys.stderr)
        return 2
    diff = np.abs(hf - ms)
    denom = np.maximum(np.abs(hf), 1e-12)
    rel = diff / denom
    nonfinite = int(np.sum(~np.isfinite(hf)) + np.sum(~np.isfinite(ms)))
    result = {
        "hf_shape": list(hf.shape),
        "ms_values": meta.get("values"),
        "ms_grids": meta.get("grids"),
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "max_rel_diff": float(rel.max()) if rel.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "nonfinite": nonfinite,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    print(json.dumps(result, indent=2))
    ok = nonfinite == 0 and bool(np.allclose(hf, ms, atol=args.atol, rtol=args.rtol))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
