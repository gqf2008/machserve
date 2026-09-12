#!/usr/bin/env python3
"""Compare HF full-model first-step/full-token artifacts with mach-server output.

HF was run on CPU with disk offload (BF16); the mach-server artifact uses
Q4-on-device. The hard gates are schema completeness, prompt length, image grid,
image-pad count, and the greedy token sequence. Top-5 logprobs are diagnostic
only because quantization changes the tail distribution.
"""

import argparse
import copy
import json
import sys


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require_int(obj, key, where, errors):
    value = obj.get(key) if isinstance(obj, dict) else None
    if not is_int(value):
        errors.append(f"{where}.{key} is missing or not an int")
        return None
    return value


def require_number(obj, key, where, errors):
    value = obj.get(key) if isinstance(obj, dict) else None
    if not is_number(value):
        errors.append(f"{where}.{key} is missing or not a number")
        return None
    return value


def require_nonempty_list(obj, key, where, errors):
    value = obj.get(key) if isinstance(obj, dict) else None
    if not isinstance(value, list) or not value:
        errors.append(f"{where}.{key} is missing or not a non-empty list")
        return None
    return value


def require_int_list(obj, key, where, errors):
    value = require_nonempty_list(obj, key, where, errors)
    if value is None:
        return None
    if any(not is_int(item) for item in value):
        errors.append(f"{where}.{key} must contain only ints")
        return None
    return value


def require_grid(obj, key, where, errors):
    grid = require_nonempty_list(obj, key, where, errors)
    if grid is None:
        return None
    for row_idx, row in enumerate(grid):
        if (
            not isinstance(row, list)
            or len(row) != 3
            or any(not is_int(item) or item <= 0 for item in row)
        ):
            errors.append(f"{where}.{key}[{row_idx}] must be a positive [t,h,w] int triple")
            return None
    return grid


def require_hf_top5(obj, errors):
    top5 = require_nonempty_list(obj, "top5", "hf", errors)
    if top5 is None:
        return None
    for idx, entry in enumerate(top5):
        if not isinstance(entry, dict):
            errors.append(f"hf.top5[{idx}] is not an object")
            return None
        if require_int(entry, "token_id", f"hf.top5[{idx}]", errors) is None:
            return None
        if require_number(entry, "logprob", f"hf.top5[{idx}]", errors) is None:
            return None
    return top5


def first_ms_topk(choice, errors):
    logprobs = choice.get("logprobs") if isinstance(choice, dict) else None
    if not isinstance(logprobs, dict):
        errors.append("ms.choices[0].logprobs is missing or not an object")
        return None
    rows = require_nonempty_list(logprobs, "top_logprobs", "ms.choices[0].logprobs", errors)
    if rows is None:
        return None
    if not isinstance(rows[0], list) or not rows[0]:
        errors.append("ms.choices[0].logprobs.top_logprobs[0] must be a non-empty list")
        return None
    for idx, entry in enumerate(rows[0]):
        where = f"ms.choices[0].logprobs.top_logprobs[0][{idx}]"
        if not isinstance(entry, dict):
            errors.append(f"{where} is not an object")
            return None
        if not isinstance(entry.get("token"), str):
            errors.append(f"{where}.token is missing or not a string")
            return None
        if require_number(entry, "logprob", where, errors) is None:
            return None
    return rows[0]


def compare_first_step(hf, ms, ms_features, ms_meta, hf_full=None):
    errors = []
    hf_token = require_int(hf, "greedy_token_id", "hf", errors)
    require_number(hf, "greedy_logprob", "hf", errors)
    hf_prompt = require_int(hf, "prompt_input_ids_len", "hf", errors)
    hf_pad = require_int(hf, "image_pad_tokens", "hf", errors)
    if hf_prompt is not None and hf_prompt <= 0:
        errors.append("hf.prompt_input_ids_len must be positive")
    if hf_pad is not None and hf_pad <= 0:
        errors.append("hf.image_pad_tokens must be positive")
    hf_grid = require_grid(hf, "image_grid_thw", "hf", errors)
    require_hf_top5(hf, errors)

    choices = require_nonempty_list(ms, "choices", "ms", errors)
    choice = choices[0] if choices else None
    if choice is not None and not isinstance(choice, dict):
        errors.append("ms.choices[0] is not an object")
        choice = None
    ms_tokens = require_int_list(choice or {}, "tokens", "ms.choices[0]", errors)
    ms_token = ms_tokens[0] if ms_tokens else None
    usage = ms.get("usage") if isinstance(ms, dict) else None
    if not isinstance(usage, dict):
        errors.append("ms.usage is missing or not an object")
        usage = {}
    ms_prompt = require_int(usage, "prompt_tokens", "ms.usage", errors)
    ms_topk = first_ms_topk(choice or {}, errors) if choice is not None else None

    ms_grid = require_grid(ms_features, "grids", "ms_features", errors)
    features_shape = require_int_list(ms_meta or {}, "features_shape", "ms_meta", errors)
    if features_shape is not None and any(value <= 0 for value in features_shape):
        errors.append("ms_meta.features_shape entries must be positive")
        features_shape = None
    ms_merged = features_shape[0] if features_shape else None
    if ms_merged is not None and not is_int(ms_merged):
        errors.append("ms_meta.features_shape[0] is not an int")
        ms_merged = None

    prompt_ok = hf_prompt is not None and ms_prompt is not None and hf_prompt == ms_prompt
    grid_ok = hf_grid is not None and ms_grid is not None and hf_grid == ms_grid
    pad_ok = hf_pad is not None and ms_merged is not None and hf_pad == ms_merged
    token_ok = hf_token is not None and ms_token is not None and hf_token == ms_token
    if not prompt_ok:
        errors.append(f"prompt length mismatch: hf={hf_prompt} ms={ms_prompt}")
    if not grid_ok:
        errors.append(f"vision grid mismatch: hf={hf_grid} ms={ms_grid}")
    if not pad_ok:
        errors.append(f"image pad token mismatch: hf={hf_pad} ms={ms_merged}")
    if not token_ok:
        errors.append(f"first token mismatch: hf={hf_token} ms={ms_token}")

    gates = {
        "prompt_length": prompt_ok,
        "vision_grid": grid_ok,
        "image_pad_tokens": pad_ok,
        "greedy_first_token": token_ok,
    }

    if hf_full is not None:
        full_ids = require_int_list(hf_full, "generated_ids", "hf_full", errors)
        full_ok = full_ids is not None and ms_tokens is not None and full_ids == ms_tokens
        gates["full_token_sequence"] = full_ok
        if not full_ok:
            errors.append(f"full token mismatch: hf={full_ids} ms={ms_tokens}")
        full_prompt = require_int(hf_full, "prompt_input_ids_len", "hf_full", errors)
        full_grid = require_grid(hf_full, "image_grid_thw", "hf_full", errors)
        full_pad = require_int(hf_full, "image_pad_tokens", "hf_full", errors)
        if full_prompt is not None and hf_prompt is not None and full_prompt != hf_prompt:
            errors.append(f"hf_full prompt length disagrees with hf: {full_prompt} != {hf_prompt}")
        if full_grid is not None and hf_grid is not None and full_grid != hf_grid:
            errors.append(f"hf_full grid disagrees with hf: {full_grid} != {hf_grid}")
        if full_pad is not None and hf_pad is not None and full_pad != hf_pad:
            errors.append(f"hf_full pad disagrees with hf: {full_pad} != {hf_pad}")

    report = {
        "gates": gates,
        "prompt_tokens": {"hf": hf_prompt, "ms": ms_prompt},
        "vision_grid": {"hf": hf_grid, "ms": ms_grid},
        "image_pad_tokens": {"hf": hf_pad, "ms": ms_merged},
        "greedy_token": {"hf": hf_token, "ms": ms_token},
        "greedy_tokens": {"hf": (hf_full or {}).get("generated_ids"), "ms": ms_tokens},
        "hf_top5": hf.get("top5") if isinstance(hf, dict) else None,
        "ms_top5": ms_topk,
        "logit_note": (
            "HF is BF16 (CPU/disk offload); mach-server artifact is Q4-on-device. "
            "Top-5 logprobs are diagnostic only and are not a parity gate."
        ),
        "pass": not errors,
        "errors": errors,
    }
    return report, errors


def selftest():
    hf = {
        "greedy_token_id": 248068,
        "greedy_logprob": 0.0,
        "prompt_input_ids_len": 81,
        "image_pad_tokens": 64,
        "image_grid_thw": [[1, 16, 16]],
        "top5": [{"token_id": 248068, "logprob": 0.0, "text": "<think>"}],
    }
    ms = {
        "choices": [
            {
                "tokens": [248068, 271, 248069, 271],
                "logprobs": {
                    "top_logprobs": [
                        [{"token": "<think>", "logprob": 0.0}],
                        [{"token": "\n\n", "logprob": -0.1}],
                        [{"token": "</think>", "logprob": 0.0}],
                        [{"token": "\n\n", "logprob": 0.0}],
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 81},
    }
    features = {"grids": [[1, 16, 16]], "values": 327680}
    meta = {"features_shape": [64, 5120]}
    full = {
        "prompt_input_ids_len": 81,
        "image_pad_tokens": 64,
        "image_grid_thw": [[1, 16, 16]],
        "generated_ids": [248068, 271, 248069, 271],
    }
    report, errors = compare_first_step(hf, ms, features, meta, full)
    assert not errors and report["pass"], errors

    cases = []
    bad_token = copy.deepcopy(ms)
    bad_token["choices"][0]["tokens"][0] = 1
    cases.append((hf, bad_token, features, meta, full, "first token mismatch"))
    bad_grid = copy.deepcopy(features)
    bad_grid["grids"] = [[1, 8, 8]]
    cases.append((hf, ms, bad_grid, meta, full, "vision grid mismatch"))
    bad_prompt = copy.deepcopy(ms)
    bad_prompt["usage"]["prompt_tokens"] = 80
    cases.append((hf, bad_prompt, features, meta, full, "prompt length mismatch"))
    bad_pad = copy.deepcopy(hf)
    bad_pad["image_pad_tokens"] = 63
    cases.append((bad_pad, ms, features, meta, full, "image pad token mismatch"))
    bad_full = copy.deepcopy(full)
    bad_full["generated_ids"][2] = 1
    cases.append((hf, ms, features, meta, bad_full, "full token mismatch"))
    for args in cases:
        _, errs = compare_first_step(*args[:5])
        assert any(args[5] in e for e in errs), (args[5], errs)

    bad_nested_grid = copy.deepcopy(features)
    bad_nested_grid["grids"] = [[1, "x", 16]]
    _, errs = compare_first_step(hf, ms, bad_nested_grid, meta, full)
    assert any("positive [t,h,w] int triple" in e for e in errs), errs

    bad_grid_shape = copy.deepcopy(features)
    bad_grid_shape["grids"] = [[1, 16]]
    _, errs = compare_first_step(hf, ms, bad_grid_shape, meta, full)
    assert any("positive [t,h,w] int triple" in e for e in errs), errs

    bad_features_tail = copy.deepcopy(meta)
    bad_features_tail["features_shape"] = [64, "bad"]
    _, errs = compare_first_step(hf, ms, features, bad_features_tail, full)
    assert any("features_shape" in e for e in errs), errs

    bad_hf_top5 = copy.deepcopy(hf)
    bad_hf_top5["top5"] = [None]
    _, errs = compare_first_step(bad_hf_top5, ms, features, meta, full)
    assert any("top5[0]" in e for e in errs), errs

    bad_ms_topk = copy.deepcopy(ms)
    bad_ms_topk["choices"][0]["logprobs"]["top_logprobs"] = [["x"]]
    _, errs = compare_first_step(hf, bad_ms_topk, features, meta, full)
    assert any("top_logprobs[0][0]" in e for e in errs), errs

    no_choices = copy.deepcopy(ms)
    no_choices["choices"] = []
    _, errs = compare_first_step(hf, no_choices, features, meta, full)
    assert any("choices" in e for e in errs), errs

    no_logprobs = copy.deepcopy(ms)
    del no_logprobs["choices"][0]["logprobs"]
    _, errs = compare_first_step(hf, no_logprobs, features, meta, full)
    assert any("logprobs" in e for e in errs), errs

    missing_both = copy.deepcopy(hf)
    del missing_both["prompt_input_ids_len"]
    missing_ms = copy.deepcopy(ms)
    missing_ms["usage"] = {}
    _, errs = compare_first_step(missing_both, missing_ms, features, meta, full)
    assert any("prompt_input_ids_len" in e for e in errs), errs
    assert any("prompt_tokens" in e for e in errs), errs

    print("vision_c4_compare_first_step selftest ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", default="artifacts/vision-c4/hf_first_step.json")
    parser.add_argument("--hf-full", default="artifacts/vision-c4/hf_full_tokens.json")
    parser.add_argument("--ms", default="artifacts/vision-c4/ms_logprobs.json")
    parser.add_argument("--ms-features", default="artifacts/vision-c4/ms_features.json")
    parser.add_argument("--ms-meta", default="artifacts/vision-c4/parity_meta.json")
    parser.add_argument("--out", default="")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0

    hf = load_json(args.hf)
    hf_full = load_json(args.hf_full) if args.hf_full else None
    ms = load_json(args.ms)
    features = load_json(args.ms_features)
    meta = load_json(args.ms_meta) if args.ms_meta else None
    report, errors = compare_first_step(hf, ms, features, meta, hf_full)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
