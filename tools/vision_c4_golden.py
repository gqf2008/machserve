#!/usr/bin/env python3
"""HF reference dump for the C4 vision parity check.

Always dumps the processor golden (pixel_values/grid). With --tower it also
tries to load model.visual.* into the transformers Qwen3_5VisionModel and
dumps reference merged features. The tower path needs the full checkpoint,
torch and safetensors; failures are reported instead of silently skipped.

Example:
  python tools/vision_c4_golden.py --model-dir .models/qwen3.8-27b \
      --image artifacts/vision-c4/input.png --out artifacts/vision-c4/hf_golden.json --tower
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def load_processor(model_dir):
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_dir)
        return proc, "AutoProcessor"
    except Exception as exc:
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
            Qwen2VLImageProcessorPil,
        )

        with open(os.path.join(model_dir, "preprocessor_config.json"), encoding="utf-8") as handle:
            cfg = json.load(handle)
        proc = Qwen2VLImageProcessorPil(
            size=cfg["size"],
            patch_size=cfg["patch_size"],
            temporal_patch_size=cfg["temporal_patch_size"],
            merge_size=cfg["merge_size"],
            image_mean=cfg["image_mean"],
            image_std=cfg["image_std"],
        )
        return proc, "Qwen2VLImageProcessorPil (AutoProcessor failed: " + str(exc) + ")"


def process(proc, img):
    try:
        inputs = proc(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].detach().cpu().numpy()
        grid = inputs["image_grid_thw"].detach().cpu().numpy().tolist()
        return pixel_values, grid
    except Exception:
        inputs = proc(images=img, return_tensors="np", input_data_format="channels_last")
        return inputs["pixel_values"], inputs["image_grid_thw"].tolist()


def run_tower(model_dir, pixel_values, grid):
    import torch
    from safetensors.torch import load_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
        raw = json.load(handle)
    vcfg = Qwen3_5VisionConfig(**raw["vision_config"])
    model = Qwen3_5VisionModel(vcfg).eval()
    with open(os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf-8") as handle:
        index = json.load(handle)
    state = {}
    for shard in sorted(set(index["weight_map"].values())):
        state.update(load_file(os.path.join(model_dir, shard)))
    visual = {
        key[len("model.visual."):]: value
        for key, value in state.items()
        if key.startswith("model.visual.")
    }
    if not visual:
        raise RuntimeError("no model.visual.* tensors found in the checkpoint")
    missing, unexpected = model.load_state_dict(visual, strict=False)
    if missing:
        raise RuntimeError("missing vision tensors: " + str(list(missing)[:8]))
    if unexpected:
        raise RuntimeError("unexpected vision tensors: " + str(list(unexpected)[:8]))
    with torch.no_grad():
        out = model(
            pixel_values=torch.from_numpy(np.asarray(pixel_values, dtype=np.float32)),
            grid_thw=torch.tensor(grid, dtype=torch.long),
        )
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tower", action="store_true")
    args = parser.parse_args()

    img = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    proc, kind = load_processor(args.model_dir)
    pixel_values, grid = process(proc, img)
    flat = np.asarray(pixel_values, dtype=np.float64).flatten()
    golden = {
        "image": {"height": int(img.shape[0]), "width": int(img.shape[1])},
        "processor": kind,
        "grid": grid,
        "pixel_values_shape": list(np.asarray(pixel_values).shape),
        "pixel_values_sum": float(flat.sum()),
        "pixel_values_first8": [float(x) for x in flat[:8]],
    }
    if args.tower:
        features = run_tower(args.model_dir, pixel_values, grid)
        fflat = features.astype(np.float64).flatten()
        golden["features_shape"] = list(features.shape)
        golden["features_sum"] = float(fflat.sum())
        golden["features_first8"] = [float(x) for x in fflat[:8]]
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(golden, handle, indent=2)
    print(json.dumps(golden, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
