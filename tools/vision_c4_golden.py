#!/usr/bin/env python3
"""HF reference dump for the C4 vision parity check.

Always dumps the processor golden (pixel_values/grid). With --tower it loads
only the shards that contain model.visual.* and dumps the merged features to
a .npy file for the comparator. The tower path needs torch + safetensors;
failures are reported instead of silently skipped.

Example:
  python tools/vision_c4_golden.py --model-dir .models/qwen3.8-27b \
      --image artifacts/vision-c4/input.png --out artifacts/vision-c4/hf_golden.json \
      --tower --allow-pil-fallback
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image


def load_processor(model_dir, allow_pil_fallback):
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        return proc, "AutoProcessor"
    except Exception as exc:
        if not allow_pil_fallback:
            raise RuntimeError(
                "AutoProcessor failed and --allow-pil-fallback was not set: " + str(exc)
            )
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
            Qwen2VLImageProcessorPil,
        )

        with open(
            os.path.join(model_dir, "preprocessor_config.json"), encoding="utf-8"
        ) as handle:
            cfg = json.load(handle)
        proc = Qwen2VLImageProcessorPil(
            size=cfg["size"],
            patch_size=cfg["patch_size"],
            temporal_patch_size=cfg["temporal_patch_size"],
            merge_size=cfg["merge_size"],
            image_mean=cfg["image_mean"],
            image_std=cfg["image_std"],
        )
        return proc, "Qwen2VLImageProcessorPil (fallback: " + str(exc) + ")"


def process(proc, img):
    try:
        inputs = proc(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].detach().cpu().numpy()
        grid = inputs["image_grid_thw"].detach().cpu().numpy().tolist()
        return pixel_values, grid
    except Exception:
        inputs = proc(
            images=img, return_tensors="np", input_data_format="channels_last"
        )
        return inputs["pixel_values"], inputs["image_grid_thw"].tolist()


def run_tower(model_dir, hidden_states, grid, out_prefix):
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
        raw = json.load(handle)
    vcfg = Qwen3_5VisionConfig(**raw["vision_config"])
    model = Qwen3_5VisionModel(vcfg).eval()
    with open(
        os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf-8"
    ) as handle:
        index = json.load(handle)
    visual_keys = [
        key for key in index["weight_map"] if key.startswith("model.visual.")
    ]
    if not visual_keys:
        raise RuntimeError("no model.visual.* tensors found in the checkpoint index")
    shards = sorted({index["weight_map"][key] for key in visual_keys})
    state = {}
    for shard in shards:
        with safe_open(
            os.path.join(model_dir, shard), framework="pt", device="cpu"
        ) as handle:
            for key in handle.keys():
                if key.startswith("model.visual."):
                    state[key[len("model.visual."):]] = handle.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError("missing vision tensors: " + str(list(missing)[:8]))
    if unexpected:
        raise RuntimeError("unexpected vision tensors: " + str(list(unexpected)[:8]))
    del state
    with torch.no_grad():
        out = model(
            hidden_states=torch.from_numpy(np.asarray(hidden_states, dtype=np.float32)),
            grid_thw=torch.tensor(grid, dtype=torch.long),
        )
    features = out.pooler_output
    if features is None:
        raise RuntimeError("Qwen3_5VisionModel returned no pooler_output")
    arr = features.detach().cpu().numpy()
    np.save(out_prefix + "_features.npy", arr)
    return arr, shards, len(visual_keys)


def tower_smoke():
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    cfg = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=8,
        intermediate_size=16,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=4,
        out_hidden_size=12,
    )
    model = Qwen3_5VisionModel(cfg).eval()
    hidden = torch.zeros(16, 3 * 2 * 2 * 2)
    grid = torch.tensor([[1, 4, 4]])
    with torch.no_grad():
        out = model(hidden_states=hidden, grid_thw=grid)
    features = out.pooler_output
    assert features is not None, "tower smoke: no pooler_output"
    assert tuple(features.shape) == (4, cfg.out_hidden_size), tuple(features.shape)
    print("tower smoke ok:", tuple(features.shape))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir")
    parser.add_argument("--image")
    parser.add_argument("--out")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tower", action="store_true")
    parser.add_argument("--allow-pil-fallback", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        tower_smoke()
        return 0
    if not args.model_dir or not args.image or not args.out:
        parser.error("--model-dir/--image/--out are required unless --smoke")

    with open(args.image, "rb") as handle:
        image_sha256 = hashlib.sha256(handle.read()).hexdigest()
    with Image.open(args.image) as handle:
        img = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    proc, kind = load_processor(args.model_dir, args.allow_pil_fallback)
    hidden_states, grid = process(proc, img)
    flat = np.asarray(hidden_states, dtype=np.float64).flatten()
    golden = {
        "image": {
            "path": os.path.abspath(args.image),
            "height": int(img.shape[0]),
            "width": int(img.shape[1]),
        },
        "processor": kind,
        "image_sha256": image_sha256,
        "grid": grid,
        "hidden_states_shape": list(np.asarray(hidden_states).shape),
        "hidden_states_sum": float(flat.sum()),
        "hidden_states_first8": [float(x) for x in flat[:8]],
    }
    if args.tower:
        prefix = os.path.splitext(args.out)[0]
        features, shards, keys = run_tower(
            args.model_dir, hidden_states, grid, prefix
        )
        fflat = features.astype(np.float64).flatten()
        golden["tower_shards"] = shards
        golden["tower_keys"] = keys
        golden["features_shape"] = list(features.shape)
        golden["features_sum"] = float(fflat.sum())
        golden["features_first8"] = [float(x) for x in fflat[:8]]
        golden["features_npy"] = prefix + "_features.npy"
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(golden, handle, indent=2)
    print(json.dumps(golden, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
