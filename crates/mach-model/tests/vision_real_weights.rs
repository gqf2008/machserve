//! Real-checkpoint CPU parity for the Qwen3.8 vision tower.
//!
//! `vision.rs` pins the maths against a tiny synthetic config, which cannot
//! catch shape/scale mistakes that only show up with the real 27-layer tower.
//! This test replays the same fixed image as the HF reference and compares the
//! merged features, so the CPU reference is pinned to transformers on the real
//! checkpoint before anyone spends a GPU window on it.
//!
//! Opt-in: without `MACH_VISION_GOLDEN` *and* `MACH_VISION_MODEL` the test
//! prints a SKIP line and returns. libtest still reports `1 passed`, so a real
//! verification must be read from `--nocapture` output (the SKIP line, or the
//! parity line with the measured ratios). Setting only one of the two
//! variables is a configuration error and fails loudly.
//!
//! `docs/vision-e2e.md` documents the recipe; in short:
//!
//! ```text
//! python tools/vision_c4_golden.py --model-dir .models/qwen3.8-27b \
//!     --image <input.png> --out <dir>/hf_golden.json --tower \
//!     --allow-pil-fallback --parity-export
//! ```
//!
//! Measured with the 128x128 fixture (grid `[1,16,16]`, 64x5120 merged
//! features): worst tolerance ratio 0.4488 (feature 320103: -1.2215794 vs
//! -1.2210314, allowed 1.22e-3), no non-finite values, ~11.5 min
//! single-threaded - hence the opt-in gate.

use mach_model::loader::load_vision_weights;
use mach_model::vision::{VisionConfig, VisionGrid, vision_forward};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// A feature passes when `|got - want| <= MAX_ABS * max(1, |want|)`: 1e-3
/// absolute for small features, 0.1% relative above 1. transformers also runs
/// f32, but sums in a different order, so this is the noise floor rather than
/// an exact-equality bound.
const MAX_ABS: f32 = 1e-3;

/// How much of the allowed tolerance one feature uses: `<= 1.0` passes. Both
/// the long-running parity test and the fast positive control go through this
/// function, so `tolerance_flags_a_perturbed_feature` protects the exact
/// expression the parity run asserts on.
fn tolerance_ratio(got: f32, want: f32) -> f32 {
    (got - want).abs() / (MAX_ABS * want.abs().max(1.0))
}

fn within_tolerance(got: f32, want: f32) -> bool {
    tolerance_ratio(got, want) <= 1.0
}

fn env_path(name: &str) -> Option<PathBuf> {
    match std::env::var_os(name) {
        Some(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => None,
    }
}

fn read_f32_le(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(bytes.len() % 4, 0, "{} is not f32 aligned", path.display());
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn shape_of(meta: &serde_json::Value, key: &str) -> Vec<usize> {
    meta[key]
        .as_array()
        .unwrap_or_else(|| panic!("parity_meta.json is missing {key}"))
        .iter()
        .map(|v| v.as_u64().expect("shape entry") as usize)
        .collect()
}

fn product(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Fast positive control for the tolerance rule: a 1e-2 perturbation of a
/// near-zero reference (as well as a relative miss above 1) must be rejected.
#[test]
fn tolerance_flags_a_perturbed_feature() {
    assert!(within_tolerance(0.5, 0.5));
    assert!(within_tolerance(1.0005, 1.0));
    assert!(within_tolerance(10.005, 10.0));
    // Near-zero reference: only the 1e-3 absolute term applies, so 0.01 fails.
    assert!(!within_tolerance(0.01, 0.0));
    assert!(!within_tolerance(-0.01, 0.0));
    // Above 1 the relative term applies, but 0.2% is still too much.
    assert!(!within_tolerance(10.02, 10.0));
}

#[test]
fn cpu_vision_real_weights_match_hf_golden() {
    let golden = env_path("MACH_VISION_GOLDEN");
    let model_dir = env_path("MACH_VISION_MODEL");
    match (&golden, &model_dir) {
        (None, None) => {
            eprintln!(
                "SKIP cpu_vision_real_weights_match_hf_golden: set MACH_VISION_GOLDEN \
                 (golden dir) and MACH_VISION_MODEL (checkpoint dir), then rerun with \
                 --nocapture to confirm this test really ran"
            );
            return;
        }
        (Some(_), None) => panic!(
            "MACH_VISION_GOLDEN is set but MACH_VISION_MODEL is missing; set both or neither"
        ),
        (None, Some(_)) => panic!(
            "MACH_VISION_MODEL is set but MACH_VISION_GOLDEN is missing; set both or neither"
        ),
        (Some(_), Some(_)) => {}
    }
    let golden = golden.expect("checked above");
    let model_dir = model_dir.expect("checked above");

    let meta_path = golden.join("parity_meta.json");
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&meta_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", meta_path.display())),
    )
    .expect("parse parity_meta.json");

    assert_eq!(
        meta["dtype"].as_str(),
        Some("float32-le"),
        "parity_meta.json dtype must be float32-le"
    );
    let pixel_shape = shape_of(&meta, "pixel_shape");
    let features_shape = shape_of(&meta, "features_shape");
    let grid: Vec<VisionGrid> = meta["grid"]
        .as_array()
        .expect("meta.grid")
        .iter()
        .map(|row| {
            let row = row.as_array().expect("grid row");
            assert_eq!(row.len(), 3, "each grid row must have 3 entries");
            let mut g = [0usize; 3];
            for (slot, value) in g.iter_mut().zip(row) {
                *slot = value.as_u64().expect("grid entry") as usize;
            }
            assert!(g.iter().all(|v| *v > 0), "grid entries must be positive");
            g
        })
        .collect();
    let image_sha = meta["image_sha256"]
        .as_str()
        .expect("meta.image_sha256")
        .to_owned();

    // Bind the golden to the image it came from when it is alongside it.
    let image_path = golden.join("input.png");
    if image_path.exists() {
        let digest = Sha256::digest(std::fs::read(&image_path).expect("read input.png"));
        assert_eq!(
            digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
            image_sha,
            "input.png does not match parity_meta.json image_sha256"
        );
    }

    let pixel_file = meta["pixel_file"].as_str().unwrap_or("ms_input_pixel.bin");
    let features_file = meta["features_file"].as_str().unwrap_or("hf_features.bin");
    let pixel = read_f32_le(&golden.join(pixel_file));
    let want = read_f32_le(&golden.join(features_file));
    assert_eq!(
        pixel.len(),
        product(&pixel_shape),
        "{} length does not match pixel_shape {pixel_shape:?}",
        golden.join(pixel_file).display()
    );
    assert_eq!(
        want.len(),
        product(&features_shape),
        "{} length does not match features_shape {features_shape:?}",
        golden.join(features_file).display()
    );

    let cfg_path = model_dir.join("config.json");
    let raw: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&cfg_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", cfg_path.display())),
    )
    .expect("parse config.json");
    let cfg = VisionConfig::from_hf_json(&raw).expect("vision config");
    let patch_volume = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let patches: usize = grid.iter().map(|g| g[0] * g[1] * g[2]).sum();
    assert_eq!(
        pixel.len(),
        patches * patch_volume,
        "pixel buffer is not {patches} patches x {patch_volume} values"
    );

    let weights = load_vision_weights(&model_dir, &cfg).expect("load vision weights");
    let started = std::time::Instant::now();
    let got = vision_forward(&cfg, &weights, &pixel, &grid).expect("cpu vision forward");
    let elapsed = started.elapsed();

    assert_eq!(
        got.len(),
        want.len(),
        "merged feature count (image sha {image_sha})"
    );
    assert_eq!(
        got.len(),
        product(&features_shape),
        "merged feature count does not match features_shape {features_shape:?}"
    );

    let mut worst_ratio = 0f32;
    let mut worst = (0usize, 0.0f32, 0.0f32);
    let mut nonfinite = 0usize;
    for (index, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        if !a.is_finite() || !b.is_finite() {
            nonfinite += 1;
            continue;
        }
        let ratio = tolerance_ratio(*a, *b);
        if ratio > worst_ratio {
            worst_ratio = ratio;
            worst = (index, *a, *b);
        }
    }
    let (worst_index, worst_got, worst_want) = worst;
    let worst_allowed = MAX_ABS * worst_want.abs().max(1.0);
    eprintln!(
        "real-weight vision parity: features={} grid={grid:?} worst_ratio={worst_ratio:.4} \
         worst_index={worst_index} got={worst_got:e} want={worst_want:e} \
         allowed={worst_allowed:e} nonfinite={nonfinite} cpu_elapsed={elapsed:?} sha={image_sha}",
        got.len()
    );
    assert_eq!(nonfinite, 0, "non-finite vision features");
    assert!(
        worst_ratio <= 1.0,
        "feature {worst_index}: {worst_got} vs {worst_want} exceeds tolerance by \
         {worst_ratio:.3}x (allowed {worst_allowed:e})"
    );
}
