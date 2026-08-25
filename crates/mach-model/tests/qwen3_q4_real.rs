//! Qwen3-8B storage-Q4 real-model smoke test (skippable).
//!
//! Loads `Qwen/Qwen3-8B` shards through the storage-Q4 loader and builds the
//! GPU model via `GpuModel::from_q4` (dequantize to f16 on upload), verifying
//! finite + deterministic decode. Host memory stays ~= packed Q4 (~5GB) +
//! one tensor's f16 buffer instead of the ~48GB f32 path.

#![cfg(feature = "hip")]

use mach_kernel_sys::hip;
use mach_model::config::ModelDType;
use mach_model::loader::load_safetensors_q4;
use mach_model::model::GpuModel;
use mach_model::{Config, WeightsQ4};

fn model_dir() -> Option<std::path::PathBuf> {
    mach_model::real_test_model_path()
}

fn hip_ctx() -> Option<std::sync::Arc<hip::Hip>> {
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

#[test]
fn qwen3_8b_q4_decodes_finite_and_deterministic() {
    let Some(dir) = model_dir() else {
        eprintln!("skipping qwen3_8b_q4: set MACH_TEST_MODEL to the model dir");
        return;
    };
    let Some(hip) = hip_ctx() else { return };

    let mut cfg = Config::qwen3(4096, 36, 32, 8, 151936, 2048);
    cfg.dtype = ModelDType::F16;
    let w: WeightsQ4 = load_safetensors_q4(&dir, &cfg, false).expect("load Qwen3-8B Q4");

    let mut m = GpuModel::from_q4(hip.clone(), cfg, &w).expect("build model from q4");
    let tokens = [1u32, 100, 200, 300, 400];
    let a = m.forward(&tokens).expect("decode");
    assert!(a.iter().all(|v| v.is_finite()), "logits must be finite");
    assert_eq!(a.len(), cfg.vocab_size, "logits must cover the vocab");

    // Free the first model before building a second one so VRAM peak stays at
    // one F16 model (~16GB) instead of two (~32GB).
    drop(m);
    let mut fresh = GpuModel::from_q4(hip, cfg, &w).expect("fresh");
    let b = fresh.forward(&tokens).expect("decode fresh");
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(max, 0.0, "decode must be deterministic");
    eprintln!("qwen3_8b_q4 OK: {} logits", a.len());
}
