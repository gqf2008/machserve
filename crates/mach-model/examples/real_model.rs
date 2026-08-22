//! Loads a real safetensors Llama checkpoint and decodes a few tokens on the
//! GPU, dumping logits to a JSON file for numeric comparison against an
//! independent reference (`tools/ref_llama.py`).
//!
//! Run with:
//!   cargo run -p mach-model --release --features hip --example real_model
//!   python tools/ref_llama.py .models/tiny-llama.safetensors .models/rust_logits.json

#[cfg(feature = "hip")]
fn main() {
    use mach_kernel_sys::hip;
    use mach_model::loader::load_safetensors;
    use mach_model::model::GpuModel;
    use mach_model::{Config, Weights};
    use std::path::PathBuf;

    let root = std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into());
    let model_path = PathBuf::from(&root).join("tiny-llama.safetensors");
    let out_path = PathBuf::from(&root).join("rust_logits.json");

    // hf-internal-testing/tiny-random-LlamaForCausalLM
    let cfg = Config::llama(16, 2, 4, 4, 32000, 2048);
    let w: Weights = load_safetensors(&model_path, &cfg, false).expect("load weights");
    println!(
        "loaded {} weights, {:.1} MB",
        w.byte_size() / 4,
        w.byte_size() as f64 / 1e6
    );

    let hip = hip::hip().expect("HIP runtime");
    assert!(hip::device_count().expect("devices") > 0, "no HIP device");
    let mut model = GpuModel::new(hip, cfg, &w).expect("build model");

    let tokens = [1u32, 2, 3, 4, 5];
    let logits = model.forward(&tokens).expect("decode");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "logits must be finite"
    );

    let json = serde_json::to_string(&logits).expect("serialize");
    std::fs::write(&out_path, json).expect("write logits");
    let max = logits.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!(
        "tokens {tokens:?} -> logits len {} (max |x| {max:.3})",
        logits.len()
    );
    println!("wrote {}", out_path.display());
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!(
        "real_model requires the `hip` feature: cargo run -p mach-model --features hip --example real_model"
    );
}
