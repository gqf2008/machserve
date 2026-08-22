//! Debug: dump a few loaded weight values to compare loader vs reference.

use mach_model::loader::load_safetensors;
use mach_model::{Config, Weights};
use std::path::PathBuf;

fn main() {
    let root = std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into());
    let root = PathBuf::from(&root);
    let mut cfg = Config::llama(896, 24, 14, 2, 151936, 2048);
    cfg.intermediate_size = 4864;
    let w: Weights =
        load_safetensors(&root.join("qwen-0.5b.safetensors"), &cfg, true).expect("load");
    println!("embed[0,0..8] = {:?}", &w.tok_emb[0..8]);
    println!("rms_final[0..4] = {:?}", &w.rms_final[0..4]);
    println!("layer0 wq[0,0..4] = {:?}", &w.layers[0].wq[0..4]);
}
