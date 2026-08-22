//! Diagnoses real-model chat quality end to end: encodes the Qwen chat
//! template with the real tokenizer, feeds it through the batched engine path
//! (greedy) and prints the continuation decoded back. A coherent answer (e.g.
//! "Paris") confirms the forward is correct against a real checkpoint.
//!
//!   cargo run -p mach-model --release --features hip --example chat_check
#[cfg(feature = "hip")]
fn main() {
    use mach_kernel_sys::hip;
    use mach_model::config::ModelDType;
    use mach_model::loader::load_safetensors;
    use mach_model::tokenizer::Tokenizer;
    use mach_model::{Config, Weights};
    use std::path::PathBuf;

    let root = PathBuf::from(".models");
    let cfg_path = root.join("qwen-config.json");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cfg_path).expect("config")).expect("parse");
    let hidden = v["hidden_size"].as_u64().unwrap() as usize;
    let layers = v["num_hidden_layers"].as_u64().unwrap() as usize;
    let heads = v["num_attention_heads"].as_u64().unwrap() as usize;
    let kv = v["num_key_value_heads"].as_u64().unwrap() as usize;
    let vocab = v["vocab_size"].as_u64().unwrap() as usize;
    let inter = v["intermediate_size"].as_u64().unwrap() as usize;
    let max_seq = 2048usize;
    let mut cfg = Config::llama(hidden, layers, heads, kv, vocab, max_seq);
    cfg.intermediate_size = inter;
    cfg.rms_eps = v["rms_norm_eps"].as_f64().unwrap() as f32;
    cfg.rope_theta = v["rope_theta"].as_f64().unwrap() as f32;
    cfg.dtype = ModelDType::F32; // isolate: F16 vs F32

    let w: Weights =
        load_safetensors(&root.join("qwen-0.5b.safetensors"), &cfg, true).expect("weights");
    let tok = Tokenizer::from_path(&root.join("tokenizer.json")).expect("tokenizer");

    let prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";
    let ids = tok.encode(prompt);
    println!("prompt tokens ({}): {:?}", ids.len(), ids);

    use mach_model::batched::BatchedModel;
    use mach_model::sampling::SamplingParams;
    let hip = hip::hip().expect("hip");
    let mut model = BatchedModel::new(hip, cfg, &w, 64).expect("batched model");
    // Teacher-force the whole prompt as rows in one batched step (greedy);
    // the last row's sample is the first generated token.
    let lens: Vec<u32> = (0..ids.len() as u32).collect();
    let slots = vec![0u32; ids.len()];
    let mut params = vec![SamplingParams::greedy(0); ids.len()];
    let (sampled, _) = model
        .decode_step_explicit(
            &ids,
            &lens,
            &slots,
            &mut params,
            &vec![Vec::new(); ids.len()],
            &vec![Vec::new(); ids.len()],
        )
        .expect("prefill");
    let mut tok_id = sampled[ids.len() - 1];
    let mut out = vec![tok_id];
    for _ in 0..40 {
        let mut p = [SamplingParams::greedy(0)];
        tok_id = model
            .decode_step_explicit(
                &[tok_id],
                &[ids.len() as u32],
                &[0],
                &mut p,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
            )
            .expect("gen")
            .0[0];
        if tok_id == 151645 {
            out.push(tok_id);
            break;
        }
        out.push(tok_id);
    }
    println!("generated: {:?}", out);
    println!("decoded: {:?}", tok.decode(&out));
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!("chat_check requires the hip feature");
}
