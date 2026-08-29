//! Diagnoses real-model chat quality end to end: encodes the Qwen chat
//! template with the real tokenizer, feeds it through the batched engine path
//! (greedy) and prints the continuation decoded back. A coherent answer (e.g.
//! "Paris") confirms the forward is correct against a real checkpoint.
//!
//!   cargo run -p mach-model --release --features hip --example chat_check
//! Env: MACH_MODEL (default qwen-0.5b.safetensors), MACH_CONFIG (default
//! qwen-config.json), MACH_DTYPE (f16/f32); max sequence length comes from the
//! config (capped 8192).
#[cfg(feature = "hip")]
fn main() {
    use mach_kernel_sys::hip;
    use mach_model::config::ModelDType;
    use mach_model::loader::load_safetensors;
    use mach_model::tokenizer::Tokenizer;
    use mach_model::{Config, Weights};
    use std::path::PathBuf;

    let root = PathBuf::from(".models");
    let model_name = std::env::var("MACH_MODEL").unwrap_or_else(|_| "qwen-0.5b.safetensors".into());
    let config_name = std::env::var("MACH_CONFIG").unwrap_or_else(|_| "qwen-config.json".into());
    let cfg_path = root.join(&config_name);
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cfg_path).expect("config")).expect("parse");
    let hidden = v["hidden_size"].as_u64().unwrap() as usize;
    let layers = v["num_hidden_layers"].as_u64().unwrap() as usize;
    let heads = v["num_attention_heads"].as_u64().unwrap() as usize;
    let kv = v["num_key_value_heads"].as_u64().unwrap() as usize;
    let vocab = v["vocab_size"].as_u64().unwrap() as usize;
    let inter = v["intermediate_size"].as_u64().unwrap() as usize;
    let max_seq = v["max_position_embeddings"]
        .as_u64()
        .unwrap_or(2048)
        .min(8192) as usize;
    let mut cfg = Config::llama(hidden, layers, heads, kv, vocab, max_seq);
    cfg.intermediate_size = inter;
    cfg.rms_eps = v["rms_norm_eps"].as_f64().unwrap() as f32;
    cfg.rope_theta = v["rope_theta"].as_f64().unwrap() as f32;
    // MACH_DTYPE=f16 exercises the production fp16 path (default f32 here).
    cfg.dtype = match std::env::var("MACH_DTYPE")
        .unwrap_or_else(|_| "f32".into())
        .as_str()
    {
        "f16" => ModelDType::F16,
        _ => ModelDType::F32,
    };

    let w: Weights = load_safetensors(&root.join(&model_name), &cfg, true).expect("weights");
    println!(
        "model {model_name}: d_model={} layers={} heads={} kv={} head_dim={} vocab={} max_seq={max_seq} dtype={:?}",
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.vocab_size,
        cfg.dtype
    );
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
    let (sampled, _, _) = model
        .decode_step_explicit(
            &ids,
            &lens,
            &slots,
            &mut params,
            &vec![Vec::new(); ids.len()],
            &vec![Vec::new(); ids.len()],
            false,
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
                true,
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
