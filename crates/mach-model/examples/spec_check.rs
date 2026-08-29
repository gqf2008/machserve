//! Speculative decoding: 0.5B draft accelerates 1.5B greedy decode.
//!
//! Uses [`mach_model::speculative::SpeculativeDecoder`]; argmax acceptance
//! keeps the output identical to plain greedy (see tests/spec_decode.rs).
//!
//!   cargo run -p mach-model --release --features hip --example spec_check
//! Env: MACH_DRAFT / MACH_TARGET / MACH_CONFIG / MACH_K (default 4) /
//!      MACH_MAX_NEW (default 30) / MACH_PROMPT_LEN (default 26).
#[cfg(feature = "hip")]
fn main() {
    use mach_kernel_sys::hip;
    use mach_model::config::ModelDType;
    use mach_model::loader::load_safetensors;
    use mach_model::sampling::SamplingParams;
    use mach_model::speculative::SpeculativeDecoder;
    use mach_model::{Config, Weights};
    use std::path::PathBuf;
    use std::time::Instant;

    let root = PathBuf::from(std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into()));
    let draft_name = std::env::var("MACH_DRAFT").unwrap_or_else(|_| "qwen-0.5b.safetensors".into());
    let target_name =
        std::env::var("MACH_TARGET").unwrap_or_else(|_| "qwen-1.5b.safetensors".into());
    let config_name =
        std::env::var("MACH_CONFIG").unwrap_or_else(|_| "qwen-1.5b-config.json".into());
    let k: usize = std::env::var("MACH_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let max_new: usize = std::env::var("MACH_MAX_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let parse_cfg = |path: &std::path::Path| -> Config {
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("config")).expect("parse");
        let vocab = v["vocab_size"].as_u64().unwrap() as usize;
        let max_seq = v["max_position_embeddings"]
            .as_u64()
            .unwrap_or(2048)
            .min(8192) as usize;
        let mut c = Config::llama(
            v["hidden_size"].as_u64().unwrap() as usize,
            v["num_hidden_layers"].as_u64().unwrap() as usize,
            v["num_attention_heads"].as_u64().unwrap() as usize,
            v["num_key_value_heads"].as_u64().unwrap() as usize,
            vocab,
            max_seq,
        );
        c.intermediate_size = v["intermediate_size"].as_u64().unwrap() as usize;
        c.rms_eps = v["rms_norm_eps"].as_f64().unwrap() as f32;
        c.rope_theta = v["rope_theta"].as_f64().unwrap() as f32;
        c.dtype = ModelDType::F16;
        c
    };

    let dcfg = parse_cfg(&root.join("qwen-config.json"));
    let tcfg = parse_cfg(&root.join(&config_name));
    let hip = hip::hip().expect("hip");
    let dw: Weights =
        load_safetensors(&root.join(&draft_name), &dcfg, true).expect("draft weights");
    let tw: Weights =
        load_safetensors(&root.join(&target_name), &tcfg, true).expect("target weights");
    println!(
        "draft {draft_name} (d={} L={}) target {target_name} (d={} L={}) K={k}",
        dcfg.d_model, dcfg.n_layers, tcfg.d_model, tcfg.n_layers
    );

    let prompt: Vec<u32> = vec![
        151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838,
        374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198,
    ];

    // Plain greedy reference (target only).
    let mut t_ref = mach_model::batched::BatchedModel::new(hip.clone(), tcfg, &tw, 64).unwrap();
    let plen = prompt.len();
    let lens: Vec<u32> = (0..plen as u32).collect();
    let slots = vec![0u32; plen];
    let mut rp = vec![SamplingParams::greedy(0); plen];
    t_ref
        .decode_step_explicit(
            &prompt,
            &lens,
            &slots,
            &mut rp,
            &vec![Vec::new(); plen],
            &vec![Vec::new(); plen],
            false,
        )
        .unwrap();
    let mut rnext = t_ref
        .decode_step_explicit(
            &[prompt[plen - 1]],
            &[(plen - 1) as u32],
            &[0],
            &mut [SamplingParams::greedy(0)],
            &vec![Vec::new(); 1],
            &vec![Vec::new(); 1],
            true,
        )
        .unwrap()
        .0[0];
    let mut reference = Vec::new();
    let t0 = Instant::now();
    for rpos in plen..plen + max_new {
        reference.push(rnext);
        let mut p = [SamplingParams::greedy(0)];
        rnext = t_ref
            .decode_step_explicit(
                &[rnext],
                &[rpos as u32],
                &[0],
                &mut p,
                &vec![Vec::new(); 1],
                &vec![Vec::new(); 1],
                true,
            )
            .unwrap()
            .0[0];
    }
    let ref_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!(
        "plain greedy: {ref_ms:.1} ms for {max_new} tokens ({:.1} ms/tok)",
        ref_ms / max_new as f64
    );
    // Free the reference engine (its KV cache is the biggest GPU allocation
    // besides the draft/target models) before building the spec engines, so
    // peak device+commit memory is lower on memory-tight machines.
    drop(t_ref);

    // Spec-decode.
    let draft = mach_model::batched::BatchedModel::new(hip.clone(), dcfg, &dw, 64).unwrap();
    let target = mach_model::batched::BatchedModel::new(hip.clone(), tcfg, &tw, 64).unwrap();
    // Host weights are only needed for upload; drop the f32 copies (~8GB for
    // 0.5B+1.5B) now that both engines hold device copies.
    drop(dw);
    drop(tw);
    let mut dec = SpeculativeDecoder::new(draft, target, k, &prompt).unwrap();
    let mut generated = Vec::new();
    let mut rounds = 0usize;
    let t1 = Instant::now();
    while generated.len() < max_new {
        for t in dec.step().unwrap() {
            if generated.len() < max_new {
                generated.push(t);
            }
        }
        rounds += 1;
    }
    let spec_ms = t1.elapsed().as_secs_f64() * 1000.0;
    println!(
        "spec-decode: {spec_ms:.1} ms for {max_new} tokens ({:.1} ms/tok, {} rounds)",
        spec_ms / max_new as f64,
        rounds
    );
    println!(
        "  parity: {}",
        if generated == reference {
            "MATCH"
        } else {
            "DIFFER"
        }
    );
    if generated != reference {
        println!("  spec: {generated:?}\n  ref : {reference:?}");
    }
    println!("  speedup: {:.2}x", ref_ms / spec_ms);
}
#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!("spec_check requires the hip feature");
}
