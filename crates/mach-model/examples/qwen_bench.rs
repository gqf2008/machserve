//! Loads a real Qwen2.5-0.5B-Instruct checkpoint (BF16 safetensors) and
//! benchmarks decode TPOT on the GPU, eager vs HIP graph.
//!
//! Also dumps the first tokens' logits to `.models/qwen_rust_logits.json` for
//! numeric validation against `tools/ref_llama.py` (fp64 reference).
//!
//! Run with:
//!   cargo run -p mach-model --release --features hip --example qwen_bench
//!
//! Download first:
//!   curl -L -o .models/qwen-0.5b.safetensors \
//!     https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors
//!   curl -L -o .models/qwen-config.json \
//!     https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json

#[cfg(feature = "hip")]
use mach_model::Config;

#[cfg(feature = "hip")]
fn main() {
    use mach_kernel_sys::hip;
    use mach_model::Weights;
    use mach_model::loader::load_safetensors;
    use mach_model::model::GpuModel;
    use std::path::PathBuf;
    use std::time::Instant;

    let root = std::env::var("MACH_MODELS").unwrap_or_else(|_| ".models".into());
    let root = PathBuf::from(&root);
    let model_path = root.join("qwen-0.5b.safetensors");
    let cfg_path = root.join("qwen-config.json");
    assert!(
        model_path.exists(),
        "missing {model_path:?} (see doc comment)"
    );
    assert!(cfg_path.exists(), "missing {cfg_path:?}");

    let cfg = config_from_json(&cfg_path);
    println!(
        "model: Qwen2.5-0.5B-Instruct (d_model={} layers={} heads={} kv={} head_dim={} inter={} vocab={})",
        cfg.d_model,
        cfg.n_layers,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.head_dim,
        cfg.intermediate_size,
        cfg.vocab_size
    );

    let w: Weights = load_safetensors(&model_path, &cfg, true).expect("load weights");
    println!("weights: {:.1} MB fp32", w.byte_size() as f64 / 1e6);

    let hip = hip::hip().expect("HIP runtime");
    assert!(hip::device_count().expect("devices") > 0, "no HIP device");
    let mut model = GpuModel::new(hip, cfg, &w).expect("build model");

    // Numeric validation: tokens from args (default 1 2 3), dump logits.
    let check_tokens: Vec<u32> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let check_tokens = if check_tokens.is_empty() {
        vec![1, 2, 3]
    } else {
        check_tokens
    };
    let logits = model.forward(&check_tokens).expect("decode");
    assert!(logits.iter().all(|v| v.is_finite()));
    std::fs::write(
        root.join("qwen_rust_logits.json"),
        serde_json::to_string(&logits).unwrap(),
    )
    .expect("write");
    println!("validated tokens {check_tokens:?} -> logits finite, dumped for ref_llama.py");

    let n_tokens = 100usize;
    let seq: Vec<u32> = (0..n_tokens).map(|i| (i % 977) as u32).collect();

    // --- full step (input copy + kernels + logits readback): the single-seq TPOT path ---
    model.reset_state().expect("reset");
    for _ in 0..5 {
        model.decode_step(1).expect("warmup");
    }
    model.reset_state().expect("reset");
    let t0 = Instant::now();
    for &t in &seq {
        model.decode_step(t).expect("eager");
    }
    let eager_full_ms = t0.elapsed().as_secs_f64() * 1000.0 / n_tokens as f64;

    let graph = model.capture_decode().expect("capture");
    let t1 = Instant::now();
    for &t in &seq {
        model.decode_step_graph(&*graph, t).expect("graph");
    }
    let graph_full_ms = t1.elapsed().as_secs_f64() * 1000.0 / n_tokens as f64;

    // --- launch-only path ---
    model.reset_state().expect("reset");
    let t2 = Instant::now();
    for &t in &seq {
        model.step_eager(t).expect("eager launch");
    }
    let eager_launch_ms = t2.elapsed().as_secs_f64() * 1000.0 / n_tokens as f64;

    let graph2 = model.capture_decode().expect("capture2");
    let t3 = Instant::now();
    for &t in &seq {
        model.step_graph(&*graph2, t).expect("graph launch");
    }
    let graph_launch_ms = t3.elapsed().as_secs_f64() * 1000.0 / n_tokens as f64;

    println!("\n=== decode TPOT (Qwen2.5-0.5B, 7900 XTX) ===");
    println!(
        "full step (incl. logits readback): eager {eager_full_ms:.2} ms/tok | graph {graph_full_ms:.2} ms/tok | {:.2}x",
        eager_full_ms / graph_full_ms
    );
    println!(
        "launch-only: eager {eager_launch_ms:.2} ms/tok | graph {graph_launch_ms:.2} ms/tok | {:.2}x",
        eager_launch_ms / graph_launch_ms
    );

    // --- real generation demo: 8 tokens, host argmax (full logits readback) ---
    model.reset_state().expect("reset");
    let mut token = 151643u32; // <|im_start|>
    let mut out = Vec::new();
    let t4 = Instant::now();
    for _ in 0..8 {
        let logits = model.decode_step(token).expect("gen");
        token = argmax(&logits) as u32;
        out.push(token);
    }
    let gen_host_ms = t4.elapsed().as_secs_f64() * 1000.0 / 8.0;
    println!(
        "\ngeneration (host argmax, full logits readback): {gen_host_ms:.2} ms/token, tokens {out:?}"
    );

    // --- Verify launch-only is submission rate: 50 eager steps + ONE final sync ---
    model.reset_state().expect("reset");
    let m = 50usize;
    let t6 = Instant::now();
    for i in 0..m {
        model.step_eager((i % 977) as u32).expect("eager");
    }
    model.sync().expect("sync");
    let batch_ms = t6.elapsed().as_secs_f64() * 1000.0 / m as f64;
    println!("eager 50 steps + 1 sync: {batch_ms:.2} ms/token (GPU completion rate)");

    // --- GPU-sampled generation: only 4 bytes read back per token ---
    let n_gen = 200usize;
    model.reset_state().expect("reset");
    let mut token = 151643u32;
    let mut out2 = Vec::new();
    let t5 = Instant::now();
    for _ in 0..n_gen {
        token = model.decode_step_sampled(token).expect("gen-gpu");
        out2.push(token);
    }
    let gen_gpu_ms = t5.elapsed().as_secs_f64() * 1000.0 / n_gen as f64;
    println!("generation (GPU argmax, 4B readback): {gen_gpu_ms:.2} ms/token, tokens {out2:?}");
    println!(
        "\nend-to-end TPOT: host-readback {gen_host_ms:.2} ms | GPU-sampled {gen_gpu_ms:.2} ms | speedup {:.2}x | llama.cpp Vulkan reference 1.55 ms",
        gen_host_ms / gen_gpu_ms
    );

    // --- batched decode (continuous batching): B sequences share one forward ---
    {
        use mach_model::batched::BatchedModel;
        println!("\n=== batched decode scaling (Qwen2.5-0.5B, 7900 XTX) ===");
        println!("batch |  ms/step | ms/seq-tok |   tok/s");
        for &b in &[1usize, 2, 4, 8, 16, 32, 64] {
            let mut bm = BatchedModel::new(hip::hip().unwrap(), cfg, &w, b).expect("batched model");
            bm.reset_state().expect("reset");
            let mut tokens: Vec<u32> = (0..b).map(|i| (i % 977) as u32).collect();
            for _ in 0..5 {
                tokens = bm.decode_step(&tokens).expect("warmup");
            }
            bm.reset_state().expect("reset");
            tokens = (0..b).map(|i| (i % 977) as u32).collect();
            let steps = 50usize;
            let t7 = Instant::now();
            for _ in 0..steps {
                tokens = bm.decode_step(&tokens).expect("batched step");
            }
            let step_ms = t7.elapsed().as_secs_f64() * 1000.0 / steps as f64;
            let per = step_ms / b as f64;
            let tps = 1000.0 / per;
            println!("{b:>4} | {step_ms:>8.2} ms | {per:>11.3} ms/seq-tok | {tps:>13.0} tok/s");
        }
        println!(
            "  reference: llama.cpp Vulkan 1.55 ms/seq-tok (643 tok/s), single-seq MachServe ~7 ms"
        );
    }

    // --- fp16 compute path: same batched decode, fp16 weights + fp32 accumulate ---
    {
        use mach_model::batched::BatchedModel;
        use mach_model::config::ModelDType;
        println!("\n=== batched decode scaling, fp16 compute (Qwen2.5-0.5B, 7900 XTX) ===");
        let mut cfg16 = cfg;
        cfg16.dtype = ModelDType::F16;
        println!("batch |  ms/step | ms/seq-tok |   tok/s");
        for &b in &[16usize, 32, 64] {
            let mut bm =
                BatchedModel::new(hip::hip().unwrap(), cfg16, &w, b).expect("batched fp16");
            bm.reset_state().expect("reset");
            let mut tokens: Vec<u32> = (0..b).map(|i| (i % 977) as u32).collect();
            for _ in 0..5 {
                tokens = bm.decode_step(&tokens).expect("warmup");
            }
            bm.reset_state().expect("reset");
            tokens = (0..b).map(|i| (i % 977) as u32).collect();
            let steps = 50usize;
            let t = Instant::now();
            for _ in 0..steps {
                tokens = bm.decode_step(&tokens).expect("batched fp16 step");
            }
            let step_ms = t.elapsed().as_secs_f64() * 1000.0 / steps as f64;
            let per = step_ms / b as f64;
            let tps = 1000.0 / per;
            println!("{b:>4} | {step_ms:>8.2} ms | {per:>11.3} ms/seq-tok | {tps:>13.0} tok/s");
        }
    }

    // --- chunked prefill: TTFT for a long prompt (continuous engine) ---
    {
        use mach_model::config::ModelDType;
        use mach_model::continuous::ContinuousModel;
        let mut cfg16 = cfg;
        cfg16.dtype = ModelDType::F16;
        println!("\n=== chunked prefill TTFT (Qwen2.5-0.5B, capacity 64) ===");
        let prompt_len = 512usize;
        let prompt: Vec<u32> = (0..prompt_len).map(|i| (i % 977) as u32).collect();
        let mut eng = ContinuousModel::new(hip::hip().unwrap(), cfg16, &w, 64).unwrap();
        let id = eng
            .add(
                &prompt,
                1,
                None,
                mach_model::sampling::SamplingParams::default(),
            )
            .unwrap();
        let mut steps = 0usize;
        let t = Instant::now();
        while !eng.is_done(id) {
            eng.step().unwrap();
            steps += 1;
        }
        let ttft_ms = t.elapsed().as_secs_f64() * 1000.0;
        let prefill_tps = prompt_len as f64 / (ttft_ms / 1000.0);
        println!(
            "prompt {prompt_len} tokens: {steps} steps, TTFT {ttft_ms:.1} ms, prefill {prefill_tps:.0} tok/s (single-token prefill would be ~{} ms)",
            prompt_len as f64 * 5.0
        );
    }
}

#[cfg(not(feature = "hip"))]
fn main() {
    eprintln!(
        "qwen_bench requires the `hip` feature: cargo run -p mach-model --features hip --example qwen_bench"
    );
}

#[cfg(feature = "hip")]
fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[cfg(feature = "hip")]
fn config_from_json(path: &std::path::Path) -> Config {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read config"))
            .expect("parse config");
    let hidden = v["hidden_size"].as_u64().unwrap_or(896) as usize;
    let layers = v["num_hidden_layers"].as_u64().unwrap_or(24) as usize;
    let heads = v["num_attention_heads"].as_u64().unwrap_or(14) as usize;
    let kv = v["num_key_value_heads"].as_u64().unwrap_or(heads as u64) as usize;
    let vocab = v["vocab_size"].as_u64().unwrap_or(151936) as usize;
    let inter = v["intermediate_size"].as_u64().unwrap_or(4 * hidden as u64) as usize;
    let max_seq = v["max_position_embeddings"]
        .as_u64()
        .unwrap_or(2048)
        .min(2048) as usize;
    let eps = v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
    let theta = v["rope_theta"].as_f64().unwrap_or(10000.0) as f32;
    let mut cfg = Config::llama(hidden, layers, heads, kv, vocab, max_seq);
    cfg.intermediate_size = inter;
    cfg.rms_eps = eps;
    cfg.rope_theta = theta;
    cfg
}
