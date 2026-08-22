import io
p = "E:/Users/gxh/Documents/GitHub/machserve/crates/mach-model/examples/qwen_bench.rs"
s = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
old = """    // --- real generation demo: 8 tokens, argmax sampling ---
    model.reset_state().expect("reset");
    let mut token = 151643u32; // <|im_start|>
    let mut out = Vec::new();
    let t4 = Instant::now();
    for _ in 0..8 {
        let logits = model.decode_step(token).expect("gen");
        token = argmax(&logits) as u32;
        out.push(token);
    }
    let gen_ms = t4.elapsed().as_secs_f64() * 1000.0 / 8.0;
    println!("\\ngeneration (argmax): {gen_ms:.2} ms/token, tokens {out:?}");
}"""
new = """    // --- real generation demo: 8 tokens, host argmax (full logits readback) ---
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
    println!("\\ngeneration (host argmax, full logits readback): {gen_host_ms:.2} ms/token, tokens {out:?}");

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
        "\\nend-to-end TPOT: host-readback {gen_host_ms:.2} ms | GPU-sampled {gen_gpu_ms:.2} ms | speedup {:.2}x | llama.cpp Vulkan reference 1.55 ms",
        gen_host_ms / gen_gpu_ms
    );
}"""
assert old in s, "bench gen anchor"
s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("qwen_bench updated")
