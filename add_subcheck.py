import io
p = "E:/Users/gxh/Documents/GitHub/machserve/crates/mach-model/examples/qwen_bench.rs"
s = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
old = """    // --- GPU-sampled generation: only 4 bytes read back per token ---"""
new = """    // --- Verify launch-only is submission rate: 50 eager steps + ONE final sync ---
    model.reset_state().expect("reset");
    let m = 50usize;
    let t6 = Instant::now();
    for i in 0..m {
        model.step_eager((i % 977) as u32).expect("eager");
    }
    model.sync().expect("sync");
    let batch_ms = t6.elapsed().as_secs_f64() * 1000.0 / m as f64;
    println!("eager 50 steps + 1 sync: {batch_ms:.2} ms/token (GPU completion rate)");

    // --- GPU-sampled generation: only 4 bytes read back per token ---"""
assert old in s, "anchor"
s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
