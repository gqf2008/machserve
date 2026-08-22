import io
def rd(p):
    return io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
def wr(p, s):
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)

# 1) lib.rs: pub mod sampling
p = "E:/Users/gxh/Documents/GitHub/machserve/crates/mach-model/src/lib.rs"
s = rd(p)
s = s.replace("pub mod ref_model;\npub mod weights;", "pub mod ref_model;\npub mod sampling;\npub mod weights;")
wr(p, s)

# 2) model.rs: add sampler field + init + decode_step_sampled
p = "E:/Users/gxh/Documents/GitHub/machserve/crates/mach-model/src/model.rs"
s = rd(p)

s = s.replace("use crate::kernels::HipKernels;", "use crate::kernels::HipKernels;\nuse crate::sampling::HipSampler;")

# field
s = s.replace("    /// Host copy of layer-0 rms weights (debug).\n    dbg_rms_host: Vec<f32>,\n}", "")  # not present in restored version
s = s.replace("    /// Number of tokens stored so far.\n    pos: usize,\n    host_pins: Vec<*mut core::ffi::c_void>,\n}",
              "    /// Number of tokens stored so far.\n    pos: usize,\n    host_pins: Vec<*mut core::ffi::c_void>,\n    /// GPU-side greedy sampler (reads only the sampled token).\n    sampler: HipSampler,\n}")

# init in new(): after k created, before Self { ... }
old_init = "        let k = Arc::new(HipKernels::new(Arc::clone(&hip))?);\n        let mut m = Self {"
new_init = "        let k = Arc::new(HipKernels::new(Arc::clone(&hip))?);\n        let sampler = HipSampler::new(Arc::clone(&hip), k.stream)?;\n        let mut m = Self {"
assert old_init in s, "init anchor"
s = s.replace(old_init, new_init)

s = s.replace("            pos: 0,\n            host_pins: Vec::new(),\n        };",
              "            pos: 0,\n            host_pins: Vec::new(),\n            sampler,\n        };")

# decode_step_sampled: add after decode_step
anchor = "    /// Runs `tokens` one by one and returns logits of the final token.\n    pub fn forward("
add = """    /// One decode step returning the greedy-sampled next token, reading back
    /// only 4 bytes instead of the full logits vector.
    pub fn decode_step_sampled(&mut self, token: u32) -> Result<u32, Error> {
        if self.pos >= self.cfg.max_seq_len {
            return Err(Error::Model("sequence length exceeded".into()));
        }
        self.update_inputs(token)?;
        self.run_kernels()?;
        let next = self.sampler.argmax(self.logits, self.cfg.vocab_size)?;
        self.pos += 1;
        Ok(next)
    }

    /// Runs `tokens` one by one and returns logits of the final token.
    pub fn forward("""
assert anchor in s, "decode_step_sampled anchor"
s = s.replace(anchor, add)
wr(p, s)
print("wired")
