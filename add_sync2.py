import io
p = "E:/Users/gxh/Documents/GitHub/machserve/crates/mach-model/src/model.rs"
s = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
anchor = "    /// One decode step returning the greedy-sampled next token"
add = """    /// Syncs the model stream (debug/measurement).
    pub fn sync(&self) -> Result<(), Error> {
        self.k.sync()
    }

    /// One decode step returning the greedy-sampled next token"""
assert anchor in s, "anchor"
s = s.replace(anchor, add)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("sync method added")
