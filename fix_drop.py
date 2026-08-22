import io
p = "E:/Users/gxh/Documents/GitHub/machserve/crates/mach-model/src/sampling.rs"
s = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
old = """impl Drop for HipSampler {
    fn drop(&mut self) {
        unsafe {
            let _ = hip::free(&self.hip, self.block_val as *mut _);
            let _ = hip::free(&self.hip, self.block_idx as *mut _);
            let _ = hip::free(&self.hip, self.out_val as *mut _);
            let _ = hip::free(&self.hip, self.out_idx as *mut _);
            let _ = hip::host_free(&self.hip, self.host_idx as *mut _);
        }
    }
}"""
new = """impl Drop for HipSampler {
    fn drop(&mut self) {
        let _ = hip::free(&self.hip, self.block_val as *mut _);
        let _ = hip::free(&self.hip, self.block_idx as *mut _);
        let _ = hip::free(&self.hip, self.out_val as *mut _);
        let _ = hip::free(&self.hip, self.out_idx as *mut _);
        let _ = hip::host_free(&self.hip, self.host_idx as *mut _);
    }
}"""
assert old in s, "drop"
s = s.replace(old, new)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
