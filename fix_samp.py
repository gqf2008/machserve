import io
p = "E:/Users/gxh/Documents/GitHub/machserve/crates/mach-model/src/sampling.rs"
s = io.open(p, encoding="utf-8", newline="").read().replace("\r\n", "\n")
s = s.replace("""        let mut p1 = vec![
            &lp as *const *const f32 as *mut core::ffi::c_void,
            &bv as *const *mut f32 as *mut core::ffi::c_void,
            &bi as *const *mut i32 as *mut core::ffi::c_void,
            &vocab_i as *const i32 as *mut core::ffi::c_void,
        ];""",
"""        let mut p1 = vec![
            &lp as *const *const f32 as *mut core::ffi::c_void,
            &bv as *const *mut f32 as *mut core::ffi::c_void,
            &bi as *const *mut i32 as *mut core::ffi::c_void,
            &vocab_i as *const i32 as *mut core::ffi::c_void,
        ];""")
s = s.replace("""        let mut p2 = vec![
            &iv as *const *const f32 as *mut core::ffi::c_void,
            &ii as *const *const i32 as *mut core::ffi::c_void,
            &ov as *const *mut f32 as *mut core::ffi::c_void,
            &oi as *const *mut i32 as *mut core::ffi::c_void,
            &n as *const i32 as *mut core::ffi::c_void,
        ];""",
"""        let mut p2 = vec![
            &iv as *const *mut f32 as *mut core::ffi::c_void,
            &ii as *const *mut i32 as *mut core::ffi::c_void,
            &ov as *const *mut f32 as *mut core::ffi::c_void,
            &oi as *const *mut i32 as *mut core::ffi::c_void,
            &n as *const i32 as *mut core::ffi::c_void,
        ];""")
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("fixed")
