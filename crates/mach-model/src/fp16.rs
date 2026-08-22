//! Host-side fp16/bf16 conversion (no external dependencies).
//!
//! Used to prepare half-precision weights and activations for the device. The
//! GPU kernels implement the same conversions natively (`__float_as_uint` /
//! `__uint_as_float`), so host and device agree on the exact bit patterns.

/// Rounds an f32 to fp16 (IEEE 754 binary16, round-to-nearest-even).
#[must_use]
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;

    if exp == 0xff {
        // NaN / Inf: keep payload bit set so NaN round-trips.
        return sign | 0x7c00 | if mant != 0 { 0x0200 } else { 0 };
    }
    // Rebias exponent from fp32 (127) to fp16 (15).
    let mut e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        // Subnormal / zero. Value = (1.mant) * 2^(e-15); a subnormal fp16
        // encodes value = m16 * 2^-24.
        if e < -10 {
            return sign; // too small -> zero
        }
        let m = mant | 0x80_0000;
        let shift = 14 - e; // >= 14 (e <= 0)
        let half = 1u32 << (shift - 1);
        let mut m16 = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        if rem > half || (rem == half && (m16 & 1) == 1) {
            m16 += 1;
        }
        return sign | m16 as u16;
    }
    // Normal fp16: 10 mantissa bits.
    let mut m16 = mant >> 13;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (m16 & 1) == 1) {
        m16 += 1;
        if m16 == 0x400 {
            // Mantissa overflow: carry into the exponent.
            m16 = 0;
            e += 1;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | (((e as u16) & 0x1f) << 10) | (m16 as u16)
}

/// Expands an fp16 to f32 (exact).
#[must_use]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x03ff) as u32;
    let b = if e == 0 {
        if m == 0 {
            sign
        } else {
            // Normalize the subnormal mantissa.
            let mut m2 = m;
            let mut e2 = 127u32 - 15 + 1;
            while (m2 & 0x0400) == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            sign | (e2 << 23) | ((m2 & 0x03ff) << 13)
        }
    } else if e == 0x1f {
        sign | 0x7f80_0000 | (m << 13) // inf / nan
    } else {
        sign | (((e as i32 - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(b)
}

/// Rounds an f32 to bf16 (round-to-nearest-even on the high 16 bits).
#[must_use]
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    let mut h = (b >> 16) as u16;
    let rem = b & 0xffff;
    if rem > 0x8000 || (rem == 0x8000 && (h & 1) == 1) {
        h = h.wrapping_add(1);
    }
    h
}

/// Expands a bf16 to f32 (exact).
#[must_use]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-1.0), 0xbc00);
        assert_eq!(f32_to_f16(2.0), 0x4000);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(65504.0), 0x7bff); // max normal
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // -> inf
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_f16(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_f16(6.1e-5), 0x03ff); // just below 2^-14 -> max subnormal
        assert_eq!(f32_to_f16(1.0 / 16384.0), 0x0400); // min normal = 2^-14
        assert_eq!(f32_to_f16(6.0e-8), 0x0001); // min subnormal = 2^-24
        assert_eq!(f32_to_f16(-6.0e-8), 0x8001);
    }

    #[test]
    fn f16_round_trip() {
        let cases = [
            0.0f32,
            1.0,
            -1.0,
            0.5,
            std::f32::consts::PI,
            -std::f32::consts::E,
            1e-3,
            1e-5,
            123.456,
            -0.25,
            65504.0,
            0.000061,
            -0.000061,
            100.0,
        ];
        for c in cases {
            let rt = f16_to_f32(f32_to_f16(c));
            let err = (rt - c).abs();
            let rel = err / c.abs().max(1e-30);
            // fp16 has ~3 decimal digits; subnormals (e.g. 1e-5) are looser.
            assert!(
                rel < 1e-2,
                "f16 round-trip rel err too large for {c}: {rt} (err {err})"
            );
        }
    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn f16_round_to_nearest_even() {
        // 1.0000001192 (just above 1.0) -> 1.0 (rounds down, tie-to-even on 0).
        assert_eq!(f32_to_f16(1.0000001192), 0x3c00);
        // Just below 1.5 -> rounds to 1.5; just above -> 1.5 (RNE).
        let below = 1.4999389648f32; // 1.5 - 2^-11-ish
        let above = 1.5000610352f32; // 1.5 + 2^-11-ish
        assert_eq!(f16_to_f32(f32_to_f16(below)), 1.5);
        assert_eq!(f16_to_f32(f32_to_f16(above)), 1.5);
    }

    #[test]
    fn bf16_round_trip() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0)), 1.0);
        for c in [0.0f32, 1.0, -1.0, 0.5, std::f32::consts::PI, 1e-5, 1e5] {
            let rt = bf16_to_f32(f32_to_bf16(c));
            assert!(
                (rt - c).abs() / c.abs().max(1e-30) < 1e-2,
                "bf16 {c} -> {rt}"
            );
        }
    }
}
