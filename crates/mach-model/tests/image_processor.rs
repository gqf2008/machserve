//! Qwen3.8 image-preprocessing parity against transformers 5.16.1
//! `Qwen2VLImageProcessorPil` (Pillow 12.3.0 BICUBIC).
//!
//! Goldens were generated on 2026-09-11 with the checkpoint
//! `preprocessor_config.json` values and a deterministic 64-bit LCG image,
//! then embedded as raw `f32` bit patterns so the assertions are exact.

use mach_model::image_processor::{ImageProcessorConfig, preprocess_image, smart_resize};

fn lcg_image(height: usize, width: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut out = Vec::with_capacity(height * width * 3);
    for _ in 0..height * width * 3 {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        out.push((s >> 33) as u8);
    }
    out
}

fn real_cfg() -> ImageProcessorConfig {
    ImageProcessorConfig::default()
}

#[test]
fn parses_qwen38_preprocessor_config() {
    let json: serde_json::Value = serde_json::from_str(
        r#"{
            "size": {"longest_edge": 16777216, "shortest_edge": 65536},
            "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
            "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]
        }"#,
    )
    .unwrap();
    let cfg = ImageProcessorConfig::from_hf_json(&json).unwrap();
    assert_eq!(cfg, real_cfg());
    assert_eq!(cfg.factor(), 32);
}

#[test]
fn smart_resize_matches_hf_reference() {
    let cases = [
        ((48usize, 64usize), (224usize, 320usize)),
        ((1024, 1024), (1024, 1024)),
        ((8, 8), (256, 256)),
        ((5000, 4000), (4576, 3648)),
        ((200, 1), (3648, 32)),
    ];
    for ((h, w), (want_h, want_w)) in cases {
        let got = smart_resize(h, w, 32, 65_536, 16_777_216).unwrap();
        assert_eq!(got, (want_h, want_w), "smart_resize({h}, {w})");
    }
    let err = smart_resize(201, 1, 32, 65_536, 16_777_216)
        .unwrap_err()
        .to_string();
    assert!(err.contains("aspect ratio"), "{err}");
}

#[test]
fn tiny_preprocess_matches_hf_golden() {
    let cfg = ImageProcessorConfig {
        patch_size: 1,
        temporal_patch_size: 1,
        merge_size: 1,
        min_pixels: 4,
        max_pixels: 64,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
    };
    let img = lcg_image(2, 2, 0xabcdef01_23456789);
    let out = preprocess_image(&cfg, &img, 2, 2).unwrap();
    assert_eq!(out.grid, [1, 2, 2]);
    let want_bits: [u32; 12] = [
        3205994392, 3192177860, 3193756892, 3201224398, 3205468048, 3206520736, 1056109300,
        1046273248, 1062721496, 3187967108, 1063905770, 3210468316,
    ];
    let got_bits: Vec<u32> = out.pixel_values.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got_bits, want_bits);
}

#[test]
fn small_preprocess_matches_hf_golden() {
    let cfg = ImageProcessorConfig {
        patch_size: 1,
        temporal_patch_size: 1,
        merge_size: 1,
        min_pixels: 64,
        max_pixels: 4096,
        image_mean: [0.5; 3],
        image_std: [0.5; 3],
    };
    let img = lcg_image(5, 7, 0x0f1e2d3c_4b5a6978);
    let out = preprocess_image(&cfg, &img, 5, 7).unwrap();
    assert_eq!(out.grid, [1, 7, 10]);
    assert_eq!(out.tokens(), 70);
    assert_eq!(out.pixel_values.len(), 210);
    let samples: [(usize, [u32; 6]); 3] = [
        (
            0,
            [
                1056898816, 3197276818, 1061142464, 1061142464, 3208099768, 1034463408,
            ],
        ),
        (
            105,
            [
                3204678532, 3168854240, 1061274050, 3205204876, 3193756892, 1061537222,
            ],
        ),
        (
            204,
            [
                3211389418, 1046799592, 1051635376, 3207441838, 1047325936, 1059300260,
            ],
        ),
    ];
    for (start, want) in samples {
        for (i, &bits) in want.iter().enumerate() {
            assert_eq!(
                out.pixel_values[start + i].to_bits(),
                bits,
                "index {}",
                start + i
            );
        }
    }
    assert_checksums(
        &out.pixel_values,
        -1.709800124168396,
        -1105.5878486037254,
        51.6396176529629,
    );
}

fn assert_checksums(values: &[f32], want_sum: f64, want_wsum: f64, want_sq: f64) {
    let mut sum = 0.0f64;
    let mut wsum = 0.0f64;
    let mut sq = 0.0f64;
    for (i, &v) in values.iter().enumerate() {
        let v = v as f64;
        sum += v;
        wsum += i as f64 * v;
        sq += v * v;
    }
    assert!((sum - want_sum).abs() < 1e-3, "sum {sum} vs {want_sum}");
    assert!(
        (wsum - want_wsum).abs() < 0.05,
        "wsum {wsum} vs {want_wsum}"
    );
    assert!((sq - want_sq).abs() < 1e-2, "sq {sq} vs {want_sq}");
}

#[test]
fn real_preprocess_matches_hf_golden() {
    let cfg = real_cfg();
    let img = lcg_image(48, 64, 0x11223344_55667788);
    let out = preprocess_image(&cfg, &img, 48, 64).unwrap();
    assert_eq!(out.grid, [1, 14, 20]);
    assert_eq!(out.tokens(), 280);
    assert_eq!(out.pixel_values.len(), 280 * 1536);
    let samples: [(usize, [u32; 6]); 5] = [
        (
            0,
            [
                1065353216, 1065353216, 1065353216, 1064826872, 1062326738, 1059300260,
            ],
        ),
        (
            1536,
            [
                3196224130, 3192704204, 3193230548, 3195862268, 3198066334, 3199382194,
            ],
        ),
        (
            19 * 1536,
            [
                1053477580, 1053214408, 1052688064, 1050582688, 1045220560, 1024495776,
            ],
        ),
        (
            20 * 1536,
            [
                1053477580, 1058115986, 1058510744, 1053214408, 1021370624, 3201487570,
            ],
        ),
        (
            279 * 1536,
            [
                1057458056, 1060616120, 1063116254, 1064432114, 1064300528, 1063247840,
            ],
        ),
    ];
    for (start, want) in samples {
        for (i, &bits) in want.iter().enumerate() {
            assert_eq!(
                out.pixel_values[start + i].to_bits(),
                bits,
                "index {}",
                start + i
            );
        }
    }
    assert_checksums(
        &out.pixel_values,
        703.7260987758636,
        510_562_086.0040089,
        96_497.95109243425,
    );
}

#[test]
fn preprocess_rejects_bad_inputs() {
    let cfg = real_cfg();
    let err = preprocess_image(&cfg, &[0u8; 10], 4, 4)
        .unwrap_err()
        .to_string();
    assert!(err.contains("expected 48"), "{err}");

    let bad = ImageProcessorConfig {
        min_pixels: 100,
        max_pixels: 50,
        ..real_cfg()
    };
    assert!(preprocess_image(&bad, &[0u8; 48], 4, 4).is_err());
    assert!(smart_resize(4, 4, 0, 1, 16).is_err());
    assert!(smart_resize(0, 4, 2, 1, 16).is_err());
}

#[test]
fn downscale_nondefault_preprocess_matches_hf_golden() {
    let cfg = ImageProcessorConfig {
        patch_size: 2,
        temporal_patch_size: 3,
        merge_size: 2,
        min_pixels: 64,
        max_pixels: 1024,
        image_mean: [0.25, 0.5, 0.75],
        image_std: [0.5, 0.25, 0.2],
    };
    let img = lcg_image(40, 60, 0x5a5a1234_deadbeef);
    let out = preprocess_image(&cfg, &img, 40, 60).unwrap();
    assert_eq!(out.grid, [1, 12, 18]);
    assert_eq!(out.tokens(), 216);
    assert_eq!(out.pixel_values.len(), 216 * 36);
    let samples: [(usize, [u32; 6]); 3] = [
        (
            0,
            [
                1058083090, 1053148614, 1058741020, 1062688600, 1058083090, 1053148614,
            ],
        ),
        (
            100 * 36,
            [
                1054464474, 1061504326, 1065057148, 1058741020, 1054464474, 1061504326,
            ],
        ),
        (
            215 * 36,
            [
                1042457252, 1042457252, 1047720692, 1055517162, 1042457252, 1042457252,
            ],
        ),
    ];
    for (start, want) in samples {
        for (i, &bits) in want.iter().enumerate() {
            assert_eq!(
                out.pixel_values[start + i].to_bits(),
                bits,
                "index {}",
                start + i
            );
        }
    }
    assert_checksums(
        &out.pixel_values,
        -1781.3878052830696,
        -7_005_965.9489017725,
        7097.543473012898,
    );
}

#[test]
fn rejects_unsupported_preprocessor_flags() {
    for bad in [
        r#"{"do_normalize": false}"#,
        r#"{"do_rescale": false}"#,
        r#"{"rescale_factor": 0.5}"#,
        r#"{"merge_size": 0}"#,
        r#"{"image_std": [0.5, 0.0, 0.5]}"#,
    ] {
        let json: serde_json::Value = serde_json::from_str(bad).unwrap();
        assert!(
            ImageProcessorConfig::from_hf_json(&json).is_err(),
            "config should be rejected: {bad}"
        );
    }
}
