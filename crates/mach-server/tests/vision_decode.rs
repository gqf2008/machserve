//! EXIF-orientation parity for the server image decoder.
//!
//! HF `load_image` applies `PIL.ImageOps.exif_transpose`; the fixtures under
//! `tests/data/exif` are the same 16x24 four-quadrant JPEG (red/green/blue/
//! yellow) saved with EXIF orientation 1..8 by Pillow 12.3.0. The golden
//! colours below are the quadrant centres of the `exif_transpose` result, so a
//! wrong direction moves a colour by >180 per channel. The image crate decodes
//! JPEG with zune-jpeg instead of libjpeg, which measures <=2 per channel on
//! these fixtures; the tolerance keeps headroom for decoder-version drift.

use mach_server::multimodal::{DecodedImage, decode_rgb8};

/// Quadrant-centre colours of the transposed image, in TL, TR, BL, BR order.
type Quadrants = [(u8, u8, u8); 4];

fn quadrants(image: &DecodedImage) -> Quadrants {
    let xs = [image.width / 4, image.width * 3 / 4];
    let ys = [image.height / 4, image.height * 3 / 4];
    let sample = |x: usize, y: usize| {
        let i = (y * image.width + x) * 3;
        (image.rgb8[i], image.rgb8[i + 1], image.rgb8[i + 2])
    };
    [
        sample(xs[0], ys[0]),
        sample(xs[1], ys[0]),
        sample(xs[0], ys[1]),
        sample(xs[1], ys[1]),
    ]
}

fn assert_close(got: (u8, u8, u8), want: (u8, u8, u8), what: &str) {
    let got = [got.0 as i32, got.1 as i32, got.2 as i32];
    let want = [want.0 as i32, want.1 as i32, want.2 as i32];
    for channel in 0..3 {
        assert!(
            (got[channel] - want[channel]).abs() <= 16,
            "{what} channel {channel}: {got:?} vs {want:?}"
        );
    }
}

#[test]
fn decode_rgb8_applies_exif_orientation() {
    // EXIF orientation 1..8 -> (displayed size, Pillow `exif_transpose` quadrants).
    let cases: [(&[u8], (usize, usize), Quadrants); 8] = [
        (
            include_bytes!("data/exif/o1.jpg"),
            (16, 24),
            [(222, 24, 23), (28, 207, 20), (23, 19, 217), (219, 210, 17)],
        ),
        (
            include_bytes!("data/exif/o2.jpg"),
            (16, 24),
            [(34, 203, 22), (229, 20, 25), (216, 211, 23), (22, 19, 222)],
        ),
        (
            include_bytes!("data/exif/o3.jpg"),
            (16, 24),
            [(216, 211, 23), (22, 19, 222), (23, 209, 20), (236, 15, 32)],
        ),
        (
            include_bytes!("data/exif/o4.jpg"),
            (16, 24),
            [(23, 19, 217), (219, 210, 17), (233, 17, 30), (20, 211, 19)],
        ),
        (
            include_bytes!("data/exif/o5.jpg"),
            (24, 16),
            [(222, 24, 23), (23, 19, 217), (28, 207, 20), (219, 210, 17)],
        ),
        (
            include_bytes!("data/exif/o6.jpg"),
            (24, 16),
            [(23, 19, 217), (233, 17, 30), (219, 210, 17), (20, 211, 19)],
        ),
        (
            include_bytes!("data/exif/o7.jpg"),
            (24, 16),
            [(216, 211, 23), (23, 209, 20), (22, 19, 222), (236, 15, 32)],
        ),
        (
            include_bytes!("data/exif/o8.jpg"),
            (24, 16),
            [(34, 203, 22), (216, 211, 23), (229, 20, 25), (22, 19, 222)],
        ),
    ];

    for (index, (bytes, size, want)) in cases.iter().enumerate() {
        let orientation = index + 1;
        let got = decode_rgb8(bytes).unwrap();
        assert_eq!(
            (got.width, got.height),
            *size,
            "EXIF orientation {orientation} must transpose the image"
        );
        for (corner, (got, want)) in quadrants(&got).iter().zip(want.iter()).enumerate() {
            assert_close(
                *got,
                *want,
                &format!("orientation {orientation} corner {corner}"),
            );
        }
    }
}
