//! OpenAI chat content parts → MachServe image inputs.
//!
//! C3e scope: parse `content` arrays (`text` / `image_url`), decode base64
//! data-URL PNG/JPEG payloads to RGB8, run the C3d image processor, and
//! expand `<|image_pad|>` placeholders to the merged vision-token count.
//! HTTP(S) fetching, engine vision execution and per-request embedding
//! injection are C3f; the chat handler fails fast for images until then.

use base64::Engine as _;
use image::GenericImageView as _;
use mach_model::image_processor::{ImageProcessorConfig, ProcessedImage, preprocess_image};
use mach_model::vision::VisionGrid;
use serde::Deserialize;

/// `<|vision_start|><|image_pad|><|vision_end|>`, exactly what the Qwen3.8
/// chat template renders for one image.
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// Largest accepted data-URL payload (encoded bytes).
pub const MAX_ENCODED_IMAGE_BYTES: usize = 64 << 20;
/// Largest decoded RGBA/RGB allocation accepted from the image decoder.
const MAX_DECODE_BYTES: u64 = 512 << 20;
/// Largest accepted HTTP(S) image response body.
pub const MAX_FETCH_BYTES: usize = 64 << 20;
/// Largest accepted RGB8 buffer after channel expansion. 256 MiB covers
/// an ~89 MP RGB photo while bounding grayscale/CMYK decode bombs.
pub const MAX_RGB8_BYTES: usize = 256 << 20;

/// OpenAI `content`: a plain string or an array of typed parts.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// One element of an OpenAI multimodal `content` array.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

/// OpenAI `image_url` object.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default)]
    pub detail: Option<String>,
}

/// Errors from parsing/decoding/preprocessing multimodal content.
#[derive(Debug, thiserror::Error)]
pub enum MultimodalError {
    #[error("image_url must be a data URL or http(s) URL")]
    UnsupportedUrl,
    #[error("invalid image data URL: {0}")]
    BadDataUrl(String),
    #[error("unsupported image media type: {0}")]
    UnsupportedMedia(String),
    #[error("image data is too large: {0} bytes")]
    TooLarge(usize),
    #[error("image base64 decode failed: {0}")]
    Base64(String),
    #[error("image fetch failed: {0}")]
    Fetch(String),
    #[error("image decode failed: {0}")]
    Decode(String),
    #[error(transparent)]
    Preprocess(#[from] mach_model::Error),
    #[error("image pad expansion failed: {0}")]
    PadExpansion(String),
}

/// Render `content` into the chat text and collect image URLs in order.
///
/// A plain string is appended verbatim; each `image_url` part appends the
/// chat template placeholder and pushes the URL.
pub fn render_content(content: &ChatContent, text: &mut String, images: &mut Vec<String>) {
    match content {
        ChatContent::Text(t) => text.push_str(t),
        ChatContent::Parts(parts) => {
            for part in parts {
                match part {
                    ContentPart::Text { text: t } => text.push_str(t),
                    ContentPart::ImageUrl { image_url } => {
                        images.push(image_url.url.clone());
                        text.push_str(IMAGE_PLACEHOLDER);
                    }
                }
            }
        }
    }
}

/// A decoded image ready for [`mach_model::image_processor::preprocess_image`].
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedImage {
    /// Row-major RGB8 pixels.
    pub rgb8: Vec<u8>,
    pub height: usize,
    pub width: usize,
}

/// Decode a `data:image/png;base64,...` (or JPEG) URL to RGB8.
pub fn decode_data_url(url: &str) -> Result<DecodedImage, MultimodalError> {
    let rest = url
        .strip_prefix("data:")
        .ok_or(MultimodalError::UnsupportedUrl)?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| MultimodalError::BadDataUrl("missing comma".into()))?;
    let mut fields = meta.split(';');
    let mime = fields.next().unwrap_or_default().to_ascii_lowercase();
    let is_base64 = fields.any(|f| f.eq_ignore_ascii_case("base64"));
    if !is_base64 {
        return Err(MultimodalError::BadDataUrl(
            "only base64 data URLs are supported".into(),
        ));
    }
    match mime.as_str() {
        "image/png" | "image/jpeg" | "image/jpg" => {}
        other => return Err(MultimodalError::UnsupportedMedia(other.into())),
    }
    if payload.len() > MAX_ENCODED_IMAGE_BYTES {
        return Err(MultimodalError::TooLarge(payload.len()));
    }
    let compact: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.len() > MAX_ENCODED_IMAGE_BYTES {
        return Err(MultimodalError::TooLarge(compact.len()));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(compact.as_bytes())
        .map_err(|e| MultimodalError::Base64(e.to_string()))?;
    decode_rgb8(&bytes)
}

/// Decode PNG/JPEG bytes to RGB8 with an allocation limit.
pub fn decode_rgb8(bytes: &[u8]) -> Result<DecodedImage, MultimodalError> {
    decode_rgb8_with_limit(bytes, MAX_RGB8_BYTES)
}

fn decode_rgb8_with_limit(
    bytes: &[u8],
    max_rgb8_bytes: usize,
) -> Result<DecodedImage, MultimodalError> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes));
    reader = reader
        .with_guessed_format()
        .map_err(|e| MultimodalError::Decode(e.to_string()))?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(MAX_DECODE_BYTES);
    reader.limits(limits);
    let img = reader
        .decode()
        .map_err(|e| MultimodalError::Decode(e.to_string()))?;
    let (width, height) = img.dimensions();
    let rgb_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| MultimodalError::Decode("RGB8 size overflow".into()))?;
    if rgb_bytes > max_rgb8_bytes {
        return Err(MultimodalError::TooLarge(rgb_bytes));
    }
    let rgb = img.into_rgb8();
    let (width, height) = rgb.dimensions();
    Ok(DecodedImage {
        rgb8: rgb.into_raw(),
        height: height as usize,
        width: width as usize,
    })
}

/// Decode a data URL and run the C3d image processor.
pub fn preprocess_data_url(
    url: &str,
    cfg: &ImageProcessorConfig,
) -> Result<ProcessedImage, MultimodalError> {
    let image = decode_data_url(url)?;
    Ok(preprocess_image(
        cfg,
        &image.rgb8,
        image.height,
        image.width,
    )?)
}

/// Fetch an `http(s)://` image (bounded) or decode a data URL, then run the
/// C3d image processor.
pub async fn fetch_image_url(
    url: &str,
    cfg: &ImageProcessorConfig,
) -> Result<ProcessedImage, MultimodalError> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| MultimodalError::Fetch(e.to_string()))?;
    fetch_with_client(&client, url, cfg).await
}

async fn fetch_with_client(
    client: &reqwest::Client,
    url: &str,
    cfg: &ImageProcessorConfig,
) -> Result<ProcessedImage, MultimodalError> {
    if url.starts_with("data:") {
        return preprocess_data_url(url, cfg);
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(MultimodalError::UnsupportedUrl);
    }
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| MultimodalError::Fetch(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(MultimodalError::Fetch(format!("HTTP {}", resp.status())));
    }
    if let Some(len) = resp.content_length()
        && len > MAX_FETCH_BYTES as u64
    {
        return Err(MultimodalError::TooLarge(len as usize));
    }
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    if !content_type.is_empty()
        && !matches!(
            content_type.as_str(),
            "image/png" | "image/jpeg" | "image/jpg"
        )
    {
        return Err(MultimodalError::UnsupportedMedia(content_type));
    }
    let mut resp = resp;
    let mut bytes = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| MultimodalError::Fetch(e.to_string()))?
    {
        let total = bytes.len() + chunk.len();
        if total > MAX_FETCH_BYTES {
            return Err(MultimodalError::TooLarge(total));
        }
        bytes.extend_from_slice(&chunk);
    }
    let image = decode_rgb8(&bytes)?;
    Ok(preprocess_image(
        cfg,
        &image.rgb8,
        image.height,
        image.width,
    )?)
}

/// Expand each `<|image_pad|>` to `grid_t * grid_h * grid_w / merge^2` pad
/// tokens, consuming `grids` in order (HF `replace_image_token`).
pub fn expand_image_pads(
    tokens: &[u32],
    pad_token_id: u32,
    grids: &[VisionGrid],
    merge_size: usize,
) -> Result<Vec<u32>, MultimodalError> {
    if merge_size == 0 {
        return Err(MultimodalError::PadExpansion(
            "merge_size must be positive".into(),
        ));
    }
    let merge_unit = merge_size
        .checked_mul(merge_size)
        .ok_or_else(|| MultimodalError::PadExpansion("merge unit overflow".into()))?;
    let mut out = Vec::with_capacity(tokens.len());
    let mut next_grid = 0usize;
    for &token in tokens {
        if token != pad_token_id {
            out.push(token);
            continue;
        }
        let grid = grids.get(next_grid).ok_or_else(|| {
            MultimodalError::PadExpansion(format!("image pad token {next_grid} has no grid entry"))
        })?;
        next_grid += 1;
        let count = grid[0]
            .checked_mul(grid[1])
            .and_then(|v| v.checked_mul(grid[2]))
            .ok_or_else(|| MultimodalError::PadExpansion("grid size overflow".into()))?;
        if count == 0 || !count.is_multiple_of(merge_unit) {
            return Err(MultimodalError::PadExpansion(format!(
                "grid {grid:?} does not expand under merge_size {merge_size}"
            )));
        }
        out.extend(std::iter::repeat_n(pad_token_id, count / merge_unit));
    }
    if next_grid != grids.len() {
        return Err(MultimodalError::PadExpansion(format!(
            "{} image grids were not consumed by pad tokens",
            grids.len() - next_grid
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png_bytes(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([x as u8 * 10, y as u8 * 20, 7])
        });
        let mut bytes = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
        bytes
    }

    fn png_data_url(width: u32, height: u32) -> String {
        let payload = base64::engine::general_purpose::STANDARD.encode(png_bytes(width, height));
        format!("data:image/png;base64,{payload}")
    }

    #[test]
    fn decodes_png_data_url() {
        let url = png_data_url(3, 2);
        let image = decode_data_url(&url).unwrap();
        assert_eq!((image.width, image.height), (3, 2));
        assert_eq!(image.rgb8.len(), 3 * 2 * 3);
        assert_eq!(&image.rgb8[0..3], &[0, 0, 7]);
        assert_eq!(&image.rgb8[3..6], &[10, 0, 7]);
    }

    #[test]
    fn preprocesses_data_url_with_c3d_processor() {
        let cfg = ImageProcessorConfig {
            patch_size: 1,
            temporal_patch_size: 1,
            merge_size: 1,
            min_pixels: 1,
            max_pixels: 64,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
        };
        let out = preprocess_data_url(&png_data_url(2, 3), &cfg).unwrap();
        assert_eq!(out.grid, [1, 3, 2]);
        assert_eq!(out.pixel_values.len(), 3 * 2 * 3);
    }

    #[test]
    fn expands_image_pads_in_grid_order() {
        let tokens = [1, 7, 2, 7];
        let grids = [[1, 4, 4], [1, 2, 2]];
        let got = expand_image_pads(&tokens, 7, &grids, 2).unwrap();
        assert_eq!(got, [1, 7, 7, 7, 7, 2, 7]);
    }

    #[test]
    fn rejects_bad_image_inputs() {
        assert!(matches!(
            decode_data_url("https://example.com/a.png"),
            Err(MultimodalError::UnsupportedUrl)
        ));
        assert!(matches!(
            decode_data_url("data:image/png,notbase64"),
            Err(MultimodalError::BadDataUrl(_))
        ));
        assert!(matches!(
            decode_data_url("data:image/gif;base64,AAAA"),
            Err(MultimodalError::UnsupportedMedia(_))
        ));
        assert!(matches!(
            decode_data_url("data:image/png;base64,****"),
            Err(MultimodalError::Base64(_))
        ));
    }

    #[test]
    fn rejects_bad_pad_expansion() {
        assert!(expand_image_pads(&[7], 7, &[], 2).is_err());
        assert!(expand_image_pads(&[1], 7, &[[1, 4, 4]], 2).is_err());
        assert!(expand_image_pads(&[7], 7, &[[1, 3, 3]], 2).is_err());
        assert!(expand_image_pads(&[7], 7, &[[1, 4, 4]], 0).is_err());
    }

    #[test]
    fn rejects_oversized_rgb8_expansion() {
        let img = image::GrayImage::from_pixel(4, 4, image::Luma([1u8]));
        let mut bytes = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
        let err = decode_rgb8_with_limit(&bytes, 8).unwrap_err();
        assert!(matches!(err, MultimodalError::TooLarge(48)), "{err}");
    }

    #[test]
    fn rejects_oversized_data_url_payload() {
        let url = format!(
            "data:image/png;base64,{}",
            "A".repeat(MAX_ENCODED_IMAGE_BYTES + 4)
        );
        assert!(matches!(
            decode_data_url(&url),
            Err(MultimodalError::TooLarge(_))
        ));
    }

    fn small_cfg() -> ImageProcessorConfig {
        ImageProcessorConfig {
            patch_size: 1,
            temporal_patch_size: 1,
            merge_size: 1,
            min_pixels: 1,
            max_pixels: 64,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
        }
    }

    fn no_proxy_client() -> reqwest::Client {
        reqwest::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap()
    }

    fn http_response(status: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
        let mut out = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        out.extend_from_slice(body);
        out
    }

    async fn spawn_raw(response: Vec<u8>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((mut socket, _)) = listener.accept().await {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = [0u8; 4096];
                let _ = socket.read(&mut buf).await;
                let _ = socket.write_all(&response).await;
            }
        });
        format!("http://{addr}/image")
    }

    #[tokio::test]
    async fn fetch_data_url_uses_processor() {
        let out = fetch_with_client(&no_proxy_client(), &png_data_url(2, 3), &small_cfg())
            .await
            .unwrap();
        assert_eq!(out.grid, [1, 3, 2]);
        assert_eq!(out.pixel_values.len(), 3 * 2 * 3);
    }

    #[tokio::test]
    async fn fetches_http_png_image() {
        let url = spawn_raw(http_response("200 OK", "image/png", &png_bytes(3, 2))).await;
        let out = fetch_with_client(&no_proxy_client(), &url, &small_cfg())
            .await
            .unwrap();
        assert_eq!(out.grid, [1, 2, 3]);
        assert_eq!(out.pixel_values.len(), 3 * 2 * 3);
    }

    #[tokio::test]
    async fn rejects_http_error_status() {
        let url = spawn_raw(http_response("404 Not Found", "text/plain", b"nope")).await;
        let err = fetch_with_client(&no_proxy_client(), &url, &small_cfg())
            .await
            .unwrap_err();
        assert!(matches!(err, MultimodalError::Fetch(_)), "{err}");
    }

    #[tokio::test]
    async fn rejects_non_image_content_type() {
        let url = spawn_raw(http_response("200 OK", "text/html", b"<html/>")).await;
        let err = fetch_with_client(&no_proxy_client(), &url, &small_cfg())
            .await
            .unwrap_err();
        assert!(matches!(err, MultimodalError::UnsupportedMedia(_)), "{err}");
    }

    #[tokio::test]
    async fn rejects_oversized_http_content_length() {
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            MAX_FETCH_BYTES + 1
        );
        let url = spawn_raw(head.into_bytes()).await;
        let err = fetch_with_client(&no_proxy_client(), &url, &small_cfg())
            .await
            .unwrap_err();
        assert!(matches!(err, MultimodalError::TooLarge(_)), "{err}");
    }

    #[tokio::test]
    async fn rejects_unsupported_scheme() {
        let err = fetch_with_client(&no_proxy_client(), "ftp://example.com/a.png", &small_cfg())
            .await
            .unwrap_err();
        assert!(matches!(err, MultimodalError::UnsupportedUrl), "{err}");
    }
}
