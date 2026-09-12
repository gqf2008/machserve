//! OpenAI chat content parts → MachServe image inputs.
//!
//! C3e scope: parse `content` arrays (`text` / `image_url`), decode base64
//! data-URL PNG/JPEG payloads to RGB8, run the C3d image processor, and
//! expand `<|image_pad|>` placeholders to the merged vision-token count.
//! HTTP(S) fetching, engine vision execution and per-request embedding
//! injection are C3f; the chat handler fails fast for images until then.

use base64::Engine as _;
use image::ImageDecoder as _;
use mach_model::image_processor::{
    ImageProcessorConfig, ProcessedImage, preprocess_image_limited,
    preprocess_image_limited_downscaling,
};
use mach_model::vision::VisionGrid;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// `<|vision_start|><|image_pad|><|vision_end|>`, exactly what the Qwen3.8
/// chat template renders for one image.
pub const IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// Largest accepted data-URL payload (encoded bytes).
pub const MAX_ENCODED_IMAGE_BYTES: usize = 64 << 20;
/// Largest decoded RGBA/RGB allocation accepted from the image decoder.
const MAX_DECODE_BYTES: u64 = 512 << 20;
/// Largest accepted HTTP(S) image response body.
pub const MAX_FETCH_BYTES: usize = 64 << 20;
/// Overall wall-clock limit for one image fetch (all redirect hops).
pub const FETCH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
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
    decode_rgb8_with_limits(bytes, MAX_DECODE_BYTES, max_rgb8_bytes)
}

fn decode_rgb8_with_limits(
    bytes: &[u8],
    max_decode_bytes: u64,
    max_rgb8_bytes: usize,
) -> Result<DecodedImage, MultimodalError> {
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes));
    reader = reader
        .with_guessed_format()
        .map_err(|e| MultimodalError::Decode(e.to_string()))?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(max_decode_bytes);
    reader.limits(limits.clone());
    let mut decoder = reader
        .into_decoder()
        .map_err(|e| MultimodalError::Decode(e.to_string()))?;
    // `ImageReader::decode` charges `total_bytes` against the allocation budget
    // before reading any pixels; `into_decoder` + `from_decoder` skips that, so
    // repeat the decode-time check here.
    limits
        .reserve(decoder.total_bytes())
        .map_err(|e| MultimodalError::Decode(e.to_string()))?;
    // Reject an oversized RGB8 result from the header, before `from_decoder`
    // allocates the decoded buffer. Orientation only permutes pixels, so
    // `width * height * 3` is the final size.
    let (width, height) = decoder.dimensions();
    let rgb_bytes = (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| MultimodalError::Decode("RGB8 size overflow".into()))?;
    if rgb_bytes > max_rgb8_bytes {
        return Err(MultimodalError::TooLarge(rgb_bytes));
    }
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let img = image::DynamicImage::from_decoder(decoder)
        .map_err(|e| MultimodalError::Decode(e.to_string()))?;
    // Orient the RGB8 buffer rather than the decoder-native one: rotations copy
    // the whole image, and doing it after the RGB8 conversion keeps that copy
    // inside `max_rgb8_bytes`. Peak allocation is therefore bounded by
    // `max_decode_bytes` (decoded buffer) + `max_rgb8_bytes` (converted and
    // rotated buffers), never the 2x decoded buffer the native path would need.
    let mut oriented = image::DynamicImage::ImageRgb8(img.into_rgb8());
    oriented.apply_orientation(orientation);
    let rgb = oriented.into_rgb8();
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
    preprocess_data_url_limited(url, cfg, usize::MAX)
}

/// Like preprocess_data_url but applies the patch budget before the
/// patch buffer is allocated.
pub fn preprocess_data_url_limited(
    url: &str,
    cfg: &ImageProcessorConfig,
    max_patches: usize,
) -> Result<ProcessedImage, MultimodalError> {
    preprocess_data_url_capped(url, cfg, max_patches, false)
}

/// Opt-in variant (`MACH_VISION_DOWNSCALE=1`): an image over the patch budget
/// is downscaled into it instead of being rejected.
pub fn preprocess_data_url_limited_downscaling(
    url: &str,
    cfg: &ImageProcessorConfig,
    max_patches: usize,
) -> Result<ProcessedImage, MultimodalError> {
    preprocess_data_url_capped(url, cfg, max_patches, true)
}

fn preprocess_data_url_capped(
    url: &str,
    cfg: &ImageProcessorConfig,
    max_patches: usize,
    downscale: bool,
) -> Result<ProcessedImage, MultimodalError> {
    let image = decode_data_url(url)?;
    let processed = if downscale {
        preprocess_image_limited_downscaling(
            cfg,
            &image.rgb8,
            image.height,
            image.width,
            max_patches,
        )
    } else {
        preprocess_image_limited(cfg, &image.rgb8, image.height, image.width, max_patches)
    };
    Ok(processed?)
}

/// Fetch an `http(s)://` image (bounded, public addresses only) or decode a
/// data URL, then run the C3d image processor.
pub async fn fetch_image_url(
    url: &str,
    cfg: &ImageProcessorConfig,
) -> Result<ProcessedImage, MultimodalError> {
    fetch_image_url_limited(url, cfg, usize::MAX).await
}

/// Like [`fetch_image_url`] but rejects images above `max_patches` before
/// allocating the patch buffer.
pub async fn fetch_image_url_limited(
    url: &str,
    cfg: &ImageProcessorConfig,
    max_patches: usize,
) -> Result<ProcessedImage, MultimodalError> {
    fetch_image_url_capped(url, cfg, max_patches, false).await
}

/// Opt-in variant (`MACH_VISION_DOWNSCALE=1`): an image over the patch budget
/// is downscaled into it instead of being rejected.
pub async fn fetch_image_url_limited_downscaling(
    url: &str,
    cfg: &ImageProcessorConfig,
    max_patches: usize,
) -> Result<ProcessedImage, MultimodalError> {
    fetch_image_url_capped(url, cfg, max_patches, true).await
}

async fn fetch_image_url_capped(
    url: &str,
    cfg: &ImageProcessorConfig,
    max_patches: usize,
    downscale: bool,
) -> Result<ProcessedImage, MultimodalError> {
    tokio::time::timeout(
        FETCH_TIMEOUT,
        fetch_impl_limited(url, cfg, false, max_patches, downscale),
    )
    .await
    .map_err(|_| MultimodalError::Fetch("image fetch timed out".into()))?
}

/// Max redirect hops followed manually (each hop is re-validated).
const MAX_REDIRECTS: usize = 5;

#[cfg(test)]
async fn fetch_impl(
    url: &str,
    cfg: &ImageProcessorConfig,
    allow_private: bool,
) -> Result<ProcessedImage, MultimodalError> {
    fetch_impl_limited(url, cfg, allow_private, usize::MAX, false).await
}

/// Concurrency cap for CPU-bound image jobs. `spawn_blocking` grows its pool to
/// hundreds of threads on demand, and one decode can peak at
/// `MAX_DECODE_BYTES + MAX_RGB8_BYTES`, so the pool must not be allowed to run
/// many of them at once. The engine is single-threaded anyway, so a small cap
/// only serialises work that would queue later regardless.
const IMAGE_JOB_PERMITS: usize = 2;

/// Take a slot for one CPU-bound image job. Call this *before* materialising the
/// input buffer and hand the permit to [`run_image_job`], which keeps it inside
/// the blocking closure: a blocking job cannot be cancelled, so a permit owned
/// by the caller's future would be released by a timeout while the job (and its
/// memory) keeps running, silently exceeding the cap.
async fn acquire_image_job() -> Result<tokio::sync::OwnedSemaphorePermit, MultimodalError> {
    static JOBS: OnceLock<std::sync::Arc<tokio::sync::Semaphore>> = OnceLock::new();
    let jobs =
        JOBS.get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(IMAGE_JOB_PERMITS)));
    jobs.clone()
        .acquire_owned()
        .await
        .map_err(|e| MultimodalError::Decode(format!("image job queue closed: {e}")))
}

/// Run a CPU-bound decode/preprocess job on the blocking pool. Base64 decoding,
/// PNG/JPEG decoding and the 22-bit bicubic resize can take hundreds of
/// milliseconds on a large image, which must not stall a tokio worker thread.
async fn run_image_job<T, F>(
    permit: tokio::sync::OwnedSemaphorePermit,
    job: F,
) -> Result<T, MultimodalError>
where
    F: FnOnce() -> Result<T, MultimodalError> + Send + 'static,
    T: Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        job()
    })
    .await
    .map_err(|e| MultimodalError::Decode(format!("image processing task failed: {e}")))?
}

async fn fetch_impl_limited(
    url: &str,
    cfg: &ImageProcessorConfig,
    allow_private: bool,
    max_patches: usize,
    downscale: bool,
) -> Result<ProcessedImage, MultimodalError> {
    if url.starts_with("data:") {
        let permit = acquire_image_job().await?;
        let url = url.to_owned();
        let cfg = cfg.clone();
        return run_image_job(permit, move || {
            if downscale {
                preprocess_data_url_limited_downscaling(&url, &cfg, max_patches)
            } else {
                preprocess_data_url_limited(&url, &cfg, max_patches)
            }
        })
        .await;
    }
    let mut current = parse_http_url(url)?;
    let mut hops = 0usize;
    loop {
        let addr = resolve_public(&current, allow_private).await?;
        let client = shared_fetch_client(&current, addr)?;
        let resp = client
            .get(current.clone())
            .send()
            .await
            .map_err(|e| MultimodalError::Fetch(e.to_string()))?;
        if resp.status().is_redirection() {
            if hops >= MAX_REDIRECTS {
                return Err(MultimodalError::Fetch(format!(
                    "too many redirects (> {MAX_REDIRECTS})"
                )));
            }
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| MultimodalError::Fetch("redirect without Location".into()))?;
            let next = current
                .join(location)
                .map_err(|e| MultimodalError::Fetch(format!("bad redirect target: {e}")))?;
            if current.scheme() == "https" && next.scheme() != "https" {
                return Err(MultimodalError::Fetch(
                    "refusing HTTPS to HTTP redirect".into(),
                ));
            }
            current = parse_http_url(next.as_str())?;
            hops += 1;
            continue;
        }
        if !resp.status().is_success() {
            return Err(MultimodalError::Fetch(format!("HTTP {}", resp.status())));
        }
        if let Some(len) = resp.content_length() {
            let len = usize::try_from(len).map_err(|_| MultimodalError::TooLarge(usize::MAX))?;
            if len > MAX_FETCH_BYTES {
                return Err(MultimodalError::TooLarge(len));
            }
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
        // Hold the slot across the body read so a burst of requests cannot pile
        // up 64 MiB input buffers waiting for a decode slot.
        let permit = acquire_image_job().await?;
        let bytes = read_body_limited(resp, MAX_FETCH_BYTES).await?;
        let cfg = cfg.clone();
        return run_image_job(permit, move || {
            let image = decode_rgb8(&bytes)?;
            let processed = if downscale {
                preprocess_image_limited_downscaling(
                    &cfg,
                    &image.rgb8,
                    image.height,
                    image.width,
                    max_patches,
                )
            } else {
                preprocess_image_limited(&cfg, &image.rgb8, image.height, image.width, max_patches)
            };
            Ok(processed?)
        })
        .await;
    }
}

fn parse_http_url(url: &str) -> Result<reqwest::Url, MultimodalError> {
    let parsed = reqwest::Url::parse(url).map_err(|_| MultimodalError::UnsupportedUrl)?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err(MultimodalError::UnsupportedUrl);
    }
    Ok(parsed)
}

async fn resolve_public(
    url: &reqwest::Url,
    allow_private: bool,
) -> Result<std::net::SocketAddr, MultimodalError> {
    let host = url.host_str().ok_or(MultimodalError::UnsupportedUrl)?;
    let port = url
        .port_or_known_default()
        .ok_or(MultimodalError::UnsupportedUrl)?;
    let addrs: Vec<_> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| MultimodalError::Fetch(e.to_string()))?
        .collect();
    let addr = addrs
        .first()
        .copied()
        .ok_or_else(|| MultimodalError::Fetch("image host resolved to no addresses".into()))?;
    if !allow_private {
        for candidate in &addrs {
            if !is_public_ip(candidate.ip()) {
                return Err(MultimodalError::Fetch(format!(
                    "image URL resolves to non-public address {}",
                    candidate.ip()
                )));
            }
        }
    }
    Ok(addr)
}

/// Cached HTTP clients keyed by scheme + host + port + pinned address.
///
/// `reqwest::Client` owns the connection pool, and this path used to build a
/// fresh client for every redirect hop, so each image paid for a new TCP/TLS
/// handshake and nothing was reused across requests. The key includes the
/// resolved address, so a DNS change (or an SSRF rebinding attempt) can never
/// reuse a client pinned to a different address. Bounded because the key space
/// is attacker-controlled.
const CLIENT_CACHE_CAP: usize = 64;

type ClientKey = (String, String, u16, std::net::SocketAddr);

#[derive(Default)]
struct ClientCache {
    entries: HashMap<ClientKey, (reqwest::Client, u64)>,
    tick: u64,
}

static FETCH_CLIENTS: OnceLock<Mutex<ClientCache>> = OnceLock::new();

#[cfg(test)]
fn cached_client_count() -> usize {
    FETCH_CLIENTS
        .get()
        .map(|cache| {
            cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entries
                .len()
        })
        .unwrap_or(0)
}

fn shared_fetch_client(
    url: &reqwest::Url,
    addr: std::net::SocketAddr,
) -> Result<reqwest::Client, MultimodalError> {
    let clients = FETCH_CLIENTS.get_or_init(|| Mutex::new(ClientCache::default()));
    let key = (
        url.scheme().to_owned(),
        url.host_str().unwrap_or_default().to_owned(),
        url.port_or_known_default().unwrap_or(0),
        addr,
    );
    {
        let mut cache = clients.lock().unwrap_or_else(|e| e.into_inner());
        cache.tick += 1;
        let tick = cache.tick;
        if let Some((client, used)) = cache.entries.get_mut(&key) {
            *used = tick;
            return Ok(client.clone());
        }
    }
    // Build outside the lock; a racing thread may build the same key twice,
    // which only wastes one client.
    let client = build_fetch_client(url, addr)?;
    let mut cache = clients.lock().unwrap_or_else(|e| e.into_inner());
    cache.tick += 1;
    let tick = cache.tick;
    if cache.entries.len() >= CLIENT_CACHE_CAP
        && let Some(oldest) = cache
            .entries
            .iter()
            .min_by_key(|(_, (_, used))| *used)
            .map(|(key, _)| key.clone())
    {
        cache.entries.remove(&oldest);
    }
    cache.entries.insert(key, (client.clone(), tick));
    Ok(client)
}

fn build_fetch_client(
    url: &reqwest::Url,
    addr: std::net::SocketAddr,
) -> Result<reqwest::Client, MultimodalError> {
    let mut builder = reqwest::Client::builder()
        .timeout(FETCH_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        // Cached clients keep their pool alive; without a cap a burst of
        // requests to one origin would leave that many idle sockets open.
        .pool_max_idle_per_host(2)
        .pool_idle_timeout(std::time::Duration::from_secs(30));
    if let Some(host) = url.host_str()
        && host.parse::<std::net::IpAddr>().is_err()
    {
        builder = builder.resolve(host, addr);
    }
    builder = builder.no_proxy();
    builder
        .build()
        .map_err(|e| MultimodalError::Fetch(e.to_string()))
}

async fn read_body_limited(
    mut resp: reqwest::Response,
    limit: usize,
) -> Result<Vec<u8>, MultimodalError> {
    let mut bytes = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| MultimodalError::Fetch(e.to_string()))?
    {
        let total = bytes
            .len()
            .checked_add(chunk.len())
            .ok_or(MultimodalError::TooLarge(usize::MAX))?;
        if total > limit {
            return Err(MultimodalError::TooLarge(total));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// SSRF guard: reject every non-publicly-routable IPv4/IPv6 range we can
/// name, including private/loopback/link-local/CGNAT/documentation/reserved
/// blocks and IPv6 ULA/site-local/NAT64/6to4/Teredo/ORCHID/discard prefixes.
#[must_use]
pub fn is_public_ip(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            !(v4.is_private()
                || v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_multicast()
                || v4.is_broadcast()
                || v4.is_documentation()
                || o[0] == 0
                || (o[0] == 192 && o[1] == 0 && o[2] == 0)
                || (o[0] == 192 && o[1] == 88 && o[2] == 99)
                || (o[0] == 198 && (o[1] & 0xfe) == 18)
                || (o[0] & 0xf0) == 240
                || (o[0] == 100 && (o[1] & 0xc0) == 64))
        }
        IpAddr::V6(v6) => {
            let seg = v6.segments();
            let is_ula = (seg[0] & 0xfe00) == 0xfc00;
            let is_link_local = (seg[0] & 0xffc0) == 0xfe80;
            let is_site_local = (seg[0] & 0xffc0) == 0xfec0;
            let is_doc = (seg[0] == 0x2001 && seg[1] == 0x0db8) || (seg[0] & 0xfff0) == 0x3ff0;
            let is_nat64 =
                seg[0] == 0x0064 && seg[1] == 0xff9b && (seg[2] == 0x0000 || seg[2] == 0x0001);
            let is_6to4 = seg[0] == 0x2002;
            let is_teredo = seg[0] == 0x2001 && seg[1] == 0x0000;
            let is_benchmark = seg[0] == 0x2001 && seg[1] == 0x0002;
            let is_orchid = seg[0] == 0x2001 && (seg[1] & 0xfff0) == 0x0010;
            let is_orchid_v2 = seg[0] == 0x2001 && (seg[1] & 0xfff0) == 0x0020;
            let is_discard = seg[0] == 0x0100 && seg[1] == 0 && seg[2] == 0 && seg[3] == 0;
            let embedded_public = v6
                .to_ipv4_mapped()
                .is_none_or(|v4| is_public_ip(IpAddr::V4(v4)));
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || is_ula
                || is_link_local
                || is_site_local
                || is_doc
                || is_nat64
                || is_6to4
                || is_teredo
                || is_benchmark
                || is_orchid
                || is_orchid_v2
                || is_discard
                || !embedded_public)
        }
    }
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
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    fn jpeg_bytes(width: u32, height: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x * 3) as u8, (y * 5) as u8, 7])
        });
        let mut bytes = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Jpeg,
        )
        .unwrap();
        bytes
    }

    #[test]
    fn rejects_decode_bomb_from_metadata_before_decoding() {
        // The JPEG decoder only maps max_image_width/height and ignores
        // max_alloc, so without the header-level reserve guard this 12 KiB
        // image would be fully decoded (and allocated) despite the 1 KiB budget.
        let bytes = jpeg_bytes(64, 64);
        let err = decode_rgb8_with_limits(&bytes, 1024, MAX_RGB8_BYTES).unwrap_err();
        assert!(matches!(err, MultimodalError::Decode(_)), "{err}");
    }

    #[test]
    fn rejects_oversized_rgb8_from_header_before_decoding() {
        // Build a 64x64 PNG and cut its IDAT short: the header still reports
        // 12288 RGB8 bytes but the pixels cannot be decoded. A permissive RGB8
        // budget therefore surfaces the decode failure, while a tight one must
        // fail from the header - proving the size check runs before any decode
        // (and before `from_decoder` allocates the pixel buffer).
        let img = image::RgbImage::from_pixel(64, 64, image::Rgb([120, 130, 140]));
        let mut bytes = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
        bytes.truncate(60);

        let Err(decode_err) = decode_rgb8_with_limits(&bytes, MAX_DECODE_BYTES, 64 * 64 * 3) else {
            panic!("truncated PNG must not decode");
        };
        assert!(
            matches!(decode_err, MultimodalError::Decode(_)),
            "{decode_err}"
        );
        let err = decode_rgb8_with_limits(&bytes, MAX_DECODE_BYTES, 64 * 64 * 3 - 1).unwrap_err();
        assert!(matches!(err, MultimodalError::TooLarge(12288)), "{err}");
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

    /// Tests that poke the process-wide client cache must not interleave.
    fn cache_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn client_cache_is_bounded() {
        let _guard = cache_test_lock();
        for i in 0..CLIENT_CACHE_CAP + 8 {
            let url = reqwest::Url::parse(&format!("http://cache-{i}.example/img")).unwrap();
            let addr = format!("127.0.0.1:{}", 20_000 + i).parse().unwrap();
            shared_fetch_client(&url, addr).unwrap();
        }
        let len = cached_client_count();
        assert!(len <= CLIENT_CACHE_CAP, "client cache grew to {len}");
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

    fn test_client() -> reqwest::Client {
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
        let out = fetch_impl(&png_data_url(2, 3), &small_cfg(), true)
            .await
            .unwrap();
        assert_eq!(out.grid, [1, 3, 2]);
        assert_eq!(out.pixel_values.len(), 3 * 2 * 3);
    }

    #[tokio::test]
    async fn fetches_http_png_image() {
        let url = spawn_raw(http_response("200 OK", "image/png", &png_bytes(3, 2))).await;
        let out = fetch_impl(&url, &small_cfg(), true).await.unwrap();
        assert_eq!(out.grid, [1, 2, 3]);
        assert_eq!(out.pixel_values.len(), 3 * 2 * 3);
    }

    #[tokio::test]
    async fn rejects_http_error_status() {
        let url = spawn_raw(http_response("404 Not Found", "text/plain", b"nope")).await;
        let err = fetch_impl(&url, &small_cfg(), true).await.unwrap_err();
        assert!(matches!(err, MultimodalError::Fetch(_)), "{err}");
    }

    #[tokio::test]
    async fn rejects_non_image_content_type() {
        let url = spawn_raw(http_response("200 OK", "text/html", b"<html/>")).await;
        let err = fetch_impl(&url, &small_cfg(), true).await.unwrap_err();
        assert!(matches!(err, MultimodalError::UnsupportedMedia(_)), "{err}");
    }

    #[tokio::test]
    async fn rejects_oversized_http_content_length() {
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            MAX_FETCH_BYTES + 1
        );
        let url = spawn_raw(head.into_bytes()).await;
        let err = fetch_impl(&url, &small_cfg(), true).await.unwrap_err();
        assert!(matches!(err, MultimodalError::TooLarge(_)), "{err}");
    }

    /// A gate parked image jobs wait on. Opening it (explicitly or on drop)
    /// releases them so a failing assertion cannot hang runtime shutdown.
    struct JobGate(Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>);

    impl JobGate {
        fn new() -> Self {
            Self(Arc::new((
                std::sync::Mutex::new(false),
                std::sync::Condvar::new(),
            )))
        }

        fn share(&self) -> Arc<(std::sync::Mutex<bool>, std::sync::Condvar)> {
            self.0.clone()
        }

        fn open(&self) {
            let (lock, wake) = &*self.0;
            *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
            wake.notify_all();
        }
    }

    impl Drop for JobGate {
        fn drop(&mut self) {
            self.open();
        }
    }

    async fn wait_for(ms: u64, mut ready: impl FnMut() -> bool) {
        tokio::time::timeout(std::time::Duration::from_millis(ms), async {
            while !ready() {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("condition not reached in time");
    }

    /// Park `IMAGE_JOB_PERMITS` production jobs on a gate, filling the
    /// process-wide semaphore, and return the caller tasks plus the job counter.
    fn park_image_jobs(
        gate: &JobGate,
        running: &Arc<AtomicUsize>,
    ) -> Vec<tokio::task::JoinHandle<Result<(), MultimodalError>>> {
        let mut callers = Vec::new();
        for _ in 0..IMAGE_JOB_PERMITS {
            let gate = gate.share();
            let running = running.clone();
            callers.push(tokio::spawn(async move {
                let permit = acquire_image_job().await.unwrap();
                run_image_job(permit, move || {
                    running.fetch_add(1, Ordering::SeqCst);
                    let (lock, wake) = &*gate;
                    let mut open = lock.lock().unwrap();
                    while !*open {
                        open = wake.wait(open).unwrap();
                    }
                    running.fetch_sub(1, Ordering::SeqCst);
                    Ok(())
                })
                .await
            }));
        }
        callers
    }

    #[tokio::test]
    async fn run_image_job_enforces_permit_cap() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        // A *local* semaphore keeps this deterministic: the process-wide one is
        // shared with other tests, whose short jobs would otherwise keep the
        // observed peak below the cap.
        const EXPECTED_CAP: usize = 2;
        assert_eq!(IMAGE_JOB_PERMITS, EXPECTED_CAP);
        let jobs = Arc::new(tokio::sync::Semaphore::new(EXPECTED_CAP));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for _ in 0..EXPECTED_CAP * 3 {
            let in_flight = in_flight.clone();
            let peak = peak.clone();
            let jobs = jobs.clone();
            tasks.push(tokio::spawn(async move {
                let permit = jobs.acquire_owned().await.unwrap();
                run_image_job(permit, move || {
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    Ok(())
                })
                .await
                .unwrap();
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(
            peak.load(Ordering::SeqCst),
            EXPECTED_CAP,
            "jobs must overlap but never exceed the cap"
        );
    }

    /// Serialises the tests that park the process-wide image-job semaphore;
    /// without it two such tests can each hold one permit and wait forever.
    fn parked_job_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn production_image_jobs_are_capped() {
        let _guard = parked_job_test_lock();
        // A current-thread runtime keeps the third task and `wait_for` on the
        // same thread, so observing `attempted` proves the acquire was polled.
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                use std::sync::atomic::{AtomicBool, Ordering};
                let gate = JobGate::new();
                let running = Arc::new(AtomicUsize::new(0));
                let callers = park_image_jobs(&gate, &running);
                wait_for(5_000, || {
                    running.load(Ordering::SeqCst) == IMAGE_JOB_PERMITS
                })
                .await;

                let attempted = Arc::new(AtomicBool::new(false));
                let acquired = Arc::new(AtomicBool::new(false));
                let third = {
                    let attempted = attempted.clone();
                    let acquired = acquired.clone();
                    tokio::spawn(async move {
                        attempted.store(true, Ordering::SeqCst);
                        let permit = acquire_image_job().await.unwrap();
                        acquired.store(true, Ordering::SeqCst);
                        run_image_job(permit, || Ok(())).await.unwrap();
                    })
                };
                // Once the third task has been polled, an available permit would
                // already have been granted in that same poll.
                wait_for(5_000, || attempted.load(Ordering::SeqCst)).await;
                assert!(
                    !acquired.load(Ordering::SeqCst),
                    "a third job acquired a permit while {IMAGE_JOB_PERMITS} were parked"
                );
                assert!(!third.is_finished());
                gate.open();
                for caller in callers {
                    let _ = caller.await;
                }
                third.await.unwrap();
                assert!(acquired.load(Ordering::SeqCst));
            });
    }

    #[test]
    fn cancelled_jobs_keep_their_permit() {
        let _guard = parked_job_test_lock();
        // A current-thread runtime keeps the third task and `wait_for` on the
        // same thread, so observing `attempted` proves the acquire was polled.
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                use std::sync::atomic::{AtomicBool, Ordering};
                // A cancelled `spawn_blocking` job keeps running, so its permit must
                // be owned by the job, not by the dropped caller future.
                let gate = JobGate::new();
                let running = Arc::new(AtomicUsize::new(0));
                let callers = park_image_jobs(&gate, &running);
                wait_for(5_000, || {
                    running.load(Ordering::SeqCst) == IMAGE_JOB_PERMITS
                })
                .await;

                for caller in &callers {
                    caller.abort();
                }
                // Await the aborted tasks so their futures are really dropped.
                for caller in callers {
                    let _ = caller.await;
                }

                let attempted = Arc::new(AtomicBool::new(false));
                let acquired = Arc::new(AtomicBool::new(false));
                let third = {
                    let attempted = attempted.clone();
                    let acquired = acquired.clone();
                    tokio::spawn(async move {
                        attempted.store(true, Ordering::SeqCst);
                        let permit = acquire_image_job().await.unwrap();
                        acquired.store(true, Ordering::SeqCst);
                        run_image_job(permit, || Ok(())).await.unwrap();
                    })
                };
                wait_for(5_000, || attempted.load(Ordering::SeqCst)).await;
                assert!(
                    !acquired.load(Ordering::SeqCst),
                    "a cancelled job released its permit while still running"
                );
                assert!(!third.is_finished());
                gate.open();
                third.await.unwrap();
                assert!(acquired.load(Ordering::SeqCst));
            });
    }
    #[tokio::test]
    async fn decode_jobs_run_off_the_async_worker() {
        let worker = std::thread::current().id();
        let permit = acquire_image_job().await.unwrap();
        let ran_on = run_image_job(permit, move || Ok(std::thread::current().id()))
            .await
            .unwrap();
        assert_ne!(
            ran_on, worker,
            "decode/preprocess must run on the blocking pool, not a tokio worker"
        );
    }

    /// HTTP server that keeps connections alive and counts accepted sockets, so
    /// a test can observe client-side connection reuse.
    async fn spawn_keepalive_server(
        body: Vec<u8>,
        connections: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    ) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                connections.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let body = body.clone();
                tokio::spawn(async move {
                    let mut pending = Vec::new();
                    let mut chunk = [0u8; 1024];
                    loop {
                        match socket.read(&mut chunk).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => pending.extend_from_slice(&chunk[..n]),
                        }
                        while let Some(end) = pending.windows(4).position(|w| w == b"\r\n\r\n") {
                            pending.drain(..end + 4);
                            let head = format!(
                                "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\n\r\n",
                                body.len()
                            );
                            if socket.write_all(head.as_bytes()).await.is_err()
                                || socket.write_all(&body).await.is_err()
                            {
                                return;
                            }
                        }
                    }
                });
            }
        });
        format!("http://{addr}/image")
    }

    #[test]
    fn reuses_http_connections_across_fetches() {
        // A manual runtime so the cache lock is held across `block_on` rather
        // than across `await` points (clippy::await_holding_lock).
        let _guard = cache_test_lock();
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                use std::sync::Arc;
                use std::sync::atomic::{AtomicUsize, Ordering};
                let connections = Arc::new(AtomicUsize::new(0));
                let url = spawn_keepalive_server(png_bytes(3, 2), connections.clone()).await;
                for _ in 0..3 {
                    let out = fetch_impl(&url, &small_cfg(), true).await.unwrap();
                    assert_eq!(out.grid, [1, 2, 3]);
                }
                assert_eq!(
                    connections.load(Ordering::SeqCst),
                    1,
                    "fetches to the same origin must reuse the pooled connection"
                );
            });
    }

    #[tokio::test]
    async fn rejects_unsupported_scheme() {
        let err = fetch_impl("ftp://example.com/a.png", &small_cfg(), true)
            .await
            .unwrap_err();
        assert!(matches!(err, MultimodalError::UnsupportedUrl), "{err}");
    }

    #[test]
    fn rejects_private_and_special_ips() {
        use std::net::IpAddr;
        for ip in [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "255.255.255.255",
            "::1",
            "fc00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "192.0.0.1",
            "198.18.0.1",
            "240.0.0.1",
            "64:ff9b::7f00:1",
            "2002:7f00:1::",
            "2001:0:0:0:0:0:0:1",
            "2001:2::1",
            "fec0::1",
            "64:ff9b:1:7f00:1::",
            "100::1",
            "2001:10::1",
            "2001:20::1",
            "3fff::1",
            "192.88.99.1",
        ] {
            let ip: IpAddr = ip.parse().unwrap();
            assert!(!is_public_ip(ip), "{ip} must be rejected");
        }
        for ip in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
            let ip: IpAddr = ip.parse().unwrap();
            assert!(is_public_ip(ip), "{ip} must be allowed");
        }
    }

    #[tokio::test]
    async fn rejects_private_host_without_connecting() {
        let err = fetch_impl("http://127.0.0.1:9/image", &small_cfg(), false)
            .await
            .unwrap_err();
        assert!(matches!(err, MultimodalError::Fetch(_)), "{err}");
        assert!(err.to_string().contains("non-public"), "{err}");
    }

    async fn spawn_redirect_server(hops: usize) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            for i in 0..=hops {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let mut buf = [0u8; 4096];
                let _ = socket.read(&mut buf).await;
                if i < hops {
                    let response = format!(
                        "HTTP/1.1 302 Found\r\nLocation: /hop{}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                        i + 1
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                } else {
                    let body = png_bytes(3, 2);
                    let mut response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .into_bytes();
                    response.extend_from_slice(&body);
                    let _ = socket.write_all(&response).await;
                }
            }
        });
        format!("http://{addr}/start")
    }

    #[tokio::test]
    async fn follows_five_redirects_and_rejects_sixth() {
        let ok_url = spawn_redirect_server(5).await;
        let out = fetch_impl(&ok_url, &small_cfg(), true).await.unwrap();
        assert_eq!(out.grid, [1, 2, 3]);

        let too_many = spawn_redirect_server(6).await;
        let err = fetch_impl(&too_many, &small_cfg(), true).await.unwrap_err();
        assert!(err.to_string().contains("too many redirects"), "{err}");
    }

    async fn spawn_chunked_server() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            if let Ok((mut socket, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = socket.read(&mut buf).await;
                let payload = [7u8; 16];
                let mut response = b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n10\r\n".to_vec();
                response.extend_from_slice(&payload);
                response.extend_from_slice(b"\r\n0\r\n\r\n");
                let _ = socket.write_all(&response).await;
            }
        });
        format!("http://{addr}/image")
    }

    #[tokio::test]
    async fn read_body_limited_rejects_chunked_overflow() {
        let url = spawn_chunked_server().await;
        let resp = test_client().get(&url).send().await.unwrap();
        let err = read_body_limited(resp, 8).await.unwrap_err();
        assert!(matches!(err, MultimodalError::TooLarge(_)), "{err}");
    }

    #[tokio::test]
    async fn accepts_content_type_with_parameters() {
        let url = spawn_raw(http_response(
            "200 OK",
            "image/png; charset=binary",
            &png_bytes(3, 2),
        ))
        .await;
        let out = fetch_impl(&url, &small_cfg(), true).await.unwrap();
        assert_eq!(out.grid, [1, 2, 3]);
    }

    #[tokio::test]
    async fn fetch_limit_rejects_oversized_grid() {
        let url = spawn_raw(http_response("200 OK", "image/png", &png_bytes(4, 4))).await;
        let err = fetch_impl_limited(&url, &small_cfg(), true, 8, false)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("exceeds limit"), "{err}");
    }

    #[test]
    fn preprocess_data_url_limit_rejects_oversized_grid() {
        let err = preprocess_data_url_limited(&png_data_url(4, 4), &small_cfg(), 8)
            .unwrap_err()
            .to_string();
        assert!(err.contains("exceeds limit"), "{err}");
    }

    #[tokio::test]
    async fn fetch_data_url_limit_rejects_oversized_grid() {
        let err = fetch_image_url_limited(&png_data_url(4, 4), &small_cfg(), 8)
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("exceeds limit"), "{err}");
    }

    #[test]
    fn preprocess_data_url_downscaling_fits_oversized_grid() {
        let out =
            preprocess_data_url_limited_downscaling(&png_data_url(8, 8), &small_cfg(), 16).unwrap();
        assert_eq!(out.grid, [1, 4, 4]);
        assert_eq!(out.pixel_values.len(), 4 * 4 * 3);
    }

    #[tokio::test]
    async fn fetch_data_url_downscaling_fits_oversized_grid() {
        let out = fetch_image_url_limited_downscaling(&png_data_url(8, 8), &small_cfg(), 16)
            .await
            .unwrap();
        assert_eq!(out.grid, [1, 4, 4]);
    }

    #[tokio::test]
    async fn fetch_http_downscaling_fits_oversized_grid() {
        let url = spawn_raw(http_response("200 OK", "image/png", &png_bytes(8, 8))).await;
        let out = fetch_impl_limited(&url, &small_cfg(), true, 16, true)
            .await
            .unwrap();
        assert_eq!(out.grid, [1, 4, 4]);
    }
}
