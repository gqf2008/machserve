//! HTTP routes: /v1/completions, /v1/chat/completions, /healthz.
//!
//! Prompts are accepted as raw token ids (`prompt_tokens`) or text. Text is
//! encoded with the real byte-level BPE tokenizer when one is configured
//! (falls back to a naive byte-per-token mapping otherwise). Both endpoints
//! support OpenAI-shaped `stream: true` -> SSE with per-token deltas.

use crate::engine::ServerEngine;
use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use mach_model::sampling::SamplingParams;
use mach_model::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::ReceiverStream;

/// Shared router state.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<ServerEngine>,
    pub model: String,
    /// Real tokenizer when `tokenizer.json` is available (else naive bytes).
    pub tok: Option<Arc<Tokenizer>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    /// Text, encoded with the configured tokenizer.
    Text(String),
    /// Raw token ids.
    Tokens(Vec<u32>),
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub prompt: Prompt,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Temperature; omitted or 0 means greedy. OpenAI-shaped.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Top-p (nucleus); omitted means disabled (1.0).
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k; omitted means disabled. Nonstandard extension field.
    #[serde(default)]
    pub top_k: Option<usize>,
    /// RNG seed for deterministic sampling (OpenAI-shaped).
    #[serde(default)]
    pub seed: Option<u64>,
    /// Stream tokens over SSE as they are generated.
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub tokens: Vec<u32>,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
}

fn default_max_tokens() -> usize {
    32
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Naive fallback: token id -> byte (lossy UTF-8, no tokenizer configured).
/// Kept consistent with the streaming path (byte-level + `from_utf8_lossy`).
fn naive_decode(tokens: &[u32]) -> String {
    let bytes: Vec<u8> = tokens
        .iter()
        .map(|&t| if t < 256 { t as u8 } else { b'?' })
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Naive ASCII fallback: byte -> token id.
fn naive_encode(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

fn prompt_tokens(tok: &Option<Arc<Tokenizer>>, prompt: &Prompt) -> Vec<u32> {
    match prompt {
        Prompt::Text(s) => match tok {
            Some(t) => t.encode(s),
            None => naive_encode(s),
        },
        Prompt::Tokens(t) => t.clone(),
    }
}

fn decode_text(tok: &Option<Arc<Tokenizer>>, tokens: &[u32]) -> String {
    match tok {
        Some(t) => t.decode(tokens),
        None => naive_decode(tokens),
    }
}

/// Builds sampling params from optional request fields (greedy by default).
fn sampling_params(
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<u64>,
) -> SamplingParams {
    SamplingParams {
        temperature: temperature.unwrap_or(0.0),
        top_k: top_k.unwrap_or(0),
        top_p: top_p.unwrap_or(1.0),
        seed: seed.unwrap_or(0),
    }
}

fn sse_chunk(
    id: &str,
    object: &str,
    model: &str,
    created: u64,
    delta: &str,
    finish: Option<&str>,
) -> String {
    use serde_json::json;
    let finish_reason = finish.map(|f| json!(f)).unwrap_or(json!(null));
    let ev = if object == "chat.completion.chunk" {
        json!({
            "id": id,
            "object": object,
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": finish_reason}],
        })
    } else {
        json!({
            "id": id,
            "object": object,
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": delta, "finish_reason": finish_reason}],
        })
    };
    format!("data: {ev}\n\n")
}

/// Drains `acc`, returning the longest valid UTF-8 prefix (keeps a partial
/// multi-byte tail for the next token).
fn emit_valid_prefix(acc: &mut Vec<u8>) -> String {
    match std::str::from_utf8(acc) {
        Ok(s) => {
            let s = s.to_string();
            acc.clear();
            s
        }
        Err(e) => {
            let up = e.valid_up_to();
            let valid = String::from_utf8_lossy(&acc[..up]).into_owned();
            acc.drain(..up);
            valid
        }
    }
}

/// Reads per-token ids, decodes incrementally and pushes SSE events.
async fn stream_tokens(
    state: AppState,
    id: String,
    object: &'static str,
    created: u64,
    mut rx: tokio::sync::mpsc::Receiver<u32>,
    tx: tokio::sync::mpsc::Sender<Result<Bytes, Infallible>>,
) {
    let mut acc: Vec<u8> = Vec::new();
    while let Some(tok) = rx.recv().await {
        let bytes = match &state.tok {
            Some(t) => t.decode_bytes(&[tok]),
            None => vec![if tok < 256 { tok as u8 } else { b'?' }],
        };
        acc.extend_from_slice(&bytes);
        let text = emit_valid_prefix(&mut acc);
        if !text.is_empty() {
            let ev = sse_chunk(&id, object, &state.model, created, &text, None);
            if tx.send(Ok(Bytes::from(ev))).await.is_err() {
                return; // client disconnected
            }
        }
    }
    let tail = String::from_utf8_lossy(&acc).into_owned();
    if !tail.is_empty() {
        let ev = sse_chunk(&id, object, &state.model, created, &tail, None);
        let _ = tx.send(Ok(Bytes::from(ev))).await;
    }
    let ev = sse_chunk(&id, object, &state.model, created, "", Some("stop"));
    let _ = tx.send(Ok(Bytes::from(ev))).await;
    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
}

fn sse_response(rx: tokio::sync::mpsc::Receiver<Result<Bytes, Infallible>>) -> Response {
    Response::builder()
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .expect("valid sse response")
}

/// POST /v1/completions
pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let tokens = prompt_tokens(&state.tok, &req.prompt);
    let params = sampling_params(req.temperature, req.top_k, req.top_p, req.seed);
    let id = format!("cmpl-{}", now());
    let created = now();

    if req.stream.unwrap_or(false) {
        let (rx_final, rx_tokens) = match state
            .engine
            .submit_stream(tokens, req.max_tokens, None, params)
            .await
        {
            Ok(x) => x,
            Err(_) => return axum::http::StatusCode::SERVICE_UNAVAILABLE.into_response(),
        };
        let st = state.clone();
        let id2 = id.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(16);
        tokio::spawn(async move {
            let _ = rx_final.await; // keep the request alive until completion
            stream_tokens(st, id2, "text_completion", created, rx_tokens, tx).await;
        });
        return sse_response(rx);
    }

    let output = match state
        .engine
        .submit(tokens, req.max_tokens, None, params)
        .await
    {
        Ok(o) => o,
        Err(_) => return axum::http::StatusCode::SERVICE_UNAVAILABLE.into_response(),
    };
    Json(CompletionResponse {
        id,
        object: "text_completion".into(),
        created,
        model: state.model.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            text: decode_text(&state.tok, &output),
            tokens: output,
            finish_reason: "length".into(),
        }],
    })
    .into_response()
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let text: String = req
        .messages
        .iter()
        .map(|m| format!("{}: {}\n", m.role, m.content))
        .collect();
    let tokens = match &state.tok {
        Some(t) => t.encode(&text),
        None => naive_encode(&text),
    };
    let params = sampling_params(req.temperature, req.top_k, req.top_p, req.seed);
    let id = format!("chatcmpl-{}", now());
    let created = now();

    if req.stream.unwrap_or(false) {
        let (rx_final, rx_tokens) = match state
            .engine
            .submit_stream(tokens, req.max_tokens, None, params)
            .await
        {
            Ok(x) => x,
            Err(_) => return axum::http::StatusCode::SERVICE_UNAVAILABLE.into_response(),
        };
        let st = state.clone();
        let id2 = id.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(16);
        tokio::spawn(async move {
            let _ = rx_final.await;
            stream_tokens(st, id2, "chat.completion.chunk", created, rx_tokens, tx).await;
        });
        return sse_response(rx);
    }

    let output = match state
        .engine
        .submit(tokens, req.max_tokens, None, params)
        .await
    {
        Ok(o) => o,
        Err(_) => return axum::http::StatusCode::SERVICE_UNAVAILABLE.into_response(),
    };
    Json(CompletionResponse {
        id,
        object: "chat.completion".into(),
        created,
        model: state.model.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            text: decode_text(&state.tok, &output),
            tokens: output,
            finish_reason: "length".into(),
        }],
    })
    .into_response()
}

/// GET /healthz
pub async fn healthz() -> &'static str {
    "ok"
}

/// Builds the axum router.
pub fn router(state: AppState) -> axum::Router {
    use axum::routing::{get, post};
    axum::Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}
