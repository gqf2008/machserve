//! HTTP routes: /v1/completions, /v1/chat/completions, /healthz.
//!
//! Prompts are accepted as raw token ids (`prompt_tokens`) or text. Text is
//! encoded with the real byte-level BPE tokenizer when one is configured
//! (falls back to a naive byte-per-token mapping otherwise). Both endpoints
//! support OpenAI-shaped `stream: true` -> SSE with per-token deltas.

use crate::engine::{EngineError, ServerEngine};
use axum::Json;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::StatusCode;
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

/// OpenAI `stop`: a single string or an array of strings.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
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
    /// Stop generation when the output ends with any of these strings.
    #[serde(default)]
    pub stop: Option<StopSpec>,
    /// Number of independent completions to generate (default 1).
    #[serde(default)]
    pub n: Option<usize>,
    /// Include per-token log-probabilities in the response.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Penalize tokens that have appeared (OpenAI presence_penalty).
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Penalize tokens by occurrence count (OpenAI frequency_penalty).
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Add bias to specific token logits (OpenAI logit_bias: {token_id: bias}).
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// Report the top-`n` tokens + log-probs per generated position (OpenAI
    /// `logprobs.top_logprobs`; requires `logprobs: true`).
    #[serde(default)]
    pub top_logprobs: Option<usize>,
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
    #[serde(default)]
    pub stop: Option<StopSpec>,
    #[serde(default)]
    pub n: Option<usize>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// Report the top-`n` tokens + log-probs per generated position (OpenAI
    /// `logprobs.top_logprobs`; requires `logprobs: true`).
    #[serde(default)]
    pub top_logprobs: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub tokens: Vec<u32>,
    /// OpenAI logprobs (present when requested).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
    pub finish_reason: String,
}

/// One reported alternative token (OpenAI `top_logprobs` entry).
#[derive(Debug, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
}

/// OpenAI `logprobs` payload (tokens + per-token log-probabilities).
#[derive(Debug, Serialize)]
pub struct Logprobs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    /// Per-position top-k alternatives (present when `top_logprobs` > 0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<Vec<TopLogprob>>>,
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

/// OpenAI-shaped error body: `{"error": {"message", "type", "code"}}`.
fn err_response(message: &str, err_type: &str, code: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": err_type,
            "code": code,
        }
    });
    (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
}

/// Maps an engine submission error to an OpenAI-shaped 503 response.
fn busy_response(e: EngineError) -> Response {
    match e {
        EngineError::Busy => err_response(
            "engine capacity reached; retry later",
            "server_error",
            "engine_busy",
        ),
        EngineError::ShuttingDown => err_response(
            "engine is shutting down; retry later",
            "server_error",
            "engine_shutting_down",
        ),
        EngineError::Model(m) => {
            err_response(&format!("model error: {m}"), "server_error", "model_error")
        }
    }
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

/// Converts an OpenAI logit_bias map ({token_id: bias}) into (token, bias) pairs.
fn logit_bias_pairs(m: &Option<std::collections::HashMap<String, f32>>) -> Vec<(u32, f32)> {
    let Some(map) = m else {
        return Vec::new();
    };
    map.iter()
        .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|t| (t, v)))
        .collect()
}

/// Encodes OpenAI `stop` strings into token sequences.
fn stop_seqs(tok: &Option<Arc<Tokenizer>>, stop: &Option<StopSpec>) -> Vec<Vec<u32>> {
    let Some(spec) = stop else {
        return Vec::new();
    };
    let strs: Vec<&String> = match spec {
        StopSpec::One(s) => vec![s],
        StopSpec::Many(v) => v.iter().collect(),
    };
    strs.into_iter()
        .map(|s| match tok {
            Some(t) => t.encode(s),
            None => naive_encode(s),
        })
        .collect()
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

/// Formats chat messages with the Qwen chat template
/// (`<|im_start|>role\ncontent<|im_end|>\n...<|im_start|>assistant\n`).
fn qwen_chat_text(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        out.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            m.role, m.content
        ));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Builds sampling params from optional request fields (greedy by default).
fn sampling_params(
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<u64>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    top_logprobs: usize,
) -> SamplingParams {
    SamplingParams {
        temperature: temperature.unwrap_or(0.0),
        top_k: top_k.unwrap_or(0),
        top_p: top_p.unwrap_or(1.0),
        seed: seed.unwrap_or(0),
        presence_penalty: presence_penalty.unwrap_or(0.0),
        frequency_penalty: frequency_penalty.unwrap_or(0.0),
        top_logprobs: top_logprobs.min(20),
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
    reason: &'static str,
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
    let ev = sse_chunk(&id, object, &state.model, created, "", Some(reason));
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
    let stop = stop_seqs(&state.tok, &req.stop);
    let bias = logit_bias_pairs(&req.logit_bias);
    // top_logprobs only applies when logprobs are requested (OpenAI).
    let top_logprobs = if req.logprobs.unwrap_or(false) {
        req.top_logprobs.unwrap_or(0).min(20)
    } else {
        0
    };
    let params = sampling_params(
        req.temperature,
        req.top_k,
        req.top_p,
        req.seed,
        req.presence_penalty,
        req.frequency_penalty,
        top_logprobs,
    );
    let id = format!("cmpl-{}", now());
    let created = now();

    if req.stream.unwrap_or(false) {
        let (rx_final, rx_tokens) = match state
            .engine
            .submit_stream(tokens, req.max_tokens, None, stop, bias, params)
            .await
        {
            Ok(x) => x,
            Err(e) => return busy_response(e),
        };
        let st = state.clone();
        let id2 = id.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(16);
        tokio::spawn(async move {
            let reason = rx_final.await.map(|(_, _, _, r)| r).unwrap_or("length");
            stream_tokens(st, id2, "text_completion", created, reason, rx_tokens, tx).await;
        });
        return sse_response(rx);
    }

    let n = req.n.unwrap_or(1).max(1);
    let mut choices = Vec::with_capacity(n);
    for i in 0..n {
        // Distinct seeds per choice so n > 1 produces independent samples
        // (n = 1 keeps the caller seed unchanged).
        let seed = if n > 1 {
            Some(req.seed.unwrap_or(1_000_000).wrapping_add(i as u64))
        } else {
            req.seed
        };
        let params = sampling_params(
            req.temperature,
            req.top_k,
            req.top_p,
            seed,
            req.presence_penalty,
            req.frequency_penalty,
            top_logprobs,
        );
        let (output, lps, tlps, reason) = match state
            .engine
            .submit(
                tokens.clone(),
                req.max_tokens,
                None,
                stop.clone(),
                bias.clone(),
                params,
            )
            .await
        {
            Ok(o) => o,
            Err(e) => return busy_response(e),
        };
        let logprobs = if req.logprobs.unwrap_or(false) {
            Some(Logprobs {
                tokens: output
                    .iter()
                    .map(|&t| decode_text(&state.tok, &[t]))
                    .collect(),
                token_logprobs: lps.clone(),
                top_logprobs: (top_logprobs > 0).then(|| {
                    tlps.iter()
                        .map(|row| {
                            row.iter()
                                .map(|&(t, lp)| TopLogprob {
                                    token: decode_text(&state.tok, &[t]),
                                    logprob: lp,
                                })
                                .collect()
                        })
                        .collect()
                }),
            })
        } else {
            None
        };
        choices.push(CompletionChoice {
            index: i,
            text: decode_text(&state.tok, &output),
            tokens: output,
            logprobs,
            finish_reason: reason.into(),
        });
    }
    Json(CompletionResponse {
        id,
        object: "text_completion".into(),
        created,
        model: state.model.clone(),
        choices,
    })
    .into_response()
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let text = qwen_chat_text(&req.messages);
    let tokens = match &state.tok {
        Some(t) => t.encode(&text),
        None => naive_encode(&text),
    };
    // Stop at the chat end token when the real tokenizer is configured.
    let eos = state
        .tok
        .as_ref()
        .and_then(|t| t.special_token_id("<|im_end|>"));
    let stop = stop_seqs(&state.tok, &req.stop);
    let bias = logit_bias_pairs(&req.logit_bias);
    // top_logprobs only applies when logprobs are requested (OpenAI).
    let top_logprobs = if req.logprobs.unwrap_or(false) {
        req.top_logprobs.unwrap_or(0).min(20)
    } else {
        0
    };
    let params = sampling_params(
        req.temperature,
        req.top_k,
        req.top_p,
        req.seed,
        req.presence_penalty,
        req.frequency_penalty,
        top_logprobs,
    );
    let id = format!("chatcmpl-{}", now());
    let created = now();

    if req.stream.unwrap_or(false) {
        let (rx_final, rx_tokens) = match state
            .engine
            .submit_stream(tokens, req.max_tokens, eos, stop, bias, params)
            .await
        {
            Ok(x) => x,
            Err(e) => return busy_response(e),
        };
        let st = state.clone();
        let id2 = id.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(16);
        tokio::spawn(async move {
            let reason = rx_final.await.map(|(_, _, _, r)| r).unwrap_or("length");
            stream_tokens(
                st,
                id2,
                "chat.completion.chunk",
                created,
                reason,
                rx_tokens,
                tx,
            )
            .await;
        });
        return sse_response(rx);
    }

    let n = req.n.unwrap_or(1).max(1);
    let mut choices = Vec::with_capacity(n);
    for i in 0..n {
        let seed = if n > 1 {
            Some(req.seed.unwrap_or(1_000_000).wrapping_add(i as u64))
        } else {
            req.seed
        };
        let params = sampling_params(
            req.temperature,
            req.top_k,
            req.top_p,
            seed,
            req.presence_penalty,
            req.frequency_penalty,
            top_logprobs,
        );
        let (output, lps, tlps, reason) = match state
            .engine
            .submit(
                tokens.clone(),
                req.max_tokens,
                eos,
                stop.clone(),
                bias.clone(),
                params,
            )
            .await
        {
            Ok(o) => o,
            Err(e) => return busy_response(e),
        };
        let logprobs = if req.logprobs.unwrap_or(false) {
            Some(Logprobs {
                tokens: output
                    .iter()
                    .map(|&t| decode_text(&state.tok, &[t]))
                    .collect(),
                token_logprobs: lps.clone(),
                top_logprobs: (top_logprobs > 0).then(|| {
                    tlps.iter()
                        .map(|row| {
                            row.iter()
                                .map(|&(t, lp)| TopLogprob {
                                    token: decode_text(&state.tok, &[t]),
                                    logprob: lp,
                                })
                                .collect()
                        })
                        .collect()
                }),
            })
        } else {
            None
        };
        choices.push(CompletionChoice {
            index: i,
            text: decode_text(&state.tok, &output),
            tokens: output,
            logprobs,
            finish_reason: reason.into(),
        });
    }
    Json(CompletionResponse {
        id,
        object: "chat.completion".into(),
        created,
        model: state.model.clone(),
        choices,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    #[test]
    fn qwen_chat_template_format() {
        let text = qwen_chat_text(&[msg("system", "You are helpful."), msg("user", "hi")]);
        assert_eq!(
            text,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen_chat_template_no_system() {
        let text = qwen_chat_text(&[msg("user", "hello")]);
        assert_eq!(
            text,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
