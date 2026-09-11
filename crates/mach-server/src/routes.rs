//! HTTP routes: /v1/completions, /v1/chat/completions, /healthz.
//!
//! Prompts are accepted as raw token ids (`prompt_tokens`) or text. Text is
//! encoded with the real byte-level BPE tokenizer when one is configured
//! (falls back to a naive byte-per-token mapping otherwise). Both endpoints
//! support OpenAI-shaped `stream: true` -> SSE with per-token deltas.

use crate::engine::{DoneReceiver, EngineError, ServerEngine};
use crate::multimodal::{ChatContent, render_content};
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

/// Which chat template `/v1/chat/completions` renders.
///
/// The template belongs with the checkpoint, not the request: feeding a
/// DeepSeek checkpoint ChatML markers (`<|im_start|>`) makes it decode as if
/// those tokens were user prose, so the server picks the format from the
/// loaded model rather than guessing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChatFormat {
    /// Qwen / ChatML: `<|im_start|>role\ncontent<|im_end|>\n…`.
    #[default]
    Qwen,
    /// DeepSeek-V2 (`tokenizer_config.json` chat template): a BOS token, then
    /// `User: …\n\n` / `Assistant: …<eos>` turns, then a bare `Assistant:`
    /// generation prompt. Note the template emits no trailing space — the
    /// model's own first token supplies it.
    DeepSeek,
}

/// Shared router state.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<ServerEngine>,
    pub model: String,
    /// Real tokenizer when `tokenizer.json` is available (else naive bytes).
    pub tok: Option<Arc<Tokenizer>>,
    /// Chat template for `/v1/chat/completions`.
    pub chat_format: ChatFormat,
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
    pub content: ChatContent,
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

/// OpenAI token usage (`usage`).
#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
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
fn err_response(status: StatusCode, message: &str, err_type: &str, code: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": err_type,
            "code": code,
        }
    });
    (status, Json(body)).into_response()
}

/// OpenAI-shaped 400 response for invalid request parameters.
fn bad_request(message: &str) -> Response {
    err_response(
        StatusCode::BAD_REQUEST,
        message,
        "invalid_request_error",
        "invalid_request",
    )
}

/// Request-parameter limits (mirroring OpenAI's API surface).
const MAX_N: usize = 128;
const MAX_TOP_LOGPROBS: usize = 20;

/// Validates request parameters; returns a 400 response for the first invalid
/// one (OpenAI `invalid_request_error`).
/// Remaining vision patch budget for the next image, or None when exhausted.
fn remaining_patch_budget(total: usize, max: usize) -> Option<usize> {
    (max > total).then(|| max - total)
}

fn validate_request(
    max_tokens: usize,
    top_logprobs: Option<usize>,
    n: Option<usize>,
) -> Option<Response> {
    if max_tokens == 0 {
        return Some(bad_request("max_tokens must be greater than 0"));
    }
    if let Some(k) = top_logprobs
        && k > MAX_TOP_LOGPROBS
    {
        return Some(bad_request(&format!(
            "top_logprobs must be between 0 and {MAX_TOP_LOGPROBS}"
        )));
    }
    if let Some(n) = n
        && (n == 0 || n > MAX_N)
    {
        return Some(bad_request(&format!("n must be between 1 and {MAX_N}")));
    }
    None
}

/// Maps engine errors to OpenAI-shaped responses (400 invalid, 503 busy/model, 500 startup).
fn busy_response(e: EngineError) -> Response {
    let status = StatusCode::SERVICE_UNAVAILABLE;
    match e {
        EngineError::Busy => err_response(
            status,
            "engine capacity reached; retry later",
            "server_error",
            "engine_busy",
        ),
        EngineError::ShuttingDown => err_response(
            status,
            "engine is shutting down; retry later",
            "server_error",
            "engine_shutting_down",
        ),
        EngineError::InvalidRequest(m) => err_response(
            StatusCode::BAD_REQUEST,
            &m,
            "invalid_request_error",
            "invalid_request",
        ),
        EngineError::Model(m) => err_response(
            status,
            &format!("model error: {m}"),
            "server_error",
            "model_error",
        ),
        EngineError::Startup(m) => err_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("engine startup failed: {m}"),
            "server_error",
            "engine_startup",
        ),
    }
}

/// Emits an OpenAI-style SSE error frame followed by `[DONE]`.
async fn stream_error(tx: tokio::sync::mpsc::Sender<Result<Bytes, Infallible>>, message: &str) {
    let body = serde_json::json!({"error": {"message": message, "type": "server_error"}});
    let frame = format!("data: {body}\n\ndata: [DONE]\n\n");
    let _ = tx.send(Ok(Bytes::from(frame))).await;
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
fn qwen_chat_text(messages: &[ChatMessage]) -> (String, Vec<String>) {
    let mut out = String::new();
    let mut images = Vec::new();
    for m in messages {
        out.push_str(&format!("<|im_start|>{}\n", m.role));
        render_content(&m.content, &mut out, &mut images);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    (out, images)
}

/// DeepSeek-V2 special tokens. The delimiters are U+FF5C FULLWIDTH VERTICAL
/// LINE (not ASCII `|`) and U+2581 LOWER ONE EIGHTH BLOCK (the SentencePiece
/// word separator), exactly as `tokenizer_config.json` spells them; ASCII
/// look-alikes would not be in the vocabulary and would encode to junk.
const DS_BOS: &str = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>";
const DS_EOS: &str = "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>";

/// Formats chat messages with the DeepSeek-V2 chat template, transcribed from
/// the checkpoint `tokenizer_config.json`:
/// `{{ bos_token }}` then, per message, `system` -> `{content}\n\n`, `user` ->
/// `User: {content}\n\n`, `assistant` -> `Assistant: {content}{eos}`, closing
/// with the `Assistant:` generation prompt.
fn deepseek_chat_text(messages: &[ChatMessage]) -> (String, Vec<String>) {
    let mut out = String::from(DS_BOS);
    let mut images = Vec::new();
    for m in messages {
        let mut content = String::new();
        render_content(&m.content, &mut content, &mut images);
        match m.role.as_str() {
            "user" => out.push_str(&format!("User: {content}\n\n")),
            "assistant" => out.push_str(&format!("Assistant: {content}{DS_EOS}")),
            // The template falls through to the system spelling for any other
            // role (e.g. `system`, `tool`), as the Jinja source does.
            _ => out.push_str(&format!("{content}\n\n")),
        }
    }
    out.push_str("Assistant:");
    (out, images)
}

/// The end-of-turn token string for `format`, used to derive the request
/// stop token id from the tokenizer specials.
fn chat_eos_text(format: ChatFormat) -> &'static str {
    match format {
        ChatFormat::Qwen => "<|im_end|>",
        ChatFormat::DeepSeek => DS_EOS,
    }
}

/// Renders `messages` in the server configured chat format, returning the
/// text and the image URLs in message order.
fn chat_text(format: ChatFormat, messages: &[ChatMessage]) -> (String, Vec<String>) {
    match format {
        ChatFormat::Qwen => qwen_chat_text(messages),
        ChatFormat::DeepSeek => deepseek_chat_text(messages),
    }
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

/// Frames buffered while the client is slower than the engine. The engine's
/// own token channel is best-effort (`try_send`), so the consumer here must
/// never stop draining; an exceeded backlog is reported as an explicit error
/// instead of silently truncating the stream.
const STREAM_FRAME_BACKLOG: usize = 4096;

/// Upper bound for draining the tail of a finished stream. A client that keeps
/// stalling is disconnected instead of hanging the streaming task.
const STREAM_FLUSH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Reads per-token ids, decodes incrementally and pushes SSE events as they
/// arrive, then a finish chunk once the engine reports completion.
///
/// The token stream must **not** be drained behind `done`: awaiting the
/// completion signal first turns SSE into a single write at the end of the
/// request, so clients cannot show tokens (and TTFT becomes the full
/// generation time). Frames also must not block on the client: a stalled
/// `tx.send` would stop draining `rx`, fill the engine channel and make it drop
/// tokens.
async fn stream_tokens(
    state: AppState,
    id: String,
    object: &'static str,
    created: u64,
    mut done: DoneReceiver,
    mut rx: tokio::sync::mpsc::Receiver<u32>,
    tx: tokio::sync::mpsc::Sender<Result<Bytes, Infallible>>,
) {
    let mut acc: Vec<u8> = Vec::new();
    let mut pending: std::collections::VecDeque<Bytes> = std::collections::VecDeque::new();
    let mut reason: Option<&'static str> = None;
    let mut failure: Option<String> = None;
    loop {
        tokio::select! {
            biased;
            res = &mut done, if reason.is_none() && failure.is_none() => {
                match res {
                    Ok(Ok((_, _, _, r))) => reason = Some(r),
                    // A failed completion ends the stream: waiting for more
                    // tokens would leave the task parked on `rx.recv()`.
                    Ok(Err(e)) => {
                        failure = Some(e.to_string());
                        break;
                    }
                    Err(_) => {
                        failure = Some("engine stopped before completion".into());
                        break;
                    }
                }
            }
            maybe = rx.recv() => {
                let Some(tok) = maybe else { break };
                let bytes = match &state.tok {
                    Some(t) => t.decode_bytes(&[tok]),
                    None => vec![if tok < 256 { tok as u8 } else { b'?' }],
                };
                acc.extend_from_slice(&bytes);
                let text = emit_valid_prefix(&mut acc);
                if !text.is_empty() {
                    if pending.len() >= STREAM_FRAME_BACKLOG {
                        failure = Some("client too slow: stream backlog exhausted".into());
                        break;
                    }
                    let ev = sse_chunk(&id, object, &state.model, created, &text, None);
                    pending.push_back(Bytes::from(ev));
                }
            }
            permit = tx.reserve(), if !pending.is_empty() => {
                match permit {
                    Ok(permit) => {
                        if let Some(frame) = pending.pop_front() {
                            permit.send(Ok(frame));
                        }
                    }
                    Err(_) => return, // client disconnected
                }
            }
        }
    }
    if let Some(message) = failure {
        // A stalled client must not hang this task: bounded wait, and pending
        // frames are dropped (the stream is being failed anyway).
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            stream_error(tx, &message),
        )
        .await;
        return;
    }
    if reason.is_none() {
        // The token channel closed without the engine reporting completion:
        // that is an error, not a normal stop.
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            stream_error(tx, "token stream ended before completion"),
        )
        .await;
        return;
    }
    // Everything left (buffered frames, the UTF-8 tail, the finish chunk and
    // `[DONE]`) is sent under one deadline: a client that stops reading must
    // not park this task forever.
    let flush = async move {
        while let Some(frame) = pending.pop_front() {
            if tx.send(Ok(frame)).await.is_err() {
                return;
            }
        }
        let tail = String::from_utf8_lossy(&acc).into_owned();
        if !tail.is_empty() {
            let ev = sse_chunk(&id, object, &state.model, created, &tail, None);
            if tx.send(Ok(Bytes::from(ev))).await.is_err() {
                return;
            }
        }
        let ev = sse_chunk(
            &id,
            object,
            &state.model,
            created,
            "",
            Some(reason.unwrap_or("stop")),
        );
        if tx.send(Ok(Bytes::from(ev))).await.is_err() {
            return;
        }
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await;
    };
    let _ = tokio::time::timeout(STREAM_FLUSH_TIMEOUT, flush).await;
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
            stream_tokens(st, id2, "text_completion", created, rx_final, rx_tokens, tx).await;
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
    let completion_tokens: usize = choices.iter().map(|c| c.tokens.len()).sum();
    Json(CompletionResponse {
        id,
        object: "text_completion".into(),
        created,
        model: state.model.clone(),
        choices,
        usage: Usage {
            prompt_tokens: tokens.len(),
            completion_tokens,
            total_tokens: tokens.len() + completion_tokens,
        },
    })
    .into_response()
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let (text, image_urls) = chat_text(state.chat_format, &req.messages);
    let tokens = match &state.tok {
        Some(t) => t.encode(&text),
        None => naive_encode(&text),
    };
    if let Some(resp) = validate_request(req.max_tokens, req.top_logprobs, req.n) {
        return resp;
    }
    let (tokens, images) = if image_urls.is_empty() {
        (tokens, Vec::new())
    } else {
        let Some(image_cfg) = state.engine.image_runtime() else {
            return err_response(
                StatusCode::NOT_IMPLEMENTED,
                "multimodal image requests require MACH_VISION=1",
                "invalid_request_error",
                "multimodal_not_implemented",
            );
        };
        let mut images = Vec::with_capacity(image_urls.len());
        let mut total_patches = 0usize;
        for url in &image_urls {
            let Some(remaining) = remaining_patch_budget(total_patches, image_cfg.max_patches)
            else {
                return bad_request(&format!(
                    "images exceed the {} vision patch budget",
                    image_cfg.max_patches
                ));
            };
            match crate::multimodal::fetch_image_url_limited(url, &image_cfg.processor, remaining)
                .await
            {
                Ok(image) => {
                    let patches = image.grid[0] * image.grid[1] * image.grid[2];
                    total_patches = match total_patches.checked_add(patches) {
                        Some(total) => total,
                        None => {
                            return bad_request(&format!(
                                "images exceed the {} vision patch budget",
                                image_cfg.max_patches
                            ));
                        }
                    };
                    images.push(image)
                }
                Err(e) => return bad_request(&format!("image: {e}")),
            }
        }
        let grids: Vec<mach_model::vision::VisionGrid> = images.iter().map(|i| i.grid).collect();
        match crate::multimodal::expand_image_pads(
            &tokens,
            image_cfg.image_token_id,
            &grids,
            image_cfg.spatial_merge_size,
        ) {
            Ok(tokens) => (tokens, images),
            Err(e) => return bad_request(&format!("image: {e}")),
        }
    };
    // Stop at the chat end token when the real tokenizer is configured.
    let eos = state
        .tok
        .as_ref()
        .and_then(|t| t.special_token_id(chat_eos_text(state.chat_format)));
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
            .submit_stream_multimodal(tokens, images, req.max_tokens, eos, stop, bias, params)
            .await
        {
            Ok(x) => x,
            Err(e) => return busy_response(e),
        };
        let st = state.clone();
        let id2 = id.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(16);
        tokio::spawn(async move {
            stream_tokens(
                st,
                id2,
                "chat.completion.chunk",
                created,
                rx_final,
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
            .submit_multimodal(
                tokens.clone(),
                images.clone(),
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
    let completion_tokens: usize = choices.iter().map(|c| c.tokens.len()).sum();
    Json(CompletionResponse {
        id,
        object: "chat.completion".into(),
        created,
        model: state.model.clone(),
        choices,
        usage: Usage {
            prompt_tokens: tokens.len(),
            completion_tokens,
            total_tokens: tokens.len() + completion_tokens,
        },
    })
    .into_response()
}

/// GET /healthz
pub async fn healthz() -> &'static str {
    "ok"
}

/// Builds the axum router.
/// Largest accepted chat body when multimodal serving is enabled: one
/// 64MiB encoded base64 payload plus JSON overhead. Text-only keeps the
/// axum default.
pub const MAX_CHAT_BODY_BYTES: usize = 96 << 20;

/// Chat request body limit: multimodal serving allows one encoded image.
fn chat_body_limit(image_enabled: bool) -> usize {
    if image_enabled {
        MAX_CHAT_BODY_BYTES
    } else {
        2 << 20
    }
}

pub fn router(state: AppState) -> axum::Router {
    use axum::routing::{get, post};
    let limit = chat_body_limit(state.engine.image_runtime().is_some());
    let chat = axum::Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .layer(axum::extract::DefaultBodyLimit::max(limit));
    axum::Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/completions", post(completions))
        .merge(chat)
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: ChatContent::Text(content.into()),
        }
    }

    #[test]
    fn qwen_chat_template_format() {
        let (text, images) =
            qwen_chat_text(&[msg("system", "You are helpful."), msg("user", "hi")]);
        assert!(images.is_empty());
        assert_eq!(
            text,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen_chat_template_no_system() {
        let (text, images) = qwen_chat_text(&[msg("user", "hello")]);
        assert!(images.is_empty());
        assert_eq!(
            text,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// The DeepSeek template transcribes the checkpoint's Jinja verbatim: BOS,
    /// `User: …\n\n`, then a bare `Assistant:` prompt with no trailing space.
    #[test]
    fn deepseek_chat_template_user_only() {
        let (text, images) = deepseek_chat_text(&[msg("user", "hello")]);
        assert!(images.is_empty());
        assert_eq!(text, format!("{DS_BOS}User: hello\n\nAssistant:"));
    }

    #[test]
    fn deepseek_chat_template_with_system_and_history() {
        let (text, images) = deepseek_chat_text(&[
            msg("system", "You are helpful."),
            msg("user", "hi"),
            msg("assistant", "Hello!"),
            msg("user", "again"),
        ]);
        assert!(images.is_empty());
        assert_eq!(
            text,
            format!(
                "{DS_BOS}You are helpful.\n\nUser: hi\n\nAssistant: Hello!{DS_EOS}User: again\n\nAssistant:"
            )
        );
    }

    /// The delimiters are U+FF5C (fullwidth vertical line) and U+2581 — an
    /// ASCII `|` would not be an in-vocabulary special token.
    #[test]
    fn deepseek_special_tokens_use_fullwidth_delimiters() {
        assert_eq!(DS_BOS, "<｜begin▁of▁sentence｜>");
        assert_eq!(DS_EOS, "<｜end▁of▁sentence｜>");
        assert!(!DS_EOS.contains('|'));
    }

    /// `chat_text` dispatches on the configured format, so a DeepSeek
    /// checkpoint never gets ChatML markers rendered into its prompt.
    #[test]
    fn chat_text_dispatches_on_format() {
        let msgs = [msg("user", "hi")];
        assert_eq!(
            chat_text(ChatFormat::DeepSeek, &msgs),
            deepseek_chat_text(&msgs)
        );
        assert_eq!(chat_text(ChatFormat::Qwen, &msgs), qwen_chat_text(&msgs));
        assert_ne!(
            chat_text(ChatFormat::DeepSeek, &msgs),
            chat_text(ChatFormat::Qwen, &msgs)
        );
        assert_eq!(chat_eos_text(ChatFormat::Qwen), "<|im_end|>");
        assert_eq!(chat_eos_text(ChatFormat::DeepSeek), DS_EOS);
    }

    /// `AppState` defaults to the Qwen format so existing callers are
    /// unaffected.
    #[test]
    fn chat_format_defaults_to_qwen() {
        assert_eq!(ChatFormat::default(), ChatFormat::Qwen);
    }

    #[test]
    fn chat_request_parses_multimodal_parts() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"messages":[{"role":"user","content":[
                {"type":"text","text":"what is this? "},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}
            ]}]}"#,
        )
        .unwrap();
        match &req.messages[0].content {
            ChatContent::Parts(parts) => assert_eq!(parts.len(), 2),
            other => panic!("expected parts, got {other:?}"),
        }
    }

    #[test]
    fn qwen_chat_template_renders_image_placeholder() {
        use crate::multimodal::{ContentPart, ImageUrl};
        let msgs = [ChatMessage {
            role: "user".into(),
            content: ChatContent::Parts(vec![
                ContentPart::Text {
                    text: "look ".into(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,AA==".into(),
                        detail: None,
                    },
                },
                ContentPart::Text {
                    text: " now".into(),
                },
            ]),
        }];
        let (text, images) = qwen_chat_text(&msgs);
        assert_eq!(images, vec!["data:image/png;base64,AA==".to_string()]);
        assert_eq!(
            text,
            "<|im_start|>user\nlook <|vision_start|><|image_pad|><|vision_end|> now<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn engine_errors_map_to_http_status() {
        assert_eq!(
            busy_response(EngineError::InvalidRequest("bad".into())).status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            busy_response(EngineError::Model(mach_model::Error::Model("m".into()))).status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(
            busy_response(EngineError::Startup("s".into())).status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn router_builds_for_text_and_vision_states() {
        let text = AppState {
            engine: ServerEngine::new(1),
            model: "test".into(),
            tok: None,
            chat_format: ChatFormat::Qwen,
        };
        let _ = router(text);
        let engine = ServerEngine::new(1);
        engine.set_image_runtime(crate::engine::ImageRuntimeConfig {
            processor: mach_model::image_processor::ImageProcessorConfig::default(),
            image_token_id: 1,
            spatial_merge_size: 2,
            max_patches: 16,
        });
        let vision = AppState {
            engine,
            model: "test".into(),
            tok: None,
            chat_format: ChatFormat::Qwen,
        };
        let _ = router(vision);
    }

    #[test]
    fn remaining_patch_budget_shrinks_per_image() {
        assert_eq!(remaining_patch_budget(0, 8192), Some(8192));
        assert_eq!(remaining_patch_budget(5000, 8192), Some(3192));
        assert_eq!(remaining_patch_budget(8192, 8192), None);
        assert_eq!(remaining_patch_budget(9000, 8192), None);
    }

    #[tokio::test]
    async fn chat_body_limit_rejects_oversized_text_request() {
        use axum::body::Body;
        use tower::ServiceExt;
        let state = AppState {
            engine: ServerEngine::new(1),
            model: "test".into(),
            tok: None,
            chat_format: ChatFormat::Qwen,
        };
        let app = router(state);
        let text = "a".repeat((2 << 20) + 1024);
        let body = serde_json::json!({"messages": [{"role": "user", "content": text}]});
        let req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn chat_body_limit_selects_text_and_vision_budgets() {
        assert_eq!(chat_body_limit(false), 2 << 20);
        assert_eq!(chat_body_limit(true), MAX_CHAT_BODY_BYTES);
    }

    /// Regression: SSE used to await the engine's completion signal before
    /// draining the token stream, so every frame arrived at the end of the
    /// request (TTFT == full generation time). The first frame must reach the
    /// client while the request is still running.
    #[tokio::test]
    async fn stream_tokens_emits_before_completion() {
        use tokio::sync::{mpsc, oneshot};
        let state = AppState {
            engine: ServerEngine::new(1),
            model: "test".into(),
            tok: None,
            chat_format: ChatFormat::Qwen,
        };
        let (tok_tx, tok_rx) = mpsc::channel(4);
        let (done_tx, done_rx) = oneshot::channel();
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let task = tokio::spawn(stream_tokens(
            state,
            "id".into(),
            "text_completion",
            0,
            done_rx,
            tok_rx,
            out_tx,
        ));

        tok_tx.send(b'a' as u32).await.unwrap();
        let frame = tokio::time::timeout(std::time::Duration::from_secs(5), out_rx.recv())
            .await
            .expect("first SSE frame must arrive before the engine reports completion")
            .expect("frame present")
            .unwrap();
        let text = String::from_utf8_lossy(&frame).into_owned();
        assert!(text.contains("\"a\""), "{text}");

        done_tx
            .send(Ok((Vec::new(), Vec::new(), Vec::new(), "stop")))
            .unwrap();
        drop(tok_tx);
        let mut rest = String::new();
        while let Some(frame) = out_rx.recv().await {
            rest.push_str(&String::from_utf8_lossy(&frame.unwrap()));
        }
        assert!(rest.contains("\"finish_reason\":\"stop\""), "{rest}");
        assert!(rest.contains("[DONE]"), "{rest}");
        task.await.unwrap();
    }

    fn stream_test_state() -> AppState {
        AppState {
            engine: ServerEngine::new(1),
            model: "test".into(),
            tok: None,
            chat_format: ChatFormat::Qwen,
        }
    }

    async fn collect_frames(
        out_rx: &mut tokio::sync::mpsc::Receiver<Result<Bytes, Infallible>>,
    ) -> String {
        let mut all = String::new();
        while let Some(frame) = out_rx.recv().await {
            all.push_str(&String::from_utf8_lossy(&frame.unwrap()));
        }
        all
    }

    /// Tokens queued before the completion signal must all be delivered, and
    /// the finish chunk keeps the engine's reason.
    #[tokio::test]
    async fn stream_tokens_drains_tokens_before_done() {
        use tokio::sync::{mpsc, oneshot};
        let (tok_tx, tok_rx) = mpsc::channel(4);
        let (done_tx, done_rx) = oneshot::channel();
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let task = tokio::spawn(stream_tokens(
            stream_test_state(),
            "id".into(),
            "text_completion",
            0,
            done_rx,
            tok_rx,
            out_tx,
        ));
        for b in b"abc" {
            tok_tx.send(*b as u32).await.unwrap();
        }
        let first = tokio::time::timeout(std::time::Duration::from_secs(5), out_rx.recv())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(String::from_utf8_lossy(&first).contains("\"a\""));
        done_tx
            .send(Ok((Vec::new(), Vec::new(), Vec::new(), "length")))
            .unwrap();
        drop(tok_tx);
        let rest = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            collect_frames(&mut out_rx),
        )
        .await
        .unwrap();
        assert!(rest.contains("\"b\"") && rest.contains("\"c\""), "{rest}");
        assert!(rest.contains("\"finish_reason\":\"length\""), "{rest}");
        assert!(rest.contains("[DONE]"), "{rest}");
        task.await.unwrap();
    }

    /// An engine error must surface as an SSE error frame, not a finish chunk.
    #[tokio::test]
    async fn stream_tokens_reports_engine_error() {
        use tokio::sync::{mpsc, oneshot};
        let (_tok_tx, tok_rx) = mpsc::channel(4);
        let (done_tx, done_rx) = oneshot::channel();
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let task = tokio::spawn(stream_tokens(
            stream_test_state(),
            "id".into(),
            "text_completion",
            0,
            done_rx,
            tok_rx,
            out_tx,
        ));
        done_tx
            .send(Err(EngineError::Model(mach_model::Error::Model(
                "boom".into(),
            ))))
            .unwrap();
        let all = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            collect_frames(&mut out_rx),
        )
        .await
        .unwrap();
        assert!(all.contains("boom"), "{all}");
        assert!(all.contains("[DONE]"), "{all}");
        assert!(!all.contains("finish_reason"), "{all}");
        task.await.unwrap();
    }

    /// A closed token channel without completion is an error, not a `stop`.
    #[tokio::test]
    async fn stream_tokens_reports_close_without_done() {
        use tokio::sync::mpsc;
        let (tok_tx, tok_rx) = mpsc::channel(4);
        let (_done_tx, done_rx) = tokio::sync::oneshot::channel();
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let task = tokio::spawn(stream_tokens(
            stream_test_state(),
            "id".into(),
            "text_completion",
            0,
            done_rx,
            tok_rx,
            out_tx,
        ));
        drop(tok_tx);
        let all = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            collect_frames(&mut out_rx),
        )
        .await
        .unwrap();
        assert!(all.contains("ended before completion"), "{all}");
        assert!(all.contains("[DONE]"), "{all}");
        task.await.unwrap();
    }

    /// A slow client must not hang the streaming task, and an exhausted backlog
    /// must surface as an explicit error instead of a silent truncation.
    #[tokio::test]
    async fn stream_tokens_stalled_client_terminates() {
        use tokio::sync::{mpsc, oneshot};
        let (tok_tx, tok_rx) = mpsc::channel(4);
        let (_done_tx, done_rx) = oneshot::channel();
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let task = tokio::spawn(stream_tokens(
            stream_test_state(),
            "id".into(),
            "text_completion",
            0,
            done_rx,
            tok_rx,
            out_tx,
        ));
        // Drain one frame every 50ms: slow enough that the local backlog
        // overflows, fast enough to receive the explicit error frame.
        let reader = tokio::spawn(async move {
            let mut all = String::new();
            while let Some(frame) = out_rx.recv().await {
                all.push_str(&String::from_utf8_lossy(&frame.unwrap()));
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
            all
        });
        for i in 0..(STREAM_FRAME_BACKLOG + 64) {
            if tok_tx.send(b'a' as u32 + (i % 26) as u32).await.is_err() {
                break;
            }
        }
        let all = tokio::time::timeout(std::time::Duration::from_secs(60), reader)
            .await
            .expect("slow client must still receive a terminal frame")
            .unwrap();
        assert!(all.contains("client too slow"), "{all}");
        assert!(all.contains("[DONE]"), "{all}");
        tokio::time::timeout(std::time::Duration::from_secs(30), task)
            .await
            .expect("streaming task must terminate for a slow client")
            .unwrap();
    }
    /// A stalled client with a response *below* the backlog cap must terminate
    /// too: the tail flush is bounded by `STREAM_FLUSH_TIMEOUT`.
    #[tokio::test]
    async fn stream_tokens_stalled_client_small_response_terminates() {
        use tokio::sync::{mpsc, oneshot};
        let (tok_tx, tok_rx) = mpsc::channel(8);
        let (done_tx, done_rx) = oneshot::channel();
        let (out_tx, _out_rx) = mpsc::channel(8); // never read
        let task = tokio::spawn(stream_tokens(
            stream_test_state(),
            "id".into(),
            "text_completion",
            0,
            done_rx,
            tok_rx,
            out_tx,
        ));
        for _ in 0..64 {
            tok_tx.send(b'a' as u32).await.unwrap();
        }
        done_tx
            .send(Ok((Vec::new(), Vec::new(), Vec::new(), "stop")))
            .unwrap();
        drop(tok_tx);
        tokio::time::timeout(std::time::Duration::from_secs(30), task)
            .await
            .expect("tail flush must be bounded for a stalled client")
            .unwrap();
    }
}
