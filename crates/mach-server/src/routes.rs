//! HTTP routes: /v1/completions, /v1/chat/completions, /healthz.
//!
//! Prompts are accepted either as raw token ids (`prompt_tokens`) or as a
//! string (naively mapped byte-per-token for ASCII; a real tokenizer is a
//! follow-up). Responses are OpenAI-shaped.

use crate::engine::ServerEngine;
use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Shared router state.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<ServerEngine>,
    pub model: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    /// ASCII text, naively mapped byte-per-token.
    Text(String),
    /// Raw token ids.
    Tokens(Vec<u32>),
}

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub prompt: Prompt,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_max_tokens() -> usize {
    32
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

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Naive ASCII decode: token id -> byte.
fn naive_decode(tokens: &[u32]) -> String {
    tokens
        .iter()
        .map(|&t| if t < 256 { t as u8 } else { b'?' })
        .collect::<Vec<u8>>()
        .iter()
        .map(|&b| b as char)
        .collect()
}

/// Naive ASCII encode: byte -> token id.
fn naive_encode(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

fn prompt_tokens(prompt: &Prompt) -> Vec<u32> {
    match prompt {
        Prompt::Text(s) => naive_encode(s),
        Prompt::Tokens(t) => t.clone(),
    }
}

/// POST /v1/completions
pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, axum::http::StatusCode> {
    let tokens = prompt_tokens(&req.prompt);
    let output = state
        .engine
        .submit(tokens, req.max_tokens, None)
        .await
        .map_err(|_| axum::http::StatusCode::SERVICE_UNAVAILABLE)?;
    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", now()),
        object: "text_completion".into(),
        created: now(),
        model: state.model.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            text: naive_decode(&output),
            tokens: output,
            finish_reason: "length".into(),
        }],
    }))
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<CompletionResponse>, axum::http::StatusCode> {
    let text: String = req
        .messages
        .iter()
        .map(|m| {
            format!(
                "{}: {}
",
                m.role, m.content
            )
        })
        .collect();
    let output = state
        .engine
        .submit(naive_encode(&text), req.max_tokens, None)
        .await
        .map_err(|_| axum::http::StatusCode::SERVICE_UNAVAILABLE)?;
    Ok(Json(CompletionResponse {
        id: format!("chatcmpl-{}", now()),
        object: "chat.completion".into(),
        created: now(),
        model: state.model.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            text: naive_decode(&output),
            tokens: output,
            finish_reason: "length".into(),
        }],
    }))
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
