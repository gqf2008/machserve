//! OpenAI-shaped request/response types and pure prompt helpers.
//!
//! These carry no axum/engine dependency, so they compile in both feature
//! modes and their unit tests run in the default (no-`hip`) test面.
//! `routes` re-exports them, so `crate::routes::…` paths are unchanged.
//!
//! Without `hip` the axum layer in `routes` is not compiled, so these
//! helpers are reachable only from this module's own tests. Keep them
//! compiled (the tests are the point) without per-item attributes.
#![cfg_attr(not(feature = "hip"), allow(dead_code))]

use crate::multimodal::{ChatContent, render_content};
use mach_model::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Largest accepted chat body when multimodal serving is enabled: one
/// 64MiB encoded base64 payload plus JSON overhead. Text-only keeps the
/// axum default.
pub const MAX_CHAT_BODY_BYTES: usize = 96 << 20;

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

pub(crate) fn default_max_tokens() -> usize {
    32
}

pub(crate) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Validates request parameters; returns a 400 response for the first invalid
/// one (OpenAI `invalid_request_error`).
/// Remaining vision patch budget for the next image, or None when exhausted.
pub(crate) fn remaining_patch_budget(total: usize, max: usize) -> Option<usize> {
    (max > total).then(|| max - total)
}

/// Naive fallback: token id -> byte (lossy UTF-8, no tokenizer configured).
/// Kept consistent with the streaming path (byte-level + `from_utf8_lossy`).
pub(crate) fn naive_decode(tokens: &[u32]) -> String {
    let bytes: Vec<u8> = tokens
        .iter()
        .map(|&t| if t < 256 { t as u8 } else { b'?' })
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Naive ASCII fallback: byte -> token id.
pub(crate) fn naive_encode(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

/// Converts an OpenAI logit_bias map ({token_id: bias}) into (token, bias) pairs.
pub(crate) fn logit_bias_pairs(
    m: &Option<std::collections::HashMap<String, f32>>,
) -> Vec<(u32, f32)> {
    let Some(map) = m else {
        return Vec::new();
    };
    map.iter()
        .filter_map(|(k, &v)| k.parse::<u32>().ok().map(|t| (t, v)))
        .collect()
}

/// Encodes OpenAI `stop` strings into token sequences.
pub(crate) fn stop_seqs(tok: &Option<Arc<Tokenizer>>, stop: &Option<StopSpec>) -> Vec<Vec<u32>> {
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

pub(crate) fn prompt_tokens(tok: &Option<Arc<Tokenizer>>, prompt: &Prompt) -> Vec<u32> {
    match prompt {
        Prompt::Text(s) => match tok {
            Some(t) => t.encode(s),
            None => naive_encode(s),
        },
        Prompt::Tokens(t) => t.clone(),
    }
}

pub(crate) fn decode_text(tok: &Option<Arc<Tokenizer>>, tokens: &[u32]) -> String {
    match tok {
        Some(t) => t.decode(tokens),
        None => naive_decode(tokens),
    }
}

/// Formats chat messages with the Qwen chat template
/// (`<|im_start|>role\ncontent<|im_end|>\n...<|im_start|>assistant\n`).
pub(crate) fn qwen_chat_text(messages: &[ChatMessage]) -> (String, Vec<String>) {
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
pub(crate) fn deepseek_chat_text(messages: &[ChatMessage]) -> (String, Vec<String>) {
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
pub(crate) fn chat_eos_text(format: ChatFormat) -> &'static str {
    match format {
        ChatFormat::Qwen => "<|im_end|>",
        ChatFormat::DeepSeek => DS_EOS,
    }
}

/// Renders `messages` in the server configured chat format, returning the
/// text and the image URLs in message order.
pub(crate) fn chat_text(format: ChatFormat, messages: &[ChatMessage]) -> (String, Vec<String>) {
    match format {
        ChatFormat::Qwen => qwen_chat_text(messages),
        ChatFormat::DeepSeek => deepseek_chat_text(messages),
    }
}

/// Chat request body limit: multimodal serving allows one encoded image.
pub(crate) fn chat_body_limit(image_enabled: bool) -> usize {
    if image_enabled {
        MAX_CHAT_BODY_BYTES
    } else {
        2 << 20
    }
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
    fn remaining_patch_budget_shrinks_per_image() {
        assert_eq!(remaining_patch_budget(0, 8192), Some(8192));
        assert_eq!(remaining_patch_budget(5000, 8192), Some(3192));
        assert_eq!(remaining_patch_budget(8192, 8192), None);
        assert_eq!(remaining_patch_budget(9000, 8192), None);
    }

    #[test]
    fn chat_body_limit_selects_text_and_vision_budgets() {
        assert_eq!(chat_body_limit(false), 2 << 20);
        assert_eq!(chat_body_limit(true), MAX_CHAT_BODY_BYTES);
    }
}
