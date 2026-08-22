//! HTTP server integration tests: engine + router respond like a direct
//! engine run.
#![cfg(feature = "hip")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use mach_kernel_sys::hip;
use mach_model::continuous::ContinuousModel;
use mach_model::sampling::SamplingParams;
use mach_model::tokenizer::Tokenizer;
use mach_model::{Config, Weights};
use mach_server::{AppState, ServerEngine, router};
use tower::ServiceExt;

fn hip_ctx() -> Option<std::sync::Arc<hip::Hip>> {
    match hip::hip() {
        Ok(h) => match hip::device_count() {
            Ok(n) if n > 0 => Some(h),
            _ => {
                eprintln!("skipping HIP test: no device");
                None
            }
        },
        Err(e) => {
            eprintln!("skipping HIP test: {e}");
            None
        }
    }
}

#[tokio::test]
async fn completions_endpoint_matches_direct_engine() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 61).unwrap();
    let prompt = vec![5u32, 9, 3];
    let max_new = 4usize;

    // Expected output from a direct engine run.
    let mut cm = ContinuousModel::new(hip.clone(), cfg, &w, 4).unwrap();
    let id = cm
        .add(
            &prompt,
            max_new,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !cm.is_done(id) {
        cm.step().unwrap();
    }
    let want = cm.generated(id);

    // Server path.
    let engine = ServerEngine::new(4);
    let _handle = engine.clone().spawn(hip, cfg, w).unwrap();
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: None,
    };
    let app = router(state);

    let body = serde_json::json!({ "prompt": prompt, "max_tokens": max_new });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let got: Vec<u32> = json["choices"][0]["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(got, want, "server output must equal direct engine run");
}

#[tokio::test]
async fn healthz_and_text_prompt() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 73).unwrap();
    let engine = ServerEngine::new(4);
    let _handle = engine.clone().spawn(hip, cfg, w).unwrap();
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: None,
    };
    let app = router(state);

    // healthz
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // text prompt -> naive byte tokens -> response tokens decode back to text
    let body = serde_json::json!({ "prompt": "hi", "max_tokens": 3 });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json["choices"][0]["tokens"].as_array().is_some());
}

#[tokio::test]
async fn sampling_params_flow_through_http() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 67).unwrap();
    let prompt = vec![5u32, 9, 3];
    let max_new = 6usize;
    let params = SamplingParams {
        temperature: 0.85,
        top_k: 50,
        top_p: 0.92,
        seed: 1234,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
    };

    // Direct engine reference with the same params + seed.
    let mut cm = ContinuousModel::new(hip.clone(), cfg, &w, 4).unwrap();
    let id = cm
        .add(&prompt, max_new, None, Vec::new(), Vec::new(), params)
        .unwrap();
    while !cm.is_done(id) {
        cm.step().unwrap();
    }
    let want = cm.generated(id);

    // Server path with the params passed over HTTP.
    let engine = ServerEngine::new(4);
    let _handle = engine.clone().spawn(hip, cfg, w).unwrap();
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: None,
    };
    let app = router(state);
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": max_new,
        "temperature": 0.85,
        "top_k": 50,
        "top_p": 0.92,
        "seed": 1234,
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let got: Vec<u32> = json["choices"][0]["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(
        got, want,
        "HTTP sampling must equal a direct engine run with the same params+seed"
    );
}

/// Collects the `content`/`text` deltas from an SSE response body.
async fn collect_sse_deltas(resp: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(resp.into_body(), 4 << 20)
        .await
        .expect("read sse body");
    let text = String::from_utf8_lossy(&bytes);
    let mut out = String::new();
    for line in text.lines() {
        if let Some(payload) = line.strip_prefix("data: ") {
            if payload == "[DONE]" {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(payload).expect("sse json");
            if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                out.push_str(delta);
            } else if let Some(t) = v["choices"][0]["text"].as_str() {
                out.push_str(t);
            }
        }
    }
    out
}

#[tokio::test]
async fn streaming_matches_non_streaming_with_same_seed() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 89).unwrap();
    let prompt = vec![5u32, 9, 3];
    let max_new = 6usize;

    let engine = ServerEngine::new(4);
    let _handle = engine.clone().spawn(hip, cfg, w).unwrap();
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: None,
    };
    let app = router(state);

    // Non-streaming with a fixed seed.
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": max_new,
        "temperature": 0.9,
        "top_p": 0.95,
        "seed": 4321,
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let want = json["choices"][0]["text"].as_str().unwrap().to_string();

    // Streaming with the same seed: concatenated deltas must equal `want`.
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": max_new,
        "temperature": 0.9,
        "top_p": 0.95,
        "seed": 4321,
        "stream": true,
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()["content-type"].to_str().unwrap(),
        "text/event-stream",
        "stream must be served as SSE"
    );
    let got = collect_sse_deltas(resp).await;
    assert_eq!(got, want, "SSE deltas must reconstruct the non-stream text");
}

#[tokio::test]
async fn text_prompt_uses_real_tokenizer_when_available() {
    let Some(hip) = hip_ctx() else { return };
    // tokenizer.json at the repo root .models (skips when absent).
    let candidates = [
        std::path::PathBuf::from("../../.models").join("tokenizer.json"),
        std::path::PathBuf::from(".models").join("tokenizer.json"),
    ];
    let Some(tok_path) = candidates.into_iter().find(|p| p.exists()) else {
        eprintln!("skipping tokenizer test: tokenizer.json missing");
        return;
    };
    let tok = std::sync::Arc::new(Tokenizer::from_path(&tok_path).expect("load tokenizer"));
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 97).unwrap();

    let engine = ServerEngine::new(4);
    let _handle = engine.clone().spawn(hip, cfg, w).unwrap();
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: Some(tok.clone()),
    };
    let app = router(state);

    // Text prompt with a fixed seed must equal the same prompt sent as raw
    // token ids produced by the tokenizer.
    let text = "Hello, world!";
    let body_text = serde_json::json!({
        "prompt": text,
        "max_tokens": 5,
        "temperature": 0.9,
        "top_p": 0.9,
        "seed": 777,
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body_text.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let text_tokens: Vec<u32> = json["choices"][0]["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    let ids = tok.encode(text);
    let body_ids = serde_json::json!({
        "prompt": ids,
        "max_tokens": 5,
        "temperature": 0.9,
        "top_p": 0.9,
        "seed": 777,
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body_ids.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let id_tokens: Vec<u32> = json["choices"][0]["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(
        text_tokens, id_tokens,
        "text prompt must encode to the same ids as the tokenizer"
    );
}

#[tokio::test]
async fn n_returns_multiple_distinct_choices() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 79).unwrap();
    let prompt = vec![5u32, 9, 3];
    let engine = ServerEngine::new(8);
    let _handle = engine.clone().spawn(hip, cfg, w).unwrap();
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: None,
    };
    let app = router(state);
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": 4,
        "temperature": 0.9,
        "top_p": 0.95,
        "seed": 42,
        "n": 2,
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let choices = json["choices"].as_array().expect("choices array");
    assert_eq!(choices.len(), 2, "n=2 must return two choices");
    let t0: Vec<u32> = choices[0]["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let t1: Vec<u32> = choices[1]["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert!(!t0.is_empty() && !t1.is_empty());
    assert_ne!(
        t0, t1,
        "n>1 choices must use distinct seeds -> distinct samples"
    );
}

#[tokio::test]
async fn logprobs_are_returned_when_requested() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 91).unwrap();
    let prompt = vec![5u32, 9, 3];
    let engine = ServerEngine::new(4);
    let _handle = engine.clone().spawn(hip, cfg, w).unwrap();
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: None,
    };
    let app = router(state);
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": 4,
        "temperature": 0,
        "logprobs": true,
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let lp = &json["choices"][0]["logprobs"];
    let tokens = lp["tokens"].as_array().expect("logprobs tokens");
    let tlog = lp["token_logprobs"].as_array().expect("token_logprobs");
    assert_eq!(tokens.len(), 4);
    assert_eq!(tlog.len(), 4);
    // Greedy -> each token logprob is 0.
    assert!(tlog.iter().all(|v| v.as_f64().unwrap() == 0.0));
}
