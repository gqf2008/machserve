//! HTTP server integration tests: engine + router respond like a direct
//! engine run.
#![cfg(feature = "hip")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use mach_kernel_sys::hip;
use mach_model::config::ModelDType;
use mach_model::continuous::ContinuousModel;
use mach_model::sampling::SamplingParams;
use mach_model::tokenizer::Tokenizer;
use mach_model::{Config, Weights, WeightsFp8, WeightsQ4};
use mach_server::{AppState, ChatFormat, ServerEngine, router};
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
        chat_format: ChatFormat::default(),
    };
    let app = router(state);

    let got = post_completions(&app, prompt, max_new).await;
    assert_eq!(got, want, "server output must equal direct engine run");
}

fn moe_cfg() -> Config {
    let mut cfg = Config::tiny();
    cfg.intermediate_size = 64;
    cfg.num_experts = 4;
    cfg.num_experts_per_tok = 2;
    cfg
}

/// Posts one completions request and returns the served token ids.
async fn post_completions(app: &axum::Router, prompt: Vec<u32>, max_new: usize) -> Vec<u32> {
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
    json["choices"][0]["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect::<Vec<u32>>()
}

#[tokio::test]
async fn completions_endpoint_moe_matches_direct_engine() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = moe_cfg();
    let w = Weights::random(&cfg, 91).unwrap();
    let prompt = vec![5u32, 9, 3];
    let max_new = 4usize;

    // Expected output from a direct engine run (MoE config through
    // ContinuousModel -> BatchedModel grouped expert GEMMs).
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
    assert!(!want.is_empty(), "MoE engine must generate tokens");

    // Server path.
    let engine = ServerEngine::new(4);
    let _handle = engine.clone().spawn(hip, cfg, w).unwrap();
    let state = AppState {
        engine,
        model: "tiny-moe".into(),
        tok: None,
        chat_format: ChatFormat::default(),
    };
    let app = router(state);

    let got = post_completions(&app, prompt, max_new).await;
    assert_eq!(got, want, "server MoE output must equal direct engine run");
}

/// Server-path Q4: `ServerEngine::spawn_q4` (storage-int4 -> f16 on device)
/// must serve HTTP completions identically to a direct Q4 engine run. MoE
/// config exercises the per-expert Q4 upload. Ignored: GPU test, run serially
/// with `-- --ignored --test-threads=1`.
#[ignore]
#[tokio::test]
async fn completions_endpoint_q4_matches_direct_engine() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = moe_cfg();
    cfg.dtype = ModelDType::F16;
    let w = Weights::random(&cfg, 91).unwrap();
    let wq4 = WeightsQ4::from_weights(&w, &cfg);
    let prompt = vec![5u32, 9, 3];
    let max_new = 4usize;

    // Expected output from a direct Q4 engine run.
    let mut cm = ContinuousModel::with_prefill_rows_q4(hip.clone(), cfg, &wq4, 4, 4).unwrap();
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
    assert!(!want.is_empty(), "Q4 engine must generate tokens");

    // Server path (Q4 engine thread).
    let engine = ServerEngine::new(4);
    let _handle = engine.clone().spawn_q4(hip, cfg, wq4).unwrap();
    let state = AppState {
        engine,
        model: "tiny-moe-q4".into(),
        tok: None,
        chat_format: ChatFormat::default(),
    };
    let app = router(state);

    let got = post_completions(&app, prompt, max_new).await;
    assert_eq!(
        got, want,
        "server Q4 output must equal direct Q4 engine run"
    );
}

/// Server-path FP8: `ServerEngine::spawn_fp8` (storage-E4M3 -> f16 on device)
/// must serve HTTP completions identically to a direct FP8 engine run. MoE
/// config exercises the per-expert FP8 upload. Ignored: GPU test, run serially
/// with `-- --ignored --test-threads=1`.
#[ignore]
#[tokio::test]
async fn completions_endpoint_fp8_matches_direct_engine() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = moe_cfg();
    cfg.dtype = ModelDType::F16;
    let w = Weights::random(&cfg, 91).unwrap();
    let wfp8 = WeightsFp8::from_weights(&w, &cfg);
    let prompt = vec![5u32, 9, 3];
    let max_new = 4usize;

    // Expected output from a direct FP8 engine run.
    let mut cm = ContinuousModel::with_prefill_rows_fp8(hip.clone(), cfg, &wfp8, 4, 4).unwrap();
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
    assert!(!want.is_empty(), "FP8 engine must generate tokens");

    // Server path (FP8 engine thread).
    let engine = ServerEngine::new(4);
    let _handle = engine.clone().spawn_fp8(hip, cfg, wfp8).unwrap();
    let state = AppState {
        engine,
        model: "tiny-moe-fp8".into(),
        tok: None,
        chat_format: ChatFormat::default(),
    };
    let app = router(state);

    let got = post_completions(&app, prompt, max_new).await;
    assert_eq!(
        got, want,
        "server FP8 output must equal direct FP8 engine run"
    );
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
        chat_format: ChatFormat::default(),
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
        top_logprobs: 0,
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
        chat_format: ChatFormat::default(),
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
        chat_format: ChatFormat::default(),
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
        chat_format: ChatFormat::default(),
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
        chat_format: ChatFormat::default(),
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
        chat_format: ChatFormat::default(),
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

#[tokio::test]
async fn busy_engine_returns_openai_error_json() {
    // No GPU work happens: a zero-capacity engine rejects submissions before
    // the model is ever touched.
    let engine = ServerEngine::new(0);
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: None,
        chat_format: ChatFormat::default(),
    };
    let app = router(state);
    let body = serde_json::json!({ "prompt": [1, 2, 3], "max_tokens": 4 });
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
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let err = &json["error"];
    assert_eq!(err["type"], "server_error");
    assert_eq!(err["code"], "engine_busy");
    assert!(
        err["message"].as_str().unwrap().contains("capacity"),
        "message should explain the capacity error: {err}"
    );
}

#[tokio::test]
async fn engine_shutdown_drains_queued_work_then_exits() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 53).unwrap();
    let engine = ServerEngine::new(4);
    let handle = engine.clone().spawn(hip, cfg, w).unwrap();

    // Two requests complete even though shutdown is requested while they are
    // queued or in flight (the engine drains before exiting).
    let a = {
        let e = engine.clone();
        tokio::spawn(async move {
            e.submit(
                vec![5u32, 9, 3],
                4,
                None,
                Vec::new(),
                Vec::new(),
                SamplingParams::default(),
            )
            .await
        })
    };
    let b = {
        let e = engine.clone();
        tokio::spawn(async move {
            e.submit(
                vec![3u32, 9, 5],
                4,
                None,
                Vec::new(),
                Vec::new(),
                SamplingParams::default(),
            )
            .await
        })
    };
    // Let the engine thread pick both up before requesting shutdown.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    engine.shutdown();

    let (oa, _, _, ra) = a.await.unwrap().unwrap();
    let (ob, _, _, rb) = b.await.unwrap().unwrap();
    assert_eq!(oa.len(), 4);
    assert_eq!(ob.len(), 4);
    assert_eq!(ra, "length");
    assert_eq!(rb, "length");

    // The engine thread must exit on its own after draining.
    handle
        .join()
        .expect("engine thread must exit after shutdown");
}

#[tokio::test]
async fn top_logprobs_returned_when_requested() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 101).unwrap();
    let prompt = vec![5u32, 9, 3];
    let engine = ServerEngine::new(4);
    let _handle = engine.clone().spawn(hip, cfg, w).unwrap();
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: None,
        chat_format: ChatFormat::default(),
    };
    let app = router(state);

    // top_logprobs=3 with logprobs: true -> one sorted list per position.
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": 4,
        "temperature": 0.9,
        "top_p": 0.95,
        "seed": 99,
        "logprobs": true,
        "top_logprobs": 3,
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
    let top = lp["top_logprobs"].as_array().expect("top_logprobs array");
    assert_eq!(top.len(), 4, "one top_logprobs list per generated token");
    for row in top {
        let entries = row.as_array().expect("top_logprobs row");
        assert_eq!(entries.len(), 3, "top_logprobs=3 per position");
        let lps: Vec<f64> = entries
            .iter()
            .map(|e| e["logprob"].as_f64().unwrap())
            .collect();
        assert!(lps.windows(2).all(|w| w[0] >= w[1]), "descending logprobs");
        for e in entries {
            assert!(e["token"].as_str().is_some(), "token string present");
        }
    }

    // Without logprobs: true, top_logprobs is ignored (no logprobs payload).
    let body = serde_json::json!({
        "prompt": prompt,
        "max_tokens": 2,
        "temperature": 0.9,
        "top_p": 0.95,
        "seed": 99,
        "top_logprobs": 3,
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
    assert!(
        json["choices"][0]["logprobs"].is_null(),
        "logprobs must be absent without logprobs: true"
    );
}

#[tokio::test]
async fn invalid_request_returns_openai_error_json() {
    // Validation happens before any engine work, so no GPU/model is needed
    // (a zero-capacity engine never gets a chance to run).
    let engine = ServerEngine::new(0);
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: None,
        chat_format: ChatFormat::default(),
    };
    let app = router(state);
    let cases = [
        (
            serde_json::json!({ "prompt": [1, 2, 3], "max_tokens": 0 }),
            "max_tokens",
        ),
        (
            serde_json::json!({
                "prompt": [1, 2, 3],
                "max_tokens": 4,
                "logprobs": true,
                "top_logprobs": 21,
            }),
            "top_logprobs",
        ),
        (
            serde_json::json!({ "prompt": [1, 2, 3], "max_tokens": 4, "n": 0 }),
            "n",
        ),
    ];
    for (body, needle) in cases {
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
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "case {needle}");
        let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["code"], "invalid_request");
        assert!(
            json["error"]["message"].as_str().unwrap().contains(needle),
            "message should mention {needle}: {json}"
        );
    }
}

#[tokio::test]
async fn usage_is_reported() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let w = Weights::random(&cfg, 113).unwrap();
    let prompt = vec![5u32, 9, 3];
    let engine = ServerEngine::new(4);
    let _handle = engine.clone().spawn(hip, cfg, w).unwrap();
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: None,
        chat_format: ChatFormat::default(),
    };
    let app = router(state);
    let body = serde_json::json!({ "prompt": prompt, "max_tokens": 4 });
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
    let u = &json["usage"];
    assert_eq!(u["prompt_tokens"], 3, "prompt token count");
    assert_eq!(u["completion_tokens"], 4, "generated token count");
    assert_eq!(u["total_tokens"], 7, "total = prompt + completion");
    // choices[0].tokens must match completion_tokens.
    let toks = json["choices"][0]["tokens"].as_array().unwrap();
    assert_eq!(toks.len(), 4);
}

#[tokio::test]
async fn spec_mode_serves_greedy_and_rejects_non_greedy() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny();
    let dw = Weights::random(&cfg, 61).unwrap();
    let tw = Weights::random(&cfg, 73).unwrap();
    let engine = ServerEngine::with_spec(4, 4);
    let _handle = engine
        .clone()
        .spawn_spec(hip.clone(), cfg, tw, cfg, dw)
        .unwrap();
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: None,
        chat_format: ChatFormat::default(),
    };
    let app = router(state);

    // Greedy (default params) request is served.
    let body = serde_json::json!({ "prompt": [5, 9, 3, 200], "max_tokens": 6 });
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
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "greedy request must be served"
    );
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let toks = json["choices"][0]["tokens"].as_array().unwrap();
    assert_eq!(toks.len(), 6, "spec mode must generate max_tokens");

    // Non-greedy request is rejected with a 400 invalid_request_error.
    let body = serde_json::json!({
        "prompt": [5, 9, 3, 200],
        "max_tokens": 6,
        "temperature": 0.9,
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
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["error"]["type"], "invalid_request_error");
}

/// MLA (DeepSeek-V2 style) config: low-rank Q + compressed KV.
fn mla_cfg() -> Config {
    Config::mla(128, 2, 4, 1024, 64, 32, 16, 16, 8, 16)
}

#[tokio::test]
async fn completions_endpoint_mla_matches_direct_engine() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = mla_cfg();
    let w = Weights::random(&cfg, 92).unwrap();
    let prompt = vec![5u32, 9, 3];
    let max_new = 4usize;

    // Expected output from a direct engine run (MLA through ContinuousModel ->
    // BatchedModel expanded per-head KV decode).
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
    assert!(!want.is_empty(), "MLA engine must generate tokens");

    // Server path.
    let engine = ServerEngine::new(4);
    let _handle = engine.clone().spawn(hip, cfg, w).unwrap();
    let state = AppState {
        engine,
        model: "tiny-mla".into(),
        tok: None,
        chat_format: ChatFormat::default(),
    };
    let app = router(state);

    let got = post_completions(&app, prompt, max_new).await;
    assert_eq!(got, want, "server MLA output must equal direct engine run");
}

/// Paged-KV server e2e (#78 C5b): two HTTP requests sharing a system-prompt
/// prefix alias the same physical KV pages — the second request's output
/// equals a direct paged-engine run and the engine reports the exact
/// prompt-token savings.
#[tokio::test]
async fn completions_endpoint_paged_shared_prefix_matches_direct_engine() {
    let Some(hip) = hip_ctx() else { return };
    let cfg = Config::tiny(); // max_seq 256
    let tpp = 64usize;
    let w = Weights::random(&cfg, 61).unwrap();

    let prefix: Vec<u32> = (0..tpp as u32).map(|i| (i * 37 + 3) % 1024 + 1).collect();
    let d_a: Vec<u32> = vec![42, 99];
    let d_b: Vec<u32> = vec![300, 17, 8];
    let a: Vec<u32> = prefix.iter().chain(&d_a).copied().collect();
    let b: Vec<u32> = prefix.iter().chain(&d_b).copied().collect();

    // Reference: the same prompts through a direct paged engine (sequential —
    // A fully materializes its pages before B reuses them).
    let mut cm = ContinuousModel::with_paged_prefill_rows(hip.clone(), cfg, &w, 4, 4, tpp).unwrap();
    let id_a = cm
        .add(
            &a,
            3,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !cm.is_done(id_a) {
        cm.step().unwrap();
    }
    let id_b = cm
        .add(
            &b,
            3,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !cm.all_done() {
        cm.step().unwrap();
    }
    let want_a = cm.generated(id_a);
    let want_b = cm.generated(id_b);

    // Server path (paged engine). A second engine handle reads the reuse
    // stats snapshot after both requests complete.
    let engine = ServerEngine::with_paged(4, 4, tpp);
    let stats_handle = engine.clone();
    let _handle = engine.clone().spawn(hip, cfg, w).unwrap();
    let state = AppState {
        engine,
        model: "tiny".into(),
        tok: None,
        chat_format: ChatFormat::default(),
    };
    let app = router(state);

    // First request writes the shared prefix pages; the second reuses them.
    let got_a = post_completions(&app, a, 3).await;
    let got_b = post_completions(&app, b, 3).await;
    assert_eq!(
        got_a, want_a,
        "paged HTTP output A must equal direct engine"
    );
    assert_eq!(
        got_b, want_b,
        "reused HTTP output B must equal direct engine"
    );

    let stats = stats_handle
        .paged_reuse_stats()
        .expect("paged engine reports reuse stats");
    assert_eq!(stats.reused_tokens, tpp, "B reuses exactly the shared page");
    assert_eq!(stats.requests, 2);
    assert_eq!(
        stats.reused_tokens, tpp,
        "prompt-token savings must equal the shared prefix"
    );
}

/// Paged Q4 serving (#80 review fix): the MACH_Q4=1 MACH_PAGED=1
/// configuration — `ServerEngine::with_paged` + `spawn_q4` — must serve
/// cross-request prefix reuse over the dequantized f16 path.
#[tokio::test]
async fn completions_endpoint_q4_paged_reuses_shared_prefix() {
    let Some(hip) = hip_ctx() else { return };
    let mut cfg = Config::tiny(); // max_seq 256
    cfg.dtype = ModelDType::F16;
    let tpp = 64usize;
    let w = Weights::random(&cfg, 63).unwrap();
    let wq4 = WeightsQ4::from_weights(&w, &cfg);

    let prefix: Vec<u32> = (0..tpp as u32).map(|i| (i * 37 + 3) % 1024 + 1).collect();
    let d_a: Vec<u32> = vec![42, 99];
    let d_b: Vec<u32> = vec![300, 17, 8];
    let a: Vec<u32> = prefix.iter().chain(&d_a).copied().collect();
    let b: Vec<u32> = prefix.iter().chain(&d_b).copied().collect();

    // Reference: the same prompts through the direct paged Q4 engine.
    let mut cm =
        ContinuousModel::with_paged_prefill_rows_q4(hip.clone(), cfg, &wq4, 4, 4, tpp).unwrap();
    let id_a = cm
        .add(
            &a,
            3,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !cm.is_done(id_a) {
        cm.step().unwrap();
    }
    let id_b = cm
        .add(
            &b,
            3,
            None,
            Vec::new(),
            Vec::new(),
            SamplingParams::default(),
        )
        .unwrap();
    while !cm.all_done() {
        cm.step().unwrap();
    }
    let want_a = cm.generated(id_a);
    let want_b = cm.generated(id_b);

    // Server path: paged engine over spawn_q4 (the Q4 paged branch).
    let engine = ServerEngine::with_paged(4, 4, tpp);
    let stats_handle = engine.clone();
    let _handle = engine.clone().spawn_q4(hip, cfg, wq4).unwrap();
    let state = AppState {
        engine,
        model: "tiny-q4".into(),
        tok: None,
        chat_format: ChatFormat::default(),
    };
    let app = router(state);

    let got_a = post_completions(&app, a, 3).await;
    let got_b = post_completions(&app, b, 3).await;
    assert_eq!(
        got_a, want_a,
        "Q4 paged HTTP output A must equal direct engine"
    );
    assert_eq!(
        got_b, want_b,
        "Q4 paged reused HTTP output B must equal direct engine"
    );
    let stats = stats_handle
        .paged_reuse_stats()
        .expect("paged Q4 engine reports reuse stats");
    assert_eq!(stats.reused_tokens, tpp, "Q4 server reuses the shared page");
}
