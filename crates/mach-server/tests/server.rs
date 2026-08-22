//! HTTP server integration tests: engine + router respond like a direct
//! engine run.
#![cfg(feature = "hip")]

use axum::body::Body;
use axum::http::{Request, StatusCode};
use mach_kernel_sys::hip;
use mach_model::continuous::ContinuousModel;
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
    let id = cm.add(&prompt, max_new, None).unwrap();
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
