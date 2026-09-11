#!/usr/bin/env python3
"""C4 negative-case + determinism driver (runbook step 6).

Starts mach-server twice and checks the failure paths around vision:

  vision on  : image request 200, the same image twice identical at
               temperature=0, over-budget image 400, text-only 200
  vision off : image request 501 multimodal_not_implemented

Writes a JSON report (default `artifacts/vision-c4/negatives.json`) and exits
non-zero unless every expectation holds. stdlib only; the over-budget PNG is
synthesised in-process so no fixture is needed.
"""

import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib


def solid_png(width, height, rgb):
    """Deterministic solid-colour PNG (stdlib only)."""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag, data):
        body = struct.pack(">I", len(data)) + tag + data
        return body + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def data_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()


def image_payload(path):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image in one sentence."},
                    {"type": "image_url", "image_url": {"url": data_url(path)}},
                ],
            }
        ],
        "max_tokens": 24,
        "temperature": 0,
        "stream": True,
    }


def text_payload():
    return {
        "messages": [{"role": "user", "content": "Say hi in one word."}],
        "max_tokens": 8,
        "temperature": 0,
    }


def sse_text(body):
    out = []
    for line in body.splitlines():
        if line.startswith("data: ") and not line.rstrip().endswith("[DONE]"):
            try:
                out.append(json.loads(line[6:])["choices"][0]["delta"].get("content") or "")
            except Exception:
                pass
    return "".join(out)


def selftest():
    body = (
        'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    assert sse_text(body) == "hi", sse_text(body)
    assert sse_text("data: [DONE]\n\n") == ""
    png = solid_png(4, 4, (1, 2, 3))
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and png.endswith(b"IEND\xaeB`\x82")
    print("vision_c4_negative selftest ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="target/release/mach-server.exe")
    parser.add_argument("--model-dir", default=".models/qwen3.8-27b")
    parser.add_argument("--image", default="artifacts/vision-c4/input.png")
    parser.add_argument("--out", default="artifacts/vision-c4/negatives.json")
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--max-patches", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--extra-env", action="append", default=[])
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0

    base = "http://127.0.0.1:" + str(args.port)
    big_path = os.path.splitext(args.out)[0] + "_big.png"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(big_path, "wb") as handle:
        handle.write(solid_png(2048, 2048, (30, 90, 30)))

    def wait_health():
        end = time.time() + args.timeout
        while time.time() < end:
            try:
                with urllib.request.urlopen(base + "/healthz", timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                time.sleep(2)
        return False

    def post(payload):
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")

    def start(vision, log_name):
        env = dict(os.environ)
        env.update(
            {
                "MACH_MODEL": os.path.basename(os.path.normpath(args.model_dir)),
                "MACH_MODELS": os.path.dirname(os.path.normpath(args.model_dir)) or ".",
                "MACH_ADDR": "127.0.0.1:" + str(args.port),
                "MACH_VISION_MAX_TOKENS": str(args.max_patches),
            }
        )
        if vision:
            env["MACH_VISION"] = "1"
        else:
            env.pop("MACH_VISION", None)
        for item in args.extra_env:
            key, _, value = item.partition("=")
            env[key] = value
        log = open(os.path.join(os.path.dirname(os.path.abspath(args.out)), log_name), "wb")
        return subprocess.Popen(
            [args.binary], env=env, stdout=log, stderr=subprocess.STDOUT
        )

    def stop(proc):
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    checks = {}
    report = {}
    proc = start(True, "negatives_vision_on.log")
    try:
        if not wait_health():
            print("vision-on server never became healthy", file=sys.stderr)
            return 2
        s1, b1 = post(image_payload(args.image))
        s2, b2 = post(image_payload(args.image))
        checks["vision_on_image_200"] = s1 == 200
        checks["vision_on_image_deterministic"] = bool(sse_text(b1)) and sse_text(b1) == sse_text(b2)
        s3, b3 = post(image_payload(big_path))
        checks["oversized_image_400"] = s3 == 400 and "exceeds limit" in b3
        s4, b4 = post(text_payload())
        checks["text_only_200"] = s4 == 200
        report["vision_on"] = {
            "image_status": s1,
            "image_text": sse_text(b1),
            "second_image_status": s2,
            "second_image_text": sse_text(b2),
            "oversized_status": s3,
            "oversized_body": b3.strip()[:300],
            "text_status": s4,
            "text_body": b4.strip()[:200],
        }
    finally:
        stop(proc)

    proc = start(False, "negatives_vision_off.log")
    try:
        if not wait_health():
            print("vision-off server never became healthy", file=sys.stderr)
            return 2
        s5, b5 = post(image_payload(args.image))
        checks["vision_off_image_501"] = s5 == 501 and "multimodal_not_implemented" in b5
        report["vision_off"] = {"image_status": s5, "image_body": b5.strip()[:300]}
    finally:
        stop(proc)

    report["checks"] = checks
    report["pass"] = all(checks.values())
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())