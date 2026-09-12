#!/usr/bin/env python3
"""Real-model cross-request prefix-reuse A/B driver (one arm per run).

Starts mach-server, warms up (hiprtc / hipBLAS lazy init), then issues
`--requests` chat completions that share one long system block and differ only
in the trailing user turn. For each request it records the per-frame arrival
times of the SSE stream, so "TTFT dropped because the prefix was reused" can be
told apart from "the response arrived in one burst".

    paged      : MACH_PAGED=1 (+ MACH_TPP) -- cross-request page aliasing
    contiguous : MACH_PAGED=0              -- every request full-recomputes

Run ONE arm per invocation and leave an observation window (>=10 min) between
the two arms: on this machine back-to-back heavy GPU arms have twice been
followed by a driver TDR / dGPU dropping off the HIP device list. After each
arm check `Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Display'}`
(4101 = TDR) and compare `mach-server doctor`'s `device_count` with the value
from before the run.

The paged arm also sets MACH_PAGED_DEBUG=1, so the server log carries one
`paged: admit ...` line per request (prompt tokens / full pages / reused pages)
and one `paged: register ...` line per materialised prefill. A request that
should have hit the cache but shows `reused_pages=0` is the smoking gun.

Writes a JSON report (default `artifacts/paged-prefix/ab_real_<arm>.json`).
stdlib only.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

# A fixed, self-similar system block. `--shared-repeats 22` measures 1937
# tokens with the Qwen3-8B checkpoint tokenizer (the chat template adds a few
# more); a third of that is enough to make a full prefill visible against a
# delta-only one on any 8B-class model.
PARA = (
    "You are a careful assistant. Answer using only the information given. "
    "The following reference passage is provided once and must be treated as stable context for every turn. "
    "It describes a fictional lighthouse: the keeper maintains a ledger of every ship that passes the reef, "
    "the lamp burns oil measured in quarts, and the fog horn sounds when visibility drops below one mile. "
    "Keep all of this in mind while answering, but do not repeat it back. "
)


def stream_chat(host, port, payload, timeout=1800):
    """POST one streaming completion; return per-frame arrival times + text."""
    body = json.dumps(payload).encode()
    head = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode()
    s = socket.create_connection((host, port), timeout=timeout)
    t0 = time.time()
    s.sendall(head + body)
    buf = b""
    frames = []
    text = []
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        now = time.time() - t0
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                continue
            frames.append(now)
            try:
                doc = json.loads(data)
                delta = doc["choices"][0].get("delta", {})
                if delta.get("content"):
                    text.append(delta["content"])
            except Exception:
                pass
    total = time.time() - t0
    s.close()
    return {
        "ttft_s": round(frames[0] if frames else total, 4),
        "total_s": round(total, 4),
        "frames": len(frames),
        "frame_times_s": [round(t, 4) for t in frames],
        "reply": "".join(text)[:160],
    }


def wait_health(host, port, timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def build_messages(shared, i):
    return [
        {"role": "system", "content": shared},
        {
            "role": "user",
            "content": f"Ledger question {i}: how many quarts does the lamp burn? Answer in one word.",
        },
    ]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=("paged", "contiguous"), required=True)
    ap.add_argument("--exe", default=os.path.join("target", "release", "mach-server.exe"))
    ap.add_argument("--models-dir", default=".models")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8131)
    ap.add_argument("--capacity", type=int, default=4)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--tokens-per-page", type=int, default=64)
    ap.add_argument("--shared-repeats", type=int, default=22)
    ap.add_argument("--requests", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--out", default=os.path.join("artifacts", "paged-prefix"))
    ap.add_argument("--health-timeout", type=int, default=900)
    args = ap.parse_args(argv)

    paged = args.arm == "paged"
    shared = PARA * args.shared_repeats
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f"ab_real_{args.arm}.log")

    env = dict(os.environ)
    env.update(
        {
            "MACH_MODEL": args.model,
            "MACH_MODELS": os.path.abspath(args.models_dir),
            "MACH_ADDR": f"{args.host}:{args.port}",
            "MACH_Q4": "1",
            "MACH_Q4_DEVICE": "2",
            "MACH_CAPACITY": str(args.capacity),
            "MACH_MAX_SEQ": str(args.max_seq),
            "MACH_PAGED": "1" if paged else "0",
        }
    )
    if paged:
        env["MACH_TPP"] = str(args.tokens_per_page)
        env["MACH_PAGED_DEBUG"] = "1"

    exe = os.path.abspath(args.exe)
    if not os.path.exists(exe):
        print(f"error: server binary not found: {exe}", file=sys.stderr)
        return 2

    log = open(log_path, "wb")
    proc = subprocess.Popen([exe], env=env, stdout=log, stderr=subprocess.STDOUT)
    report = {"arm": args.arm, "paged": paged, "model": args.model,
              "requests": args.requests, "max_new": args.max_new,
              "shared_repeats": args.shared_repeats}
    try:
        t0 = time.time()
        if not wait_health(args.host, args.port, args.health_timeout):
            print("error: server never became healthy", file=sys.stderr)
            return 1
        report["load_plus_health_s"] = round(time.time() - t0, 1)

        def chat(i):
            return stream_chat(
                args.host,
                args.port,
                {
                    "messages": build_messages(shared, i),
                    "max_tokens": args.max_new,
                    "temperature": 0,
                    "stream": True,
                },
            )

        report["warmup"] = chat(0)  # pays hiprtc / hipBLAS lazy init
        runs = [chat(i) for i in range(1, args.requests + 1)]
        report["runs"] = runs
        ttfts = [r["ttft_s"] for r in runs]
        rest = ttfts[1:] or ttfts
        report["ttft_first_s"] = ttfts[0]
        report["ttft_median_rest_s"] = round(sorted(rest)[len(rest) // 2], 4)
        report["ttft_ratio_first_over_rest"] = round(
            ttfts[0] / max(report["ttft_median_rest_s"], 1e-6), 2
        )
        report["wall_total_s"] = round(sum(r["total_s"] for r in runs), 4)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    # Self-check: a paged arm that silently degraded to contiguous would make
    # the A/B meaningless, so fail loudly instead of reporting numbers.
    with open(log_path, "r", encoding="utf-8", errors="replace") as h:
        log_text = h.read()
    if paged and "MACH_PAGED is unsupported" in log_text:
        print("error: MACH_PAGED was rejected (see log); arm is not paged",
              file=sys.stderr)
        return 1
    report["admit_lines"] = [l for l in log_text.splitlines() if l.startswith("paged: admit")]
    report["register_lines"] = [l for l in log_text.splitlines() if l.startswith("paged: register")]

    out_path = os.path.join(args.out, f"ab_real_{args.arm}.json")
    with open(out_path, "w", encoding="utf-8") as h:
        json.dump(report, h, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())