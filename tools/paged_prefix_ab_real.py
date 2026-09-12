#!/usr/bin/env python3
"""Real-model cross-request prefix-reuse A/B driver (one arm per run).

Starts mach-server, warms up (hiprtc / hipBLAS lazy init), then issues
`--requests` chat completions that share one long system block and differ only
in the trailing user turn. For each request it records the SSE arrival
timestamps (per `recv`, so frames delivered in one read share a timestamp),
which is what tells "TTFT dropped because the prefix was reused" apart from
"the response arrived in one burst".

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
and one `paged: register ...` line per materialised prefill. The arm fails
unless those lines appear AND at least one request reports `reused_pages>0`:
a run that silently degraded to contiguous (or whose shared prefix never
matched a page) must not produce plausible-looking numbers.

Fails loudly (non-zero, no JSON) on: non-200 status, an `error` SSE frame, a
stream without data frames, a stream without `finish_reason`, a missing model
directory, a busy port, or a server that exits while being polled.

Writes a JSON report (default `artifacts/paged-prefix/ab_real_<arm>.json`).
stdlib only.
"""

import argparse
import json
import os
import re
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

REUSED_PAGES_RE = re.compile(r"reused_pages=(\d+)")


class ArmError(RuntimeError):
    """A condition that makes the arm's numbers meaningless."""


def stream_chat(host, port, payload, timeout=1800):
    """POST one streaming completion; return arrival times (per recv) + text.

    Raises `ArmError` on anything that is not a complete, successful stream:
    a failed arm must never be recorded as a fast one.
    """
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
    status = None
    frames = []
    text = []
    saw_finish = False
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            now = time.time() - t0
            buf += chunk
            if status is None:
                # Split the status line/headers off before treating the rest
                # as an SSE body: an error response is plain JSON.
                if b"\r\n\r\n" not in buf:
                    continue
                head_blob, buf = buf.split(b"\r\n\r\n", 1)
                first_line = head_blob.split(b"\r\n", 1)[0].decode("latin-1")
                status = first_line
                if " 200 " not in first_line:
                    raise ArmError(f"HTTP status {first_line!r}")
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
                except Exception as exc:
                    raise ArmError(f"undecodable SSE frame {data[:120]!r}: {exc}") from exc
                if "error" in doc:
                    raise ArmError(f"server error frame: {doc['error']}")
                choices = doc.get("choices") or []
                if not choices:
                    raise ArmError(f"SSE frame without choices: {data[:120]!r}")
                choice = choices[0]
                if choice.get("finish_reason"):
                    saw_finish = True
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text.append(delta["content"])
    finally:
        s.close()
    total = time.time() - t0
    if status is None:
        raise ArmError("no HTTP response")
    if not frames:
        raise ArmError("stream produced no data frames")
    if not saw_finish:
        raise ArmError("stream ended without finish_reason")
    return {
        "ttft_s": round(frames[0], 4),
        "total_s": round(total, 4),
        "frames": len(frames),
        "frame_times_s": [round(t, 4) for t in frames],
        "reply": "".join(text)[:160],
    }


def wait_health(host, port, proc, timeout=900):
    """Wait for /healthz, failing early if `proc` exits or never listens."""
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise ArmError(f"server exited early (code {proc.returncode})")
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise ArmError(f"server never became healthy within {timeout}s")


def port_is_busy(host, port):
    """True when something already listens on `host:port`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def positive_int(raw):
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


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
    ap.add_argument("--capacity", type=positive_int, default=4)
    ap.add_argument("--max-seq", type=positive_int, default=4096)
    ap.add_argument("--tokens-per-page", type=positive_int, default=64)
    ap.add_argument("--shared-repeats", type=positive_int, default=22)
    ap.add_argument("--requests", type=positive_int, default=5)
    ap.add_argument("--max-new", type=positive_int, default=8)
    ap.add_argument("--out", default=os.path.join("artifacts", "paged-prefix"))
    ap.add_argument("--health-timeout", type=positive_int, default=900)
    args = ap.parse_args(argv)

    paged = args.arm == "paged"
    shared = PARA * args.shared_repeats
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, f"ab_real_{args.arm}.log")

    exe = os.path.abspath(args.exe)
    if not os.path.exists(exe):
        print(f"error: server binary not found: {exe}", file=sys.stderr)
        return 2
    model_dir = os.path.join(os.path.abspath(args.models_dir), args.model)
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print(f"error: model config not found: {model_dir}", file=sys.stderr)
        return 2
    if port_is_busy(args.host, args.port):
        print(f"error: {args.host}:{args.port} is already in use; "
              "a stale server would be measured instead of this arm", file=sys.stderr)
        return 2

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

    report = {"arm": args.arm, "paged": paged, "model": args.model,
              "requests": args.requests, "max_new": args.max_new,
              "shared_repeats": args.shared_repeats}
    log = open(log_path, "wb")
    proc = subprocess.Popen([exe], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        wait_health(args.host, args.port, proc, args.health_timeout)
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
    except ArmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    with open(log_path, "r", encoding="utf-8", errors="replace") as h:
        log_text = h.read()
    report["admit_lines"] = [l for l in log_text.splitlines() if l.startswith("paged: admit")]
    report["register_lines"] = [l for l in log_text.splitlines() if l.startswith("paged: register")]

    # Positive invariant instead of a blacklist of degradation messages: the
    # paged arm must have admitted through the paged engine and actually hit
    # the page cache. Any (present or future) silent fallback to contiguous
    # fails this, whatever wording it uses.
    if paged:
        if "MACH_PAGED" in log_text and "serving continuous" in log_text:
            print("error: server log reports MACH_PAGED degraded to continuous; "
                  "arm is not paged", file=sys.stderr)
            return 1
        if not report["admit_lines"]:
            print("error: no 'paged: admit' lines — the paged engine never "
                  "admitted these requests (see log)", file=sys.stderr)
            return 1
        reused = [
            int(REUSED_PAGES_RE.search(l).group(1))
            for l in report["admit_lines"]
            if REUSED_PAGES_RE.search(l)
        ]
        if not reused or max(reused) == 0:
            print("error: no request aliased a single cached page "
                  f"(reused_pages per admit: {reused}); the arm measures no "
                  "reuse — check the shared prefix length/page alignment",
                  file=sys.stderr)
            return 1

    out_path = os.path.join(args.out, f"ab_real_{args.arm}.json")
    with open(out_path, "w", encoding="utf-8") as h:
        json.dump(report, h, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())