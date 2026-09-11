#!/usr/bin/env python3
"""C4 vision E2E driver for mach-server (stdlib only).

Starts mach-server with MACH_VISION=1, waits for /healthz, sends one OpenAI
chat request with a base64 data-URL image, records the response and timings,
then stops the server. Run one server at a time; watch VRAM while it loads.

Example:
  python tools/vision_c4_e2e.py --binary target/release/mach-server.exe \
      --model-dir .models/qwen3.8-27b --extra-env MACH_Q4=1 --extra-env MACH_CAPACITY=1
"""

import argparse
import base64
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib


def tiny_png(width=8, height=8):
    """Build a deterministic RGB PNG without third-party deps."""
    raw = b""
    for y in range(height):
        raw += b"\x00"
        for x in range(width):
            raw += bytes([(x * 31) % 256, (y * 47) % 256, 128])

    def chunk(tag, data):
        body = struct.pack(">I", len(data)) + tag + data
        return body + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )




# Model/quantization env vars that would make `mach-server doctor` read the
# checkpoint. Everything else - notably MACH_HIP_PATH - must survive so doctor
# resolves the same ROCm install as the server.
DOCTOR_ENV_DROP = (
    "MACH_MODELS",
    "MACH_MODEL",
    "MACH_CONFIG",
    "MACH_TOKENIZER",
    "MACH_Q4",
    "MACH_Q4_DEVICE",
    "MACH_FP8",
    "MACH_KV_INT8",
    "MACH_CAPACITY",
    "MACH_PREFILL_ROWS",
)


def doctor_env():
    return {k: v for k, v in os.environ.items() if k not in DOCTOR_ENV_DROP}


def parse_doctor_vram(text):
    """Return (value, error) from `mach-server doctor` output.

    `vram: mem_info error (...)` is a failure, not a sample.
    """
    line = next(
        (l.strip() for l in text.splitlines() if l.strip().startswith("vram:")),
        None,
    )
    if line is None:
        return None, None
    if "mem_info error" in line:
        return None, line
    return "mach-server doctor " + line, None


def probe_rocm_smi(timeout=10):
    """Return (value, error) for the rocm-smi VRAM query."""
    exe = shutil.which("rocm-smi")
    if not exe:
        return None, "rocm-smi not installed"
    try:
        out = subprocess.check_output(
            [exe, "--showmeminfo", "vram", "--csv"],
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except Exception as exc:
        return None, f"rocm-smi failed: {exc}"
    text = out.decode("utf-8", "replace").strip()
    if not text:
        return None, "rocm-smi produced no output"
    return text, None


def probe_doctor(binary, timeout=120):
    """Return (value, error) for the `mach-server doctor` VRAM line."""
    if not binary or not os.path.isfile(binary):
        return None, "mach-server doctor unavailable (binary path is not a file)"
    try:
        out = subprocess.check_output(
            [binary, "doctor"],
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env=doctor_env(),
        )
    except Exception as exc:
        return None, f"mach-server doctor failed: {exc}"
    value, error = parse_doctor_vram(out.decode("utf-8", "replace"))
    if value is None and error is None:
        return None, "mach-server doctor printed no vram line"
    return value, error


def sample_vram(binary=None):
    """Sample VRAM as {"source", "value", "error"}.

    Prefers rocm-smi; Windows ROCm installs often ship without it, so fall back
    to `mach-server doctor` (hip::mem_info). `value` stays None whenever
    sampling failed and `error` says why - an error is never recorded as a
    sample.
    """
    value, error = probe_rocm_smi()
    if value is not None:
        return {"source": "rocm-smi", "value": value, "error": None}
    smi_error = error
    value, doctor_error = probe_doctor(binary)
    if value is not None:
        return {"source": "mach-server doctor", "value": value, "error": None}
    detail = "; ".join(x for x in (smi_error, doctor_error) if x)
    return {"source": "none", "value": None, "error": detail}


def preflight_problems(args):
    """Everything that must hold before a GPU window is worth opening."""
    problems = []
    if not os.path.isfile(args.binary):
        problems.append("binary not found (or not a file): " + args.binary)
    if not os.path.isdir(args.model_dir):
        problems.append("model dir not found: " + args.model_dir)
    else:
        for name in ("config.json", "preprocessor_config.json", "tokenizer.json"):
            if not os.path.isfile(os.path.join(args.model_dir, name)):
                problems.append(f"model dir is missing {name}: " + args.model_dir)
        try:
            entries = os.listdir(args.model_dir)
        except OSError as exc:
            problems.append(f"model dir is not readable: {exc}")
            entries = []
        if not any(entry.endswith(".safetensors") for entry in entries):
            problems.append("model dir has no .safetensors shard: " + args.model_dir)
    if args.image and not os.path.isfile(args.image):
        problems.append("image not found: " + args.image)
    if args.max_patches <= 0:
        problems.append("--max-patches must be positive")
    out_dir = os.path.abspath(args.out_dir)
    try:
        os.makedirs(out_dir, exist_ok=True)
        probe = os.path.join(out_dir, ".preflight-write-test")
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok")
        os.remove(probe)
    except OSError as exc:
        problems.append(f"out dir not writable: {out_dir} ({exc})")
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
    except OSError as exc:
        problems.append(f"port {args.port} is not bindable: {exc}")
    return problems

def wait_health(base, timeout):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen(base + "/healthz", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(2)
    return False


def parse_sse_data(lines, started):
    """Return (ttft_seconds, stream_error) parsed from decoded SSE lines."""
    ttft = None
    error = None
    for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except Exception as exc:
            error = error or ("invalid SSE JSON: " + str(exc))
            continue
        if isinstance(data, dict) and data.get("error"):
            message = data["error"]
            if isinstance(message, dict):
                message = message.get("message", message)
            error = error or str(message)
            continue
        if ttft is None and isinstance(data, dict):
            choices = data.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    ttft = time.time() - started
    return ttft, error


def dump_status(prefix, before):
    """Validate MACH_VISION_DUMP files exist, are fresh and non-empty."""
    status = {}
    for suffix in (".bin", ".json"):
        path = prefix + suffix
        if not os.path.exists(path):
            status[suffix] = {"error": "missing"}
            continue
        size = os.path.getsize(path)
        mtime = os.path.getmtime(path)
        entry = {"size": size, "mtime": mtime}
        if size == 0:
            entry["error"] = "empty"
        if before.get(suffix) is not None and mtime <= before[suffix]:
            entry["error"] = "stale"
        status[suffix] = entry
    return status


def selftest():
    content = [
        "data: {\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n",
        "data: [DONE]\n",
    ]
    ttft, error = parse_sse_data(content, time.time())
    assert ttft is not None and error is None, (ttft, error)
    failure = [
        "data: {\"error\":{\"message\":\"boom\"}}\n",
        "data: [DONE]\n",
    ]
    ttft, error = parse_sse_data(failure, time.time())
    assert ttft is None and error == "boom", (ttft, error)
    empty = ["data: {\"choices\":[{\"delta\":{\"content\":\"\"}}]}\n"]
    ttft, error = parse_sse_data(empty, time.time())
    assert ttft is None and error is None, (ttft, error)
    import tempfile

    prefix = os.path.join(tempfile.gettempdir(), "vision_c4_dump_selftest")
    for suffix in (".bin", ".json"):
        with open(prefix + suffix, "wb") as handle:
            handle.write(b"x")
    status = dump_status(prefix, {})
    assert all("error" not in entry for entry in status.values()), status
    stale = {suffix: os.path.getmtime(prefix + suffix) + 1 for suffix in (".bin", ".json")}
    status = dump_status(prefix, stale)
    assert any(entry.get("error") == "stale" for entry in status.values()), status
    for suffix in (".bin", ".json"):
        os.remove(prefix + suffix)
    value, error = parse_doctor_vram("host\nvram: 1.50 GiB free / 24.00 GiB total\n")
    assert value == "mach-server doctor vram: 1.50 GiB free / 24.00 GiB total"
    assert error is None
    value, error = parse_doctor_vram("vram: mem_info error (boom)\n")
    assert value is None, value
    assert error == "vram: mem_info error (boom)", error
    assert parse_doctor_vram("no vram line here") == (None, None)
    os.environ["MACH_HIP_PATH"] = r"C:\fake\rocm"
    os.environ["MACH_MODEL"] = "should-be-dropped"
    try:
        assert doctor_env()["MACH_HIP_PATH"] == r"C:\fake\rocm"
        assert "MACH_MODEL" not in doctor_env()
    finally:
        del os.environ["MACH_HIP_PATH"]
        del os.environ["MACH_MODEL"]
    print("vision_c4_e2e selftest ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="target/release/mach-server.exe")
    parser.add_argument("--model-dir", default=".models/qwen3.8-27b")
    parser.add_argument("--image", default="")
    parser.add_argument("--out-dir", default="artifacts/vision-c4")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--max-patches", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--extra-env", action="append", default=[])
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0


    problems = preflight_problems(args)
    if args.dry_run:
        probe = sample_vram(args.binary)
        plan = {
            "binary": args.binary,
            "model_dir": args.model_dir,
            "image": args.image or "(built-in tiny png)",
            "port": args.port,
            "max_patches": args.max_patches,
            "out_dir": os.path.abspath(args.out_dir),
            "vram_probe": probe,
            "problems": problems,
        }
        print(json.dumps(plan, indent=2))
        return 1 if problems or probe["value"] is None else 0
    if problems:
        print(json.dumps({"problems": problems}, indent=2), file=sys.stderr)
        return 1

    if args.image:
        with open(args.image, "rb") as handle:
            data = handle.read()
        mime = "image/png" if args.image.lower().endswith(".png") else "image/jpeg"
    else:
        data = tiny_png()
        mime = "image/png"
    image_sha256 = hashlib.sha256(data).hexdigest()
    data_url = "data:" + mime + ";base64," + base64.b64encode(data).decode()
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image in one sentence."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": args.max_new,
        "temperature": 0,
        "stream": not args.no_stream,
    }

    os.makedirs(args.out_dir, exist_ok=True)

    model_dir = os.path.normpath(args.model_dir)
    env = dict(os.environ)
    env["MACH_VISION"] = "1"
    env["MACH_VISION_MAX_TOKENS"] = str(args.max_patches)
    env["MACH_MODEL"] = os.path.basename(model_dir)
    env["MACH_MODELS"] = os.path.dirname(model_dir) or "."
    env["MACH_ADDR"] = "127.0.0.1:" + str(args.port)
    for item in args.extra_env:
        key, _, value = item.partition("=")
        env[key] = value

    log_path = os.path.join(args.out_dir, "server.log")
    with open(log_path, "wb") as log:
        started = time.time()
        proc = subprocess.Popen(
            [args.binary], env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            base = "http://127.0.0.1:" + str(args.port)
            if not wait_health(base, args.timeout):
                print("server did not become healthy; see " + log_path, file=sys.stderr)
                return 2
            vram_before = sample_vram(args.binary)
            request_started = time.time()
            req = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps(request).encode(),
                headers={"content-type": "application/json"},
            )
            dump_prefix = env.get("MACH_VISION_DUMP")
            dump_before = {}
            if dump_prefix:
                for suffix in (".bin", ".json"):
                    path = dump_prefix + suffix
                    if os.path.exists(path):
                        dump_before[suffix] = os.path.getmtime(path)
            first_data = None
            stream_error = None
            try:
                with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                    status = resp.status
                    if args.no_stream:
                        body = resp.read().decode("utf-8", "replace")
                    else:
                        parts = []
                        for raw_line in resp:
                            parts.append(raw_line.decode("utf-8", "replace"))
                        body = "".join(parts)
                        first_data, stream_error = parse_sse_data(parts, request_started)
            except urllib.error.HTTPError as exc:
                status = exc.code
                body = exc.read().decode("utf-8", "replace")
            elapsed = time.time() - request_started
            dump_files = dump_status(dump_prefix, dump_before) if dump_prefix else None
            dump_error = None
            if dump_files:
                for entry in dump_files.values():
                    if entry.get("error"):
                        dump_error = entry["error"]
                        break
            vram_after = sample_vram(args.binary)
            vram_ok = (
                vram_before["value"] is not None and vram_after["value"] is not None
            )
            summary = {
                "status": status,
                "vram_ok": vram_ok,
                "elapsed_seconds": elapsed,
                "ttft_seconds": first_data,
                "load_plus_wait_seconds": request_started - started,
                "stream": not args.no_stream,
                "stream_error": stream_error,
                "image_bytes": len(data),
                "image_sha256": image_sha256,
                "vram_before": vram_before,
                "vram_after": vram_after,
                "dump_prefix": dump_prefix,
                "dump_files": dump_files,
                "dump_error": dump_error,
                "response": body,
            }
            with open(os.path.join(args.out_dir, "response.json"), "w", encoding="utf-8") as out:
                out.write(body)
            with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as out:
                json.dump(summary, out, indent=2)
            print(json.dumps(summary, indent=2))
            return (
                0
                if status == 200
                and stream_error is None
                and dump_error is None
                and vram_ok
                else 1
            )
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    sys.exit(main())
