"""Measure matched HTTP endpoints and create blinded, loudness-matched clips.

Outputs belong outside the repository: generated voice audio and answer keys
are private trial data. This script never changes VoiceServer routing.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
from importlib.util import find_spec
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
from urllib.parse import urlsplit
import wave

import numpy as np

LINES = {
    "short": "The nightly backup finished cleanly. Every scheduled job completed without errors.",
    "longer": "Before we move on, here is the summary. The build finished successfully, all the tests passed, and the backup completed without errors. The next scheduled check will run tomorrow morning.",
}
LISTENING = [
    ("A short update", "Good morning. The backup finished successfully, and every scheduled job completed without errors."),
    ("A little reassurance", "There is no need to rush this decision. We can look at the evidence together, take a breath, and choose the next step when you are ready."),
    ("A question", "The result is close, but I want to check one thing before we commit. Does that explanation match what you expected?"),
    ("A longer thought", "When several people are speaking, a familiar voice should remain clear, natural, and easy to recognize. Please listen to the ending of this sentence as well as the beginning."),
]


def health(base):
    url = urlsplit(base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
    conn.request("GET", "/health")
    r = conn.getresponse()
    data = json.loads(r.read())
    conn.close()
    if r.status != 200:
        raise RuntimeError(f"unhealthy endpoint: {base}")
    return data


def post(base, body, cancel=False):
    url = urlsplit(base)
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=60)
    started = time.perf_counter()
    conn.request("POST", "/generate", json.dumps(body), {"Content-Type": "application/json"})
    response = conn.getresponse()
    first = None
    data = bytearray()
    while True:
        chunk = response.read1(65536)
        if not chunk:
            break
        if first is None:
            first = time.perf_counter() - started
        data.extend(chunk)
        if cancel:
            break
    elapsed = time.perf_counter() - started
    headers = dict(response.getheaders())
    headers = {k.lower(): v for k, v in headers.items()}
    status = response.status
    conn.close()
    duration = len(data) / 2 / int(headers.get("x-sample-rate", "24000"))
    return {"status": status, "headers": headers, "pcm": bytes(data),
            "first_ms": None if first is None else first * 1000,
            "total_s": elapsed, "audio_s": duration, "speed_x": duration / elapsed}


def metrics(result):
    return {k: result[k] for k in ("first_ms", "total_s", "audio_s", "speed_x")}


def longest_silence(pcm, rate=24000):
    best = run = 0
    for sample in np.frombuffer(pcm, dtype="<i2"):
        run = run + 1 if sample == 0 else 0
        best = max(best, run)
    return best / rate


def rss_mb(base):
    port = urlsplit(base).port
    pid = subprocess.run(["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.split()
    if not pid:
        return None
    value = subprocess.check_output(["ps", "-o", "rss=", "-p", pid[0]], text=True).strip()
    return round(int(value) / 1024, 1)


def contract(base, voice):
    common = {"voice": voice, "temperature": .7}
    unknown = post(base, {**common, "voice": "unknown-trial-voice", "text": "Hello.", "stream": False})
    if unknown["status"] != 404:
        raise RuntimeError("unknown voice was not rejected")
    pause = post(base, {**common, "text": "One.[1.5s]Two.", "stream": False})
    if pause["status"] != 200 or longest_silence(pause["pcm"]) < 1.45:
        raise RuntimeError("explicit pause marker failed")
    normal = post(base, {**common, "text": LINES["short"], "stream": False, "targetRmsDb": -20})
    if normal["status"] != 200 or normal["headers"].get("x-sample-rate") != "24000" or normal["headers"].get("x-channels") != "1":
        raise RuntimeError("PCM contract failed")
    samples = np.frombuffer(normal["pcm"], dtype="<i2").astype(float) / 32768
    rms = float(20 * np.log10(np.sqrt(np.mean(samples**2))))
    if abs(rms + 20) > .5:
        raise RuntimeError(f"RMS normalization failed: {rms}")
    return {"unknown_voice_404": True, "pause_seconds": longest_silence(pause["pcm"]), "rms_dbfs": round(rms, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="http://127.0.0.1:8892")
    parser.add_argument("--candidate", default="http://127.0.0.1:8893")
    parser.add_argument("--voice", required=True)
    parser.add_argument("--voice-state", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    args.output.chmod(0o700)
    endpoints = {"resident_pytorch": args.reference, "candidate_mlx": args.candidate}
    initial = {name: health(base) for name, base in endpoints.items()}
    assert initial["resident_pytorch"]["model"] == initial["candidate_mlx"]["model"]
    import yaml
    config_name = initial["candidate_mlx"]["model"]
    candidate_path = Path(find_spec("pocket_tts_mlx").origin).parent / "config" / f"{config_name}.yaml"
    candidate_config = yaml.safe_load(candidate_path.read_text())
    reference_config = yaml.safe_load(args.reference_config.read_text())
    pin_check = {
        "weights": candidate_config["weights_path"] == reference_config["weights_path"],
        "public_weights": candidate_config["weights_path_without_voice_cloning"] == reference_config["weights_path_without_voice_cloning"],
        "tokenizer": candidate_config["flow_lm"]["lookup_table"]["tokenizer_path"] == reference_config["flow_lm"]["lookup_table"]["tokenizer_path"],
    }
    if not all(pin_check.values()):
        raise RuntimeError(f"model/tokenizer pins differ: {pin_check}")
    report = {"method": "same checkpoint, tokenizer pins, voice state, temperature 0.7; sequential localhost HTTP; warm models; independent stochastic draws",
              "health_before": initial, "pin_check": pin_check,
              "voice_state_sha256": hashlib.sha256(args.voice_state.read_bytes()).hexdigest(),
              "runs_per_case": args.runs, "contracts": {}, "benchmarks": {}}
    for name, base in endpoints.items():
        report["contracts"][name] = contract(base, args.voice)
        print(json.dumps({"contract": name, "passed": True}), flush=True)
        post(base, {"text": LINES["short"], "voice": args.voice, "stream": True, "temperature": .7})

    for case, text in LINES.items():
        for streaming in (True, False):
            name = f"{case}_{'streaming' if streaming else 'buffered'}"
            raw = {engine: [] for engine in endpoints}
            for index in range(args.runs):
                order = list(endpoints)
                if index % 2:
                    order.reverse()
                for engine in order:
                    result = post(endpoints[engine], {"text": text, "voice": args.voice, "stream": streaming, "temperature": .7})
                    if result["status"] != 200 or not result["pcm"]:
                        raise RuntimeError(f"generation failed: {engine}/{name}")
                    raw[engine].append(metrics(result))
            medians = {engine: {key: round(statistics.median(r[key] for r in rows), 3) for key in rows[0]} for engine, rows in raw.items()}
            report["benchmarks"][name] = {"text": text, "raw": raw, "medians": medians}
            print(json.dumps({"case": name, "medians": medians}), flush=True)

    before_cancel = health(args.candidate)
    post(args.candidate, {"text": " ".join([LINES["longer"]] * 8), "voice": args.voice, "stream": True}, cancel=True)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        after_cancel = health(args.candidate)
        if after_cancel["active"] == 0 and after_cancel["cancelled"] > before_cancel["cancelled"]:
            break
        time.sleep(.05)
    else:
        raise RuntimeError("candidate kept generating after client disconnect")
    report["cancellation"] = {"passed": True, "active_after_disconnect": after_cancel["active"]}
    report["rss_mb"] = {name: rss_mb(base) for name, base in endpoints.items()}

    roles = ["candidate_mlx", "resident_pytorch"] * 2
    random.SystemRandom().shuffle(roles)
    pairs, answer_key = [], {}
    trial_id = f"pocket-{time.time_ns()}"
    for index, ((title, text), a_engine) in enumerate(zip(LISTENING, roles), 1):
        pair_id = f"pair-{index:02d}"
        assignment = {"A": a_engine, "B": next(name for name in endpoints if name != a_engine)}
        pair = {"id": pair_id, "trial": trial_id, "title": title, "text": text}
        for label, engine in assignment.items():
            result = post(endpoints[engine], {"text": text, "voice": args.voice, "temperature": .7, "stream": False, "targetRmsDb": -20})
            if result["status"] != 200 or not result["pcm"]:
                raise RuntimeError(f"listening generation failed: {engine}/{pair_id}")
            filename = f"{pair_id}-{label}.wav"
            with wave.open(str(args.output / filename), "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(24000)
                out.writeframes(result["pcm"])
            (args.output / filename).chmod(0o600)
            pair[label] = filename
        answer_key[pair_id] = assignment
        pairs.append(pair)
    for filename, data in (("pairs.json", pairs), ("answer-key.json", answer_key)):
        path = args.output / filename
        path.write_text(json.dumps(data, indent=2) + "\n")
        path.chmod(0o600)
    report["health_after"] = {name: health(base) for name, base in endpoints.items()}
    (args.output / "benchmarks.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": True, "listening_pairs": len(pairs), "answer_key": "kept private; not served by the listening page", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
