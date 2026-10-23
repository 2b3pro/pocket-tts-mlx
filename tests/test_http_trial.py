"""HTTP trial contract, owner-thread behavior, and blind-data boundaries."""

import importlib.util
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="optional HTTP trial requirements are not installed")
from fastapi.testclient import TestClient

SERVER_PATH = Path(__file__).parents[1] / "tools/http_trial/server.py"
spec = importlib.util.spec_from_file_location("http_trial_server", SERVER_PATH)
server = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server
spec.loader.exec_module(server)


@pytest.fixture
def trial(tmp_path):
    (tmp_path / "nova.safetensors").touch()
    seen = []

    class Model:
        sample_rate = 24_000

        def __init__(self):
            self.owner = threading.get_ident()

        def get_state_for_audio_prompt(self, path):
            assert threading.get_ident() == self.owner
            return {}

        def generate_audio_stream(self, state, text, **kwargs):
            assert threading.get_ident() == self.owner
            seen.append((text, kwargs))
            yield np.full(2400, .01, dtype=np.float32)
            yield np.full(2400, -.01, dtype=np.float32)

    app = server.create_app(server.Settings(tmp_path, tmp_path, voice="nova"), lambda *a, **k: Model())
    with TestClient(app) as client:
        yield client, seen, tmp_path


def test_raw_pcm_headers_and_owner_thread_for_streaming_and_buffered(trial):
    client, seen, _ = trial
    for streaming, batch in ((True, 1), (False, 4)):
        r = client.post("/generate", json={"text": "Hello.", "voice": "nova", "stream": streaming})
        assert r.status_code == 200
        assert r.headers["x-sample-rate"] == "24000"
        assert r.headers["x-channels"] == "1"
        assert r.headers["x-model"] == "english_2026-04"
        assert len(r.content) == 4800 * 2
        assert seen[-1][1]["decode_batch_size"] == batch
    assert client.get("/health").json()["completed"] == 2


def test_unknown_voice_fails_instead_of_substitution(trial):
    client, seen, _ = trial
    assert client.post("/generate", json={"text": "Hello.", "voice": "missing"}).status_code == 404
    assert seen == []


@pytest.mark.parametrize("body", [{}, [], {"text": "[0s]", "voice": "nova"},
    {"text": "hi", "voice": "nova", "temperature": "nan"},
    {"text": "hi", "voice": "nova", "targetRmsDb": "invalid"}])
def test_invalid_requests_are_rejected(trial, body):
    assert trial[0].post("/generate", json=body).status_code == 400


def test_pause_markers_are_exact_silence_and_loudness_matches_target(trial):
    client, _, _ = trial
    r = client.post("/generate", json={"text": "One.[1.5s]Two.", "voice": "nova", "stream": False})
    samples = np.frombuffer(r.content, dtype="<i2")
    np.testing.assert_array_equal(samples[4800:4800+36000], 0)
    r = client.post("/generate", json={"text": "Hello.", "voice": "nova", "stream": False, "targetRmsDb": -20})
    samples = np.frombuffer(r.content, dtype="<i2").astype(float) / 32768
    assert abs(20 * np.log10(np.sqrt(np.mean(samples**2))) + 20) < .01


def test_blind_assets_never_expose_the_answer_key(trial):
    client, _, root = trial
    (root / "answer-key.json").write_text('{"private": true}')
    (root / "pairs.json").write_text(json.dumps([{"id": "pair-01"}]))
    assert client.get("/trial/audio/answer-key.json").status_code == 404
    assert client.get("/trial/answer-key.json").status_code == 404
    assert client.get("/trial/pairs.json").json() == [{"id": "pair-01"}]
    assert client.post("/trial/votes", json={"votes": {"pair-01": "A"}}).status_code == 200
    assert json.loads((root / "votes.json").read_text())["votes"] == {"pair-01": "A"}
    assert client.post("/trial/votes", json={"votes": {"pair-01": "other"}}).status_code == 400


def test_pcm_rounding_and_peak_ceiling():
    np.testing.assert_array_equal(np.frombuffer(server.pcm16(np.array([0,.5,-1,2])), dtype="<i2"), [0,16384,-32767,32767])
    out = server.level(np.array([0.,1.,0.]), -3)
    assert np.abs(out).max() <= .99
