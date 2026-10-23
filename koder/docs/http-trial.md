# Local HTTP and listening trial

`tools/http_trial/` is an opt-in comparison harness. It is not installed as a
package server, registered with launchd, or connected to VoiceServer routing.
The listener binds to `127.0.0.1` only. Keep production on its current backend
until performance and the owner's listening evaluation justify a change.

## Run the candidate

```bash
uv pip install --python .venv/bin/python -r tools/http_trial/requirements.txt
.venv/bin/python tools/http_trial/server.py \
  --states /path/to/reference/states/english_2026-04 \
  --voice voice_token \
  --artifacts /path/outside/the/repository/trial \
  --port 8893
```

The selected voice must have a matching upstream-format safetensors state in
`--states`. The trial defaults to `english_2026-04`, temperature `0.7`, one
latent per streaming decoder call, and four for buffered generation. Loading
and generation share the same event-loop thread, serialized by an async lock,
so MLX's thread-local GPU streams are respected.

Endpoints mirror the resident server: `GET /health`, `POST /generate` with
`text`, `voice`, `stream`, optional `temperature` and `targetRmsDb`, returning
raw signed 16-bit little-endian mono PCM and `x-sample-rate`, `x-channels`,
`x-model` headers. Pause markers are supported; unknown voices return 404.
Closing a stream stops generation without a background decoder worker.

## Compare and prepare clips

```bash
.venv/bin/python tools/http_trial/compare.py \
  --reference http://127.0.0.1:8892 \
  --candidate http://127.0.0.1:8893 \
  --voice voice_token \
  --voice-state /path/to/reference/states/english_2026-04/voice_token.safetensors \
  --reference-config /path/to/resident/site-packages/pocket_tts/config/english_2026-04.yaml \
  --output /path/outside/the/repository/trial \
  --runs 5
```

The script checks model/tokenizer pins, the HTTP PCM contract, explicit pauses,
unknown-voice rejection, RMS leveling, and candidate disconnect cancellation.
Both models are warmed. Sequential HTTP requests alternate backend order;
short and longer texts are measured in streaming and buffered modes. The
report keeps individual timings and medians, generated durations, and process
RSS. RSS is not a complete accounting of GPU/shared-system memory. Concurrent
LLM/GPU workloads are not simulated. Each backend draws independent samples;
identical waveforms or random streams are not assumed.

It also creates four listening pairs, randomized and balanced between A and B,
with the same text, reference state, checkpoint, temperature, and -20 dBFS
full-clip RMS target (subject to the shared 0.99 peak ceiling). No clip is
selected by preference. Generated WAVs, per-machine results, and the answer key
stay outside Git. Do not commit them.

Open `http://127.0.0.1:8893/trial`. The page exposes generic filenames and
prompts only. It never serves `answer-key.json` or the labeled benchmark data.
Votes and notes are saved locally as `votes.json`; the answer key is read only
after the owner has voted. Nothing plays automatically and no production
routing is changed by submitting votes.

## Verification and completion

```bash
.venv/bin/python -m pytest -q tests/test_http_trial.py
```

The contract tests use a fake model and verify owner-thread execution, raw PCM,
request validation, voice rejection, loudness, pauses, and the blind asset/vote
boundary. The real comparison checks cancellation on a live HTTP disconnect.

The trial has two gates: the measured HTTP behavior and the owner's blind
preferences. Finishing the benchmark does not pass the listening gate.
Starting a trial never authorizes a production routing change. Stop the trial
process with its recorded PID once the listening evaluation is complete.
