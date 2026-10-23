---
status: complete
type: analysis
tags: mlx, pocket-tts, inference, benchmark
created: 2026-10-23
updated: 2026-10-23
---

# Analysis 002: Resident HTTP trial

The warm local HTTP performance gate passed. The owner's four blind choices
split 2–2 between engines. The owner then requested promotion, and Nova's
production resident path was upgraded to MLX on 2026-10-23. This trial does not
establish general sound-quality superiority.

## Method and scope

- Apple M2 Pro; reference is the existing resident PyTorch CPU server on
  `127.0.0.1:8892`, candidate is an isolated MLX GPU process on port `8893`.
- Both use `english_2026-04`, the same weight/tokenizer pins and local voice
  state, temperature 0.7, and the same HTTP requests. Each backend samples
  independently. Five runs per case alternate backend order after warmup.
- Streaming uses decoder batch size 1. The candidate's buffered path uses 4.
- Harness: `tools/http_trial/`; reproduction: `koder/docs/http-trial.md`.
  Product base is `67f769a`; trial work is on `trial/resident-http`.
- No launchd registration, production routing, or production environment changes.
  Optional dependencies were installed only into this repo's `.venv`.

## Measured medians

| Case | PyTorch first body | MLX first body | PyTorch total | MLX total | PyTorch audio/wall | MLX audio/wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short streaming | 81 ms | 25 ms | 1.036 s | 0.785 s | 5.47x | 7.09x |
| Longer streaming | 83 ms | 37 ms | 2.062 s | 1.462 s | 5.39x | 7.60x |
| Short buffered | 996 ms | 581 ms | 0.996 s | 0.581 s | 5.30x | 9.57x |
| Longer buffered | 2059 ms | 1213 ms | 2.060 s | 1.213 s | 5.54x | 9.12x |

These are warm localhost timings, including the HTTP boundary. First body
timing is not speaker playback latency. Generated durations differ slightly
with stochastic sampling, so audio/wall ratios are also reported. Buffered
generation took about 41% less wall time in these cases; streaming about
24–29% less. Five repetitions and two texts do not establish tail latency.

Process RSS was 612.4 MiB for the resident and 907.8 MiB for the candidate.
This does not fully account for GPU or shared-system memory. Cold startup,
concurrent requests, sustained operation, and competition with an LLM/GPU
workload were not benchmarked.

## Verified behavior

- Both endpoints returned 24 kHz mono signed 16-bit PCM with the expected
  headers, rejected unknown voices with 404, honored a 1.5-second pause,
  and reached the -20 dBFS full-clip RMS target.
- The candidate stopped generation after a real client disconnect; its
  active counter returned to zero and the cancellation counter advanced.
- Four pairs were generated without preference selection, with balanced
  randomized A/B assignment. All eight WAVs are 24 kHz mono, with measured
  RMS 0.1 and peaks below 0.70. The answer key remained unread until the owner
  saved all four choices.
- The listening page rendered all four pairs and eight players in Brave;
  playback requires a click and saving is disabled until all pairs are chosen.
- `python -m pytest -q`: 142 passed, 15 opt-in model tests skipped,
  four subtests passed. The optional TestClient dependency emits one upstream
  Starlette/httpx deprecation warning.
- The isolated sdist/wheel build passed. Archive audits confirmed no trial
  tools, tests, operator files, voice states, WAVs, or bytecode are shipped.

## Production promotion

- Private PAI commit `d932092e` adds `Infrastructure/VoiceServer/pocket-server/mlx_server.py`,
  its tests, installer, launchd template, and deployment/rollback documentation.
  Only selected VoiceServer paths were committed; unrelated PAI edits remain.
- Installed package: `0.3.1.dev0`, source `67f769a`, from the audited wheel.
  A dedicated internal-SSD venv has 37 compatible packages and no Torch;
  package imports and offline April weights do not depend on external volumes.
- `com.pai.pocket-mlx-server` serves loopback port 8894, with Nova pinned and
  a bounded cache of 16 other states. Both decoder paths warm before readiness.
  The installed gateway LaunchAgent sets `POCKET_RESIDENT_URL` to port 8894.
- The April model, voice states, temperature 0.7, and resident voice allowlist
  are retained. Other voices continue on Core ML. PyTorch stays warm on 8892
  for manual rollback; Core ML remains the automatic standby.
- Five Python service tests and six Bun routing/fallback tests passed.
  Live checks covered PCM headers, pause markers, -20 dBFS leveling, a second
  voice, three simultaneous requests, a real client disconnect, and an
  eight-second synthetic GPU matrix workload. No failed/active requests remained.
- Under synthetic GPU pressure, the streaming probe returned first body in
  74 ms and produced 5.04 seconds of audio in 2.993 seconds. A separate idle
  first-use probe took 392 ms to first body; the earlier warm medians are not
  a latency bound. Sustained operation with an LLM remains unmeasured.
- An initial immediate gateway bootstrap was rejected by launchd. The saved
  original plist restored PyTorch; waiting for job removal before bootstrap
  completed the cutover successfully. The installer/rollback instructions
  include this wait.
- A fresh default-engine production `/notify` request, with cache bypass and
  playback skipped, produced 140,160 valid mono 24 kHz PCM samples in a WAV.
  The MLX completed counter advanced from 10 to 11; both the gateway and
  candidate were healthy, with zero MLX failures. PyTorch remained ready.

## Local continuation and rollback

The temporary trial on port 8893 was stopped after verifying its PID/command;
its private clips and votes are retained outside Git. The old locator at
`/tmp/pocket-mlx-http-trial-location.json` is marked stopped.

The persistent service's private receipt is
`~/Library/Application Support/pai/pocket-mlx-server/deployment.json`.
It points to the original gateway plist, wheel and hash, dependency snapshot,
and verification records. These files, WAVs, votes, state hashes, and answer
keys must stay outside Git.

For rollback, probe the retained PyTorch on port 8892, change the installed
gateway's `POCKET_RESIDENT_URL` back to `http://127.0.0.1:8892`, then unload
and reload the gateway with the documented wait. Exact commands are in PAI's
`Infrastructure/VoiceServer/pocket-server/README.md`. Preserve this environment
override when reinstalling the gateway; its source default intentionally stays
on port 8892. The MLX service can remain available while rollback is verified.
