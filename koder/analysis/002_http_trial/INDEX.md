---
status: in_progress
type: analysis
tags: mlx, pocket-tts, inference, benchmark
created: 2026-10-23
updated: 2026-10-23
---

# Analysis 002: Resident HTTP trial

The warm local HTTP performance gate passed. The owner's blind listening
evaluation is pending; sound quality and production readiness are not decided.

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
  RMS 0.1 and peaks below 0.70. The answer key has not been read.
- The listening page rendered all four pairs and eight players in Brave;
  playback requires a click and saving is disabled until all pairs are chosen.
- `python -m pytest -q`: 142 passed, 15 opt-in model tests skipped,
  four subtests passed. The optional TestClient dependency emits one upstream
  Starlette/httpx deprecation warning.
- The isolated sdist/wheel build passed. Archive audits confirmed no trial
  tools, tests, operator files, voice states, WAVs, or bytecode are shipped.

## Local continuation

The trial is available at `http://127.0.0.1:8893/trial`. Runtime files live
outside Git under the local application-support trial directory; the locator
is `/tmp/pocket-mlx-http-trial-location.json`. It records the root, PID, and
command. Private WAVs, votes, state hashes, answer keys, and raw benchmark data
must stay outside Git.

Wait for the owner to save preferences, then read `votes.json` and the private
answer key to report the comparison. A small listening trial cannot establish
general quality superiority. If the owner chooses to continue, test the
deployment's real concurrency and GPU contention before proposing promotion.
The listening trial does not itself authorize production routing changes.
Stop the recorded process when the evaluation is finished; re-check PID and
command first so an unrelated process is never stopped after PID reuse.
