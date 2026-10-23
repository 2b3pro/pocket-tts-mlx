---
updated_at: "23 Oct 2026 | 07:17 PM IST"
---

# Koder State

## Past

- The fork's July/August improvements and local `fix/upstream-parity-v3` work
  (`3f841ed`, `d0db206`) are integrated into `main`.
- `ae36b64` merges MLX upstream `8dc7a1f` (v0.3.0) and ports relevant Kyutai
  inference changes through `44cb63e` (October 22, after v4.0.1).
- The adjacent PyTorch checkout has an unrelated unfinished merge; it was
  left untouched. Comparison used a temporary clean Kyutai clone.

## Present

- Package version: `0.3.1.dev0`, unreleased. `main` contains the integrated code
  at `67f769a` and is pushed to `2b3pro/pocket-tts-mlx`.
- Active local branch: `trial/resident-http`. `8f05873` adds an optional
  loopback HTTP comparison harness and blind listening page outside the package.
- Default: October English at temperature 0.3. There are 28 shipped configs
  (all 27 current Kyutai configs plus the original `b6369a24`) and 27 voices.
- Seven languages, 24-layer previews, drifting, and six small English models
  are supported. Fork dictionaries, seeded sampling, recovery, fused kernels,
  bounded caches, and batched decoding are retained.
- Batched Mimi attention now retains keys needed before ring eviction;
  in-place missing-EOS retries restore prompt state. Stop events cancel streams.
- Local validation: 132 unit tests; 15 real-model checks including reference
  cloning, reproducibility, and batched full-size/small-model generation.
- Dependency checks, compilation, sdist/wheel and distribution audits passed.
  The installed wheel generated 61,440 finite 24 kHz samples through the small
  model CLI with batched decoding and no Torch import (validated dependencies).
- Hosted macOS 15 ARM CI passed for `ae36b64`: GitHub Actions run
  `39898393321` (Tests / pytest).
- The production VoiceServer environment and PyPI were not changed.
- Scope, compatibility changes, evidence, and commands:
  `koder/docs/upstream-alignment.md`; user guidance: `README.md`.
- A matched April-model HTTP trial favors MLX on the M2 Pro: streaming first
  body 25–37 ms vs 81–83 ms; buffered wall time about 41% lower. Process RSS
  is higher (908 vs 612 MiB); GPU contention and sustained load are untested.
- Four blinded, loudness-matched voice pairs are ready at
  `http://127.0.0.1:8893/trial`. The page is open in Brave. Owner votes are
  pending and the answer key remains unread. Production stays on PyTorch.
- The isolated candidate process remains running for listening. Private
  runtime/artifact locator: `/tmp/pocket-mlx-http-trial-location.json`.
  Verify the recorded PID/command before stopping it after evaluation.
- Trial validation: 142 tests passed, 15 opt-in tests skipped; real HTTP
  contract/disconnect checks passed; sdist/wheel and archive audits passed.
- Trial results/limits: `koder/analysis/002_http_trial/INDEX.md`;
  repeatable commands: `koder/docs/http-trial.md`. No trial branch push yet.

## Future

- No known source-alignment work remains for these pinned snapshots.
- After the owner saves listening preferences, compare votes with the private
  answer key. A production change needs an explicit deployment decision and
  realistic concurrency/GPU contention checks.
- Review future upstream changes deliberately. Model releases do not propagate
  automatically to this independent port.
- Publishing a package or upgrading a deployed VoiceServer is a separate task.
