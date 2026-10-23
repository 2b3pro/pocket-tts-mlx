---
updated_at: "23 Oct 2026 | 07:52 PM IST"
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

- Package version: `0.3.1.dev0`, unreleased. Active branch: `main`.
- `trial/resident-http` was fast-forwarded into `main` through `b6df3b9`,
  including the aligned code, trial harness, and deployment handoff. `8f05873`
  adds the optional loopback HTTP comparison harness and blind listening page
  outside the package. The merge had no conflicts.
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
- PyPI was not changed. The production upgrade is recorded below.
- Scope, compatibility changes, evidence, and commands:
  `koder/docs/upstream-alignment.md`; user guidance: `README.md`.
- A matched April-model HTTP trial favors MLX on the M2 Pro: streaming first
  body 25–37 ms vs 81–83 ms; buffered wall time about 41% lower. Process RSS
  is higher (908 vs 612 MiB). Synthetic GPU contention was checked during
  deployment; sustained LLM coexistence remains unmeasured.
- The owner's four blind choices split 2–2; the owner then requested the
  upgrade. Nova now uses the persistent MLX resident on `127.0.0.1:8894`.
- Private PAI commit `d932092e` adds the resident adapter, tests, installer,
  and launchd template; it is pushed to private `origin/main`.
  An installed wheel and isolated venv live on the
  internal SSD; no Torch or external source imports are used by that job.
- Gateway launchd sets `POCKET_RESIDENT_URL=http://127.0.0.1:8894`. April
  model, temperature 0.7, voice state, and Nova allowlist are preserved.
  PyTorch remains warm on 8892 for rollback; Core ML remains automatic standby.
- Five new Python service tests and six Bun routing/fallback tests passed.
  Real concurrency, disconnect, PCM/pause/RMS, and synthetic GPU checks passed;
  a fresh production `/notify` request advanced MLX's counter and saved valid
  audio. Sustained LLM coexistence is unmeasured; idle/GPU load can add latency.
- The port 8893 trial process was stopped; private clips and votes remain.
  Persistent service receipt/rollback preimage: local application support
  `pai/pocket-mlx-server/deployment.json`. Exact rollback: PAI's
  `Infrastructure/VoiceServer/pocket-server/README.md`.
- Merged-main validation: 142 tests passed, 15 opt-in tests skipped; real HTTP
  contract/disconnect checks passed; sdist/wheel and archive audits passed.
- Trial results/limits: `koder/analysis/002_http_trial/INDEX.md`;
  repeatable commands: `koder/docs/http-trial.md`.

## Future

- No known source-alignment work remains for these pinned snapshots.
- Observe the deployed Nova path under normal use. Roll back by restoring
  the gateway URL to port 8892 and reloading it with the documented wait.
- Review future upstream changes deliberately. Model releases do not propagate
  automatically to this independent port.
- Publishing a package remains a separate task. Unrelated PAI worktree edits
  were preserved; only the committed deployment change was pushed there.
