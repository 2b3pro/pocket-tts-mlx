---
updated_at: "23 Oct 2026 | 09:36 AM IST"
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
  and is pushed to `2b3pro/pocket-tts-mlx`.
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

## Future

- No known source-alignment work remains for these pinned snapshots.
- Review future upstream changes deliberately. Model releases do not propagate
  automatically to this independent port.
- Publishing a package or upgrading a deployed VoiceServer is a separate task.
