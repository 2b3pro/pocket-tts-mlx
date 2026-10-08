---
status: open
priority: P1
created: 2026-10-08
updated: 2026-10-08
tags: mlx, pocket-tts, inference
type: feature
issue_kind: track
context: Plan 001 rev 2 passed two reviews. Slices stay candidate until the owner asks to build.
---

# Issue 001: Update the MLX runtime to current Pocket TTS inference

## Problem

`pocket-tts-mlx` 0.2.1 loads `b6369a24` at temperature 0.7 with SentencePiece and an LSD-only sampler. Current Pocket TTS (`41cbc84`, package 3.3.0) ships `english` as the `english_2026-09` weights, `tokenizer.json`, a 32-wide Mimi inner dimension, BOS-before-voice, and generation fixes that this port does not implement. Loading an upstream YAML with the current schema fails because unknown fields are forbidden.

## Context

- Analysis: `koder/analysis/001_mlx_and_upstream/INDEX.md`
- Plan: `koder/plans/001_update_pocket_tts_mlx/INDEX.md` (`status: approved`, revision 2)
- Local product commit before the koder scaffold: `b1fe6a4`
- Upstream read at `41cbc84`. Training, the HTTP server, Docker, and torchao are out of scope.
- Review `koder/reviews/001_update_pocket_tts_mlx/01_plan.md` is `approve with fixes`. Its three fixes are in the plan.
- Review `koder/reviews/001_update_pocket_tts_mlx/02_plan_rereview.md` is `approve with fixes` (1 P1, 6 P2, 4 P3). It checked published safetensors headers and full upstream history, and its fixes are in plan revision 2.
- Kind is `track`. No slice is queued. Do not build until the owner asks.

## Proposed Direction

Follow `koder/plans/001_update_pocket_tts_mlx/INDEX.md` only after its review verdict is `approve` or `approve with fixes` and the owner asks to build. Keep `config/b6369a24.yaml` as the v0.2.1 regression. Do not treat it as a rename of `english_2026-01`.

## Acceptance Criteria

- [x] The analysis and the plan agree on speaker width, tokenizer kind, voice-state import, and the bundle weight path.
- [x] A plan review exists and names a verdict (`approve with fixes`).
- [x] A second review checked the plan against published weight and voice headers. Its fixes are applied.
- [x] No `pocket_tts_mlx/` edit was made to reach that verdict.

The track stays open until the slice ledger is implemented. Rows stay `candidate`.

## Slice Ledger

Rows stay `candidate` until a later build is authorized. The mapping work (analysis, plan, review) is done. The track closes when the ledger is implemented, not at this review.

| Slice | Status | Ref | Queue | Closure gate |
| --- | --- | --- | --- | --- |
| Test scaffold and local env | candidate | plan 001 L0 | | `pytest -q` in repo `.venv` |
| Config schema and language YAMLs | candidate | plan 001 L1 | | YAML parse test |
| Flow head, sampler, tanh GELU | candidate | plan 001 L2 | | Unit test, no weights |
| Mimi inner/outer dims and latent layout | candidate | plan 001 L3 | | Shape test plus old YAML still builds |
| Tokenizer and text chunking | candidate | plan 001 L4 | | Unit test, no weights |
| EOS, fade, end-on-pause | candidate | plan 001 L5 | | Unit test, no weights |
| Strict bundle load and voice-state import | candidate | plan 001 L6 | | Strict load, then short clips |
| API, CLI, docs, version | candidate | plan 001 L7 | | CLI help and a wav |
| Release preparation (owner-gated) | candidate | plan 001 L8 | | Wheel check and fresh-venv smoke |
