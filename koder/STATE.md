---
updated_at: "08 Oct 2026 | 02:20 PM IST"
---

# Koder State

## Past

- v0.2.1 (2026-02-11) is the published MLX port: onset cleanup, KV-cache sizing, and materialized `generate_audio()`.
- It tracks Pocket TTS around v1.0.3 (config `b6369a24`, temperature 0.7).

## Present

- Local `main` matches `origin/main` at `b1fe6a4`.
- Koder scaffold is in place. `CHANGELOG.md` remains the history surface.
- A chat sketch proposes catching up to current Pocket TTS. It is not checked against source and it is not authorized to build.

## Future

- Next: source analysis of this tree and current `kyutai-labs/pocket-tts` inference, then a draft plan and a plan review.
- Do not port architecture, weights, or generation until that review is `approve` or `approve with fixes` and a later request says to build.
