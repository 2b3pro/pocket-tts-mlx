---
updated_at: "08 Oct 2026 | 05:45 PM +08"
---

# Koder State

## Past

- v0.2.1 (`b1fe6a4`) is the published MLX port of the January 2026 model. It is on PyPI as `0.2.1`; remote tag `v0.2.1` → `1342d4d`.
- Koder scaffold committed as `0d1e841`. `CHANGELOG.md` stays the history surface.
- Analysis 001, plan 001 and reviews 01 and 02 are committed. Review 02 checked published safetensors headers (no weight download) and the full upstream history at `41cbc84`. Plan 001 is now revision 2.

## Present

- Plan 001 rev 2: L0 test scaffold → L1 config → L2 flow head and tanh GELU → L3 Mimi dims → L4 text → L5 generation → L6 strict load and voice-state import → L7 API/CLI/docs/version → L8 owner-gated release prep.
- Facts the plan rests on:
  - shipped bundles are already fused;
  - `b6369a24` matches the MLX tree on all 213 keys;
  - voice states carry `offset`/`pad`, which this port maps to `current_end`.
- `pocket_tts_mlx/` is untouched. Local `main` is ahead of `origin/main`. Nothing pushed.

## Future

- Next: L0, then L1, from `koder/plans/001_update_pocket_tts_mlx/INDEX.md`, one commit per layer with a clean tree between layers. Build only when the owner asks.
- Use a repo-local `.venv`. `a separate kyutai venv` imports a different checkout.
- Push, tag and PyPI publish wait for the owner (L8).
