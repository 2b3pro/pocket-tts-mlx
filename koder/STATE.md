---
updated_at: "08 Oct 2026 | 06:33 PM +08"
---

# Koder State

## Past

- v0.2.1 (`b1fe6a4`) is the published MLX port of the January 2026 model. It is on PyPI as `0.2.1`; remote tag `v0.2.1` → `1342d4d`.
- Koder scaffold: `0d1e841`. `CHANGELOG.md` stays the history surface.
- `d37c30b`: analysis 001, issue 001 (track), plan 001 rev 2, and reviews 01 and 02. Review 02 checked published safetensors headers (no weight download) and the full upstream history at `41cbc84`.
- `59d499f`: plan 001 L0 is done. Repo-local `.venv` (gitignored), pytest `testpaths`, and 7 weight-free tests in `tests/` pinning config parsing and v0.2.1 text-prompt behavior.

## Present

- Active plan: `koder/plans/001_update_pocket_tts_mlx/INDEX.md` (rev 2). Ledger: `koder/issues/001_update_to_current_pocket_tts/INDEX.md`. L0 is done; L1–L8 are `candidate`.
- Facts the plan rests on:
  - shipped bundles are already fused;
  - `b6369a24` matches the MLX tree on all 213 keys;
  - voice states carry `offset`/`pad`, which this port maps to `current_end`;
  - upstream switched to tanh GELU in `2dd548a`.
- `pocket_tts_mlx/` source is still untouched. Validation: `.venv/bin/python -m pytest -q` → 7 passed.
- Local `main` is 4 commits ahead of `origin/main` after this close. Nothing pushed.

## Future

- Next action: L1. Extend `utils/config.py` to upstream's schema, copy the 19 upstream YAMLs from `/tmp/pocket-tts-upstream/pocket_tts/config/` (re-clone at `41cbc84` if `/tmp` was cleared), add `pad_with_spaces_for_short_inputs: true` to `b6369a24.yaml`, and add tests. One commit.
- Then L2 → L7, one commit per layer with a clean tree between layers. The owner already authorized the build.
- Owner decisions still pending: push, tag `v0.3.0`, PyPI publish (L8). Renaming the `Untitled/` folder is optional.
- Use `.venv` only. `a separate kyutai venv` imports a different checkout.
