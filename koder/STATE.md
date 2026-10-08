---
updated_at: "08 Oct 2026 | 04:23 PM IST"
---

# Koder State

## Past

- v0.2.1 (`b1fe6a4`) is the published MLX port of the January 2026 model. It is on PyPI as `0.2.1`; remote tag `v0.2.1` → `1342d4d`.
- Koder scaffold: `0d1e841`. `CHANGELOG.md` stays the history surface.
- `d37c30b`: analysis 001, issue 001 (track), plan 001 rev 2, and reviews 01 and 02. Review 02 checked published safetensors headers (no weight download) and the full upstream history at `41cbc84`.
- `59d499f`: plan 001 L0 is done. Repo-local `.venv` (gitignored), pytest `testpaths`, and weight-free tests pinning v0.2.1 text-prompt behavior.
- `6ef0d70`: plan 001 L1 is done. The config schema matches upstream `41cbc84`. Nineteen language YAMLs are in `pocket_tts_mlx/config/`. `b6369a24` sets `pad_with_spaces_for_short_inputs: true`.

## Present

- Active plan: `koder/plans/001_update_pocket_tts_mlx/INDEX.md` (rev 2). Ledger: `koder/issues/001_update_to_current_pocket_tts/INDEX.md`. L0 and L1 are `done`; L2–L8 are `candidate`.
- Facts the plan rests on:
  - shipped bundles are already fused;
  - `b6369a24` matches the MLX tree on all 213 keys;
  - voice states carry `offset`/`pad`, which this port maps to `current_end`;
  - upstream switched to tanh GELU in `2dd548a`.
- Generation code is still the v0.2.1 sampler and exact GELU. The new schema fields parse and are unused.
- Validation: `.venv/bin/python -m pytest -q` → 9 passed. `english` and `english_2026-09` share weights revision `983151f13aaeab1b13c1e5e3c2c383d49a9edf3f` on different URLs.
- Local `main` is 6 commits ahead of `origin/main` after this close. Nothing pushed.

## Future

- Next action: L2. Dispatch `lsd`, `flow_matching`, and `drifting` the way `pocket_tts/models/flow_lm.py` does, and replace backbone `nn.gelu` with the tanh formula in `modules/mimi_transformer.py`. One commit. Stop the layer if a unit test fails.
- Then L3 → L7, one commit per layer with a clean tree between layers. The owner already authorized the build.
- Owner decisions still pending: push, tag `v0.3.0`, PyPI publish (L8). Renaming the `Untitled/` folder is optional.
- Use `.venv` only. `a separate kyutai venv` imports a different checkout.
