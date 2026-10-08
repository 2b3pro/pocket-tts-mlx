---
updated_at: "08 Oct 2026 | 07:56 PM IST"
---

# Koder State

## Past

- v0.2.1 (`b1fe6a4`) is the published MLX port. PyPI `0.2.1`; remote tag `v0.2.1` → `1342d4d`.
- Koder scaffold: `0d1e841`. `CHANGELOG.md` stays the history surface.
- `d37c30b`: analysis 001, issue 001 (track), plan 001 rev 2, reviews 01 and 02. Upstream read at `41cbc84`.
- `59d499f` L0: repo `.venv`, pytest `testpaths`, weight-free v0.2.1 text pins.
- `6ef0d70` L1: config schema matches `41cbc84`. Nineteen language YAMLs. `b6369a24` keeps `pad_with_spaces_for_short_inputs: true`.
- `5f5528e` L2: `lsd`, `flow_matching`, and `drifting` samplers. Backbone FFN uses the tanh GELU formula.
- `fd7ef3e` L3: Mimi width follows `inner_dim` / `outer_dim`. Speaker projection is `(d_model, inner or 512)`. Latents stay `[B, T, C]` and are quantized once.
- `2319155` L4: `tokenizers` / `.json` loads Hugging Face tokenizers; `.model` stays SentencePiece. Chunking honors the five text flags and calls `tokenizer.decode`. Generation receives the prepared chunk.
- `6f2c9fc`: `koder/AGENTS.md` says to run MLX commands with the sandbox disabled on the first attempt. A sandboxed `import mlx` aborts.

## Present

- Active plan: `koder/plans/001_update_pocket_tts_mlx/INDEX.md` (rev 2). Ledger: `koder/issues/001_update_to_current_pocket_tts/INDEX.md`. L0–L4 are `done`; L5–L8 are `candidate`.
- Facts the plan still rests on:
  - shipped bundles are already fused;
  - `b6369a24` matches the MLX tree on all 213 keys;
  - voice states carry `offset`/`pad`, which this port maps to `current_end`.
- EOS handling, the per-chunk fade, and the Mimi step of 16 are still the v0.2.1 path. Text flags and the new sampler are live.
- Validation: `.venv/bin/python -m pytest -q` → 31 passed, sandbox disabled. Scratch gate clean. `git diff --check` clean.
- Local `main` is 12 commits ahead of `origin/main` after this close. Nothing pushed.

## Future

- Next action: L5. Ignore EOS before frame 6. Fade the first `sample_rate // 200` samples of each chunk (120 at 24 kHz). Run `end_on_pause` on a cloned-voice waveform before encoding. When `frames_after_eos` is unset, use `model_recommended_frames_after_eos`, then the text guess + 2. Set the Mimi step from `encoder_frame_rate / frame_rate`. Defaults for `warmup_frames`, `trim_start_ms`, and `fade_in_ms` become 0. One commit. Stop the layer if a unit test fails.
- Then L6 → L7, one commit per layer with a clean tree between layers. The owner already authorized the build.
- Owner decisions still pending: push, tag `v0.3.0`, PyPI publish (L8). Renaming the `Untitled/` folder is optional.
- Use `.venv` only. `a separate kyutai venv` imports a different checkout.
- Run pytest and any other MLX import with the sandbox disabled. Skip the sandboxed run.
