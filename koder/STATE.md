---
updated_at: "08 Oct 2026 | 08:36 PM IST"
---

# Koder State

## Past

- v0.2.1 (`b1fe6a4`) is the published MLX port. PyPI `0.2.1`; remote tag `v0.2.1` → `1342d4d`.
- Koder scaffold: `0d1e841`. `CHANGELOG.md` stays the history surface.
- `d37c30b`: analysis 001, issue 001 (track), plan 001 rev 2, reviews 01 and 02. Upstream read at `41cbc84`.
- `59d499f` L0 through `2319155` L4: test scaffold, language configs, samplers, Mimi width, tokenizer and chunking.
- `6f2c9fc`: run MLX commands with the sandbox disabled. A sandboxed `import mlx` aborts.
- `686a51c` L5: ignore EOS before frame 6, fade the first `sample_rate // 200` samples of each chunk, `end_on_pause` before encoding a cloned waveform, frames-after-EOS fallback, Mimi step from `encoder_frame_rate / frame_rate`. Warmup default is 0.
- `ba13bcf` L6: strict bundle load. Voice states map `offset` to `current_end`, reject non-zero `pad`, and skip audio prompting. `b6369a24` keeps the 8-name `audio_prompt` catalog.
- `d2d8f44` L7: version `0.3.0` from `pocket_tts_mlx.__version__`. Default `load_model()` is `english` at the config temperature. CLI voice follows the language.
- `dcb7c0a` L8: `MANIFEST.in` keeps `tests/` and `koder/` out of the sdist.

## Present

- Plan 001 rev 2 is built. Ledger `koder/issues/001_update_to_current_pocket_tts/INDEX.md` is `done` for L0–L8.
- Local version is `0.3.0`. It is not tagged, not pushed, and not on PyPI. `v0.2.1` remains the published release.
- Upstream voice catalog at `41cbc84` has 27 names. The plan text said 26.
- Shipped bundles are fused. Strict load raises on a missing parameter, an unplaceable key, or a shape mismatch.
- Validation, rechecked at this close: `.venv/bin/python -m pytest -q` → 64 passed, sandbox disabled. Scratch gate clean. `git diff --check` clean. No product delta since `1ef2b53`.
- Wheel check: `twine check` passed. The wheel lists 20 YAMLs under `pocket_tts_mlx/config/` and does not package `koder/` or `tests/`. A fresh venv installed `0.3.0` and `pocket-tts-mlx --language english` wrote a 24 kHz wav.
- Local `main` is 18 commits ahead of `origin/main` after this close. Nothing pushed.

## Future

- Owner decisions: push `main`, tag `v0.3.0` on `d2d8f44`'s release commit or the later packaging commit, and publish to PyPI. Do not do those unless asked.
- Renaming the `Untitled/` folder is optional.
- Use `.venv` only. `a separate kyutai venv` imports a different checkout.
- Run pytest and any other MLX import with the sandbox disabled. Skip the sandboxed run.
