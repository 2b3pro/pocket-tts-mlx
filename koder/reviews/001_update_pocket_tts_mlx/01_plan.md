---
plan: 001
issue: 001
type: plan-review
verdict: approve with fixes
p1: 0
p2: 3
p3: 0
reviewer: other
created: 2026-10-08
---

# Review: Plan 001 — Update pocket-tts-mlx

## Summary

Reviewed `koder/plans/001_update_pocket_tts_mlx/INDEX.md` against `koder/analysis/001_mlx_and_upstream/INDEX.md` and the trees it cites: local `pocket_tts_mlx/` at `b1fe6a4`, upstream Pocket TTS `41cbc84`. No weights were downloaded. No `pocket_tts_mlx/` file was edited.

The draft matches the source on the points the chat sketch had wrong. Three gaps in the first draft were applied to the plan in this session. They are recorded here so they are not re-litigated. The port is still not authorized to start until the owner asks.

## Findings

### P2-1: A strict key check without the upstream skip list fails on the first real bundle

The first draft said every unexpected key is an error. Upstream `utils/weights_loading.py` deliberately skips `flow.w_s_t.*`, learnt padding, `num_ema_updates`, quantizer VQ, codebook keys, `weight_v`, logvar, and `wavlm_*`. Those keys are expected in published files. Treating them as failures makes L6 stop on a known layout.

Applied in L6: that skip list is part of the map. Anything outside the map, the skip list, and the fusion rename is an error. A skip counter alone is still not a pass.

### P2-2: The voice URL was unpinned

The first draft named `languages/{language}/embeddings/{name}.safetensors` and left the revision off. Upstream `get_predefined_voice` at `41cbc84` pins `@1e08e6a23401048648a9fdcfde2f89348215c2a7`. An unpinned URL can move under the smoke.

Applied in L6: use that pinned URL. The legacy `audio_prompt` catalog at `@d4fdd22` stays only for `b6369a24`.

### P2-3: `b6369a24` temperature change was easy to miss

`load_model` today defaults `temp` to 0.7. The draft said an omitted `temp` uses `default_temperature`, whose schema default is 0.3. `b6369a24.yaml` does not set the field, so the old model moves to 0.3. That is the upstream rule, and it is a break.

Applied in L7: the changelog must say so, including the `temp=0.7` escape, next to the CLI breaks (voice, max tokens, frames after EOS, warmup).

## Passing Checks

- `b6369a24` is kept, and it is not treated as `english_2026-01`. Weight revisions and tokenizer kinds differ.
- `english` vs `english_2026-09` is checked by revision `983151f13aaeab1b13c1e5e3c2c383d49a9edf3f`, not by URL equality.
- Speaker width is `(d_model, inner_dim or seanet.dimension)`: `(1024, 32)` for new English, `(1024, 512)` when `inner_dim` is absent.
- Missing `inner_dim` / `outer_dim` keeps square resample convs.
- The caller transpose and quantizer are removed when `decode_from_latent` takes `[B, T, C]`.
- MLP `__call__` grows a variable time list and does not divide by zero for drifting. `flow_matching` is implemented and is not a smoke.
- Tanh-approximate GELU is the shared `_ff_block`, not a second backbone module. No rename of `mimi_transformer.py`.
- Chunking uses `tokenizer.decode`. Padding, capitalization, and terminal punctuation follow config flags.
- The production proof is the bundle walker, not only `get_flow_lm_state_dict_mlx`.
- `alba` on `english` must be the per-language state import, and a legacy `audio_prompt` file fails that smoke.
- `config` stays the first positional argument. `load_model()` with no arguments selects `english`.
- One commit per layer. Training, serve, Docker, torchao, voice export, and community YAMLs stay deferred.
- 24-layer models already take `num_layers` from config.

## Verification

```bash
# not run: no koder plan/review validator in this repo
# manual: frontmatter, path numbers, analysis links, no secrets, no pocket_tts_mlx edits
git diff --stat -- pocket_tts_mlx
```

`git diff --stat -- pocket_tts_mlx` is empty. Weight downloads and generation were not run.

## Verdict

APPROVE WITH FIXES. P2-1, P2-2, and P2-3 are already edited into the draft. Do not start L1 until the owner asks to build. Issue 001 stays open; its slices stay `candidate`.
