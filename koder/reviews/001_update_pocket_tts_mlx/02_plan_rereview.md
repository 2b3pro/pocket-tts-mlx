---
plan: 001
issue: 001
type: plan-review
verdict: approve with fixes
p1: 1
p2: 6
p3: 4
reviewer: claude
created: 2026-10-08
supersedes: 01_plan.md (findings there stay applied; this pass adds evidence 01 did not have)
---

# Review 02: Plan 001, checked against published files and full upstream history

## Summary

Review 01 checked the plan against source trees only. This pass also:

- unshallowed the upstream clone at `/tmp/pocket-tts-upstream` (still at `41cbc84`, 180 commits) and read every inference commit since 2026-01-25;
- read safetensors **headers only** (HTTP range requests, no weight download) for the published bundles and voice files;
- diffed the MLX parameter tree against the `b6369a24` bundle without loading weights;
- checked packaging and release state (PyPI, tags, `pyproject.toml`, local envs).

No `pocket_tts_mlx/` file was edited.

## Evidence gathered

| Check | Result |
| --- | --- |
| MLX params vs `tts_b6369a24.safetensors@d4fdd22` keys | 213 vs 213, no missing or unexpected keys. The current bundle walker skips nothing for this model. |
| `languages/english/model.safetensors@e7205b6` header | 214 keys = the same 213 names + `flow_lm.bos_before_voice [1,1,1024]`. Already fused and already renamed (`flow_lm.*`, `mimi.*`). No `weight_g`/`weight_v`, `w_s_t`, learnt padding, VQ, logvar, or wavlm keys. `speaker_proj_weight [1024,32]`, `mimi.downsample.conv.conv.weight [32,512,32]`, `mimi.upsample.convtr.convtr.weight [512,1,32]`. |
| `languages/french/model.safetensors@1e08e6a` | Same 214-key layout as English. |
| `languages/english_drifting_26-09/model.safetensors@a5ce31f` | 202 keys: no `flow_net.time_embed.*` (12 keys fewer), which means 0 time conditions. |
| `languages/english/embeddings/alba.safetensors@1e08e6a` | 18 keys: `transformer.layers.{0..5}.self_attn/{cache F32 [2,1,126,16,64], offset I64 [1], pad I64 [1]}`. No `current_end`. |
| `languages/english_drifting_26-09/embeddings/alba`, `languages/french/embeddings/estelle` | Present, same state format. Voice files are per **config stem** (upstream `get_predefined_voice(language=self.origin.stem, ...)`). |
| Legacy `embeddings/alba.safetensors@d4fdd22` | Still served: `audio_prompt F32 [1,125,1024]`. |
| Upstream `2dd548a` (2026-09-01) | Switches backbone FFN to `F.gelu(..., approximate="tanh")`. The commit says the released English weights were trained with tanh, so the change is a correctness fix for all models, `b6369a24` included. |
| MLX `nn.gelu_approx` (mlx 0.30.4) | It is the tanh formula. The package floor is `mlx>=0.20.0`. |
| PyPI | Latest `0.2.1` (`0.2.0`, `0.2.1` published). Remote tag `v0.2.1` → `1342d4d`. The README follow-up `b1fe6a4` came after the tag. |
| Local envs | A separate `kyutai` venv outside the repo has an editable `pocket_tts_mlx 0.1.0` that points at a **different checkout** a checkout outside this repo (`40530b3`, not in this history). |

## Findings

### P1-1: L6 maps voice state in the wrong direction for this port

Plan L6 says to import voice states "the way `model_state.py` does, including `current_end` → `offset`." Upstream maps old files *into* its new `offset` schema. The MLX FlowLM attention still keys on `current_end` (shape-encoded length, `attention.py:127`, `tts_model.py:297`). The published files carry `offset` and `pad`, with no `current_end`. An import that copies upstream's direction loads a state that `_flow_lm_current_end` rejects, so L6 cannot pass its own gate as written.

Fix: map `offset` → `current_end = zeros((int(offset),))`, require `pad == 0` and drop it, and copy `cache` unchanged (`[2,B,T,H,D]` already matches MLX). Keep accepting a legacy `current_end` key as-is. Imported states skip the audio-prompting step entirely.

### P2-1: Tanh GELU is in the analysis but in no layer

The analysis and review 01 both name the `_ff_block` GELU site, but L1–L7 never change it. The `mimi_transformer.py:56` call to `nn.gelu` is exact (erf). Upstream says the weights were trained with tanh. Add it as an explicit step. Use the formula directly (`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x**3)))`) rather than `nn.gelu_approx`, so that `mlx>=0.20.0` does not depend on how an older MLX defined that helper. This applies to `b6369a24` too, and the changelog should say so.

### P2-2: L6 fusion and rename work is not needed for any shipped bundle

Every bundle checked is already fused and renamed, and the existing attribute walker matches `b6369a24` key-for-key. The real L6 risk is different: the walker counts skips instead of failing, and `setattr` never checks shapes. Replace the fusion, skip-list, and rename items with:

- a strict **name and shape** check on the bundle walker (compare against `tree_flatten(model.parameters())` after the conv layout transform);
- stop if a bundle has keys the walker cannot place, or any shape differs.

The separate-file loader (`flow_lm.weights_path` / `mimi.weights_path`) is used by no shipped config. Leave it as is; renames there are out of scope. If a future bundle ships `weight_g`/`weight_v`, the stop rule catches it.

### P2-3: Per-chunk prepared text is discarded

`generate_audio_stream` (`tts_model.py:350-358`) computes `prepare_text_prompt(chunk)` and then passes the raw `chunk`. Upstream fixed the same bug in `ce03c06`. Fold this into L4: generate from the prepared chunk text.

### P2-4: The voice catalog does not cover the per-language default voices

`PREDEFINED_VOICES` has 8 names. L7 makes the default voice follow the language (`estelle`, `giovanni`, `juergen`, `lola`, `rafael`, `daan`), but those names are not in the catalog. `get_state_for_audio_prompt("estelle")` would fall through to the voice-cloning path and fail. In L6:

- use upstream's name set (`_ORIGINS_OF_PREDEFINED_VOICES` keys, 26 names);
- build the URL from the config stem;
- for a model not loaded from a shipped config stem, refuse predefined names with upstream's message;
- keep the `d4fdd22` `audio_prompt` catalog for `b6369a24` only.

### P2-5: `b6369a24` stops being a v0.2.1 regression under the new defaults

v0.2.1 always pads short inputs with eight spaces. The new schema default is `pad_with_spaces_for_short_inputs: false`, and `b6369a24.yaml` sets nothing. So "still loads" would hide a behavior change. Set `pad_with_spaces_for_short_inputs: true` in `b6369a24.yaml`, as upstream's sibling `english_2026-01` does. Capitalize and terminal punctuation default on, so they match. Temperature stays per review 01 P2-3 (0.3 unless `temp=0.7`, listed as a break).

### P2-6: Release hygiene is missing from the plan

L7 only bumps the version and the changelog. A clean `pip` release also needs:

- **One version source.** The version lives in both `pyproject.toml` and `pocket_tts_mlx/__init__.py`. Either bump both in the same commit, or switch to `dynamic = ["version"]` with `version = {attr = "pocket_tts_mlx.__version__"}`.
- **Dependencies.** Add `tokenizers>=0.21`. Keep `sentencepiece` (needed for `b6369a24`).
- **Test config.** Add `[tool.pytest.ini_options] testpaths = ["tests"]`. `tests/` stays out of the wheel (`packages.find include = ["pocket_tts_mlx*"]`).
- **README.** "What's New", Quickstart, CLI and Voices currently teach `marius` and `--warmup-frames 1`. Rewrite them for 0.3.0 in L7.
- **Packaging check before tagging.** Build the sdist and wheel in a clean repo-local venv, run `twine check`, confirm the wheel contains all 20 YAMLs, then install the wheel into a fresh venv and run the CLI smoke.
- **Tag placement.** Tag `v0.3.0` on the final release commit, after README and changelog. `v0.2.1` was tagged before its README follow-up.
- **Owner-only steps.** Push and PyPI publish stay owner-only.

### P3-1: `audio_read` treats every WAV as 16-bit

`data/audio.py:28` decodes 24-bit and float WAV prompts as int16 garbage. Upstream routes those through soundfile (`ce03c06`). This is a small fix that can ride with L6.

### P3-2: Plan text is stale in places

The plan still says the parent issue is `issue_kind: mapping`; it is now a `track`. Its L6 proof also says "header/key check first", but this review has already done that check, so the build-time version becomes a regression test.

### P3-3: Local environment can test the wrong checkout

The editable install in a separate `kyutai` venv outside the repo imports a different checkout. Use a repo-local `.venv` (already gitignored) for tests and release builds. The repo folder is named `Untitled/`. Renaming it is cosmetic and up to the owner; do not rename it mid-build.

### P3-4: Unchecked residuals

- The gated `kyutai/pocket-tts` bundles were not read (auth). The strict walker covers them at load time.
- The `pad` values in voice files were not read (the header only gives shape). The import assert covers them.

## Passing checks

Everything listed in review 01 still holds against the published files:

- `english` / `english_2026-09` share revision `983151f`.
- New English speaker projection is `(1024, 32)`.
- Downsample is 512→32 and upsample input is 512 (depthwise).
- Drifting has zero time conditions.
- `english_2026-01` is not `b6369a24`.
- All 19 upstream YAMLs only use fields in the L1 list.
- The flow MLP RMSNorm (`ddof=1`) and LayerNorm (`ddof=0`) match upstream.

## Verdict

APPROVE WITH FIXES. All fixes above are edited into plan 001 in the same change. With them in, L1 can start once the owner asks.
