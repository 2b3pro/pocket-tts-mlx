---
plan: 001
title: Update pocket-tts-mlx to current Pocket TTS inference
issue: 001
type: implementation
status: approved
revision: 2
created: 2026-10-08
updated: 2026-10-09
---

# Plan 001: Update pocket-tts-mlx to current Pocket TTS inference

## User Decisions Captured

- Durable notes live under `koder/`. This file is the implementation plan. The chat sketch is not.
- Reviews: `koder/reviews/001_update_pocket_tts_mlx/01_plan.md` and `02_plan_rereview.md`, both `approve with fixes`. All fixes are in this revision. Build only when the owner asks.
- Training, HTTP `serve`, Docker, and torchao stay out. An MLX int8 path stays out unless a review says speech is blocked without it.
- Keep repo hygiene good enough for a clean `pip` release: one commit per layer, a clean tree between layers, version/changelog/README updated together, nothing pushed or published without the owner.
- `open` is read-only. `close` commits only when the owner asks. Do not push.

## Capability Statement

`pocket-tts-mlx` loads current Pocket TTS inference weights and speaks with current generation behavior on Apple Silicon, while `b6369a24` still loads and keeps its v0.2.1 text handling.

## Slice Accounting

- Slice: layers L0–L8 below. Parent issue `001` is a `track`. Its slice ledger mirrors these layers.
- Parent seam: inference runtime.
- Closure gate: the checks in Acceptance Criteria. Weight downloads belong to the build window.

## Triage

- Tier: L.
- Risk: yellow. The bundle walker can skip a key or take a wrong-shaped tensor and still return audio.
- TDD/review posture: tests that do not need weights land with each layer. Weight checks stop the layer on mismatch. One commit per layer. Do not land several layers as one commit.

## Evidence Already Gathered (review 02)

Header-only reads, no weight download:

- `tts_b6369a24` bundle: 213 keys, identical to the MLX parameter tree.
- `languages/{english,french}` bundles: the same 213 names plus `flow_lm.bos_before_voice`, already fused and renamed. `english_drifting_26-09` has no `flow_net.time_embed.*`.
- Voice states `languages/{config_stem}/embeddings/{name}.safetensors@1e08e6a` hold `transformer.layers.N.self_attn/{cache [2,1,T,16,64], offset [1], pad [1]}`.
- Legacy `embeddings/{name}.safetensors@d4fdd22` is still `audio_prompt [1,125,1024]`.

## Layers / Steps

### L0: Test scaffold and local env

- Create a repo-local `.venv` (gitignored) and `pip install -e . pytest` there. The `[dev]` extra (torch, pocket-tts) is only needed for parity tests, and none exist yet. Do not test through `a separate kyutai venv`; its editable install points at another checkout.
- Add `tests/` and `[tool.pytest.ini_options] testpaths = ["tests"]` to `pyproject.toml`.
- Add one baseline weight-free test, for example that `b6369a24.yaml` parses and `prepare_text_prompt` keeps v0.2.1 behavior.
- Proof: `python -m pytest -q` passes in `.venv`.

### L1: Config schema and language YAMLs

- Extend `pocket_tts_mlx/utils/config.py` to upstream's schema at `41cbc84`:
  - `flow.type` (default `lsd`) and `insert_bos_before_voice` (default false);
  - `inner_dim` and `outer_dim` (default None);
  - `default_temperature` (0.3);
  - `capitalize_first_letter` (true), `append_terminal_punctuation` (true), `remove_semicolons` (false), `pad_with_spaces_for_short_inputs` (false);
  - `replace_characters` (`{}`) and `model_recommended_frames_after_eos` (None).
- Copy the 19 upstream YAMLs from `pocket_tts/config/` at `41cbc84` into `pocket_tts_mlx/config/`. Upstream is MIT, like this repo.
- Keep `config/b6369a24.yaml` and add `pad_with_spaces_for_short_inputs: true`. That keeps the v0.2.1 short-input padding, which upstream's sibling `english_2026-01` also sets. It is not `english_2026-01`: its weights (`427e3d61`), tokenizer (`.model`) and voice catalog all differ.
- Proof: every YAML parses. `english` and `english_2026-09` share revision `983151f13aaeab1b13c1e5e3c2c383d49a9edf3f`; do not require equal URLs. `b6369a24` resolves to pad true, capitalize true, terminal punctuation true.

### L2: Flow head, sampler, and backbone FFN

- `FlowLMModel` takes `flow_type`, and gets a `bos_before_voice` parameter `(1, 1, d_model)` only when `insert_bos_before_voice` is true.
- Dispatch `lsd`, `flow_matching` (Euler, one time input), and `drifting` (`v_t(x_0)`, no time input) the way `pocket_tts/models/flow_lm.py` does.
- `SimpleMLPAdaLN.from_pydantic_config` sets the time-condition count to 2 / 1 / 0 from `flow.type`. Remove `assert num_time_conds != 1`. `__call__(c, *ts, x)` takes a variable number of time tensors and adds the time term only `if self.num_time_conds`.
- In `modules/mimi_transformer.py` `_ff_block`, replace `nn.gelu` with tanh GELU written out: `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x**3)))`. Do not use `nn.gelu_approx`, since the `mlx>=0.20.0` floor allows older definitions. This changes every model, `b6369a24` included (upstream `2dd548a`).
- `num_layers` is already read from config, so the 24-layer YAMLs need no new loop.
- Proof (unit tests, no weights): each flow type builds with the right time-embed count; drifting does not divide by zero; GELU matches the tanh formula to 1e-6 on a grid.

### L3: Mimi width and latent layout

- Downsample `out_dimension` is `inner_dim or seanet.dimension`. Upsample `in_dimension` is `outer_dim or seanet.dimension`, depthwise. New English gets downsample 512→32 and upsample 512. `b6369a24` stays square at 512.
- Speaker projection is `(d_model, inner_dim or seanet.dimension)`, replacing the hardcoded `(1024, 512)`.
- `decode_from_latent` takes `[B, T, C]` and applies the quantizer. `encode_to_latent` returns `[B, T, C]`. Delete the caller transpose and quantizer in `tts_model.py` (generation loop and `_warmup_mimi_decoder`) so neither is applied twice.
- Proof: shape tests for inner 32 / outer 512 and for missing inner/outer. `b6369a24` still constructs, and its parameter tree still matches the 213 bundle names and shapes.

### L4: Tokenizer and text chunking

- Use Hugging Face `tokenizers` when the config kind is `tokenizers`/`json` or the path ends in `.json`. Keep SentencePiece for `.model`. Both expose `encode`, `decode`, and `__call__`. Add `tokenizers>=0.21` to dependencies and keep `sentencepiece`.
- Port `pocket_tts/models/text_chunking.py`:
  - `prepare_text_prompt` with all five flags;
  - `_ensure_terminal_punctuation`: closers are kept, and trailing weak punctuation becomes `.`;
  - after `replace_characters`, the `[.!?…]\s*[,;:]` cleanup;
  - decimal-period skipping and comma/semicolon/colon sub-splits.
- Chunking calls `tokenizer.decode`, not `tokenizer.sp`.
- `generate_audio_stream` generates from the **prepared** chunk text, not the raw `chunk` (upstream `ce03c06`).
- Proof (no weights):
  - decimals are not split;
  - padding happens only when the flag is set;
  - `capitalize_first_letter` can be off;
  - a French-style quote rewrite deletes the quotes;
  - `"Hi,"` becomes `"Hi."`;
  - the prepared text reaches generation.

### L5: Generation behavior

- Ignore EOS before frame 6 (`_MIN_FRAMES_BEFORE_EOS`).
- Fade the first `sample_rate // 200` samples of each chunk once (5 ms, 120 samples at 24 kHz). This is not the CLI `--fade-in-ms` flag.
- Run `end_on_pause` (numpy port of `pocket_tts/data/audio_utils.py`) on a cloned-voice waveform before encoding. Prepend `bos_before_voice` only when the config says so.
- When `frames_after_eos` is None, use `model_recommended_frames_after_eos`, then the text guess + 2.
- Set the Mimi step increment from `encoder_frame_rate / frame_rate` instead of the literal 16.
- Keep `warmup_frames`, `trim_start_ms`, and `fade_in_ms`, with all defaults now 0. That is a break from v0.2.1, where warmup defaulted to 1.
- Proof (no weights): EOS on frame 0 is ignored; the fade is 120 samples; `end_on_pause` ends on 80 ms of zeros; BOS and padding follow the flags.

### L6: Strict weight load and voice-state import

- Bundle walker in `TTSModel._from_pydantic_config_with_weights`:
  - after the conv layout transform, check every tensor's **name and shape** against `tree_flatten(model.parameters())`;
  - a missing parameter, an unplaceable key, or a shape mismatch raises an error; a skip counter is not a pass;
  - convert BF16 to float32 as today;
  - published bundles are already fused and renamed, so add no fusion or rename code;
  - leave the separate-file loaders as they are (no shipped config uses them).
- Voice catalog:
  - predefined names are upstream's `_ORIGINS_OF_PREDEFINED_VOICES` keys (26 names, including `estelle`, `giovanni`, `juergen`, `lola`, `rafael`, `daan`);
  - for a model loaded from a shipped config stem, the URL is `hf://kyutai/pocket-tts-without-voice-cloning/languages/{config_stem}/embeddings/{name}.safetensors@1e08e6a23401048648a9fdcfde2f89348215c2a7`;
  - `b6369a24` keeps the `embeddings/{name}.safetensors@d4fdd22` `audio_prompt` catalog for its 8 names;
  - any other config (a custom path) refuses predefined names with upstream's message.
- State import for this port:
  - split keys on `/`;
  - map `offset` → `current_end = mx.zeros((int(offset),), int64)`;
  - require `pad == 0` and drop it;
  - keep `cache` as it is (`[2,B,T,H,D]` matches MLX);
  - accept a legacy `current_end` key unchanged;
  - module names must exist in `init_states(self.flow_lm)`, otherwise stop;
  - an imported state skips audio prompting;
  - a user-supplied `.safetensors` path (including `hf://…@rev` and `http(s)` URLs with queries) also imports as a state.
- `audio_read` sends non-16-bit WAV through soundfile (upstream `ce03c06`).
- Voice export and the `export-voice` CLI stay deferred.
- Proof:
  - a strict load of `b6369a24` and of `english`, as a test that skips without network;
  - one short non-silent 24 kHz clip from default `load_model()` with `alba`; it must fail if the `alba` file is a legacy `audio_prompt` tensor;
  - `b6369a24` with `alba` (legacy catalog) still speaks;
  - one of `french`/`dutch` with its default voice, and `english_drifting_26-09` with `alba`, each produce a short clip.

### L7: API, CLI, docs, and version

- `load_model(config=..., language=...)`:
  - `config` stays the first positional argument, so `load_model("b6369a24")` still works;
  - passing both is an error; omitting both loads `english`;
  - when `temp` is omitted, use the config's `default_temperature` (0.3 unless the YAML sets it, `b6369a24` included);
  - add `sampler_decode_steps`; `lsd_decode_steps` remains a deprecated alias;
  - keep the origin stem for the voice catalog.
- CLI:
  - `--language`, `--config`, and `--temperature`;
  - text `-` reads stdin;
  - the default voice follows the language: `alba`, `estelle`, `giovanni`, `juergen`, `lola`, `rafael`, `daan`;
  - `--max-tokens` default 50, `--frames-after-eos` unset, `--warmup-frames` 0.
- Packaging:
  - one version source: switch `pyproject.toml` to `dynamic = ["version"]` with `[tool.setuptools.dynamic] version = {attr = "pocket_tts_mlx.__version__"}`, or bump both files in the same commit;
  - set the version to `0.3.0` only after L1–L6 pass.
- `CHANGELOG.md` gets a `## v0.3.0 - <date>` entry in the existing style. It lists additions (languages, drifting, tokenizer.json, per-language voices) and every break:
  - default model `english` at temperature 0.3, including `b6369a24` unless `temp=0.7` is passed;
  - tanh GELU for all models;
  - CLI voice is no longer `marius`;
  - `--max-tokens` 500 → 50, `--frames-after-eos` 7 → unset, `--warmup-frames` 1 → 0;
  - text prompts now follow per-config flags.

  Do not add a second history file.
- `README.md`: replace "What's New in 0.2.1" with 0.3.0, update Quickstart, CLI, and Voices (per-language defaults, `--language`), drop the `marius` / `--warmup-frames 1` preset, and point the release notes at 0.3.0.
- Proof: `--help` shows the new flags; `pocket-tts-mlx --language english -o <tmp>.wav "Hello from MLX."` writes a wav; `python -c "import pocket_tts_mlx; print(pocket_tts_mlx.__version__)"` matches the built metadata.

### L8: Release preparation (owner-gated)

**On hold (owner, 9 Oct 2026).** Do not tag, push, or publish `0.3.0` until issue 002 is settled. Plan 002 (`koder/plans/002_fix_quiet_or_unclear_non_english_speech/INDEX.md`) runs first. Its S5 fixes review 03 P2-2 and P3-1 and then resumes this layer. Rerun every L8 check on the final commit, because the review 03 wheel check came before plan 002's code and doc changes.

- In a clean repo-local venv:
  - `python -m build`;
  - `twine check dist/*`;
  - confirm the wheel lists all 20 YAMLs under `pocket_tts_mlx/config/`, and that neither `koder/` nor `tests/` is packaged;
  - install the wheel into a fresh venv and run the L7 CLI smoke.
- Tag `v0.3.0` on the final release commit, after README and changelog.
- Pushing the branch and tag and publishing to PyPI happen only when the owner asks.

## Diff Budget

- Touch the existing MLX modules. Do not rename `mimi_transformer.py` to match upstream.
- Stop and split if one layer edits more than that layer's files plus its tests.

## Acceptance Criteria

- [ ] `tests/` runs with `python -m pytest -q` in a repo-local venv.
- [ ] All shipped YAMLs parse, including `b6369a24` with short-input padding on.
- [ ] `english` and `english_2026-09` share revision `983151f13aaeab1b13c1e5e3c2c383d49a9edf3f`.
- [ ] Flow type selects the time-condition count, including 0 and 1. The backbone FFN uses tanh GELU.
- [ ] New English speaker projection is `(1024, 32)`. Missing `inner_dim` stays `(1024, 512)`.
- [ ] EOS before frame 6 is ignored. Chunk fade is 120 samples at 24 kHz.
- [ ] Tokenizer kind selects JSON vs SentencePiece. Chunking does not touch `.sp`. Generation uses the prepared chunk text.
- [ ] The bundle load fails on a missing parameter, an unplaceable key, or a shape mismatch.
- [ ] Voice states import `offset` as `current_end`, reject non-zero `pad`, and cover the per-language default names.
- [ ] Default `load_model()` with `alba` writes a short non-silent 24 kHz clip from the per-language voice state.
- [ ] `b6369a24` still loads and speaks. One other shipped language and `english_drifting_26-09` each generate a short clip.
- [ ] The version has one source and reads `0.3.0`. `CHANGELOG.md` and `README.md` describe 0.3.0 and its breaks.
- [ ] The built wheel contains 20 YAMLs, installs cleanly, and its CLI writes a wav.

## Verification

Weight-free, per layer:

```bash
python -m pytest -q
```

After L6 and L7, on Apple Silicon, with Hugging Face downloads:

```bash
python -c "from pocket_tts_mlx import TTSModel; m=TTSModel.load_model(); a=m.generate_audio(m.get_state_for_audio_prompt('alba'), 'Hello from MLX.'); assert a.shape[-1] > 1000"
pocket-tts-mlx --language english -o /tmp/pocket-tts-mlx-english.wav "Hello from MLX."
```

L8:

```bash
python -m build && twine check dist/* && unzip -l dist/*.whl | grep -c "config/.*\.yaml"
```

## Deferred / Non-Goals

- HTTP server, training checkpoints, Docker, torchao, and an MLX int8 quantize path.
- `export-voice` and writing a voice state. Reading the shipped per-language state is in L6.
- Vendoring community-model YAMLs that are not in upstream `pocket_tts/config/`.
- A `flow_matching` generation smoke (no shipped YAML uses it).
- Renaming modules to match upstream paths, or porting upstream's linear Mimi KV cache. The MLX ring buffer is equivalent.
- Weight-norm fusion and separate-file renames, because no published bundle needs them.
- PyPI publish and git push without an owner request.

## Stop Rules

- A bundle key the walker cannot place, a missing parameter, or a shape mismatch.
- Speaker or conv shapes that disagree with `inner_dim` / `outer_dim`.
- A tokenizer vocab size that disagrees with `n_bins`.
- A voice-state file with module names outside `init_states(flow_lm)`, a non-zero `pad`, or an `alba` that is only `audio_prompt` when the model is `english`.
- Any of these stops the layer. Update this plan. Do not invent a second weight or voice format.
