---
status: complete
priority: P1
issue: 001
type: analysis
tags: mlx, pocket-tts, inference
created: 2026-10-08
updated: 2026-10-08
---

# Analysis 001: MLX port against current Pocket TTS

## Context

Parent issue: `koder/issues/001_update_to_current_pocket_tts/INDEX.md` (filed with this analysis).

Scope is inference only. Compared local `pocket-tts-mlx` `main` at `b1fe6a4` (v0.2.1) with `kyutai-labs/pocket-tts` `41cbc84` (2026-10-01, package version 3.3.0), cloned only for reading at `/tmp/pocket-tts-upstream`. Did not read `training/`. Did not download weights.

## Method

- Read the MLX package under `pocket_tts_mlx/` and root `CHANGELOG.md`, `README.md`, `pyproject.toml`.
- Read upstream `pocket_tts/models/tts_model.py`, `flow_lm.py`, `mimi.py`, `text_chunking.py`, `model_state.py`, `modules/transformer.py`, `modules/mlp.py`, `modules/resample.py`, `utils/config.py`, `utils/weights_loading.py`, `utils/utils.py`, `default_parameters.py`, `main.py`, and every file in `pocket_tts/config/`.
- Parsed those YAMLs with `yaml.safe_load` and compared weight URLs, `flow.type`, `inner_dim`, tokenizer kind, and text flags.

## Findings

| Item | Verdict | Evidence | Notes |
| --- | --- | --- | --- |
| MLX release is still the January model | pass | `pyproject.toml` version `0.2.1`; `default_parameters.py` `DEFAULT_VARIANT = "b6369a24"`, `DEFAULT_TEMPERATURE = 0.7`; `config/b6369a24.yaml` | No `tests/` directory. |
| Current default English is not that file | pass | `config/english.yaml` and `config/english_2026-09.yaml` share revision `983151f13aaeab1b13c1e5e3c2c383d49a9edf3f` but not the same `hf://` path | Comment in `english.yaml` says the weights are identical. String equality of the URLs is the wrong check. |
| `b6369a24` is not `english_2026-01` | fail | MLX weights `tts_b6369a24.safetensors@427e3d61`; upstream `languages/english_2026-01/model.safetensors@39592ff2`. MLX tokenizer is SentencePiece `.model`. Upstream `english_2026-01` uses `tokenizer: tokenizers`, `inner_dim: 512`, `insert_bos_before_voice: false`, `pad_with_spaces_for_short_inputs: true` | Keep the MLX file as the v0.2.1 regression. Do not delete it because it "matches" 2026-01. |
| New configs will not parse | fail | `utils/config.py` `StrictModel` has `extra="forbid"`. `FlowConfig` has no `type`. `MimiConfig` has no `inner_dim` / `outer_dim`. `Config` has no temperature or text flags | Copying upstream YAML before the schema grows raises a validation error. |
| Speaker projection is hardcoded for the old codec | fail | `models/tts_model.py` sets `speaker_proj_weight` to zeros `(1024, 512)`. Upstream uses `(d_model, inner_dim or seanet.dimension)` | New English `inner_dim` is 32, so the matrix is `(1024, 32)`. With `inner_dim` absent, the fallback is seanet dimension 512, which keeps `b6369a24`. |
| Mimi resample is square | fail | `modules/resample.py` builds both convs with one `dimension`. Upstream downsample `out_dimension=inner_dim`, upsample `in_dimension=outer_dim` (`models/mimi.py`) | Shipped new models are inner 32, outer 512. `None` must mean "same as seanet dimension" so the old YAML still builds. |
| Latent layout is still applied by the caller | watch | MLX `decode_from_latent` does not quantize. `tts_model.py` transposes, quantizes, then decodes, and `encode_to_latent` returns encoder layout. Upstream `decode_from_latent` quantizes a `[B, T, C]` tensor and `encode_to_latent` returns that layout | Moving the quantizer without deleting the caller transpose double-applies it. |
| Sampler is LSD-only, and the MLP call shape is fixed | fail | `flow_lm.py` only calls `lsd_decode`. `modules/mlp.py` hardcodes `num_time_conds = 2`, asserts `!= 1`, and `__call__(self, c, s, t, x)` always divides by `num_time_conds` | Only shipped non-LSD config is `english_drifting_26-09` (`type: drifting`, zero time inputs). No shipped YAML uses `flow_matching`. |
| GELU site is the shared block | pass | Upstream `modules/transformer.py` `_ff_block` uses `F.gelu(..., approximate="tanh")`. MLX `modules/mimi_transformer.py` line 56 uses `nn.gelu` with no approximation | One class serves Flow LM (`kind="flow_lm"`) and Mimi. Match that one site. A file rename is not required. |
| Tokenizer kind is ignored, and chunking calls SentencePiece | fail | `conditioners/text.py` always constructs `SentencePieceTokenizer`. `split_into_best_sentences` calls `tokenizer.sp.decode`. Upstream `text_chunking.py` calls `tokenizer.decode` and skips decimal periods | Every current upstream YAML sets `tokenizer: tokenizers` and a `.json` file, including `english_2026-01`. |
| Text flags are unconditional in MLX | fail | `prepare_text_prompt` always capitalizes, appends `.` when the last character is alphanumeric, and prefixes eight spaces when there are fewer than five words | Upstream does each of those only when the config says so. Only `english_2026-01` sets `pad_with_spaces_for_short_inputs: true`. French, German, Dutch, Italian, Portuguese, and Spanish set `replace_characters`. |
| EOS can fire on frame 0 | fail | MLX sets `eos_step` on the first true EOS (`tts_model.py` around the generation loop). Upstream `_MIN_FRAMES_BEFORE_EOS = 6` | |
| Onset cleanup is an MLX flag, not the upstream fade | watch | CLI `--warmup-frames` defaults to 1; `--fade-in-ms` and `--trim-start-ms` default to 0. Python `_MIMI_WARMUP_FRAMES = 1`. Upstream fades the first `sample_rate // 200` samples of each chunk once (5 ms at 24 kHz) inside `_decode_audio_worker`, and runs `end_on_pause` before encoding the voice | Turning warmup off is a behavior change from v0.2.1, not a preservation of it. |
| Mimi step increment is a literal 16 | watch | MLX `increment_steps(..., increment=16)`. Upstream uses `encoder_frame_rate / frame_rate` | With seanet ratios 6, 5, 4 that ratio is 16. Derive it when that loop is edited. |
| Production weights use the bundle walker, which skips missing keys | fail | Shipped YAML sets top-level `weights_path`. MLX assigns by attribute walk and counts `skipped_count` (`tts_model.py` around the bundle loop). `get_flow_lm_state_dict_mlx` is the separate-file path and has no `fuser.padding_value` → `bos_before_voice` rename. It also does not rename `in_proj_weight`. Upstream `weights_loading.py` does both, and fuses `weight_g` / `weight_v` with `torch._weight_norm` | A "key match" that uses the current walker cannot fail. Strict missing/unexpected keys have to be on the bundle path. Fusion has to be reimplemented without torch. |
| Built-in voices are not `audio_prompt` tensors anymore | fail | MLX `PREDEFINED_VOICES` points at `embeddings/{name}.safetensors@d4fdd22` and reads key `audio_prompt`. Upstream `get_predefined_voice` is `languages/{language}/embeddings/{name}.safetensors@1e08e6a`, and `model_state.py` imports `module/key` streaming state | `alba` on current English is a precomputed state for that model. Export can wait. Import cannot, or the default voice is the wrong object. Module names in those files were not opened. |
| CLI defaults diverge from both the Python API and upstream | fail | `main.py`: voice `marius`, required positional text, `--max-tokens` 500, `--frames-after-eos` 7, `--warmup-frames` 1. Python `generate_audio` uses `MAX_TOKEN_PER_CHUNK` 50. `default_parameters.py` `DEFAULT_AUDIO_PROMPT` is `alba`, and the README still says `marius`. Upstream CLI max tokens is 50, frames-after-eos is `None`, text `-` reads stdin, voice comes from the language | 0.3.0 has to list these breaks in `CHANGELOG.md`. |
| `load_model` positional slot is the variant name | watch | First parameter is `config`, default `b6369a24`. A stem loads `config/{stem}.yaml`. Upstream's first parameter is `language` | Keep `config` as the first positional so `load_model("b6369a24")` still resolves. Default when both `config` and `language` are omitted becomes `english`. |
| 24-layer depth is already config-driven | pass | `StreamingTransformer.from_pydantic_config` passes `config.num_layers` | `_24l` YAMLs do not need a new loop. They need the weight and codec changes, then a load check. |
| Training, server, torchao, Docker | pass | Not in the MLX tree. Upstream `quantization.py` and `serve` exist and are out of this port | Custom `config=` paths stay valid. |

## Recommendation

File the update as a draft plan that keeps v0.2.1 `b6369a24` loadable, targets current `english` / `english_2026-09`, and treats voice-state import as required. Do not start the port in this session.

The candidate chat sketch is right about the overall gap and wrong about these specifics: `b6369a24` equals `english_2026-01`; warmup flags already default off; the separate-file weight loader is the path that must be proven; speaker width stays 512; GELU is backbone-only; voice export is optional while `alba` still works.

## Verification / Reproduction

```bash
git rev-parse --short HEAD
# expected product commit before this session's scaffold: b1fe6a4
python3 -c "import yaml; from pathlib import Path; print(len(list(Path('/tmp/pocket-tts-upstream/pocket_tts/config').glob('*.yaml'))))"
# 19 configs at upstream 41cbc84
test ! -d tests && echo "no tests/"
```

No repo artifact validator exists. This file was checked by hand against `references/artifacts/analysis.md`: path, frontmatter, evidence column, no secrets.

## Residual Risk

- Safetensors headers for `english_2026-09` were not opened, so it is unproven whether the published bundle still contains `weight_g` / `weight_v` pairs or already-fused conv weights.
- Per-language voice states were not opened, so module names may not match the MLX state tree (`current_end` is remapped to `offset` upstream).
- `flow_matching` is implemented upstream and has no shipped YAML here. A generation smoke for it cannot be required.
- Whether the old `tts_b6369a24.safetensors` URL still downloads was not tried.

## Resolved Later

`koder/reviews/001_update_pocket_tts_mlx/02_plan_rereview.md` read the published headers:

- Shipped bundles are already fused and renamed. They contain no `weight_g`/`weight_v` pairs.
- Voice states hold `offset`/`pad`/`cache` under MLX-matching module names.
- The legacy `tts_b6369a24` and `audio_prompt` URLs still resolve.

That review also found the tanh GELU change (`2dd548a`), which this analysis named but the plan had not yet carried.
