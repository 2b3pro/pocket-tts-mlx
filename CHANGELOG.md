# Changelog

## Unreleased (`0.3.1.dev0`)

- Merged MLX upstream v0.3.0 and aligned runtime model/configuration behavior
  with Kyutai `44cb63e` (2026-10-22): October English default, six small English
  models, multilingual configs, JSON tokenizers, grouped attention, low-rank
  text tables, projected sampler inputs, pointwise sampling, and independent
  decoder dimensions.
- Preserved fork pronunciation dictionaries, request-local RNG/temperature,
  missing-EOS retries, seekable WAV headers, batched decoding, and fused kernels.
- Added model-specific EOS thresholds, latest character rewrites, linear-time
  decimal-boundary checks, stop-event cancellation, and opt-in integration tests.
- Corrected batched Mimi attention across ring-buffer wraps and restored prompt
  state before retrying in-place generation, preventing batch-size-dependent
  waveform errors and duplicated conditioning after a failed attempt.

### Added

- Added opt-in compatibility with upstream's 24-layer April 2026 English checkpoint.
- Ported upstream text normalization for decimals and currency so structural punctuation is converted to spoken text before tokenization.
- Added `UserDictionary` pronunciation overrides with JSON/YAML loading, language-specific and common sections, literal or regex matching, and dictionary composition.
- Added CLI `--dictionary` support and automatic loading from `~/.config/pocket-tts/dictionary.{yaml,yml,json}`.
- Added per-generation `temperature` and deterministic `seed` controls to the Python API and CLI.
- Use explicit per-generation MLX random keys so seeded requests are reproducible without mutating global RNG state.
- Detect buffered generations that exhaust their safety limit without EOS and retry with a fresh random stream.
- Added opt-in batched Mimi decoding through the Python `decode_batch_size` argument and CLI `--decode-batch-size`. The default remains `1` for bit-identical frame-at-a-time output and minimum streaming latency; `4` is the recommended starting point for buffered/offline generation.
- Added compatibility with upstream's April 2026 English checkpoint, including v2 Mimi dimensions, learned voice BOS conditioning, and serialized voice-state loading.

### Changed

- Use each model config's recommended sampling temperature; the April 2026 English models now default to upstream's evaluated `0.3`, while the original `b6369a24` setting can be restored with explicit `0.7`.
- Previously promoted `english_2026-04` to the default model (now superseded by October English) after reliability, reproducibility, pronunciation, and legacy-regression evaluation. The previous `b6369a24` checkpoint remains available through `TTSModel.load_model("b6369a24")` and CLI `--model b6369a24` for comparison or rollback.

### Fixed

- Preserve fully prepared short prompts during streaming generation instead of discarding padding and punctuation adjustments before tokenization.
- Correct seekable WAV headers, safely decode non-16-bit WAV prompts, and recognize remote safetensors and HTTP cache suffixes with revisions or query strings.
- Split oversized single sentences at commas, semicolons, and colons before generation, keeping each natural clause group within `max_tokens` when possible. Previously, a sentence without terminal punctuation before the end could bypass the chunk limit and produce skipped words, garbled speech, or an excessively long audio tail.
- Warn when a chunk still exceeds `max_tokens` because it contains no usable sentence or clause boundary.
- Aligned the CLI `--max-tokens` default with the safer 50-token library default instead of 500.
- Raised the minimum `huggingface_hub` version to 0.13.0, matching upstream's offline-loading requirement.

### Improved

- Vectorized Mimi ring-buffer KV-cache writes with `mx.put_along_axis`, removing per-frame Python loops, scalar device reads, and repeated slice updates from the decoder hot path.
- Replaced manual attention matmul/softmax sequences with `mx.fast.scaled_dot_product_attention` for FlowLM and Mimi.
- Replaced manual trigonometric RoPE construction with a fused `mx.fast.rope` call shared by query and key tensors.
- Evaluate FlowLM EOS output and Mimi audio together, reducing streaming generation from two GPU synchronizations to one per yielded frame.
- Allow offline callers to amortize Mimi decoder dispatch and synchronization overhead across completed latent frames. In the local fixed-seed benchmark, batch size `4` reduced median generation time from 0.776 s to 0.625 s (about 24%) while preserving output length; batched decoder arithmetic can produce small floating-point waveform differences.
- Validated the April 2026 English checkpoint on the original Lincoln failure case: generation completed without a garbled tail, remained exactly reproducible for a fixed seed, produced no non-finite samples, and ran about 13% faster than the legacy checkpoint in the local comparison.

### Tests

- Added macOS ARM CI for dependency validation and the complete unittest suite.
- Added regression coverage for oversized clause splitting, text normalization, dictionary loading and composition, generation-path pronunciation overrides, deterministic sampling, missing-EOS recovery, single-sync streaming, batched Mimi decoding, v2 configuration, resampler dimensions, prompt behavior, and serialized voice-state conversion.

### Documentation

- Clarified that `pocket-tts-mlx` is an independent port and does not automatically consume updates from a neighboring `pocket-tts` checkout.
- Added model-selection guidance, configured temperatures, local six-layer versus 24-layer resource measurements, and links to the corresponding upstream changes.

## v0.3.0 - 2026-10-16

### Added

- Current Pocket TTS languages, including french, german, dutch, italian, portuguese, spanish, the 24-layer previews, and `english_drifting`.
- Drifting, flow-matching, and LSD samplers. `tokenizer.json` for current models. SentencePiece remains for `b6369a24`.
- Per-language built-in voices. The CLI default is `alba`, `estelle`, `giovanni`, `juergen`, `lola`, `rafael`, or `daan`.
- `--language`, `--config`, and `--temperature`. Text `-` reads stdin. `sampler_decode_steps` replaces `lsd_decode_steps`, which still works as a deprecated alias.
- Voice-state `.safetensors` files load as voices from a local path, `hf://`, or an http(s) URL.
- Opt-in brightening: `generate_audio(..., brighten=True)` and `--brighten` add a +10 dB high shelf at 1.5 kHz. The french `estelle` voice is about 10 dB darker than `alba` above 2 kHz, in upstream as well. Off by default; with `--normalize`, brightening runs first.
- Opt-in loudness normalization: `generate_audio(..., normalize=True)` and `--normalize` scale speech to about -18 dBFS with peaks under -1 dBFS. Off by default. `juergen` and `daan` speak 6 to 9 dB quieter than the other defaults, in upstream as well.

### Changed

- The default model is `english` (the `english_2026-09` weights) at temperature 0.3. That includes `b6369a24` unless `temp=0.7` is passed.
- Every model, including `b6369a24`, uses tanh GELU in the backbone FFN.
- The CLI voice is no longer `marius`.
- `--max-tokens` defaults to 50, down from 500. `--frames-after-eos` is unset, down from 7. `--warmup-frames` defaults to 0, down from 1.
- `generate_audio` and `generate_audio_stream` default `warmup_frames` to 0, down from 1.
- Loading raises on a weight bundle with missing, extra, or wrongly shaped tensors. v0.2.1 skipped them silently.
- New runtime dependency: `tokenizers>=0.21`.
- Text prompts follow each config's flags: capitalization, terminal punctuation, short-input padding, semicolon removal, and character replacements.

### Fixed

- EOS is ignored before frame 6.
- Each chunk fades in over its first 5 ms.
- Cloned voice prompts end on a short pause.
- Non-16-bit WAV prompts load through soundfile.
- Generation uses the prepared chunk text, not the raw chunk.

## v0.2.1 - 2026-02-11

### Added

- Startup artifact cleanup controls in CLI and Python API:
  - `warmup_frames` / `--warmup-frames`
  - `trim_start_ms` / `--trim-start-ms`
  - `fade_in_ms` / `--fade-in-ms`

### Improved

- Ported upstream dynamic KV cache sizing behavior to reduce unnecessary cache preallocation.
- Materialized generated audio in `generate_audio()` to align reported timing with end-to-end usage.
- README updated with recommended clean-onset command and option explanations.

## v0.2.0 - 2026-02-03

- Initial public MLX release on PyPI.
