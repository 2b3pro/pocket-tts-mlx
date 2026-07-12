# Changelog

## Unreleased

### Added

- Ported upstream text normalization for decimals and currency so structural punctuation is converted to spoken text before tokenization.
- Added `UserDictionary` pronunciation overrides with JSON/YAML loading, language-specific and common sections, literal or regex matching, and dictionary composition.
- Added CLI `--dictionary` support and automatic loading from `~/.config/pocket-tts/dictionary.{yaml,yml,json}`.
- Added per-generation `temperature` and deterministic `seed` controls to the Python API and CLI.
- Use explicit per-generation MLX random keys so seeded requests are reproducible without mutating global RNG state.
- Detect buffered generations that exhaust their safety limit without EOS and retry with a fresh random stream.
- Added opt-in batched Mimi decoding through the Python `decode_batch_size` argument and CLI `--decode-batch-size`. The default remains `1` for bit-identical frame-at-a-time output and minimum streaming latency; `4` is the recommended starting point for buffered/offline generation.
- Added opt-in compatibility with upstream's April 2026 English checkpoint via `TTSModel.load_model("english_2026-04")` and CLI `--model english_2026-04`, including v2 Mimi dimensions, learned voice BOS conditioning, and serialized voice-state loading. The legacy checkpoint remains the default pending evaluation.

### Fixed

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
- Validated the April 2026 English checkpoint on the original Lincoln failure case: generation completed without a garbled tail, remained exactly reproducible for a fixed seed, produced no non-finite samples, and ran about 13% faster than the legacy checkpoint in the local comparison. The checkpoint remains opt-in until broader listening evaluation is complete.

### Tests

- Added regression coverage for oversized clause splitting, text normalization, dictionary loading and composition, generation-path pronunciation overrides, deterministic sampling, missing-EOS recovery, single-sync streaming, batched Mimi decoding, v2 configuration, resampler dimensions, prompt behavior, and serialized voice-state conversion.

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
