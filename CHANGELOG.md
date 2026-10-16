# Changelog

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
