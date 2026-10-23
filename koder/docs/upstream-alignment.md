# Upstream alignment — October 22, 2026

This fork targets inference compatibility with these reviewed snapshots:

| Source | Snapshot | Integration |
| --- | --- | --- |
| `jishnuvenugopal/pocket-tts-mlx` | `8dc7a1fe1eff0914749fe1e23c750d2b4c460a63` (`v0.3.0` plus documentation) | Merged with the fork's existing history |
| `kyutai-labs/pocket-tts` | `44cb63eaa4144f32ff80863968515165a5554353` (after `v4.0.1`) | Relevant inference changes manually ported |

Updating either upstream or an adjacent checkout never updates this package
automatically. PyTorch training, its HTTP server, Rust CLI, and torchao/CUDA
features are outside this MLX library's scope.

## Models and behavior

- The 27 current Kyutai YAML configs are copied byte-for-byte, plus the fork's
  original `b6369a24` config. All 27 predefined voice names match Kyutai.
- Default `english` uses `english_2026-10` weights at temperature `0.3`.
  Older named checkpoints remain selectable.
- Seven languages, 24-layer previews, September drifting, and all six October
  2M/7M/18M English models (including reflect-padding variants) are supported.
- Grouped-query attention, separate attention/FFN widths, low-rank text tables,
  projected backbone outputs, pointwise/drifting/LSD/flow-matching heads, and
  separate Mimi decoder widths follow the current schemas and weight names.
- Each model supplies its text flags, character rewrites, temperature, and EOS
  threshold. The small models use `-1.0`; other configs use `-4.0`.
- EOS is ignored before frame 6. Chunks receive a 5 ms onset fade; reference
  voice prompts end on a pause. Serialized voice states and JSON tokenizers
  are supported, with strict bundle shape/key validation.
- Decimal-boundary checks use adjacent tokens rather than decoding whole
  prefixes. Fork decimal/currency normalization and dictionaries run before
  splitting. Dutch decimal pronunciation is included.
- Streaming accepts `stop=threading.Event()` and ends when the event is set;
  closing the synchronous generator leaves no background worker running.

## Retained fork behavior

- Public dictionary exports and `--dictionary`, including automatic local
  dictionary loading; `--model` remains an alias for `--config`.
- Request-local `seed` and `temperature` overrides, buffered missing-EOS
  recovery, `decode_batch_size`, trim/fade controls, and seekable WAV headers.
- Fused MLX attention/RoPE, vectorized bounded Mimi cache writes, and one
  synchronization for EOS/audio in frame-at-a-time decoding.
- Batched attention retains keys needed by early queries before committing
  ring-cache writes. Calls larger than the small models' cache windows work.
- In-place buffered retries restore the original prompt state before retrying.
- MLX generation stays synchronous on the calling thread. Int8 is not enabled.

Temperature `0.3` now also applies to `b6369a24`, following current upstream.
Pass `temp=0.7` or `--temperature 0.7` to restore its original setting. The
upstream warmup default is now zero and the CLI's default English voice is Alba.
These are deliberate default changes; explicit caller settings remain available.

## Validation

- Weight-free suite: 132 tests passed, including numerical GQA and ring-wrap
  oracles, all six small-model forward/decoder paths, cancellation, recovery,
  pronunciation, prompt preparation, and deterministic sampling.
- 15 download-based checks passed for default/legacy English, French, drifting, all six
  small English models, raw reference-voice cloning, reproducible requests, and
  batched decoding (both full-size and small models). They require `POCKET_TTS_RUN_INTEGRATION=1`.
- Model/config and voice-name inventories were compared with the pinned source.
- Dependency checks, source compilation, sdist/wheel builds, distribution
  content checks, and a wheel-installed CLI smoke are part of delivery checks.

Commands:

```bash
.venv/bin/python -m pytest -q
POCKET_TTS_RUN_INTEGRATION=1 .venv/bin/python -m pytest -q tests/test_speech.py
uv pip check --python .venv/bin/python
.venv/bin/python -m build
```

The package version is `0.3.1.dev0`. Source integration is separate from PyPI
publication or upgrading/restarting a deployed VoiceServer environment.
No blanket waveform identity or subjective voice-quality claim is made.
