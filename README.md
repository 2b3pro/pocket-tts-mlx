# pocket-tts-mlx

MLX backend for [pocket-tts](https://github.com/kyutai-labs/pocket-tts) optimized for Apple Silicon.

Runtime is torch-free. Torch is only required for optional parity tests.

**Current Development (`Unreleased`)**

- Uses the April 2026 six-layer English model by default, with its recommended
  sampling temperature of `0.3`.
- Adds the upstream 24-layer English checkpoint as an opt-in model.
- Ports upstream prompt preparation, WAV handling, and remote asset fixes.
- Adds macOS ARM continuous integration.

**What's New in 0.2.1**

- Cleaner audio onset with `--warmup-frames`, `--trim-start-ms`, and `--fade-in-ms`.
- Upstream-aligned KV-cache sizing for better memory/runtime behavior.
- Lower perceived overhead after `generate_audio()` by materializing chunks before return.
- Release: [PyPI 0.2.1](https://pypi.org/project/pocket-tts-mlx/0.2.1/) · [GitHub v0.2.1](https://github.com/jishnuvenugopal/pocket-tts-mlx/releases/tag/v0.2.1)

**Relationship to pocket-tts**

This repository is an independent MLX port, not a wrapper around a local
`pocket-tts` checkout. Keeping the repositories next to each other on disk does
not connect them, and updating `pocket-tts` does not change this package.

The runtime implementation, configuration, and tests live in this repository.
`pocket-tts` appears only in the optional `dev` dependency group for parity
testing; it is not imported or installed by a normal `pocket-tts-mlx` install.
Upstream changes must therefore be reviewed and ported deliberately.

Recent manual ports include:

- [Prompt, WAV, and remote asset fixes](https://github.com/kyutai-labs/pocket-tts/pull/210)
- [English sampling-temperature evaluation](https://github.com/kyutai-labs/pocket-tts/pull/223)
- [The 24-layer English checkpoint](https://github.com/kyutai-labs/pocket-tts/pull/245)

**Installation**

PyPI install:

```bash
pip install pocket-tts-mlx
```

Local development:

```bash
pip install -e .
```

Model weights are downloaded from Hugging Face on first run. For voice cloning
weights, accept the model terms and authenticate:

```bash
hf auth login
```

**Quickstart**

```python
from pocket_tts_mlx import TTSModel

model = TTSModel.load_model()
state = model.get_state_for_audio_prompt("marius")
audio = model.generate_audio(
    state,
    "Hello from MLX!",
    max_tokens=200,
    warmup_frames=1,
    trim_start_ms=40,
    fade_in_ms=15,
)
```

Pronunciation overrides can be supplied programmatically:

```python
from pocket_tts_mlx import TTSModel, UserDictionary

dictionary = UserDictionary.from_dict({
    "english": [
        {"match": "MLX", "replace": "em el ex"},
        {"match": "Xarismata", "replace": "zah riss mah tah"},
    ]
})

audio = model.generate_audio(
    state,
    "MLX is used by Xarismata.",
    dictionary=dictionary,
)
```

**CLI**

Basic usage:

```bash
pocket-tts-mlx "Hello, world!" --voice marius --output output.wav
```

Reproducible generation with an explicit sampling temperature:

```bash
pocket-tts-mlx "Hello, world!" --temperature 0.5 --seed 42
```

Equivalent Python arguments are `temperature` and `seed` on
`generate_audio()` and `generate_audio_stream()`. If no temperature is passed,
the model's configured recommendation is used. Likewise, omitting
`--frames-after-eos` lets the model and prepared text choose the value; pass an
integer only when an explicit override is needed.

**Model Selection**

| Model | Role | Default temperature |
| --- | --- | ---: |
| `english_2026-04` | Recommended six-layer default | `0.3` |
| `english_2026-04_24l` | Opt-in 24-layer quality comparison | `0.3` |
| `b6369a24` | Legacy comparison and rollback | `0.7` |

The April 2026 English checkpoint is the default:

```bash
pocket-tts-mlx "Hello, world!"
```

Use the legacy checkpoint explicitly when comparison or rollback is needed:

```bash
pocket-tts-mlx "Hello, world!" --model b6369a24
```

In Python, `TTSModel.load_model()` uses `english_2026-04`; pass
`TTSModel.load_model("b6369a24")` for the legacy model.

An experimental 24-layer English checkpoint is available as an opt-in quality
comparison. It is substantially larger and slower than the default model:

```bash
pocket-tts-mlx "Hello, world!" --model english_2026-04_24l
```

In Python, use `TTSModel.load_model("english_2026-04_24l")`. The six-layer
`english_2026-04` checkpoint remains the default.

Local validation on a 32 GB Apple Silicon Mac produced the following results.
Generation time is the median of three post-load runs and excludes initial
download and model-loading time. The generated utterances had different
durations, so these measurements show practical resource cost rather than a
controlled model-quality comparison.

| Model | Peak memory | Median generation | Generated audio | Real-time speed |
| --- | ---: | ---: | ---: | ---: |
| Six-layer | 1.04 GiB | 0.301 s | 2.40 s | 8.29x |
| 24-layer | 2.39 GiB | 0.702 s | 3.44 s | 4.92x |

Both models generated finite audio faster than real time. The 24-layer model
loaded its matching weights and predefined voice state successfully, but its
quality benefit should be judged with a listening evaluation before making it
the default.

For faster offline generation, batch completed latent frames through the Mimi
decoder (four is a good starting point):

```bash
pocket-tts-mlx "Hello, world!" --decode-batch-size 4
```

The equivalent Python argument is `decode_batch_size=4`. The default is `1`
to preserve frame-at-a-time streaming latency and exact output. Larger batches
can introduce small floating-point differences in the decoded waveform.

Use a pronunciation dictionary:

```bash
pocket-tts-mlx "MLX is fast." --dictionary ~/.config/pocket-tts/dictionary.yaml
```

Dictionary files may contain language-specific and shared entries:

```yaml
english:
  - match: MLX
    replace: em el ex
  - match: Xarismata
    replace: zah riss mah tah
common:
  - match: "&"
    replace: " and "
```

When `--dictionary` is omitted, the CLI checks
`~/.config/pocket-tts/dictionary.{yaml,yml,json}` automatically. Pass
`--dictionary ""` to disable automatic loading.

Cleaner onset (recommended if startup artifacts are audible):

```bash
pocket-tts-mlx "Hello, world!" --voice marius --output output.wav --warmup-frames 1 --trim-start-ms 40 --fade-in-ms 15
```

**Onset Cleanup Options**

- `--warmup-frames`: decode and discard initial Mimi frames to reduce decoder startup transients.
- `--trim-start-ms`: trim milliseconds from start of output.
- `--fade-in-ms`: apply linear fade-in at start.

Equivalent Python args are `warmup_frames`, `trim_start_ms`, and `fade_in_ms`.

**Performance Note**

`generate_audio()` now materializes generated chunks before returning, so `np.array(audio)` overhead should be near zero in normal usage.

Buffered callers can opt into batched Mimi decoding with
`decode_batch_size=4`. Streaming callers should generally retain the default
of `1` unless throughput matters more than first-audio latency.

**Voices**

Predefined voices:

- alba
- marius
- javert
- jean
- fantine
- cosette
- eponine
- azelma

**Requirements**

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX
- Internet access for initial model downloads

**Notes**

- Voice cloning requires Hugging Face access to `kyutai/pocket-tts`.
- Non-voice-cloning weights are used automatically when voice cloning is unavailable.
- Model weights and predefined voice states are downloaded and cached on first use.
- Updating an adjacent source checkout of `pocket-tts` does not update this package.

**Release Notes**

- Latest: [`v0.2.1`](https://pypi.org/project/pocket-tts-mlx/0.2.1/)
- Full history: [`CHANGELOG.md`](CHANGELOG.md)
- Recommended clean-onset preset:

```bash
pocket-tts-mlx "Hello, world!" --voice marius --output output.wav --warmup-frames 1 --trim-start-ms 40 --fade-in-ms 15
```
