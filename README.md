# pocket-tts-mlx

MLX backend for [pocket-tts](https://github.com/kyutai-labs/pocket-tts) optimized for Apple Silicon.

Runtime is torch-free. Torch is only required for optional parity tests.

**Current Development (`0.3.1.dev0`, unreleased)**

- Aligned with MLX upstream `v0.3.0` and Kyutai inference changes through
  [`44cb63e`](https://github.com/kyutai-labs/pocket-tts/commit/44cb63eaa4144f32ff80863968515165a5554353)
  (October 22, 2026).
- Defaults to the October 2026 English checkpoint at temperature `0.3`.
- Supports English, French, German, Dutch, Italian, Portuguese, and Spanish,
  24-layer previews, drifting, and the six October small English models.
- Retains pronunciation dictionaries, per-request temperature and seed controls,
  buffered missing-EOS recovery, batched decoding, and fused MLX kernels.
- Includes opt-in output normalization/brightening, strict weight validation,
  model-specific text preparation, and cancellable synchronous streaming.

**Relationship to pocket-tts**

This is an independent MLX port. Updating an adjacent `pocket-tts` checkout
changes neither this package nor its pinned models. Upstream changes are
reviewed and ported deliberately; PyTorch is an optional development dependency.
The runtime does not import `pocket_tts` or Torch.

See [the upstream alignment record](koder/docs/upstream-alignment.md) for the
source snapshots, retained fork features, validation, and scope.

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

| Model/config | Role | Default temperature |
| --- | --- | ---: |
| `english` / `english_2026-10` | October English default | `0.3` |
| `english_2m_2026-10`, `english_7m_2026-10`, `english_18m_2026-10` | Small English models | `0.3` |
| `english_2m_refl_2026-10`, `english_7m_refl_2026-10`, `english_18m_refl_2026-10` | Reflect-padding small models | `0.3` |
| `french`, `german`, `dutch`, `italian`, `portuguese`, `spanish` | Other languages | `0.3` |
| `english_drifting_26-09` | September one-step drifting model | `0.3` |
| `english_2026-04`, `english_2026-04_24l`, `english_2026-09_24l` | Historical and quality comparisons | `0.3` |
| `b6369a24` | Original model; pass `--temperature 0.7` for its old setting | `0.3` |

```bash
pocket-tts-mlx "Hello, world!"
pocket-tts-mlx "Bonjour le monde." --language french
pocket-tts-mlx "Hello from a small model." --model english_2m_2026-10
pocket-tts-mlx "Hello, world!" --model english_2026-04_24l
```

`--config` and the backward-compatible `--model` alias select a shipped config
stem, local YAML, or remote YAML. They cannot be combined with `--language`.
`--eos-threshold` overrides the model's recommendation (`-1.0` for the small
models; `-4.0` otherwise). `--sampler-decode-steps` controls the sampler;
flow-matching configs generally need at least 16 steps, while LSD, drifting,
and pointwise configs use one.

In Python, `TTSModel.load_model()` selects `english`;
`TTSModel.load_model(language="french")` selects French.
Passing a config as the first positional argument remains supported.
The old `lsd_decode_steps` keyword is retained as a deprecated alias for
`sampler_decode_steps`.

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

The CLI chooses a voice for the selected language: `alba` (English), `estelle`
(French), `juergen` (German), `daan` (Dutch), `giovanni` (Italian), `rafael`
(Portuguese), and `lola` (Spanish). All current language configs support the
upstream 27-name voice catalog; the original `b6369a24` catalog has eight names:


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

**Development and Validation**

```bash
uv pip install --python .venv/bin/python -e . pytest
.venv/bin/python -m pytest -q
POCKET_TTS_RUN_INTEGRATION=1 .venv/bin/python -m pytest -q tests/test_speech.py
```

The regular suite runs without network access. Integration tests download
pinned weights and exercise real generation. The raw reference-voice test
requires access to the gated voice-cloning weights.

Streaming stays on the calling thread. Set a `threading.Event` passed as
`generate_audio_stream(..., stop=event)` to cancel, or close the generator.
Use separate model instances for concurrent generation.

Use `--normalize` to bring quiet speech to about -18 dBFS with peaks below
-1 dBFS, and `--brighten` for a +10 dB high shelf at 1.5 kHz. Both are off by
default. Python equivalents are `generate_audio(normalize=True, brighten=True)`.
Serialized voice `.safetensors` files can be supplied as local paths, `hf://`
paths, or HTTP(S) URLs.

**Release Notes**

- This fork is an unreleased development build.
- MLX upstream: [`v0.3.0`](https://github.com/jishnuvenugopal/pocket-tts-mlx/releases/tag/v0.3.0)
- Full history: [`CHANGELOG.md`](CHANGELOG.md)
- Recommended clean-onset preset:

```bash
pocket-tts-mlx "Hello, world!" --voice marius --output output.wav --warmup-frames 1 --trim-start-ms 40 --fade-in-ms 15
```
