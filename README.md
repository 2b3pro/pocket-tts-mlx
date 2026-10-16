# pocket-tts-mlx

MLX backend for [pocket-tts](https://github.com/kyutai-labs/pocket-tts) optimized for Apple Silicon.

Runtime is torch-free.

**What's New in 0.3.0**

- Default model is `english` (the current English weights) at temperature 0.3.
- Shipped languages include french, german, dutch, italian, portuguese, and spanish, plus 24-layer previews and a drifting English sampler.
- Current models use `tokenizer.json`. `b6369a24` still uses SentencePiece.
- Built-in voices follow the language. `--language french` uses `estelle`.
- Opt-in `--normalize` raises quiet voices to a common level. Opt-in `--brighten` lifts dark voices such as french `estelle`.
- Release notes: [v0.3.0](https://github.com/jishnuvenugopal/pocket-tts-mlx/blob/main/CHANGELOG.md).

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
state = model.get_state_for_audio_prompt("alba")
audio = model.generate_audio(state, "Hello from MLX!")
```

`load_model()` loads `english`. `load_model("b6369a24")` still loads the previous weights. Pass `temp=0.7` when you want the old temperature. `load_model(language="french")` selects another shipped config. Pass `config` or `language`, not both.

**CLI**

```bash
pocket-tts-mlx --language english -o hello.wav "Hello, world!"
pocket-tts-mlx --language french -o bonjour.wav "Bonjour."
```

`--temperature` overrides the config value. `--max-tokens` defaults to 50, `--frames-after-eos` is unset, and `--warmup-frames` defaults to 0. Text `-` reads stdin.

**Voices**

The default voice follows the language:

- english, and any other stem: `alba`
- french: `estelle`
- italian: `giovanni`
- german: `juergen`
- spanish: `lola`
- portuguese: `rafael`
- dutch: `daan`

Other built-in names (`marius`, `javert`, `jean`, and the rest of the upstream catalog) work with a model loaded from a shipped config. `b6369a24` keeps its original eight voices as audio prompts. A config from outside this package does not accept those names; pass an audio file instead.

**Loudness**

Output level depends on the voice. `juergen` (german) and `daan` (dutch) come out 6 to 9 dB quieter than `alba` or `estelle`, in upstream Pocket TTS as well. `--normalize`, or `generate_audio(..., normalize=True)`, scales speech to about -18 dBFS and holds peaks under -1 dBFS. It is off by default, so output matches upstream unless you ask. `generate_audio_stream` does not normalize.

```bash
pocket-tts-mlx --language german --normalize -o hallo.wav "Guten Morgen."
```

**Brightness**

The french `estelle` voice sounds muffled: above 2 kHz it is about 10 dB below `alba`, in upstream Pocket TTS as well. `--brighten`, or `generate_audio(..., brighten=True)`, adds a +10 dB high shelf at 1.5 kHz. It is off by default. Combined with `--normalize`, brightening runs first. `generate_audio_stream` does not brighten.

```bash
pocket-tts-mlx --language french --brighten --normalize -o bonjour.wav "Bonjour."
```

**Onset options**

`--warmup-frames`, `--trim-start-ms`, and `--fade-in-ms` still trim or fade the start of the file. They default to 0. Each chunk also fades its first 5 ms on its own.

**Requirements**

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)
- MLX
- Internet access for initial model downloads

**Notes**

- Voice cloning requires Hugging Face access to `kyutai/pocket-tts`.
- Non-voice-cloning weights are used automatically when voice cloning is unavailable.

**Release Notes**

- Latest: [v0.3.0](https://github.com/jishnuvenugopal/pocket-tts-mlx/blob/main/CHANGELOG.md)
- Previous package: [PyPI 0.2.1](https://pypi.org/project/pocket-tts-mlx/0.2.1/)
