# DRAFT, not sent: upstream report for kyutai-labs/pocket-tts

Status: drafted 16 Oct 2026 from issue 002. Do not post without the owner's go-ahead.
Suggested title: `french` voice `estelle` is about 10 dB darker than the other voices above 2 kHz

## Summary

With the `french` model, the built-in `estelle` voice renders speech with much less energy above 2 kHz than the other voices. It is intelligible (a local Whisper-medium transcribes it with 0.00 WER) but sounds muffled. The same holds for `estelle` in German and Dutch, so it looks like a property of the voice conditioning rather than the language model.

Tested at upstream commit `41cbc84af539ea78a804ffca5f9c6edc1a22ce44` (torch), temperature 0.3, seed 0. An independent MLX port gives the same numbers, so this is not a porting artifact.

## Repro

```python
import torch, soundfile as sf
torch.manual_seed(0)
from pocket_tts import TTSModel

text = "Bonjour. Aujourd'hui il fait beau, et nous allons nous promener dans le parc cet après-midi."
m = TTSModel.load_model(language="french", temp=0.3)
for voice in ("estelle", "alba", "eve"):
    torch.manual_seed(0)
    st = m.get_state_for_audio_prompt(voice)
    a = m.generate_audio(st, text).detach().cpu().numpy().reshape(-1)
    sf.write(f"{voice}.wav", a, 24000)
```

Then compare the band energies of active frames (STFT, 1024 window, frames above 10% of the loudest frame's energy; share of active energy per band, in dB):

| voice | 0.5–2 kHz | 2–4 kHz | 4–8 kHz | 95% roll-off |
| --- | --- | --- | --- | --- |
| estelle | −17.5 | −37.5 | −43.7 | 492 Hz |
| alba | −12.0 | −27.4 | −33.5 | 562 Hz |
| eve | −3.6 | −21.5 | −30.7 | 1289 Hz |

Torch at `41cbc84`, same line, `estelle` against `alba`: 2–4 kHz −37.0 against −26.4.

Raising the temperature to 0.7 moves the 2–4 kHz figure by only about 3 dB, so the sampler is not the cause. The script that computes the table is `koder/docs/parity-scripts/spec.py` in the port repo.

## What would help

- Confirm whether `estelle` is meant to be this dark, or whether its reference audio or voice embedding was band-limited (for example a low-sample-rate source).
- If it is unintended, re-export the voice from brighter source audio.

## Workaround

A +10 dB high shelf at 1.5 kHz brings `estelle`'s 2–8 kHz balance within about 1 dB of `alba` without changing intelligibility. The MLX port ships it as an opt-in `--brighten` flag.

## Before sending

- Checked 16 Oct 2026: upstream `main` (`ec569b2`) is 6 commits past `41cbc84`, and none touch the `french` config, weights or voices (changes are English configs, docs, tests and CLI). The numbers should carry over; a fresh run on `main` is optional.
- Searched the upstream issues for `estelle` and French quality: no existing report (nearest are #174 and #221, unrelated).
- Attach `short_1_estelle_plain_normalized.wav` and `short_4_ref_alba_normalized.wav` from `samples/issue002/french_clarity/` (they are outside the repo).
- Check the upstream contribution guide.
