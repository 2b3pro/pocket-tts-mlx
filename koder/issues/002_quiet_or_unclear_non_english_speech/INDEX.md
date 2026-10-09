---
status: open
priority: P2
created: 2026-10-09
updated: 2026-10-09
tags: audio-quality, languages, loudness
type: bug
issue_kind: investigation
context: Owner listened to the 0.3.0 samples. Some languages are quiet or unclear. Investigate before tagging 0.3.0.
---

# Issue 002: Some non-English samples are quiet or unclear

## Problem

The owner listened to the 0.3.0 samples (`../samples/`, outside the repo). Overall they sound good, but:

- **Dutch** (`daan`) is not loud enough.
- **German** (`juergen`) is not loud enough.
- **French** (`estelle`) is not clear, and possibly not loud enough.

## Measurements

Default voice, default temperature 0.3, one sample each, `TTSModel.load_model(language=...)`. "Active RMS" covers only the 20 ms frames above 10% of the loudest frame.

| Sample | Peak | Active RMS | vs English |
| --- | --- | --- | --- |
| english / alba | −4.0 dBFS | −18.1 dBFS | — |
| english_drifting / alba | −4.3 | −17.5 | +0.6 dB |
| b6369a24 / alba | −6.2 | −18.6 | −0.5 dB |
| french / estelle | −6.7 | −18.4 | −0.3 dB |
| italian / giovanni | −4.8 | −21.2 | −3.1 dB |
| spanish / lola | −7.3 | −21.5 | −3.4 dB |
| portuguese / rafael | −6.8 | −21.8 | −3.7 dB |
| dutch / daan | −7.6 | −25.2 | **−7.1 dB** |
| german / juergen | −11.7 | −30.1 | **−12.0 dB** |

German and Dutch really are quieter. French has normal loudness, so its complaint is clarity, not level.

## What is already known

- Review 03 compared the port with upstream torch `41cbc84` at temperature 0. French/`estelle` matched with corr 0.99999 (45.7 dB SNR), so the French model plus `estelle` sounds the same through upstream. German and Dutch were **not** parity-checked.
- No loudness normalization exists in either upstream or this port. Output level is whatever the voice state and model produce.
- Per-language voice states come from `languages/{stem}/embeddings/{name}.safetensors@1e08e6a`. Every one of the 27 names exists for every language, so each language has other voices to try.

## Investigation leads

1. **Port vs upstream.** Run the temperature-0 parity check for `german`/`juergen` and `dutch`/`daan`. If upstream torch is just as quiet, the cause is the model or voice, not the port. The parity script from review 03 was deleted with scratch; it is short to rewrite (load both, temp 0, compare waveforms).
2. **Voice choice.** Generate German, Dutch, and French with several other voices (for example `alba`, `marius`, `jean`, `eve`) and compare active RMS and clarity. The default voice may be a quiet speaker.
3. **Temperature.** Compare 0.3 against 0.5 and 0.7 for French clarity.
4. **Text.** Try longer, more natural sentences. The samples were short demo lines.
5. **24-layer variants.** Try `french_24l`, `german_24l`, and `dutch_24l` for clarity.
6. **Possible fix in the port.** An opt-in output loudness normalization: target RMS or LUFS, with a peak limit, as a CLI flag and a `generate_audio` argument. Keep it off by default to stay faithful to upstream, unless the owner prefers otherwise.

## Acceptance Criteria

- [ ] German and Dutch parity against upstream is measured, and the cause (port vs model/voice) is recorded here.
- [ ] Clarity and level for French, German, and Dutch are compared across voices, temperature, and 24-layer variants.
- [ ] The owner chooses a fix: different default voices, opt-in normalization, documentation only, or no change.
- [ ] Any change ships with a test and a `CHANGELOG.md` line before `v0.3.0` is tagged.
