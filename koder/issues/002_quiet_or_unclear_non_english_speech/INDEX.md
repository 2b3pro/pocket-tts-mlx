---
status: open
priority: P2
created: 2026-10-09
updated: 2026-10-09
tags: audio-quality, languages, loudness
type: bug
issue_kind: investigation
context: Owner listened to the 0.3.0 samples. Some languages are quiet or unclear. The 0.3.0 release is on hold until this is settled. Plan 002 drives the work.
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

## Plan

- `koder/plans/002_fix_quiet_or_unclear_non_english_speech/INDEX.md`. The owner put the `0.3.0` release on hold on 9 Oct 2026, so this issue runs before plan 001 L8.

## What is already known

- Review 03 compared the port with upstream torch `41cbc84` at temperature 0. French/`estelle` matched with corr 0.99999 (45.7 dB SNR), so the French model plus `estelle` sounds the same through upstream. German and Dutch were **not** parity-checked.
- No loudness normalization exists in either upstream or this port. Output level is whatever the voice state and model produce.
- Per-language voice states come from `languages/{stem}/embeddings/{name}.safetensors@1e08e6a`. Every one of the 27 names exists for every language, so each language has other voices to try.

## Findings (9 Oct 2026, plan 002 S0–S3)

**Not a port bug.** At temperature 0, the port matches upstream torch `41cbc84` for every language checked: same length, so EOS fires on the same frame, and the same level.

| Model / voice | Corr | SNR | Active RMS torch / port |
| --- | --- | --- | --- |
| english / alba | 0.99860 | 25.5 dB | −20.4 / −20.3 |
| german / juergen | 0.99856 | 25.4 dB | −31.1 / −31.2 |
| dutch / daan | 0.99743 | 22.9 dB | −25.9 / −25.9 |
| french / estelle | 0.99970 | 32.2 dB | −17.8 / −17.8 |

**PyPI `pocket-tts` 3.3.0 is not `41cbc84`.** The `v3.3.0` tag (`3dbee45`, 24 Sep) comes before `41cbc84` (1 Oct, which retrained the French, Italian, Dutch, German, Spanish and Portuguese models and their voice states). So 3.3.0 loads older non-English weights and voices (`@4e1e0a3`), and against it the port diverges (corr ≈ 0). English `alba` is byte-identical at both revisions. A parity check must install upstream from git at `41cbc84`, not `pocket-tts>=3.3.0` (this changes review 03 P3-2).

**The level follows the voice, not the language.** Port, temperature 0.3, one two-sentence line per language, mean of 3 seeds, active RMS in dBFS:

| Voice | German | Dutch | French |
| --- | --- | --- | --- |
| juergen | −26.4 | −25.5 | −27.1 |
| daan | −22.9 | −22.9 | −24.4 |
| estelle | −18.6 | −18.7 | −17.6 |
| giovanni | −22.7 | −23.7 | −25.2 |
| lola | −19.6 | −18.7 | −19.4 |
| rafael | −25.3 | −25.2 | −27.9 |
| alba | −17.3 | −16.9 | −16.8 |
| marius | −24.1 | −26.2 | −31.0 |
| jean | −17.4 | −17.2 | −19.0 |
| eve | −18.4 | −18.2 | −19.6 |
| george | −17.5 | −18.0 | −18.8 |
| caro_davy | −22.5 | −22.5 | −22.5 |

- `juergen` and `daan`, the German and Dutch defaults, are quiet speakers in every language. `estelle` (French) is loud, so French's complaint is clarity, not level.
- Temperature hardly moves the French level: `estelle` is −17.8 at 0, −17.6 at 0.3, −17.2 at 0.5, and −16.9 at 0.7.
- Dutch with `jean` and `george` peaks above 0 dBFS (+0.3 and +1.1). WAV output hard-clips at ±1, the same as upstream's clamp, so there is no wraparound.
- A plain gain with a −1 dBFS peak ceiling lifts German/`juergen` by +6.3 dB but Dutch/`daan` by only +1.0 dB. `daan` is peaky (peak about 20 dB above active RMS), so loudness normalization for Dutch needs a limiter, not just gain.
- 24-layer variants were not tried. Each is 672 MB and the disk has about 10 GB free.
- Listening set: `../samples/issue002/` (outside the repo):
  - German and Dutch × 7 voices, and French with `estelle` at 0.3/0.5/0.7 plus `alba` and `jean`;
  - A/B pairs from the same render: `*_t0.wav` (plain) against `*_t0_normalized.wav` (the real limiter) for German, Dutch, and French;
  - long four-sentence paragraphs with the default voice, plain and normalized (`long_*`);
  - `sweep_all_voices/`: the full 12-voice × 3-language sweep plus the French temperature runs.
  The earlier gain-only previews were replaced.
- Harness: `../parity/` (outside the repo, 914 MB) holds upstream torch at `41cbc84` and the scripts behind every number here. Its `README.txt` covers use and rebuild. Delete it once this issue and review 03 P3-2 are settled.

## Decision and fix (9 Oct 2026)

Owner choices:

- Add opt-in normalization.
- Keep `juergen` and `daan` as defaults.
- Skip the 24-layer variants for now.

What landed (uncommitted):

- `normalize_loudness` in `pocket_tts_mlx/data/audio_utils.py`: −18 dBFS active RMS, a −1 dBFS ceiling, and a 10 ms block limiter.
- `generate_audio(..., normalize=False)` and the CLI `--normalize` flag.
- Four weight-free tests, a changelog line, and a README "Loudness" section.

Measured on real output at temperature 0, active RMS / peak in dBFS:

| Model / voice | Plain | `--normalize` |
| --- | --- | --- |
| german / juergen | −31.2 / −13.2 | −18.3 / −1.0 |
| dutch / daan | −25.9 / −7.1 | −18.1 / −1.0 |
| french / estelle | −17.8 / −6.9 | −18.0 / −7.1 |

Long paragraphs (13–15 s, several chunks, default voice, temperature 0.3, seeds 0 and 1): the plain level is German −26.8/−26.5, Dutch −24.0/−24.2, French −20.1/−19.1. With `--normalize` it is −18.2 to −18.6 with −1.0 peaks. The limiter touches 1.1–1.9% of samples with at most 1.5–4.8 dB of reduction. These are numbers only. The owner has not listened to the normalized files yet.

Without the flag, output is unchanged: German still matches upstream at corr 0.99856. French clarity is the same as upstream, so the port has nothing to fix there. The 24-layer French model is the untried lead if it needs more.

## Investigation leads

1. **Port vs upstream.** Run the temperature-0 parity check for `german`/`juergen` and `dutch`/`daan`. If upstream torch is just as quiet, the cause is the model or voice, not the port. The parity script from review 03 was deleted with scratch; it is short to rewrite (load both, temp 0, compare waveforms).
2. **Voice choice.** Generate German, Dutch, and French with several other voices (for example `alba`, `marius`, `jean`, `eve`) and compare active RMS and clarity. The default voice may be a quiet speaker.
3. **Temperature.** Compare 0.3 against 0.5 and 0.7 for French clarity.
4. **Text.** Try longer, more natural sentences. The samples were short demo lines.
5. **24-layer variants.** Try `french_24l`, `german_24l`, and `dutch_24l` for clarity.
6. **Possible fix in the port.** An opt-in output loudness normalization: target RMS or LUFS, with a peak limit, as a CLI flag and a `generate_audio` argument. Keep it off by default to stay faithful to upstream, unless the owner prefers otherwise.

## Acceptance Criteria

- [x] German and Dutch parity against upstream is measured, and the cause (port vs model/voice) is recorded here. Model/voice: the port matches `41cbc84`.
- [x] Level for French, German, and Dutch is compared across voices, temperature, and long sentences. The owner skipped the 24-layer variants.
- [ ] Clarity is judged by ear: the owner listens to `samples/issue002/`, including the `--normalize` files.
- [x] The owner chooses a fix: different default voices, opt-in normalization, documentation only, or no change. Chosen: opt-in normalization, keep the default voices.
- [x] Any change ships with a test and a `CHANGELOG.md` line before `v0.3.0` is tagged. Not committed yet.
