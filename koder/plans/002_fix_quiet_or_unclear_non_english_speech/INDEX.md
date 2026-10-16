---
plan: 002
title: Fix quiet German/Dutch and unclear French before 0.3.0
issue: 002
type: investigation-then-fix
status: active
revision: 2
created: 2026-10-09
updated: 2026-10-09
---

# Plan 002: Fix quiet German/Dutch and unclear French before 0.3.0

## User Decisions Captured

- 9 Oct 2026: the owner holds the `0.3.0` release. No tag, push, or PyPI publish until issue 002 is settled. Plan 001 L8 is on hold until S5 below.
- Work on the issue 002 fixes first. The owner still chooses the fix in S4 from the measured options.
- Default output stays faithful to upstream torch unless the owner chooses otherwise. A port bug, if found, is fixed without asking.
- Scratch tooling (upstream venv, parity scripts, sweep wavs) lives outside the repo. Only conclusions, numbers, and any committed test land in the repo.

## Capability Statement

German, Dutch, and French speech from `pocket-tts-mlx` 0.3.0 is as loud and clear as the upstream models allow, and the defaults or the docs lead users to a good voice and level.

## Triage

- Tier: M. Investigation first, then a small product change chosen by the owner.
- Risk: green for measurement. Yellow for any change to default voices or output level, because it moves the port away from upstream.
- Review posture: S1 decides the branch. A port bug becomes a fix plus a regression test. A model or voice cause becomes an owner decision.

## Steps

### S0: Upstream parity harness (scratch)

- Make a scratch venv outside the repo with torch and upstream installed from git at `41cbc84`. PyPI `pocket-tts==3.3.0` is older than `41cbc84` and loads older non-English weights and voices. Do not use `a separate `kyutai` venv outside the repo`; it has `pocket-tts` 1.0.3.
- Write a short script: the same text, voice, and model at temperature 0 in both, then compare length, correlation, SNR, peak, and active RMS (20 ms frames above 10% of the loudest).
- Proof: `english`/`alba` reproduces review 03 (same length, corr ≈ 0.999).
- **Done 9 Oct:** corr 0.99860, same length.
- The harness is kept at `../parity/` (venv `upstream-41cbc84-venv/`, `scripts/`, `README.txt`), outside the repo. Delete it after S4's listen and the P3-2 choice.

### S1: German and Dutch parity

- Run `german`/`juergen` and `dutch`/`daan` at temperature 0 on two or three sentences.
- If upstream is just as quiet, the cause is the model or voice. Record it in issue 002.
- If the port is quieter or differs (corr < 0.99 or a different length), it is a port bug. Stop, find it, fix it, and add a regression test.
- **Done 9 Oct: no port bug.** German corr 0.99856, Dutch 0.99743, French 0.99970, with the same lengths and levels. Upstream German/`juergen` is also −31 dBFS active RMS.

### S2: Voice sweep

- For `german`, `dutch`, and `french`, generate the same text with the per-language defaults (`juergen`, `daan`, `estelle`, `giovanni`, `lola`, `rafael`, `alba`) and a few English voices (`marius`, `jean`, `eve`).
- Record peak and active RMS for each. Keep the wavs in scratch so the owner can listen and judge clarity.
- **Done 9 Oct** with 12 voices. The level follows the voice: `juergen` and `daan` are quiet in every language. The table and listening set are in issue 002.

### S3: Temperature, text, and 24-layer variants

- French clarity: temperatures 0.3, 0.5, and 0.7 with `estelle` and the best S2 voice.
- `french_24l`, `german_24l`, and `dutch_24l` with the default voice.
- Longer, natural sentences next to the short demo lines.
- Proof: a table in issue 002 and a short listening set in scratch.
- **Done 9 Oct, except 24-layer:** temperature barely changes the French level. The owner skipped 24-layer (3 × 672 MB with about 10 GB free).
- Long sentences: a four-sentence paragraph per language (13–15 s, several chunks) with the default voice at 0.3, seeds 0 and 1. Active RMS: German/`juergen` −26.8 and −26.5, Dutch/`daan` −24.0 and −24.2, French/`estelle` −20.1 and −19.1. Both runs end normally. The quiet voices stay quiet on long text.

### S4: Owner decision and fix

Options, depending on S1–S3:

- (a) A port fix, if S1 finds one. This is not optional.
- (b) Different per-language default voices in the CLI.
- (c) Opt-in output loudness normalization: a target level with a peak limit, as a `generate_audio` argument and a CLI flag, off by default. Gain alone is not enough for Dutch/`daan` (+1 dB before the −1 dBFS ceiling), so this needs a simple limiter. It needs the whole clip, so `generate_audio_stream` stays unnormalized.
- (d) Documentation only: the README says which voices are quiet and how to raise the level.
- (e) No change.

Any change ships with a weight-free test where possible, a `CHANGELOG.md` line, and README text.

- **Done 9 Oct (uncommitted).** The owner chose (c) and kept the default voices. 24-layer is skipped. `generate_audio_stream` is not normalized; the README says so. `normalize_loudness`, `generate_audio(normalize=False)`, and `--normalize` are in, with 4 tests, a changelog line, and a README "Loudness" section. With the flag: German −31.2 → −18.3, Dutch −25.9 → −18.1 dBFS active RMS, peaks −1.0.
- Tests check the levels, the default (`normalize=False`), `generate_audio(normalize=True)` with a stub stream, and the CLI passing `--normalize` through.
- Limiter on the long paragraphs: it touches 1.1–1.9% of samples with at most 1.5–4.8 dB of reduction. Output is −18.2 to −18.6 dBFS active RMS with −1.0 peaks. French/`estelle` needs no limiting.
- **Owner listen (open, before the hold lifts):** in `samples/issue002/`, compare `*_t0.wav` with `*_t0_normalized.wav` and `long_*.wav` with `long_*_normalized.wav`. Each pair is the same render. Check for pumping, dulling, or clicks on German/`juergen` and Dutch/`daan`, short and long. The numbers above do not replace this.

### S5: Release records, then resume plan 001 L8

- Review 03 P2-2: add the `warmup_frames` 1 → 0 API break, strict-load errors, the `tokenizers` dependency, and a `### Fixed` section. Add the issue 002 change.
- Review 03 P3-1: make the README `CHANGELOG.md` links absolute.
- Review 03 P3-2 (optional, owner choice, not done). Today the README says torch is "only required for optional parity tests", `pyproject.toml` has `dev = [..., "pocket-tts>=1.0.3"]`, and `tests/` has no parity test. PyPI `pocket-tts` 3.3.0 is older than `41cbc84`, and PyPI rejects direct git references in package metadata, so the extra cannot point at the right upstream. The branches:
  - drop the README sentence and the `pocket-tts` pin from `dev`, and keep the parity script as a dev note under `koder/` that installs upstream from git at `41cbc84` (recommended); or
  - add `tests/test_parity.py`, which skips unless upstream `41cbc84` is importable, and document the git install in the README instead of an extra.
- Optional, not scheduled for 0.3.0: review 03 P3-3 (`import_flow_state` completeness check), P3-4 (setuptools license format, deadline 2027-02-18), and P3-5 ("Current release" in `koder/AGENTS.md`, after publish). Cloning from your own audio stays untested and is outside this plan.
- `.venv/bin/python -m pytest -q` passes.
- Resume plan 001 L8 on the final commit (review 03 P2-1). Tag, push, and publish only when the owner asks.
- **Records done 9 Oct (uncommitted):** P2-2 (`### Fixed`, the API `warmup_frames` break, strict load, `tokenizers`) and P3-1 (absolute links). pytest: 70 passed.
- Still to do: the owner listen (S4), the P3-2 choice, a commit, the changelog date at tag time, and then a full L8 rerun. The review 03 wheel check came before this plan's code and doc changes. The issue 001 ledger marks L8 on hold.

## Acceptance Criteria

- [x] S0 reproduces review 03 parity for `english`/`alba`.
- [x] German and Dutch parity is measured. Issue 002 records whether the cause is the port or the model/voice.
- [ ] Issue 002 has level numbers for the S2 voices and the S3 variants, and the listening set is in scratch.
- [x] The owner has chosen a fix from S4.
- [x] The chosen fix is in, with a test, a changelog line, and README text.
- [x] Review 03 P2-2 and P3-1 are fixed and pytest passes.
- [x] Long sentences are measured (S3).
- [ ] The owner has listened to `--normalize` on German and Dutch, short and long.
- [ ] The owner has chosen a review 03 P3-2 branch, or deferred it.
- [ ] L8 is rerun on the final commit.
- [ ] The release hold is lifted only by the owner.

## Stop Rules

- A parity mismatch on German or Dutch stops the sweep until the port cause is found.
- Do not change default voices, default levels, or upstream-facing behavior without an owner choice.
- Do not tag, push, or publish.
