---
updated_at: "09 Oct 2026 | 08:02 PM +08"
---

# Koder State

## Past

- v0.2.1 (`b1fe6a4`) is the published MLX port. PyPI `0.2.1`; remote tag `v0.2.1` → `1342d4d`.
- Koder scaffold: `0d1e841`. `CHANGELOG.md` stays the history surface.
- Plan 001 (L0–L7) built `0.3.0` against upstream `41cbc84`. Build review `koder/reviews/001_update_pocket_tts_mlx/03_build_review.md` is `approve with fixes`.
- 9 Oct 2026: the owner held the `0.3.0` release for issue 002. Plan 001 L8 and the issue 001 ledger say "on hold".
- Plan 002 (`koder/plans/002_fix_quiet_or_unclear_non_english_speech/INDEX.md`, rev 2), recorded in `1c4c3e7`:
  - No port bug. At temperature 0 the port matches upstream `41cbc84`: German corr 0.99856, Dutch 0.99743, French 0.99970, with the same lengths and levels.
  - PyPI `pocket-tts` 3.3.0 is older than `41cbc84` and ships older non-English models. It is not a parity reference.
  - The level follows the voice. `juergen` and `daan` are quiet in every language. `estelle` is loud, so French's problem is clarity, which equals upstream.
- `4fb173a`: opt-in loudness normalization, as the owner chose. Default voices are kept and 24-layer is skipped.
  - `generate_audio(normalize=False)` and the CLI `--normalize` flag aim for −18 dBFS active RMS with a −1 dBFS limiter. The stream is not normalized.
  - With the flag: German −31.2 → −18.3, Dutch −25.9 → −18.1. On long text the limiter touches 1–2% of samples, cutting at most 4.8 dB.
  - The same commit fixes review 03 P2-2 (the full changelog) and P3-1 (absolute README links).

## Present

- Issue 002 (P2, open) and issue 001 (open, L8 on hold). `0.3.0` is not tagged, pushed, or published. The owner will not release yet.
- Validation:
  - `.venv/bin/python -m pytest -q` → 70 passed;
  - an offline smoke of all 9 shipped stems with `normalize=True` passes;
  - scratch gate clean; `git diff --check` clean.
- Kept for the owner's review, all outside the repo:
  - `../samples/issue002/`: A/B pairs `*_t0.wav` against `*_t0_normalized.wav`, `long_*` plain against normalized, 7-voice sets, and `sweep_all_voices/`;
  - `../samples/*.wav`: the original 0.3.0 samples;
  - `../parity/` (914 MB): upstream torch venv at `41cbc84` plus scripts, with `README.txt` for use and rebuild;
  - `.venv` and the HF cache `~/.cache/huggingface/hub/models--kyutai--pocket-tts-without-voice-cloning`. It holds every model the port loads plus 12 voices × de/nl/fr at `1e08e6a`.
- Deleted: 676 MB of stale old-revision cache files (old German/Dutch/French models and voices from PyPI 3.3.0), after checking that nothing else referenced them.
- Local `main` is ahead of `origin/main` (24 commits after this close). Nothing pushed.

## Future

- Next action (owner):
  - listen to `../samples/issue002/`. Compare `*_t0` with `*_t0_normalized` and `long_*` with `long_*_normalized` for pumping, dulling, or clicks. Judge French clarity too;
  - choose review 03 P3-2. Recommended: drop the README "parity tests" sentence and the `pocket-tts` pin in `dev`, and keep the harness note under `koder/`. Or add a skip-guarded `tests/test_parity.py`.
- After the owner signs off:
  - apply P3-2;
  - close issue 002;
  - set the `CHANGELOG.md` date;
  - rerun every plan 001 L8 check on the final commit (build, `twine check`, 20 YAMLs, no `tests/`/`koder/` in the dists, fresh-venv CLI smoke).
  - Tag, push, and PyPI publish only when the owner asks.
- Optional, unscheduled: review 03 P3-3, P3-4 (license format, deadline 2027-02-18), P3-5 (AGENTS.md "Current release" after publish). Cloning from your own audio is untested.
- Delete `../parity/` and the sweep voices once issue 002 and P3-2 are settled.
- Use `.venv` only. `a separate kyutai venv` imports a different checkout. Run pytest and any MLX import with the sandbox disabled.
