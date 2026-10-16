---
updated_at: "16 Oct 2026 | 04:30 PM +08"
---

# Koder State

## Past

- v0.2.1 (`b1fe6a4`) is the published MLX port. PyPI `0.2.1`; remote tag `v0.2.1` → `1342d4d`.
- Koder scaffold: `0d1e841`. `CHANGELOG.md` stays the history surface.
- Plan 001 (L0–L7) built `0.3.0` against upstream `41cbc84`. Build review `koder/reviews/001_update_pocket_tts_mlx/03_build_review.md` was `approve with fixes`; P2-2, P3-1 and P3-2 are done.
- Plan 002 (`koder/plans/002_fix_quiet_or_unclear_non_english_speech/INDEX.md`): no port bug. At temperature 0 the port matches upstream `41cbc84` (German corr 0.99856, Dutch 0.99743, French 0.99970). The level follows the voice (`juergen`, `daan` quiet); `estelle` is dark, not unclear (WER 0.00), identical in upstream.
- `4fb173a`: opt-in `--normalize` / `generate_audio(normalize=True)`, −18 dBFS active RMS, −1 dBFS limiter. The owner listened: "very good".
- `acba497`: opt-in `--brighten` / `brighten=True` (+10 dB high shelf at 1.5 kHz) for dark voices such as `estelle`.
- `40d14ee`: P3-2. The README parity sentence and the `pocket-tts` dev pin are dropped. Harness notes and scripts are in `koder/docs/`.
- `12f7c20` (16 Oct 2026): issue 002 closed, `CHANGELOG.md` dates `0.3.0` as 2026-10-16, and the upstream `estelle` report is drafted (`koder/docs/upstream-estelle-report.md`, not sent).

## Present

- Issue 002 is closed. Issue 001 stays open only for the release. The plan 001 L8 checks passed on the final code: build, `twine check`, 20 YAMLs, no `koder/` or `tests/` in the dists, and a fresh-venv CLI smoke with `--brighten --normalize`.
- `0.3.0` is tagged (`v0.3.0`), pushed, and published on PyPI (16 Oct 2026). History after `v0.2.1` was rewritten to drop local paths; the local `backup-prerewrite` branch holds the old history and can be deleted.
- Validation: `.venv/bin/python -m pytest -q` → 74 passed; scratch gate and `git diff --check` clean at close.
- Outside the repo, kept for the owner:
  - `../samples/issue002/` (A/B pairs, `french_clarity/`, sweeps) and `../samples/*.wav`;
  - `../parity/` (914 MB upstream torch venv). It is no longer needed in-repo: see `koder/docs/parity-harness.md`;
  - `.venv` and the HF cache `~/.cache/huggingface/hub/models--kyutai--pocket-tts-without-voice-cloning`.

## Future

- Next action (owner): optionally create a GitHub release from `v0.3.0`; delete `backup-prerewrite`.
- Upstream `estelle` report: re-run the repro against current upstream `main`, check for an existing issue, then send it if wanted (draft is in `koder/docs/`).
- Optional, unscheduled: review 03 P3-3, P3-4 (license format, deadline 2027-02-18), P3-5 (AGENTS.md "Current release" after publish). Cloning from your own audio is untested.
- Delete `../parity/` and the sweep voices whenever the owner is done with them.
- Use `.venv` only. `a separate `kyutai` venv outside the repo` imports a different checkout. Run pytest and any MLX import with the sandbox disabled.
