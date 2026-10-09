---
updated_at: "09 Oct 2026 | 10:47 AM IST"
---

# Koder State

## Past

- v0.2.1 (`b1fe6a4`) is the published MLX port. PyPI `0.2.1`; remote tag `v0.2.1` → `1342d4d`.
- Koder scaffold: `0d1e841`. `CHANGELOG.md` stays the history surface.
- `d37c30b`: analysis 001, issue 001 (track), plan 001 rev 2, reviews 01 and 02. Upstream read at `41cbc84`.
- `59d499f` L0 through `dcb7c0a` L8: plan 001 is fully built. Version `0.3.0` comes from `pocket_tts_mlx.__version__`. `MANIFEST.in` keeps `tests/` and `koder/` out of the sdist.
- `87bcf29`: build review `koder/reviews/001_update_pocket_tts_mlx/03_build_review.md`, `approve with fixes` (0 P1, 2 P2, 5 P3). Evidence:
  - temperature-0 parity with upstream torch: same lengths, corr ≥ 0.999;
  - all 20 configs strictly match their published headers;
  - all 513 voice files (27 names × 19 stems) exist;
  - wheel and sdist contents are correct.
- Samples for all 7 languages, drifting English and `b6369a24` were generated and the owner listened. Mostly good, but German and Dutch are quiet and French is unclear. Filed as `koder/issues/002_quiet_or_unclear_non_english_speech/INDEX.md`.

## Present

- Ledger `koder/issues/001_update_to_current_pocket_tts/INDEX.md`: L0–L8 `done`. Version `0.3.0` is not tagged, not pushed, and not on PyPI.
- Issue 002 (P2, open): measured active RMS vs English is German −12.0 dB and Dutch −7.1 dB. French is at normal level, so its problem is clarity. French matches upstream at temperature 0; German and Dutch are not parity-checked yet.
- Release blockers from review 03, still open:
  - P2-1: tag the final commit, not `d2d8f44`, because an sdist from that commit would include `tests/` and `koder/`;
  - P2-2: `CHANGELOG.md` lacks the Python `warmup_frames` 1 → 0 break, strict-load errors, the `tokenizers` dependency, and a `### Fixed` section.
- Optional P3 items:
  - absolute CHANGELOG links in the README;
  - a parity test plus `dev` extra `pocket-tts>=3.3.0`;
  - a completeness check in `import_flow_state`;
  - the setuptools license format (deadline 2027-02-18);
  - "Current release" in `koder/AGENTS.md`, to update after publish.
- Untested: cloning from your own audio with the gated `kyutai/pocket-tts` weights (needs `hf auth login`).
- Validation: `.venv/bin/python -m pytest -q` → 64 passed. Scratch gate clean. `git diff --check` clean.
- Disk:
  - test scratch, the `/tmp/pocket-tts-upstream` clone, `dist/`, `egg-info`, and caches are deleted;
  - still kept: `../samples/` (2.3 MB, needed for issue 002), `.venv` (449 MB), and the HF cache `~/.cache/huggingface/hub/models--kyutai--pocket-tts-without-voice-cloning` (~237 MB on disk).
- Local `main` is 21 commits ahead of `origin/main` after this close. Nothing pushed.

## Future

- Next action: investigate issue 002. Start with temperature-0 parity for `german`/`juergen` and `dutch`/`daan` against upstream torch, then try other voices, temperatures, and the 24-layer variants. The owner then picks a fix.
- Then fix P2-2 (changelog) and the README links, run pytest, and tag the final commit as `v0.3.0` (owner-gated).
- Owner decisions: the fix for issue 002; push, tag, and PyPI publish; delete `samples/` and the HF cache once issue 002 is settled.
- Use `.venv` only. `a separate kyutai venv` imports a different checkout.
- Run pytest and any other MLX import with the sandbox disabled. Skip the sandboxed run.
