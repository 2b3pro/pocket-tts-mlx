---
plan: 001
issue: 001
type: build-review
verdict: approve with fixes
p1: 0
p2: 2
p3: 5
reviewer: claude
created: 2026-10-08
reviewed_range: b1fe6a4..30c4a72
---

# Review 03: Plan 001 build (L0–L8), before tagging 0.3.0

## Summary

Reviewed the whole product delta since v0.2.1 (38 files, about 2,060 lines) against plan 001 rev 2 and upstream Pocket TTS `41cbc84`. The implementation is correct. No finding blocks the release. Two release-record items should be fixed before tagging `v0.3.0`.

## Evidence

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` | 64 passed. The live tests ran on cached weights with no skips: english+alba, b6369a24+alba, french+estelle, english_drifting+alba. |
| Temperature-0 parity vs upstream torch `41cbc84` (ad hoc script, not committed) | english: same length (3.20 s), corr 0.99912, SNR 27.5 dB. french: same length, corr 0.99999, 45.7 dB. drifting: same length, corr 0.99975, 32.9 dB. The first 120 samples (onset fade) are identical. Equal lengths mean EOS fires on the same frame. |
| Strict name+shape check, all 20 shipped configs vs published headers | All OK: 213 keys (`b6369a24`, `english_2026-01`), 214 (6-layer), 358 (24-layer), 202 (drifting). |
| Voice states at `1e08e6a` | 27 names × 19 config stems = 513 files, all present. The README claim holds. |
| `python -m build` + `twine check` | Both pass. The wheel holds 20 YAMLs. No `tests/` or `koder/` in the wheel or sdist. Metadata says `Version: 0.3.0` and requires `tokenizers>=0.21`. |
| Code read against upstream | The following match: EOS gating (frame 6), per-chunk 5 ms fade, frames-after-EOS order, prepared chunk text, `end_on_pause`, BOS-before-voice, sampler dispatch, and the voice import mapping `offset` → `current_end` with `pad` checked. |

## Findings

### P2-1: Tag the final commit, not `d2d8f44`

`koder/STATE.md` offers to tag `v0.3.0` on `d2d8f44` "or the later packaging commit." `MANIFEST.in` only arrives in `dcb7c0a`, so an sdist built from `d2d8f44` would contain `tests/` and `koder/`. Tag the final release commit, after the changelog fix below, and build the release artifacts from that tag.

### P2-2: The changelog leaves out library-facing changes

The `v0.3.0` entry lists the CLI breaks but not:

- `generate_audio` / `generate_audio_stream` `warmup_frames` default 1 → 0. This is a Python API break, not only a CLI one.
- Loading now raises on a bundle with missing, extra, or wrongly shaped tensors (v0.2.1 skipped them silently).
- The new runtime dependency `tokenizers>=0.21`.
- The fixes: EOS ignored before frame 6, 5 ms per-chunk onset fade, cloned prompts end on a pause, non-16-bit WAV prompts, the prepared chunk text now reaches generation, and voice-state `.safetensors` input.

Add a `### Fixed` section and these lines.

### P3-1: Relative `CHANGELOG.md` links break on PyPI

The README's "Release notes: [v0.3.0](CHANGELOG.md)" and "Latest" links are relative. PyPI renders the README without the repo, so they 404 there. Use `https://github.com/jishnuvenugopal/pocket-tts-mlx/blob/main/CHANGELOG.md`, as v0.2.1 used absolute links.

### P3-2: The parity reference is stale and the parity test does not exist

The README says torch is "only required for optional parity tests," but `tests/` has none. The `dev` extra still pins `pocket-tts>=1.0.3`, while the reference is `3.3.0`, which is on PyPI. Either commit the temperature-0 parity check as `tests/test_parity.py`, skipping without torch or `pocket_tts`, and bump the extra to `pocket-tts>=3.3.0`; or drop the README sentence.

### P3-3: `import_flow_state` does not check that the state is complete

A voice file that lacks a module, or a module without `cache`, imports without error and then fails later with a bare `KeyError`. Also check that every flow attention module received both `cache` and `current_end`, and that `offset <= cache.shape[2]`.

### P3-4: setuptools license deprecation

`license = {text = "MIT"}` and the MIT classifier trigger setuptools warnings. Builds stop working after 2027-02-18. Fix: `license = "MIT"`, drop the classifier, and set `requires = ["setuptools>=77"]`. Not urgent.

### P3-5: Notes that go stale at publish

`koder/AGENTS.md` still says "Current release is v0.2.1." Update it when 0.3.0 is actually published, not before.

## Pre-existing, unchanged

`noise_clamp` clips the noise, while upstream redraws it from a truncated normal. It is off by default (`None`).

## Verdict

APPROVE WITH FIXES. Apply P2-1 and P2-2 before tagging. P3 items are optional for 0.3.0. Push, tag, and PyPI publish remain owner-gated.
