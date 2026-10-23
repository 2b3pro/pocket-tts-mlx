# Parity harness (rebuild notes)

The harness lived outside the repo at `../parity/` (914 MB venv plus scripts). It was written for issue 002 and may be deleted. Rebuild with the steps below; copies of the scripts are in `parity-scripts/` (they assume the `../parity/` layout).

```

upstream-41cbc84-venv/  torch + upstream pocket-tts installed from git at 41cbc84 (the port's reference).
                        Not PyPI pocket-tts 3.3.0: that tag is older and ships older non-English models.
                        Moved from a session scratchpad, so its bin/pip shebang is stale: use
                        upstream-41cbc84-venv/bin/python -m pip ...
parity-scripts/gen.py          one clip: gen.py torch|mlx LANGUAGE VOICE TEMP OUT.wav "TEXT"
parity-scripts/compare.py      compare.py REF.wav TEST.wav   (length, corr, SNR, peak, active RMS)
                        compare.py --levels A.wav B.wav ...
parity-scripts/sweep.py        voices x seeds for german/dutch/french; writes ./sweep/ in the current directory
parity-scripts/longrun.py      long paragraphs, plain vs normalized, limiter stats; writes ../samples/issue002/long_*
parity-scripts/cache_prune.py  dry-run list of pocket-tts HF cache files the port at HEAD never loads (--delete removes them)
parity-scripts/needed.txt      hf paths referenced by pocket_tts_mlx/config/*.yaml when it was written; regenerate before pruning

Parity check (run from this folder, sandbox off for MLX):
  R=../Untitled
  upstream-41cbc84-venv/bin/python parity-scripts/gen.py torch german juergen 0 /tmp/de_t.wav "Guten Morgen."
  $R/.venv/bin/python parity-scripts/gen.py mlx german juergen 0 /tmp/de_m.wav "Guten Morgen."
  $R/.venv/bin/python parity-scripts/compare.py /tmp/de_t.wav /tmp/de_m.wav

Rebuild the venv if it breaks:
  /opt/homebrew/bin/python3.12 -m venv upstream-41cbc84-venv
  upstream-41cbc84-venv/bin/python -m pip install soundfile "git+https://github.com/kyutai-labs/pocket-tts@41cbc84af539ea78a804ffca5f9c6edc1a22ce44"


```
