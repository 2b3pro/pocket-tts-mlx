"""asr.py LANG REFTEXT wav... -> WER per file (whisper-medium-mlx, offline)."""
import os, sys, re, glob, jiwer, mlx_whisper
lang, ref = sys.argv[1], sys.argv[2]
snap = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--whisper-medium-mlx/snapshots/*")))[0]
norm = lambda s: re.sub(r"[^\w\s]", "", s.lower()).split()
for p in sys.argv[3:]:
    t = mlx_whisper.transcribe(p, path_or_hf_repo=snap, language=lang, temperature=0)["text"].strip()
    print(f"{jiwer.wer(' '.join(norm(ref)), ' '.join(norm(t))):.2f}  {p.split('/')[-1]:36s} {t}", flush=True)
