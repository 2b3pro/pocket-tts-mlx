"""sweep.py config temp voice1,voice2,... -> per-voice peak/active RMS over seeds; wav for seed 0 in sweep/"""
import sys, numpy as np, soundfile as sf, mlx.core as mx
from pocket_tts_mlx import TTSModel
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0]); from compare import levels
TEXTS = {
 "german": "Guten Morgen. Heute ist ein schöner Tag, und wir gehen am Nachmittag im Park spazieren.",
 "dutch": "Goedemorgen. Vandaag is het mooi weer, en we gaan vanmiddag een wandeling maken in het park.",
 "french": "Bonjour. Aujourd'hui il fait beau, et nous allons nous promener dans le parc cet après-midi.",
}
cfg, temp, voices = sys.argv[1], float(sys.argv[2]), sys.argv[3].split(",")
seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
lang = cfg.replace("_24l", "")
m = TTSModel.load_model(cfg, temp=temp)
for v in voices:
    st = m.get_state_for_audio_prompt(v)
    pk, ar, du = [], [], []
    for s in range(seeds):
        mx.random.seed(s)
        a = np.array(m.generate_audio(st, TEXTS[lang])).reshape(-1)
        if s == 0: sf.write(f"sweep/{cfg}_{v}_t{temp}.wav", a, 24000)
        p, r = levels(a); pk.append(p); ar.append(r); du.append(len(a) / 24000)
    print(f"{cfg:10s} t={temp} {v:9s} peak {np.mean(pk):6.1f}  activeRMS {np.mean(ar):6.1f} (±{np.std(ar):.1f})  dur {np.mean(du):.2f}s", flush=True)
