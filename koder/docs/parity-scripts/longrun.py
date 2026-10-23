"""Long natural paragraphs: default voice, temp 0.3, seeds 0-1. Levels plain vs normalized, plus limiter engagement."""
import sys, numpy as np, soundfile as sf, mlx.core as mx
from pocket_tts_mlx import TTSModel
from pocket_tts_mlx.data.audio_utils import normalize_loudness
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0]); from compare import levels
D = "../samples/issue002"
TEXTS = {
 ("german", "juergen"): "Am Wochenende sind wir mit dem Zug an die Nordsee gefahren. Das Wetter war wechselhaft, aber am Samstagnachmittag kam endlich die Sonne heraus. Wir haben lange am Strand gesessen, Fischbrötchen gegessen und den Schiffen zugeschaut. Am Abend waren alle müde, aber sehr zufrieden.",
 ("dutch", "daan"): "Afgelopen weekend zijn we met de trein naar de kust gegaan. Het weer was wisselvallig, maar zaterdagmiddag kwam eindelijk de zon tevoorschijn. We hebben lang op het strand gezeten, haring gegeten en naar de schepen gekeken. Die avond was iedereen moe, maar heel tevreden.",
 ("french", "estelle"): "Le week-end dernier, nous avons pris le train pour aller au bord de la mer. Le temps était changeant, mais samedi après-midi, le soleil est enfin apparu. Nous sommes restés longtemps sur la plage à regarder passer les bateaux. Le soir, tout le monde était fatigué, mais très content.",
}
for (lang, voice), text in TEXTS.items():
    m = TTSModel.load_model(language=lang)
    st = m.get_state_for_audio_prompt(voice)
    for seed in (0, 1):
        mx.random.seed(seed)
        a = np.array(m.generate_audio(st, text)).reshape(-1).astype(np.float32)
        n = normalize_loudness(a, 24000)
        frame = 480
        k = len(a) // frame
        rms = np.sqrt(np.square(a[: k * frame].reshape(k, frame)).mean(1))
        act = rms[rms >= 0.1 * rms.max()]
        g = 10 ** (-18 / 20) / np.sqrt(np.square(act).mean())
        mask = np.abs(a) > 1e-4
        red = n[mask] / (a[mask] * g)
        red_db = 20 * np.log10(np.clip(red, 1e-6, None))
        p0, r0 = levels(a); p1, r1 = levels(n)
        print(f"{lang:7s} {voice:8s} seed {seed}  dur {len(a)/24000:5.2f}s  plain {r0:6.1f}/{p0:5.1f}  norm {r1:6.1f}/{p1:5.1f}  "
              f"limited {100*np.mean(red < 0.99):4.1f}% of samples, max -{-red_db.min():.1f} dB", flush=True)
        if seed == 0:
            sf.write(f"{D}/long_{lang}_{voice}_t0.3.wav", a, 24000)
            sf.write(f"{D}/long_{lang}_{voice}_t0.3_normalized.wav", n, 24000)
