"""Compare wavs: compare.py ref.wav test.wav  |  levels.py-style: compare.py --levels a.wav b.wav ..."""
import sys, numpy as np, soundfile as sf
def levels(x, sr=24000):
    f = sr // 50
    n = len(x) // f
    r = np.sqrt((x[: n * f].reshape(n, f) ** 2).mean(1) + 1e-20)
    act = r[r > 0.1 * r.max()]
    return 20 * np.log10(np.abs(x).max() + 1e-12), 20 * np.log10(np.sqrt((act ** 2).mean()) + 1e-12)
if __name__ == "__main__" and sys.argv[1] == "--levels":
    for p in sys.argv[2:]:
        x, _ = sf.read(p, dtype="float32")
        pk, ar = levels(x)
        print(f"{p.split('/')[-1]:44s} dur {len(x)/24000:5.2f}s  peak {pk:6.1f}  activeRMS {ar:6.1f}")
    sys.exit()
if __name__ == "__main__":
    a, _ = sf.read(sys.argv[1], dtype="float32"); b, _ = sf.read(sys.argv[2], dtype="float32")
    n = min(len(a), len(b))
    corr = np.corrcoef(a[:n], b[:n])[0, 1]
    snr = 10 * np.log10((a[:n] ** 2).sum() / (((a[:n] - b[:n]) ** 2).sum() + 1e-20))
    pa, ra = levels(a); pb, rb = levels(b)
    print(f"len {len(a)} vs {len(b)}  corr {corr:.5f}  SNR {snr:5.1f} dB  peak {pa:.1f}/{pb:.1f}  activeRMS {ra:.1f}/{rb:.1f}")
