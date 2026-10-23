import sys, numpy as np, soundfile as sf
from scipy.signal import stft
def m(p):
    x,_=sf.read(p,dtype="float32"); x=x.reshape(-1)
    f,t,Z=stft(x,24000,nperseg=1024,noverlap=768); P=np.abs(Z)**2
    e=P.sum(0); act=e>0.1*e.max(); Pa=P[:,act]; mean=Pa.mean(1)
    cen=(f*mean).sum()/mean.sum()
    band=lambda a,b: 10*np.log10(mean[(f>=a)&(f<b)].sum()/mean.sum()+1e-12)
    flat=np.exp(np.log(Pa+1e-12).mean(0)).mean()/ (Pa.mean(0).mean()+1e-12)
    # high-frequency rolloff: freq below which 95% energy
    c=np.cumsum(mean)/mean.sum(); r95=f[np.searchsorted(c,0.95)]
    # silence floor
    sil=e[~act]; fl=10*np.log10(sil.mean()/e[act].mean()+1e-12) if sil.size else float('nan')
    return cen,band(0,500),band(500,2000),band(2000,4000),band(4000,8000),band(8000,12000),r95,flat,fl
print(f"{'file':34s} cent  <.5k  .5-2k 2-4k  4-8k  8-12k roll95 flat  floor")
for p in sys.argv[1:]:
    print(f"{p.split('/')[-1]:34s} "+" ".join(f"{v:5.0f}" if i in(0,6) else f"{v:5.1f}" if i!=7 else f"{v:5.3f}" for i,v in enumerate(m(p))))
