import sys, numpy as np, soundfile as sf
from scipy.signal import lfilter
from pocket_tts_mlx.data.audio_utils import normalize_loudness
def shelf(f0,g,fs=24000,S=1.0):
    A=10**(g/40); w=2*np.pi*f0/fs; c,s=np.cos(w),np.sin(w); al=s/2*np.sqrt((A+1/A)*(1/S-1)+2); q=2*np.sqrt(A)*al
    b=[A*((A+1)+(A-1)*c+q),-2*A*((A-1)+(A+1)*c),A*((A+1)+(A-1)*c-q)]
    a=[(A+1)-(A-1)*c+q,2*((A-1)-(A+1)*c),(A+1)-(A-1)*c-q]
    return np.array(b)/a[0],np.array(a)/a[0]
src,dst,f0,g=sys.argv[1],sys.argv[2],float(sys.argv[3]),float(sys.argv[4])
x,_=sf.read(src,dtype="float32"); b,a=shelf(f0,g)
y=lfilter(b,a,x).astype("float32")
y=np.array(normalize_loudness(y,24000))
sf.write(dst,y,24000)
