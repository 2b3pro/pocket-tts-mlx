"""Generate one clip with upstream torch or the MLX port. Usage: gen.py impl language voice temp out.wav text"""
import sys, numpy as np, soundfile as sf
impl, language, voice, temp, out, text = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5], sys.argv[6]
if impl == "torch":
    import torch
    torch.manual_seed(0)
    from pocket_tts import TTSModel
    m = TTSModel.load_model(language=language, temp=temp)
    st = m.get_state_for_audio_prompt(voice)
    a = m.generate_audio(st, text).detach().cpu().numpy()
else:
    import mlx.core as mx
    mx.random.seed(0)
    from pocket_tts_mlx import TTSModel
    m = TTSModel.load_model(language=language, temp=temp)
    st = m.get_state_for_audio_prompt(voice)
    a = np.array(m.generate_audio(st, text))
a = a.reshape(-1).astype(np.float32)
sf.write(out, a, 24000)
print(out, a.shape[0] / 24000)
