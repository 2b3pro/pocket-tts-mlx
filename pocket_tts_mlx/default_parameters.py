"""Default runtime parameters for MLX TTS generation."""

DEFAULT_AUDIO_PROMPT = "alba"
DEFAULT_LANGUAGE = "english"
DEFAULT_VARIANT = "b6369a24"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_LSD_DECODE_STEPS = 1
DEFAULT_SAMPLER_DECODE_STEPS = 1
DEFAULT_NOISE_CLAMP = None
DEFAULT_EOS_THRESHOLD = -4.0
DEFAULT_FRAMES_AFTER_EOS = None
MAX_TOKEN_PER_CHUNK = 50
# Opt-in output normalization (`normalize=True`, `--normalize`). Upstream has none,
# so it stays off by default; some built-in voices (juergen, daan) speak 6-9 dB quieter.
NORMALIZE_TARGET_DBFS = -18.0
NORMALIZE_PEAK_DBFS = -1.0

# Source recording behind the alba voice. Custom configs cannot use a precomputed
# state, so the CLI clones this file instead.
ALBA_SOURCE_WAV = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"

DEFAULT_VOICE_FOR_LANGUAGE = {
    "italian": "giovanni",
    "spanish": "lola",
    "german": "juergen",
    "portuguese": "rafael",
    "french": "estelle",
    "dutch": "daan",
}


def get_default_voice_for_language(language: str | None, *, custom_config: bool = False) -> str:
    """Voice used when the caller does not name one."""
    if custom_config:
        return ALBA_SOURCE_WAV
    for key, voice in DEFAULT_VOICE_FOR_LANGUAGE.items():
        if language is not None and key in language:
            return voice
    return DEFAULT_AUDIO_PROMPT
