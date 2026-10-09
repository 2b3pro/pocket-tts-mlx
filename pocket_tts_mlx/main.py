"""Command-line interface for pocket-tts-mlx."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from pocket_tts_mlx import TTSModel
from pocket_tts_mlx.default_parameters import (
    DEFAULT_SAMPLER_DECODE_STEPS,
    MAX_TOKEN_PER_CHUNK,
    get_default_voice_for_language,
)
from pocket_tts_mlx.models.tts_model import is_shipped_config, resolve_config_path

logger = logging.getLogger(__name__)


def read_cli_text(text: str) -> str:
    """Return `text`, or stdin when the caller passes `-`."""
    if text == "-":
        return sys.stdin.read()
    return text


def cli_voice(voice: str | None, language: str | None, config: str | None) -> str:
    """Default voice follows the language. A custom config clones Alba's recording."""
    if voice is not None:
        return voice
    language_for_voice = language
    custom_config = False
    if config is not None:
        origin = resolve_config_path(config)
        custom_config = not is_shipped_config(origin)
        if not custom_config:
            language_for_voice = origin.stem
    return get_default_voice_for_language(language_for_voice, custom_config=custom_config)


def main() -> int:
    """Parse CLI arguments, run TTS, and write WAV output."""
    parser = argparse.ArgumentParser(
        description="Generate speech from text using pocket-tts with MLX backend"
    )
    parser.add_argument("text", help="Text to convert to speech. Use - to read stdin.")
    parser.add_argument(
        "--voice",
        "-v",
        default=None,
        help=(
            "Voice name, audio file, or voice-state safetensors. "
            "Defaults from the language: estelle (french), giovanni (italian), "
            "juergen (german), lola (spanish), rafael (portuguese), daan (dutch), "
            "otherwise alba."
        ),
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Shipped language config stem, such as english or french. Incompatible with --config.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a model config YAML. Incompatible with --language.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Defaults to the value in the model config (0.3).",
    )
    parser.add_argument(
        "--sampler-decode-steps",
        type=int,
        default=DEFAULT_SAMPLER_DECODE_STEPS,
        help="Sampler steps. Drifting and LSD use 1.",
    )
    parser.add_argument("--output", "-o", default="output.wav", help="Output WAV file")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKEN_PER_CHUNK,
        help="Max tokens per chunk",
    )
    parser.add_argument(
        "--frames-after-eos",
        type=int,
        default=None,
        help="Frames after EOS. Unset uses the config recommendation, then a text guess.",
    )
    parser.add_argument(
        "--trim-start-ms",
        type=int,
        default=0,
        help="Trim this many milliseconds from start of generated audio",
    )
    parser.add_argument(
        "--fade-in-ms",
        type=int,
        default=0,
        help="Apply linear fade-in over this many milliseconds",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Number of initial Mimi frames to decode and discard",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Scale the output to about -18 dBFS speech level, peaks under -1 dBFS. Off by default.",
    )
    parser.add_argument("--verbose", "-V", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    try:
        text = read_cli_text(args.text)
        if not text.strip():
            logger.error("No input text.")
            return 1
        if args.language is not None and args.config is not None:
            raise ValueError("Cannot specify both config and language, please choose one or the other.")

        logger.info("Loading MLX model...")
        model = TTSModel.load_model(
            config=args.config,
            language=args.language,
            temp=args.temperature,
            sampler_decode_steps=args.sampler_decode_steps,
        )
        voice = cli_voice(args.voice, args.language, args.config)

        logger.info("Loading voice: %s", voice)
        model_state = model.get_state_for_audio_prompt(voice)

        logger.info("Generating audio...")
        audio = model.generate_audio(
            model_state=model_state,
            text_to_generate=text,
            max_tokens=args.max_tokens,
            frames_after_eos=args.frames_after_eos,
            trim_start_ms=args.trim_start_ms,
            fade_in_ms=args.fade_in_ms,
            warmup_frames=args.warmup_frames,
            normalize=args.normalize,
        )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(str(out_path), np.array(audio), model.config.mimi.sample_rate)
        duration = audio.shape[-1] / model.config.mimi.sample_rate
        logger.info("Wrote %s (%.2fs)", out_path, duration)
        return 0
    except Exception as exc:
        logger.error("Error: %s", exc)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
