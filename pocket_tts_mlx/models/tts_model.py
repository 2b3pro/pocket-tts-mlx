"""End-to-end TTS model for MLX with FlowLM and Mimi decoding."""

import copy
import logging
import math
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Generator, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from pocket_tts_mlx.conditioners.base import TokenizedText
from pocket_tts_mlx.data.audio import audio_read
from pocket_tts_mlx.data.audio_utils import convert_audio, end_on_pause
from pocket_tts_mlx.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    DEFAULT_VARIANT,
    MAX_TOKEN_PER_CHUNK,
)
from pocket_tts_mlx.models.flow_lm import FlowLMModel
from pocket_tts_mlx.models.mimi import build_mimi
from pocket_tts_mlx.models.text_chunking import prepare_text_prompt, split_into_best_sentences
from pocket_tts_mlx.modules.stateful_module import increment_steps, init_states
from pocket_tts_mlx.utils.config import Config, load_config
from pocket_tts_mlx.utils.utils import display_execution_time, download_if_necessary, size_of_dict
from pocket_tts_mlx.utils.weight_conversion import (
    PREDEFINED_VOICES,
    get_flow_lm_state_dict_mlx,
    get_mimi_state_dict_mlx,
    load_predefined_voice_mlx,
    load_safetensors_to_numpy,
    load_weights_to_mlx_model,
)

logger = logging.getLogger(__name__)


def speaker_proj_shape(config: Config) -> tuple[int, int]:
    """Map the codec bottleneck onto the flow transformer width."""
    return (
        config.flow_lm.transformer.d_model,
        config.mimi.inner_dim or config.mimi.seanet.dimension,
    )


def mimi_steps_per_latent(encoder_frame_rate: float, frame_rate: float) -> int:
    """How many Mimi steps one latent frame advances."""
    return int(encoder_frame_rate / frame_rate)


def note_eos_step(
    is_eos: bool,
    generation_step: int,
    eos_step: Optional[int],
    min_frames: int = 6,
) -> Optional[int]:
    """Record the first EOS at or after `min_frames`. Earlier EOS is ignored."""
    if is_eos and eos_step is None and generation_step >= min_frames:
        return generation_step
    return eos_step


def resolve_frames_after_eos(
    frames_after_eos: Optional[int],
    model_recommended: Optional[int],
    frames_after_eos_guess: int,
) -> int:
    """Prefer the caller, then the config recommendation, then the text guess + 2."""
    if frames_after_eos is None:
        frames_after_eos = model_recommended
    if frames_after_eos is not None:
        return frames_after_eos
    return frames_after_eos_guess + 2


def apply_chunk_onset_fade(
    audio: mx.array, fade: Optional[mx.array]
) -> tuple[mx.array, Optional[mx.array]]:
    """Fade the first samples of a chunk once. Later frames receive no ramp."""
    if fade is None or fade.shape[0] == 0:
        return audio, None
    n = min(int(fade.shape[0]), int(audio.shape[-1]))
    if n <= 0:
        return audio, None
    faded = audio[..., :n] * fade[:n].astype(audio.dtype)
    if n < audio.shape[-1]:
        audio = mx.concatenate([faded, audio[..., n:]], axis=-1)
    else:
        audio = faded
    return audio, None


def prepend_voice_bos(prompt: mx.array, bos: mx.array, insert: bool) -> mx.array:
    """Prepend the learned BOS voice token when the config asks for it."""
    if not insert:
        return prompt
    return mx.concatenate([bos, prompt], axis=1)

VOICE_CLONING_UNSUPPORTED = (
    f"We could not download the weights for the model with voice cloning, "
    f"but you're trying to use voice cloning. "
    f"Without voice cloning, you can use our catalog of voices {list(PREDEFINED_VOICES)}. "
    f"If you want access to the model with voice cloning, go to "
    f"https://huggingface.co/kyutai/pocket-tts and accept the terms, "
    f"then make sure you're logged in locally with `hf auth login`."
)


class TTSModel(nn.Module):
    """Text-to-speech pipeline with conditioning, FlowLM, and Mimi."""

    _TOKENS_PER_SECOND_ESTIMATE = 3.0
    _GEN_SECONDS_PADDING = 2.0
    _MIN_FRAMES_BEFORE_EOS = 6
    _MIMI_WARMUP_FRAMES = 0

    def __init__(
        self,
        flow_lm: FlowLMModel,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: Optional[float],
        eos_threshold: float,
        config: Config,
    ):
        super().__init__()
        self.flow_lm = flow_lm
        self.temp = temp
        self.lsd_decode_steps = lsd_decode_steps
        self.noise_clamp = noise_clamp
        self.eos_threshold = eos_threshold
        self.config = config
        self.has_voice_cloning = True

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def sample_rate(self) -> int:
        return self.config.mimi.sample_rate

    @classmethod
    def _from_pydantic_config(
        cls, config: Config, temp: float, lsd_decode_steps: int, noise_clamp: Optional[float], eos_threshold
    ):
        flow_lm = FlowLMModel.from_pydantic_config(
            config.flow_lm, latent_dim=config.mimi.quantizer.dimension
        )
        return cls(flow_lm, temp, lsd_decode_steps, noise_clamp, eos_threshold, config)

    @classmethod
    def _assemble(
        cls, config: Config, temp: float, lsd_decode_steps: int, noise_clamp: Optional[float], eos_threshold
    ):
        """Build FlowLM, the speaker projection, and Mimi without loading weights."""
        tts_model = cls._from_pydantic_config(config, temp, lsd_decode_steps, noise_clamp, eos_threshold)
        tts_model.flow_lm.speaker_proj_weight = mx.zeros(
            speaker_proj_shape(config), dtype=mx.float32
        )
        tts_model.mimi = build_mimi(config.mimi)
        return tts_model

    @classmethod
    def _from_pydantic_config_with_weights(
        cls, config: Config, temp: float, lsd_decode_steps: int, noise_clamp: Optional[float], eos_threshold
    ):
        tts_model = cls._assemble(config, temp, lsd_decode_steps, noise_clamp, eos_threshold)

        if config.flow_lm.weights_path is not None:
            if config.mimi.weights_path is None:
                raise ValueError(
                    "If you specify flow_lm.weights_path you should specify mimi.weights_path"
                )
            logger.info("Loading FlowLM weights from %s", config.flow_lm.weights_path)
            state_dict_flowlm = get_flow_lm_state_dict_mlx(
                download_if_necessary(config.flow_lm.weights_path)
            )
            load_weights_to_mlx_model(tts_model.flow_lm, state_dict_flowlm, strict=True)

        if config.mimi.weights_path is not None:
            if config.flow_lm.weights_path is None:
                raise ValueError(
                    "If you specify mimi.weights_path you should specify flow_lm.weights_path"
                )
            logger.info("Loading Mimi weights from %s", config.mimi.weights_path)
            mimi_state = get_mimi_state_dict_mlx(download_if_necessary(config.mimi.weights_path))
            load_weights_to_mlx_model(tts_model.mimi, mimi_state, strict=True)

        if config.weights_path is not None:
            logger.info("Loading TTSModel weights from %s", config.weights_path)
            try:
                weights_file = download_if_necessary(config.weights_path)
            except Exception:
                tts_model.has_voice_cloning = False
                weights_file = download_if_necessary(config.weights_path_without_voice_cloning)

            weights = load_safetensors_to_numpy(weights_file)
            loaded_count = 0
            skipped_count = 0
            for key, tensor in weights.items():
                if getattr(tensor.dtype, "name", "") == "bfloat16":
                    tensor = tensor.astype("float32")
                mlx_array = mx.array(tensor)

                # Resolve nested modules and assign parameters by name.
                parts = key.split(".")
                obj = tts_model
                try:
                    for part in parts[:-1]:
                        if part.isdigit():
                            obj = obj[int(part)]
                        else:
                            obj = getattr(obj, part)
                    param_name = parts[-1]
                    if param_name.isdigit():
                        skipped_count += 1
                        continue

                    if param_name == "weight" and isinstance(obj, nn.Conv1d) and len(mlx_array.shape) == 3:
                        # PyTorch Conv1d weight: (out, in, k) -> MLX: (out, k, in).
                        mlx_array = mx.transpose(mlx_array, (0, 2, 1))
                    elif param_name == "weight" and (
                        isinstance(obj, nn.ConvTranspose1d) or hasattr(obj, "is_conv_transpose")
                    ) and len(mlx_array.shape) == 3:
                        # PyTorch ConvTranspose1d weight: (in, out, k)
                        # MLX weight: (out, k, in/groups)
                        if mlx_array.shape[1] == 1:
                            mlx_array = mx.transpose(mlx_array, (0, 2, 1))
                        else:
                            mlx_array = mx.transpose(mlx_array, (1, 2, 0))

                    setattr(obj, param_name, mlx_array)
                    loaded_count += 1
                except (KeyError, IndexError, AttributeError):
                    skipped_count += 1
                    continue

            logger.info("Loaded %d weights, skipped %d", loaded_count, skipped_count)

        if config.flow_lm.weights_path is None and config.weights_path is None:
            logger.warning("No weights_path specified, model is uninitialized!")
        size_in_mb = size_of_dict(tts_model.state_dict()) // 1e6 if hasattr(tts_model, "state_dict") else 0
        logging.info("TTS Model loaded successfully. Size ~%d MB", size_in_mb)
        return tts_model

    @classmethod
    def load_model(
        cls,
        config: str | Path = DEFAULT_VARIANT,
        temp: float | int = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: float | int | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
    ):
        """Create and load a configured TTSModel with weights."""
        if str(config).endswith(".yaml"):
            config_path = Path(config)
            config = load_config(config_path)
            logger.info("Loading model from config at %s...", config_path)
        else:
            config = load_config(Path(__file__).parents[1] / f"config/{config}.yaml")
        tts_model = cls._from_pydantic_config_with_weights(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold
        )
        return tts_model

    def _run_flow_lm_and_increment_step(
        self,
        model_state: Dict,
        text_tokens: Optional[mx.array] = None,
        backbone_input_latents: Optional[mx.array] = None,
        audio_conditioning: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        """Run FlowLM and advance streaming offsets."""
        if text_tokens is None:
            text_tokens = mx.zeros((1, 0), dtype=mx.int64)
        if backbone_input_latents is None:
            backbone_input_latents = mx.zeros((1, 0, self.flow_lm.ldim), dtype=self.flow_lm.dtype)
        if audio_conditioning is None:
            audio_conditioning = mx.zeros((1, 0, self.flow_lm.dim), dtype=self.flow_lm.dtype)

        output = self._run_flow_lm(
            text_tokens=text_tokens,
            backbone_input_latents=backbone_input_latents,
            model_state=model_state,
            audio_conditioning=audio_conditioning,
        )
        increment_by = (
            text_tokens.shape[1] + backbone_input_latents.shape[1] + audio_conditioning.shape[1]
        )
        increment_steps(self.flow_lm, model_state, increment=increment_by)
        return output

    def _run_flow_lm(
        self,
        model_state: Dict,
        text_tokens: mx.array,
        backbone_input_latents: mx.array,
        audio_conditioning: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute next latent and EOS using FlowLM."""
        text_embeddings = self.flow_lm.conditioner(TokenizedText(text_tokens))
        text_embeddings = mx.concatenate([text_embeddings, audio_conditioning], axis=1)
        output_embeddings, is_eos = self.flow_lm._sample_next_latent(
            backbone_input_latents,
            text_embeddings,
            model_state=model_state,
            lsd_decode_steps=self.lsd_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )
        return output_embeddings[:, None, :], is_eos

    def _encode_audio(self, audio: mx.array) -> mx.array:
        """Encode conditioning audio into FlowLM-compatible embeddings."""
        encoded = self.mimi.encode_to_latent(audio)
        latents = encoded.astype(mx.float32)
        conditioning = mx.matmul(latents, self.flow_lm.speaker_proj_weight.T)
        return conditioning

    def _expand_kv_cache(self, model_state: Dict, sequence_length: int) -> None:
        """Expand KV cache capacity for longer sequences."""
        for module_state in model_state.values():
            if "cache" in module_state:
                cache = module_state["cache"]
                current_length = cache.shape[2]
                if current_length < sequence_length:
                    expanded_cache = mx.full(
                        (cache.shape[0], cache.shape[1], sequence_length, cache.shape[3], cache.shape[4]),
                        mx.nan,
                    )
                    expanded_cache = mx.slice_update(
                        expanded_cache,
                        cache,
                        mx.array([0, 0, 0, 0, 0]),
                        axes=[0, 1, 2, 3, 4],
                    )
                    module_state["cache"] = expanded_cache

    def _flow_lm_current_end(self, model_state: Dict) -> int:
        """Read the current decoded length from FlowLM state."""
        for module_state in model_state.values():
            current_end = module_state.get("current_end")
            if current_end is not None:
                return int(current_end.shape[0])
        raise ValueError(
            "Could not find current_end in model state. Please open an issue at "
            "https://github.com/jishnuvenugopal/pocket-tts-mlx/issues"
        )

    def generate_audio(
        self,
        model_state: Dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: Optional[int] = None,
        copy_state: bool = True,
        trim_start_ms: int = 0,
        fade_in_ms: int = 0,
        warmup_frames: int = _MIMI_WARMUP_FRAMES,
    ) -> mx.array:
        """Generate full audio array from text."""
        audio_chunks = []
        for chunk in self.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            frames_after_eos=frames_after_eos,
            copy_state=copy_state,
            max_tokens=max_tokens,
            warmup_frames=warmup_frames,
        ):
            audio_chunks.append(chunk)
        audio = mx.concatenate(audio_chunks, axis=0)
        audio = self._postprocess_audio_start(audio, trim_start_ms=trim_start_ms, fade_in_ms=fade_in_ms)
        # Materialize the array so external np.array timing reflects generation cost.
        mx.eval(audio)
        return audio

    def generate_audio_stream(
        self,
        model_state: Dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: Optional[int] = None,
        copy_state: bool = True,
        warmup_frames: int = _MIMI_WARMUP_FRAMES,
    ) -> Generator[mx.array, None, None]:
        """Yield audio chunks as they are generated."""
        prompt_kwargs = self._text_prompt_kwargs()
        chunks = split_into_best_sentences(
            self.flow_lm.conditioner.tokenizer,
            text_to_generate,
            max_tokens,
            **prompt_kwargs,
        )
        for chunk in chunks:
            prepared_text, frames_after_eos_guess = prepare_text_prompt(chunk, **prompt_kwargs)
            effective_frames = resolve_frames_after_eos(
                frames_after_eos,
                self.config.model_recommended_frames_after_eos,
                frames_after_eos_guess,
            )
            yield from self._generate_audio_stream_short_text(
                model_state=model_state,
                text_to_generate=prepared_text,
                frames_after_eos=effective_frames,
                copy_state=copy_state,
                warmup_frames=warmup_frames,
            )

    def _text_prompt_kwargs(self) -> dict:
        """Text flags from the loaded config."""
        config = self.config
        return {
            "pad_with_spaces_for_short_inputs": config.pad_with_spaces_for_short_inputs,
            "remove_semicolons": config.remove_semicolons,
            "append_terminal_punctuation": config.append_terminal_punctuation,
            "capitalize_first_letter": config.capitalize_first_letter,
            "replace_characters": config.replace_characters,
        }

    def _generate_audio_stream_short_text(
        self,
        model_state: Dict,
        text_to_generate: str,
        frames_after_eos: int,
        copy_state: bool,
        warmup_frames: int,
    ):
        """Generate audio for a short prompt with streaming FlowLM."""
        if copy_state:
            model_state = copy.deepcopy(model_state)

        prepared = self.flow_lm.conditioner.prepare(text_to_generate)
        token_count = prepared.tokens.shape[1]
        max_gen_len = self._estimate_max_gen_len(token_count)
        current_end = self._flow_lm_current_end(model_state)
        required_len = current_end + token_count + max_gen_len
        self._expand_kv_cache(model_state, sequence_length=required_len)

        mimi_context = self.config.mimi.transformer.context
        mimi_state = init_states(self.mimi, batch_size=1, sequence_length=mimi_context)
        self._warmup_mimi_decoder(mimi_state, warmup_frames)

        t_generating = time.monotonic()

        with display_execution_time("Prompting text"):
            self._run_flow_lm_and_increment_step(
                model_state=model_state, text_tokens=prepared.tokens
            )

        backbone_input = mx.full(
            (1, 1, self.flow_lm.ldim),
            float("nan"),
            dtype=self.flow_lm.dtype,
        )
        steps_times = []
        eos_step = None
        total_generated_samples = 0
        n_fade = self.sample_rate // 200
        onset_fade = mx.linspace(0.0, 1.0, n_fade) if n_fade > 0 else None

        for generation_step in range(max_gen_len):
            with display_execution_time("Generating latent", print_output=False) as timer:
                next_latent, is_eos = self._run_flow_lm_and_increment_step(
                    model_state=model_state, backbone_input_latents=backbone_input
                )

                is_eos_scalar = bool(is_eos.item())
                eos_step = note_eos_step(
                    is_eos_scalar, generation_step, eos_step, self._MIN_FRAMES_BEFORE_EOS
                )
                if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                    break

                # Decode latent frame into audio via Mimi. The codec quantizes.
                mimi_decoding_input = next_latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
                audio_frame = self.mimi.decode_from_latent(mimi_decoding_input, mimi_state)
                increment_steps(self.mimi, mimi_state, increment=self._mimi_steps_per_latent())
                audio_chunk = audio_frame[0, 0]
                audio_chunk, onset_fade = apply_chunk_onset_fade(audio_chunk, onset_fade)
                # Force eager execution per chunk for honest timing and smoother streaming.
                mx.eval(audio_chunk)
                total_generated_samples += audio_chunk.shape[-1]
                yield audio_chunk

                backbone_input = next_latent

            steps_times.append(timer.elapsed_time_ms)

        duration_generated_audio = int(total_generated_samples * 1000 / self.config.mimi.sample_rate)
        generation_time = int((time.monotonic() - t_generating) * 1000)
        real_time_factor = duration_generated_audio / max(1, generation_time)
        logger.info(
            "Generated: %d ms of audio in %d ms so %.2fx faster than real-time",
            duration_generated_audio,
            generation_time,
            real_time_factor,
        )

    def _mimi_steps_per_latent(self) -> int:
        """Mimi streaming step for one generated latent."""
        return mimi_steps_per_latent(self.mimi.encoder_frame_rate, self.mimi.frame_rate)

    def _estimate_max_gen_len(self, token_count: int) -> int:
        """Estimate max generation frames from token count."""
        gen_len_sec = token_count / self._TOKENS_PER_SECOND_ESTIMATE + self._GEN_SECONDS_PADDING
        frame_rate = self.config.mimi.frame_rate
        return math.ceil(gen_len_sec * frame_rate)

    def _postprocess_audio_start(self, audio: mx.array, trim_start_ms: int, fade_in_ms: int) -> mx.array:
        """Optionally trim and fade-in the beginning to suppress first-frame artifacts."""
        sample_rate = self.sample_rate

        if trim_start_ms > 0:
            trim_samples = int(sample_rate * trim_start_ms / 1000)
            if 0 < trim_samples < audio.shape[0]:
                audio = audio[trim_samples:]

        if fade_in_ms > 0 and audio.shape[0] > 1:
            fade_samples = int(sample_rate * fade_in_ms / 1000)
            fade_samples = min(max(0, fade_samples), audio.shape[0])
            if fade_samples > 1:
                ramp = mx.linspace(0.0, 1.0, fade_samples).astype(audio.dtype)
                audio = mx.concatenate([audio[:fade_samples] * ramp, audio[fade_samples:]], axis=0)

        return audio

    def _warmup_mimi_decoder(self, mimi_state: Dict, warmup_frames: int) -> None:
        """Prime Mimi decoder state and discard startup transients."""
        if warmup_frames <= 0:
            return

        zero_latent = mx.zeros((1, 1, self.flow_lm.ldim), dtype=self.flow_lm.dtype)
        for _ in range(warmup_frames):
            mimi_decoding_input = zero_latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
            audio_frame = self.mimi.decode_from_latent(mimi_decoding_input, mimi_state)
            mx.eval(audio_frame)
            increment_steps(self.mimi, mimi_state, increment=self._mimi_steps_per_latent())

    @lru_cache(maxsize=2)
    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning: Union[Path, str, mx.array], truncate: bool = False
    ) -> Dict:
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    def get_state_for_audio_prompt(
        self, audio_conditioning: Union[Path, str, mx.array], truncate: bool = False
    ) -> Dict:
        if isinstance(audio_conditioning, str) and audio_conditioning in PREDEFINED_VOICES:
            prompt = load_predefined_voice_mlx(audio_conditioning)
        else:
            if not self.has_voice_cloning and isinstance(audio_conditioning, (str, Path)):
                raise ValueError(VOICE_CLONING_UNSUPPORTED)

            if isinstance(audio_conditioning, str):
                audio_conditioning = download_if_necessary(audio_conditioning)

            if isinstance(audio_conditioning, Path):
                audio, conditioning_sample_rate = audio_read(audio_conditioning)
                if truncate:
                    max_samples = int(30 * conditioning_sample_rate)
                    if audio.shape[-1] > max_samples:
                        audio = audio[..., :max_samples]
                        logger.info("Audio truncated to 30 seconds")
                audio_conditioning = convert_audio(
                    audio, conditioning_sample_rate, self.config.mimi.sample_rate, 1
                )

            audio_conditioning = end_on_pause(
                np.asarray(audio_conditioning), self.config.mimi.sample_rate
            )
            with display_execution_time("Encoding audio prompt"):
                prompt = self._encode_audio(mx.array(audio_conditioning)[None, ...])
            if self.flow_lm.insert_bos_before_voice:
                prompt = prepend_voice_bos(prompt, self.flow_lm.bos_before_voice, True)

        model_state = init_states(self.flow_lm, batch_size=1, sequence_length=prompt.shape[1])
        with display_execution_time("Prompting audio"):
            self._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)

        logger.info(
            "Size of the model state for audio prompt: %d MB",
            size_of_dict(model_state) // 1e6,
        )
        return model_state
