"""End-to-end TTS model for MLX with FlowLM and Mimi decoding."""

import copy
import logging
import math
import time
from functools import lru_cache
from pathlib import Path
from threading import Event
from typing import Dict, Generator, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from pocket_tts_mlx.conditioners.base import TokenizedText
from pocket_tts_mlx.data.audio import audio_read
from pocket_tts_mlx.data.audio_utils import brighten_speech, convert_audio, end_on_pause, normalize_loudness
from pocket_tts_mlx.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_SAMPLER_DECODE_STEPS,
    MAX_TOKEN_PER_CHUNK,
)
from pocket_tts_mlx.models.flow_lm import FlowLMModel
from pocket_tts_mlx.models.mimi import build_mimi
from pocket_tts_mlx.models.text_chunking import prepare_text_prompt, split_into_best_sentences
from pocket_tts_mlx.modules.stateful_module import increment_steps, init_states
from pocket_tts_mlx.text_normalization import UserDictionary, normalize_text
from pocket_tts_mlx.utils.config import Config, load_config
from pocket_tts_mlx.utils.utils import display_execution_time, download_if_necessary, size_of_dict
from pocket_tts_mlx.utils.weight_conversion import (
    PREDEFINED_VOICE_NAMES,
    PREDEFINED_VOICES,
    get_flow_lm_state_dict_mlx,
    get_mimi_state_dict_mlx,
    import_flow_state,
    is_safetensors_source,
    language_voice_state_url,
    load_predefined_voice_mlx,
    load_safetensors_to_numpy,
    load_weights_to_mlx_model,
)

logger = logging.getLogger(__name__)

# Preserve the earlier import location for callers of the fork.
_is_safetensors_source = is_safetensors_source


class GenerationDidNotReachEOS(RuntimeError):
    """Raised when buffered generation reaches its safety limit without EOS."""


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


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "config"


def select_model_request(config: str | Path | None, language: str | None) -> str | Path:
    """Pick one config stem or path. Omitting both selects english."""
    if config is not None and language is not None:
        raise ValueError("Cannot specify both config and language, please choose one or the other.")
    if config is None and language is None:
        return DEFAULT_LANGUAGE
    if language is not None:
        return language
    assert config is not None
    return config


def resolve_temperature(temp: float | int | None, default_temperature: float) -> float:
    """Use the config temperature when the caller does not pass one."""
    if temp is None:
        return default_temperature
    return float(temp)


def resolve_sampler_steps(sampler_decode_steps: int, lsd_decode_steps: int | None) -> int:
    """`lsd_decode_steps` is the deprecated name for `sampler_decode_steps`."""
    if lsd_decode_steps is not None:
        logger.warning("lsd_decode_steps is deprecated, use sampler_decode_steps")
        return lsd_decode_steps
    return sampler_decode_steps


def resolve_config_path(config: str | Path) -> Path:
    """Map a stem or a yaml path to the file `load_model` will read."""
    text = str(config)
    if text.startswith(("hf://", "https://", "http://")):
        return download_if_necessary(text)
    if text.endswith(".yaml") or text.endswith(".yml"):
        return Path(config)
    return CONFIGS_DIR / f"{text}.yaml"


def is_shipped_config(origin: Path | None) -> bool:
    """True when the config file lives in this package's config directory."""
    if origin is None:
        return False
    return Path(origin).resolve().is_relative_to(CONFIGS_DIR.resolve())


def predefined_voice_source(origin: Path | None, name: str) -> str:
    """Return a voice-state URL, or ``legacy`` for the b6369a24 audio-prompt catalog."""
    if not is_shipped_config(origin):
        raise ValueError(
            "Cannot use predefined voices when the model "
            "is not loaded from a config associated with a language. "
            f"Here the origin is {origin}"
        )
    assert origin is not None
    if origin.stem == "b6369a24":
        if name not in PREDEFINED_VOICES:
            raise ValueError(
                f"Predefined voice {name!r} not found, available voices are {list(PREDEFINED_VOICES)}."
            )
        return "legacy"
    if name not in PREDEFINED_VOICE_NAMES:
        raise ValueError(
            f"Predefined voice {name!r} not found, available voices are {list(PREDEFINED_VOICE_NAMES)}."
        )
    return language_voice_state_url(origin.stem, name)


def _resolve_bundle_key(model: nn.Module, key: str):
    parts = key.split(".")
    if len(parts) < 2 or parts[-1].isdigit():
        raise KeyError(key)
    obj = model
    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj, parts[-1]


def _layout_bundle_weight(obj, param_name: str, value: mx.array) -> mx.array:
    """Match the bundle walker conv layout: PyTorch `(out, in, k)` becomes MLX `(out, k, in)`."""
    if param_name == "weight" and isinstance(obj, nn.Conv1d) and len(value.shape) == 3:
        return mx.transpose(value, (0, 2, 1))
    if (
        param_name == "weight"
        and (isinstance(obj, nn.ConvTranspose1d) or hasattr(obj, "is_conv_transpose"))
        and len(value.shape) == 3
    ):
        if value.shape[1] == 1:
            return mx.transpose(value, (0, 2, 1))
        return mx.transpose(value, (1, 2, 0))
    return value


def assign_bundle_weights(model: nn.Module, weights: Dict[str, np.ndarray]) -> None:
    """Place a fused bundle onto `model`. A skip is a failure."""
    from mlx.utils import tree_flatten

    transformed: Dict[str, mx.array] = {}
    unplaceable: list[str] = []
    for key, tensor in weights.items():
        array = tensor
        if getattr(array.dtype, "name", "") == "bfloat16":
            array = array.astype("float32")
        try:
            obj, param_name = _resolve_bundle_key(model, key)
            transformed[key] = _layout_bundle_weight(obj, param_name, mx.array(array))
        except (KeyError, IndexError, AttributeError, TypeError):
            unplaceable.append(key)
    params = {name: value for name, value in tree_flatten(model.parameters())}
    missing = sorted(set(params) - set(transformed))
    unexpected = sorted(set(transformed) - set(params))
    mismatched = sorted(
        key
        for key, value in transformed.items()
        if key in params
        and tuple(int(dim) for dim in value.shape) != tuple(int(dim) for dim in params[key].shape)
    )
    if unplaceable or missing or unexpected or mismatched:
        raise ValueError(
            "Bundle does not match the parameter tree: "
            f"unplaceable={unplaceable}, missing={missing}, unexpected={unexpected}, "
            f"shape_mismatch={mismatched}"
        )
    for key, value in transformed.items():
        obj, param_name = _resolve_bundle_key(model, key)
        setattr(obj, param_name, value)
    logger.info("Loaded %d weights", len(transformed))

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
        self.sampler_decode_steps = lsd_decode_steps
        self.lsd_decode_steps = lsd_decode_steps
        self.noise_clamp = noise_clamp
        self.eos_threshold = eos_threshold
        self.config = config
        self.has_voice_cloning = True
        self.origin: Path | None = None

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
            assign_bundle_weights(tts_model, weights)

        if config.flow_lm.weights_path is None and config.weights_path is None:
            logger.warning("No weights_path specified, model is uninitialized!")
        size_in_mb = size_of_dict(tts_model.state_dict()) // 1e6 if hasattr(tts_model, "state_dict") else 0
        logging.info("TTS Model loaded successfully. Size ~%d MB", size_in_mb)
        return tts_model

    @classmethod
    def load_model(
        cls,
        config: str | Path | None = None,
        temp: float | int | None = None,
        lsd_decode_steps: int | None = None,
        noise_clamp: float | int | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float | None = None,
        *,
        language: str | None = None,
        sampler_decode_steps: int = DEFAULT_SAMPLER_DECODE_STEPS,
    ):
        """Create and load a configured TTSModel with weights.

        `config` stays the first positional argument, so `load_model("b6369a24")`
        still selects that file. `language` is keyword-only. Passing both is an
        error. Omitting both loads `english`. An omitted `temp` uses the config's
        `default_temperature`.
        """
        requested = select_model_request(config, language)
        sampler_steps = resolve_sampler_steps(sampler_decode_steps, lsd_decode_steps)
        config_path = resolve_config_path(requested)
        loaded = load_config(config_path)
        temperature = resolve_temperature(temp, loaded.default_temperature)
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if sampler_steps < 1:
            raise ValueError("sampler_decode_steps must be at least 1")
        if eos_threshold is None:
            eos_threshold = loaded.default_eos_threshold
        logger.info("Loading model from config at %s...", config_path)
        tts_model = cls._from_pydantic_config_with_weights(
            loaded, temperature, sampler_steps, noise_clamp, eos_threshold
        )
        tts_model.origin = config_path.resolve()
        return tts_model

    def _run_flow_lm_and_increment_step(
        self,
        model_state: Dict,
        text_tokens: Optional[mx.array] = None,
        backbone_input_latents: Optional[mx.array] = None,
        audio_conditioning: Optional[mx.array] = None,
        temperature: float | None = None,
        sampling_key: Optional[mx.array] = None,
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
            temperature=temperature,
            sampling_key=sampling_key,
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
        temperature: float | None = None,
        sampling_key: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        """Compute next latent and EOS using FlowLM."""
        text_embeddings = self.flow_lm.conditioner(TokenizedText(text_tokens))
        text_embeddings = mx.concatenate([text_embeddings, audio_conditioning], axis=1)
        output_embeddings, is_eos = self.flow_lm._sample_next_latent(
            backbone_input_latents,
            text_embeddings,
            model_state=model_state,
            lsd_decode_steps=self.lsd_decode_steps,
            temp=self.temp if temperature is None else temperature,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
            sampling_key=sampling_key,
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
        dictionary: UserDictionary | None = None,
        language: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_retries: int = 1,
        decode_batch_size: int = 1,
        normalize: bool = False,
        brighten: bool = False,
    ) -> mx.array:
        """Generate full audio, retrying attempts that never reach EOS.

        ``decode_batch_size`` groups completed FlowLM latent frames for Mimi
        decoding. Values greater than one improve buffered-generation
        throughput at the cost of additional streaming latency.
        """
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        retry_state = copy.deepcopy(model_state) if not copy_state and max_retries else None
        audio_chunks = []
        for attempt in range(max_retries + 1):
            audio_chunks.clear()
            attempt_seed = None if seed is None else seed + attempt
            try:
                for chunk in self.generate_audio_stream(
                    model_state=model_state,
                    text_to_generate=text_to_generate,
                    frames_after_eos=frames_after_eos,
                    copy_state=copy_state,
                    max_tokens=max_tokens,
                    warmup_frames=warmup_frames,
                    dictionary=dictionary,
                    language=language,
                    temperature=temperature,
                    seed=attempt_seed,
                    decode_batch_size=decode_batch_size,
                    _raise_on_missing_eos=True,
                ):
                    audio_chunks.append(chunk)
                break
            except GenerationDidNotReachEOS:
                if retry_state is not None:
                    model_state.clear()
                    model_state.update(copy.deepcopy(retry_state))
                if attempt >= max_retries:
                    raise
                logger.warning(
                    "Generation did not reach EOS; retrying attempt %d/%d",
                    attempt + 2,
                    max_retries + 1,
                )

        audio = mx.concatenate(audio_chunks, axis=0)
        audio = self._postprocess_audio_start(audio, trim_start_ms=trim_start_ms, fade_in_ms=fade_in_ms)
        if brighten:
            audio = mx.array(brighten_speech(np.array(audio), self.sample_rate))
        if normalize:
            audio = mx.array(normalize_loudness(np.array(audio), self.sample_rate))
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
        dictionary: UserDictionary | None = None,
        language: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        decode_batch_size: int = 1,
        _raise_on_missing_eos: bool = False,
        stop: Event | None = None,
    ) -> Generator[mx.array, None, None]:
        """Yield audio chunks as they are generated."""
        if stop is not None and stop.is_set():
            return
        if temperature is not None and temperature < 0:
            raise ValueError("temperature must be non-negative")
        if decode_batch_size < 1:
            raise ValueError("decode_batch_size must be at least 1")
        if (
            frames_after_eos is None
            and self.config.model_recommended_frames_after_eos is not None
        ):
            frames_after_eos = self.config.model_recommended_frames_after_eos

        sampling_key = mx.random.key(seed) if seed is not None else None
        if language is None:
            origin = getattr(self, "origin", None)
            language = origin.stem.split("_")[0] if origin is not None else "english"
        prompt_kwargs = TTSModel._text_prompt_kwargs(self)
        chunks = split_into_best_sentences(
            self.flow_lm.conditioner.tokenizer,
            text_to_generate,
            max_tokens,
            dictionary=dictionary,
            language=language,
            **prompt_kwargs,
        )
        for chunk in chunks:
            if stop is not None and stop.is_set():
                return
            chunk_key = None
            if sampling_key is not None:
                split_keys = mx.random.split(sampling_key, 2)
                sampling_key, chunk_key = split_keys[0], split_keys[1]
            text_to_generate, frames_after_eos_guess = prepare_text_prompt(
                chunk,
                **prompt_kwargs,
            )
            frames_after_eos_guess += 2
            effective_frames = (
                frames_after_eos if frames_after_eos is not None else frames_after_eos_guess
            )
            yield from self._generate_audio_stream_short_text(
                model_state=model_state,
                text_to_generate=text_to_generate,
                frames_after_eos=effective_frames,
                copy_state=copy_state,
                warmup_frames=warmup_frames,
                temperature=temperature,
                sampling_key=chunk_key,
                decode_batch_size=decode_batch_size,
                raise_on_missing_eos=_raise_on_missing_eos,
                stop=stop,
            )

    def _text_prompt_kwargs(self) -> dict:
        """Honor the loaded model's text conventions before tokenization."""
        config = self.config
        return {
            "pad_with_spaces_for_short_inputs": config.pad_with_spaces_for_short_inputs,
            "remove_semicolons": config.remove_semicolons,
            "append_terminal_punctuation": getattr(config, "append_terminal_punctuation", True),
            "capitalize_first_letter": getattr(config, "capitalize_first_letter", True),
            "replace_characters": getattr(config, "replace_characters", {}),
        }

    def _generate_audio_stream_short_text(
        self,
        model_state: Dict,
        text_to_generate: str,
        frames_after_eos: int,
        copy_state: bool,
        warmup_frames: int,
        temperature: float | None = None,
        sampling_key: Optional[mx.array] = None,
        decode_batch_size: int = 1,
        raise_on_missing_eos: bool = False,
        stop: Event | None = None,
    ):
        """Generate audio for a short prompt with streaming FlowLM."""
        if stop is not None and stop.is_set():
            return
        if copy_state:
            model_state = copy.deepcopy(model_state)

        prepared = self.flow_lm.conditioner.prepare(text_to_generate)
        token_count = prepared.tokens.shape[1]
        max_gen_len = self._estimate_max_gen_len(token_count)
        current_end = self._flow_lm_current_end(model_state)
        required_len = current_end + token_count + max_gen_len
        self._expand_kv_cache(model_state, sequence_length=required_len)

        decoder_config = getattr(self.config.mimi, "decoder_transformer", None)
        mimi_context = (decoder_config or self.config.mimi.transformer).context
        mimi_state = init_states(self.mimi, batch_size=1, sequence_length=mimi_context)
        self._warmup_mimi_decoder(mimi_state, warmup_frames)

        t_generating = time.monotonic()

        def next_sampling_key() -> Optional[mx.array]:
            nonlocal sampling_key
            if sampling_key is None:
                return None
            split_keys = mx.random.split(sampling_key, 2)
            sampling_key = split_keys[0]
            return split_keys[1]

        with display_execution_time("Prompting text"):
            self._run_flow_lm_and_increment_step(
                model_state=model_state,
                text_tokens=prepared.tokens,
                temperature=temperature,
                sampling_key=next_sampling_key(),
            )

        backbone_input = mx.full(
            (1, 1, self.flow_lm.ldim),
            float("nan"),
            dtype=self.flow_lm.dtype,
        )
        steps_times = []
        eos_step = None
        total_generated_samples = 0
        pending_latents = []
        n_fade = self.config.mimi.sample_rate // 200
        onset_fade = mx.linspace(0.0, 1.0, n_fade) if n_fade else None

        def decode_latents(latents):
            """Decode one or more completed FlowLM frames with Mimi."""
            nonlocal onset_fade
            mimi_decoding_input = latents * self.flow_lm.emb_std + self.flow_lm.emb_mean
            audio_frame = self.mimi.decode_from_latent(mimi_decoding_input, mimi_state)
            increment_steps(self.mimi, mimi_state,
                            increment=self._mimi_steps_per_latent() * latents.shape[1])
            audio_chunk, onset_fade = apply_chunk_onset_fade(audio_frame[0, 0], onset_fade)
            return audio_chunk

        for generation_step in range(max_gen_len):
            if stop is not None and stop.is_set():
                return
            with display_execution_time("Generating latent", print_output=False) as timer:
                next_latent, is_eos = self._run_flow_lm_and_increment_step(
                    model_state=model_state,
                    backbone_input_latents=backbone_input,
                    temperature=temperature,
                    sampling_key=next_sampling_key(),
                )

                if decode_batch_size == 1:
                    audio_chunk = decode_latents(next_latent)

                    # Materialize FlowLM EOS and Mimi audio together to avoid two
                    # GPU synchronizations for every yielded frame.
                    mx.eval(is_eos, audio_chunk)
                    is_eos_scalar = bool(is_eos.item())
                    eos_step = note_eos_step(is_eos_scalar, generation_step, eos_step,
                                             self._MIN_FRAMES_BEFORE_EOS)
                    if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                        break

                    total_generated_samples += audio_chunk.shape[-1]
                    yield audio_chunk
                else:
                    # FlowLM is autoregressive, so finish each latent before the
                    # next step. Mimi can then decode completed latents in groups.
                    mx.eval(is_eos, next_latent)
                    is_eos_scalar = bool(is_eos.item())
                    eos_step = note_eos_step(is_eos_scalar, generation_step, eos_step,
                                             self._MIN_FRAMES_BEFORE_EOS)
                    if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                        break

                    pending_latents.append(next_latent)
                    if len(pending_latents) >= decode_batch_size:
                        audio_chunk = decode_latents(mx.concatenate(pending_latents, axis=1))
                        mx.eval(audio_chunk)
                        pending_latents.clear()
                        total_generated_samples += audio_chunk.shape[-1]
                        yield audio_chunk

                backbone_input = next_latent

            steps_times.append(timer.elapsed_time_ms)

        if pending_latents:
            audio_chunk = decode_latents(mx.concatenate(pending_latents, axis=1))
            mx.eval(audio_chunk)
            total_generated_samples += audio_chunk.shape[-1]
            yield audio_chunk

        duration_generated_audio = int(total_generated_samples * 1000 / self.config.mimi.sample_rate)
        generation_time = int((time.monotonic() - t_generating) * 1000)
        real_time_factor = duration_generated_audio / max(1, generation_time)
        logger.info(
            "Generated: %d ms of audio in %d ms so %.2fx faster than real-time",
            duration_generated_audio,
            generation_time,
            real_time_factor,
        )
        if eos_step is None:
            message = (
                f"Generation reached its {max_gen_len}-frame safety limit without EOS "
                f"for text: {text_to_generate[:80]!r}"
            )
            logger.warning(message)
            if raise_on_missing_eos:
                raise GenerationDidNotReachEOS(message)

    def _mimi_steps_per_latent(self) -> int:
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

    def _flow_state_modules(self) -> set[str]:
        """Module names a voice state is allowed to fill."""
        return set(init_states(self.flow_lm, batch_size=1, sequence_length=1))

    def get_state_for_audio_prompt(
        self, audio_conditioning: Union[Path, str, mx.array], truncate: bool = False
    ) -> Dict:
        if isinstance(audio_conditioning, (str, Path)) and is_safetensors_source(audio_conditioning):
            source = audio_conditioning
            if isinstance(source, str):
                source = download_if_necessary(source)
            return import_flow_state(source, self._flow_state_modules())

        prompt = None
        if isinstance(audio_conditioning, str) and audio_conditioning in PREDEFINED_VOICE_NAMES:
            source = predefined_voice_source(self.origin, audio_conditioning)
            if source == "legacy":
                prompt = load_predefined_voice_mlx(audio_conditioning)
            else:
                return import_flow_state(
                    download_if_necessary(source), self._flow_state_modules()
                )
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
