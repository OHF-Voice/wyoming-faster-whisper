"""Code for transcription using Qwen3-ASR ONNX exports.

Qwen3-ASR is a speech LLM: an audio encoder feeds projected audio embeddings into
a Qwen3 decoder that generates the transcript token by token. The ONNX export
splits it into several graphs, so unlike the other backends there is no library
that drives it for us - the greedy decode loop lives here.

The payoff is context biasing. Qwen3-ASR accepts free-form text in the chat
template's system turn to bias decoding toward specific spellings, so
``--initial-prompt`` can carry Home Assistant entity names ("Vocabulary: Ecobee,
office lamp.") and fix names that would otherwise be transcribed phonetically.

Two decoder layouts are supported, chosen by which files the model directory has:

*merged* (``decoder_merged.int4.onnx``) - one graph taking a KV cache *and* a
dynamic sequence length. Because the biasing prompt sits in the system turn,
ahead of the audio, its KV depends only on the prompt tokens, so it is computed
once and reused for every later utterance. That matters because the prompt is not
free: a 50-name list is ~180 tokens of prefill, which on a Pi 5 took a 3.2 s
command from 1.47 s to 3.42 s. Reusing it gives 2.20 s, and the graph also drops
the in-graph embedding table (1407 -> 785 MB on disk, 2.25 -> 1.55 GB peak RSS).

*split* (``decoder_init`` + ``decoder_step``) - the original export. ``decoder_init``
has no KV input and ``decoder_step`` is pinned to one token, so the prompt must be
re-prefilled every utterance. Kept as a fallback so existing model directories
keep working.

onnxruntime is imported at module scope (not lazily inside __init__) so that
importing this module raises ImportError when it is absent. ModelLoader relies on
that to detect the backend and fall back to faster-whisper, matching
onnx_asr_handler/funasr_handler.
"""

import json
import logging
import threading
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnxruntime as ort
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from .const import Transcriber, qwen3_asr_language
from .device import onnx_providers, warn_if_no_onnx_gpu

_LOGGER = logging.getLogger(__name__)

_RATE = 16000

# Mel front-end (identical to Whisper's).
_N_FFT = 400
_HOP_LENGTH = 160
_N_MELS = 128
_FMIN = 0.0
_FMAX = 8000.0

# Chat template token ids. NOTE: the reference prompt builder in
# andrewleech/qwen3-asr-onnx hardcodes 9125 and 882 here, which decode to
# " Current" and " time" rather than "system" and "user".
_IM_START = 151644
_IM_END = 151645
_SYSTEM = 8948
_USER = 872
_ASSISTANT = 77091
_NEWLINE = 198
_AUDIO_START = 151669
_AUDIO_END = 151670
_AUDIO_PAD = 151676
_ASR_TEXT = 151704  # <asr_text>: everything before it is the language preamble
_EOS_IDS = frozenset((151643, 151645))

_MAX_NEW_TOKENS = 256

# Only these files are needed for int4 inference; the repo also holds multi-GB
# FP32 weights and tarballs that we must not pull down.
_ALLOW_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "embed_tokens.bin",
    "encoder.int4.onnx",
    "encoder.int4.onnx.data",
    # Merged layout.
    "decoder_merged.int4.onnx",
    "decoder_merged.int4.onnx.data",
    # Split layout. A repo has one layout or the other, and patterns that match
    # nothing are simply skipped, so one list serves both.
    "decoder_init.int4.onnx",
    "decoder_step.int4.onnx",
    "decoder_weights.int4.data",
]

# Presence of this file selects the merged layout.
_MERGED_DECODER = "decoder_merged.int4.onnx"

# Additive attention mask: 0 where attending is allowed, this where it is not.
_MASK_BLOCKED = np.finfo(np.float32).min


def _mel_filterbank() -> np.ndarray:
    """Slaney-normalized mel filterbank, matching librosa.filters.mel(norm="slaney")."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(freq: np.ndarray) -> np.ndarray:
        mels = freq / f_sp
        above = freq >= min_log_hz
        mels[above] = min_log_mel + np.log(freq[above] / min_log_hz) / logstep
        return mels

    def mel_to_hz(mels: np.ndarray) -> np.ndarray:
        freqs = mels * f_sp
        above = mels >= min_log_mel
        freqs[above] = min_log_hz * np.exp(logstep * (mels[above] - min_log_mel))
        return freqs

    fft_freqs = np.fft.rfftfreq(_N_FFT, d=1.0 / _RATE)
    mel_points = np.linspace(
        hz_to_mel(np.array([_FMIN]))[0], hz_to_mel(np.array([_FMAX]))[0], _N_MELS + 2
    )
    band_hz = mel_to_hz(mel_points)

    diff = np.diff(band_hz)
    ramps = band_hz[:, np.newaxis] - fft_freqs[np.newaxis, :]
    weights = np.zeros((_N_MELS, len(fft_freqs)), dtype=np.float32)
    for i in range(_N_MELS):
        lower = -ramps[i] / diff[i]
        upper = ramps[i + 2] / diff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    # Slaney normalization: equal area per filter.
    weights *= (2.0 / (band_hz[2 : _N_MELS + 2] - band_hz[:_N_MELS]))[:, np.newaxis]
    return weights


def _log_mel_spectrogram(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """Whisper-compatible log-mel spectrogram, shape [1, n_mels, time]."""
    padded = np.pad(audio, _N_FFT // 2, mode="reflect")
    # Periodic Hann (np.hanning is symmetric).
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_N_FFT) / _N_FFT)

    frames = np.lib.stride_tricks.sliding_window_view(padded, _N_FFT)[::_HOP_LENGTH]
    spec = np.fft.rfft(frames * window, n=_N_FFT, axis=-1)
    magnitudes = (np.abs(spec) ** 2).T.astype(np.float32)

    mel_spec = mel_filters @ magnitudes
    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    log_spec = log_spec[:, :-1]  # match WhisperFeatureExtractor's frame count
    return log_spec[np.newaxis, :, :].astype(np.float32)


def _system_ids(context_ids: Sequence[int]) -> List[int]:
    """The chat template's system turn, which is where the biasing context goes.

    Split out because it is exactly the span whose KV the merged decoder caches:
    it precedes the audio, so nothing in it changes from one utterance to the next.
    """
    return [_IM_START, _SYSTEM, _NEWLINE, *context_ids, _IM_END, _NEWLINE]


def _causal_mask(q_len: int, past_len: int) -> np.ndarray:
    """[1, 1, q_len, past_len + q_len] additive mask for the merged decoder.

    New tokens may attend to everything already in the cache, and causally among
    themselves. Built here rather than inside the graph because deriving it from
    input shapes is the kind of shape arithmetic the ONNX exporter freezes into
    constants.
    """
    new = np.triu(np.full((q_len, q_len), _MASK_BLOCKED, dtype=np.float32), k=1)
    if past_len == 0:
        return new[np.newaxis, np.newaxis]

    past = np.zeros((q_len, past_len), dtype=np.float32)
    return np.concatenate([past, new], axis=-1)[np.newaxis, np.newaxis]


class Qwen3AsrTranscriber(Transcriber):
    """Wrapper for a Qwen3-ASR ONNX export (encoder + split decoder)."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        local_files_only: bool = False,
        cpu_threads: int = 4,
        device: str = "cpu",
    ) -> None:
        """Initialize model."""
        model_dir = Path(model_id)
        if not model_dir.is_dir():
            model_dir = Path(
                snapshot_download(
                    model_id,
                    cache_dir=str(Path(cache_dir).resolve()),
                    local_files_only=local_files_only,
                    allow_patterns=_ALLOW_PATTERNS,
                )
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # The int4 weights are MatMulNBits nodes. The CUDA provider supports
        # them, but the quantization is where most of this model's speed comes
        # from, so the GPU win here is smaller than for the fp32 backends.
        session_args = {
            "sess_options": options,
            "providers": onnx_providers(device),
        }

        self._encoder = ort.InferenceSession(
            str(model_dir / "encoder.int4.onnx"), **session_args
        )

        # Ask the session that was actually created, not onnxruntime's advertised
        # provider list: a CUDA provider that fails to load is still advertised.
        warn_if_no_onnx_gpu(device, self._encoder.get_providers())

        self._merged: Optional[ort.InferenceSession] = None
        self._decoder_init: Optional[ort.InferenceSession] = None
        self._decoder_step: Optional[ort.InferenceSession] = None

        if (model_dir / _MERGED_DECODER).is_file():
            self._merged = ort.InferenceSession(
                str(model_dir / _MERGED_DECODER), **session_args
            )
            # present_keys is [num_layers, batch, kv_heads, seq, head_dim]; the
            # non-sequence dims are static, so the empty cache that starts a
            # prefill can be shaped from the graph instead of from config.json.
            kv_shape = self._merged.get_outputs()[1].shape
            self._empty_kv = np.zeros(
                (kv_shape[0], 1, kv_shape[2], 0, kv_shape[4]), dtype=np.float32
            )
            _LOGGER.debug("Using merged decoder (biasing prompt KV is reused)")
        else:
            self._decoder_init = ort.InferenceSession(
                str(model_dir / "decoder_init.int4.onnx"), **session_args
            )
            self._decoder_step = ort.InferenceSession(
                str(model_dir / "decoder_step.int4.onnx"), **session_args
            )
            _LOGGER.debug(
                "Using split decoder; the biasing prompt is re-prefilled every "
                "utterance. A model directory with %s avoids that.",
                _MERGED_DECODER,
            )

        # Cached KV for the system turn, merged layout only. Exactly one entry:
        # Home Assistant sends the same name list every time, and each entry is
        # substantial (~42 MB for a 180-token prompt), so keeping a history would
        # cost far more memory than it could ever save.
        self._prefix_lock = threading.Lock()
        self._prefix_key: Optional[Tuple[int, ...]] = None
        self._prefix_kv: Optional[Tuple[np.ndarray, np.ndarray]] = None

        with open(model_dir / "config.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        # Keep the embedding table in fp16 and cast a single row per generated
        # token. Casting the whole table up front costs ~300MB of RSS for nothing.
        hidden_size = config["decoder"]["hidden_size"]
        self._embed_tokens = np.fromfile(
            model_dir / "embed_tokens.bin", dtype=np.float16
        ).reshape(-1, hidden_size)

        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._mel_filters = _mel_filterbank()
        self._encoder_inputs = [inp.name for inp in self._encoder.get_inputs()]
        self._prompt_cache: Dict[Tuple[str, Optional[str]], List[int]] = {}

    def count_prompt_tokens(self, text: str) -> Optional[int]:
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)

    def _context_ids(
        self, initial_prompt: Optional[str], language: Optional[str]
    ) -> Tuple[List[int], List[int]]:
        """Tokenize the biasing context and the forced-language suffix."""
        key = (initial_prompt or "", language)
        cached = self._prompt_cache.get(key)
        if cached is None:
            context = (
                self._tokenizer.encode(initial_prompt, add_special_tokens=False).ids
                if initial_prompt
                else []
            )
            language_name = qwen3_asr_language(language)
            suffix = (
                self._tokenizer.encode(
                    f"language {language_name}", add_special_tokens=False
                ).ids
                + [_ASR_TEXT]
                if language_name
                else []
            )
            # Cache as one list so a single dict lookup covers both.
            cached = [len(context)] + context + suffix
            self._prompt_cache[key] = cached

        n_context = cached[0]
        return cached[1 : 1 + n_context], cached[1 + n_context :]

    def _build_prompt_ids(
        self,
        n_audio_tokens: int,
        context_ids: Sequence[int],
        language_ids: Sequence[int],
    ) -> List[int]:
        """Build the chat-template prompt around the audio placeholders.

        <|im_start|>system\n{context}<|im_end|>\n
        <|im_start|>user\n<|audio_start|>{audio}<|audio_end|><|im_end|>\n
        <|im_start|>assistant\n{language {Name}<asr_text>}
        """
        return (
            _system_ids(context_ids)
            + [_IM_START, _USER, _NEWLINE, _AUDIO_START]
            + [_AUDIO_PAD] * n_audio_tokens
            + [_AUDIO_END, _IM_END, _NEWLINE]
            + [_IM_START, _ASSISTANT, _NEWLINE, *language_ids]
        )

    def _run_merged(
        self,
        input_embeds: np.ndarray,
        positions: Union[Sequence[int], np.ndarray],
        past_keys: np.ndarray,
        past_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One pass of the merged decoder over ``input_embeds``."""
        assert self._merged is not None
        return self._merged.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_embeds": input_embeds,
                "position_ids": np.asarray(positions, dtype=np.int64)[np.newaxis, :],
                "attention_mask": _causal_mask(
                    input_embeds.shape[1], past_keys.shape[3]
                ),
                "past_keys": past_keys,
                "past_values": past_values,
            },
        )

    def _prefix_cache(self, system_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """KV for the system turn, computed once and reused while it is unchanged."""
        key = tuple(system_ids)
        with self._prefix_lock:
            if (self._prefix_key == key) and (self._prefix_kv is not None):
                return self._prefix_kv

        # Run outside the lock: this is a model pass, and holding the lock would
        # serialize concurrent requests behind it. Two callers racing here just
        # compute the same thing twice and store the same result.
        _, keys, values = self._run_merged(
            self._embed_tokens[system_ids].astype(np.float32)[np.newaxis, :, :],
            np.arange(len(system_ids)),
            self._empty_kv,
            self._empty_kv,
        )

        with self._prefix_lock:
            self._prefix_key = key
            self._prefix_kv = (keys, values)

        _LOGGER.debug("Cached KV for a %s-token system turn", len(system_ids))
        return keys, values

    def _generate_merged(
        self,
        audio_features: np.ndarray,
        prompt_ids: List[int],
        n_system: int,
    ) -> List[int]:
        """Greedy decode reusing the system turn's KV across utterances."""
        past_keys, past_values = self._prefix_cache(prompt_ids[:n_system])

        # Everything after the system turn: the audio placeholders and the
        # assistant preamble. Audio features replace the placeholder embeddings.
        rest = prompt_ids[n_system:]
        input_embeds = self._embed_tokens[rest].astype(np.float32)[np.newaxis, :, :]
        audio_at = prompt_ids.index(_AUDIO_PAD) - n_system
        input_embeds[0, audio_at : audio_at + audio_features.shape[1]] = audio_features[
            0
        ]

        logits, past_keys, past_values = self._run_merged(
            input_embeds,
            np.arange(n_system, len(prompt_ids)),
            past_keys,
            past_values,
        )

        token = int(np.argmax(logits[0, -1, :]))
        tokens = [token]
        position = len(prompt_ids)

        while (token not in _EOS_IDS) and (len(tokens) < _MAX_NEW_TOKENS):
            logits, past_keys, past_values = self._run_merged(
                self._embed_tokens[token].astype(np.float32)[np.newaxis, np.newaxis, :],
                [position],
                past_keys,
                past_values,
            )
            token = int(np.argmax(logits[0, -1, :]))
            tokens.append(token)
            position += 1

        return tokens

    def _generate(self, audio_features: np.ndarray, prompt_ids: List[int]) -> List[int]:
        """Greedy decode on the split layout: prefill, then one token at a time."""
        assert (self._decoder_init is not None) and (self._decoder_step is not None)

        logits, past_keys, past_values = self._decoder_init.run(
            ["logits", "present_keys", "present_values"],
            {
                "input_ids": np.array(prompt_ids, dtype=np.int64)[np.newaxis, :],
                "position_ids": np.arange(len(prompt_ids), dtype=np.int64)[
                    np.newaxis, :
                ],
                "audio_features": audio_features,
                "audio_offset": np.array(
                    [prompt_ids.index(_AUDIO_PAD)], dtype=np.int64
                ),
            },
        )

        token = int(np.argmax(logits[0, -1, :]))
        tokens = [token]
        position = len(prompt_ids)

        while (token not in _EOS_IDS) and (len(tokens) < _MAX_NEW_TOKENS):
            logits, past_keys, past_values = self._decoder_step.run(
                ["logits", "present_keys", "present_values"],
                {
                    "input_embeds": self._embed_tokens[token].astype(np.float32)[
                        np.newaxis, np.newaxis, :
                    ],
                    "position_ids": np.array([[position]], dtype=np.int64),
                    "past_keys": past_keys,
                    "past_values": past_values,
                },
            )
            token = int(np.argmax(logits[0, -1, :]))
            tokens.append(token)
            position += 1

        return tokens

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Returns transcription for WAV file.

        WAV file must be 16Khz 16-bit mono audio. Decoding is greedy, so
        beam_size is ignored. initial_prompt is used as biasing context rather
        than as a transcript prefix.
        """
        wav_file: wave.Wave_read = wave.open(str(wav_path), "rb")
        with wav_file:
            assert wav_file.getframerate() == _RATE, "Sample rate must be 16Khz"
            assert wav_file.getsampwidth() == 2, "Width must be 16-bit (2 bytes)"
            assert wav_file.getnchannels() == 1, "Audio must be mono"
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        mel = _log_mel_spectrogram(audio, self._mel_filters)

        encoder_inputs = {self._encoder_inputs[0]: mel}
        if len(self._encoder_inputs) > 1:
            encoder_inputs[self._encoder_inputs[1]] = np.array(
                [mel.shape[2]], dtype=np.int64
            )
        audio_features = self._encoder.run(None, encoder_inputs)[0]

        context_ids, language_ids = self._context_ids(initial_prompt, language)
        prompt_ids = self._build_prompt_ids(
            audio_features.shape[1], context_ids, language_ids
        )

        if self._merged is not None:
            tokens = self._generate_merged(
                audio_features, prompt_ids, len(_system_ids(context_ids))
            )
        else:
            tokens = self._generate(audio_features, prompt_ids)

        # The model writes the detected language before <asr_text>; drop it.
        if _ASR_TEXT in tokens:
            tokens = tokens[tokens.index(_ASR_TEXT) + 1 :]

        return self._tokenizer.decode(tokens, skip_special_tokens=True).strip()
