"""Logic for model selection, loading, and transcription."""

import asyncio
import importlib.util
import logging
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union

from .const import SttLibrary, Transcriber, sense_voice_language
from .device import is_gpu, resolve_compute_type
from .faster_whisper_handler import FasterWhisperTranscriber

_LOGGER = logging.getLogger(__name__)

TRANSCRIBER_KEY = Tuple[SttLibrary, str, bool]  # (library, model id, streaming)


def _module_available(name: str) -> bool:
    """Report whether a module is installed without importing it.

    Importing a backend pulls in its native libraries, some of which abort at
    import time on certain CPUs (e.g. torch's LSE atomics raise SIGILL on the
    ARMv8.0 Cortex-A72 in the Raspberry Pi 4). find_spec only locates the module,
    so we can detect availability without paying that cost or risking the crash.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # A parent package failed to import, or the name is malformed.
        return False


class ModelLoader:
    """Load transcribers for models."""

    def __init__(
        self,
        preferred_stt_library: SttLibrary,
        preferred_language: Optional[str],
        download_dir: Union[str, Path],
        local_files_only: bool,
        model: Optional[str],
        compute_type: str,
        device: str,
        beam_size: int,
        cpu_threads: int,
        initial_prompt: Optional[str],
        vad_parameters: Optional[Dict[str, Any]],
        whisper_task: Optional[str] = None,
        sherpa_streaming: bool = False,
        vad_clip: bool = False,
        vad_clip_threshold: float = 0.5,
        vad_clip_pad_ms: int = 400,
        vad_clip_libraries: Optional[Set[SttLibrary]] = None,
    ) -> None:
        self.preferred_stt_library = preferred_stt_library
        self.preferred_language = preferred_language
        self.sherpa_streaming = sherpa_streaming

        self.download_dir = Path(download_dir)
        self.local_files_only = local_files_only

        self.model = model
        self.device = device

        # faster-whisper only. On a GPU an unspecified compute type becomes
        # float16 rather than the model's own (int8, for the defaults here).
        self.compute_type = resolve_compute_type(compute_type, device)

        self.beam_size = beam_size
        self.cpu_threads = cpu_threads
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters
        self.whisper_task = whisper_task
        self.vad_clip = vad_clip
        self.vad_clip_threshold = vad_clip_threshold
        self.vad_clip_pad_ms = vad_clip_pad_ms
        # None means every library when vad_clip is set; otherwise only these.
        self.vad_clip_libraries = vad_clip_libraries

        self._transcriber: Dict[TRANSCRIBER_KEY, Transcriber] = {}
        self._transcriber_lock: Dict[TRANSCRIBER_KEY, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )
        self._stt_library: Dict[Optional[str], SttLibrary] = {}

    def resolve_stt_library(self, language: Optional[str] = None) -> SttLibrary:
        """Resolve which speech-to-text library will be used for a language.

        Cached per language: backend availability cannot change while running, and
        callers (transcriber loading, the --vad-clip decision) ask repeatedly.
        """
        language = language or self.preferred_language

        cached = self._stt_library.get(language)
        if cached is not None:
            return cached

        # Detect which backends are installed *without* importing them. Importing
        # a backend loads its native libraries (sherpa-onnx, torch, onnxruntime,
        # funasr); some of those abort the whole process at import time on certain
        # CPUs (e.g. torch's LSE atomics SIGILL on the Raspberry Pi 4's ARMv8.0
        # Cortex-A72). We only want to pay that cost - and take that risk - for the
        # single backend actually selected below, so probe with find_spec here and
        # defer the real import to the branch that instantiates the transcriber.
        has_sherpa = _module_available("sherpa_onnx")
        has_transformers = _module_available("transformers") and _module_available(
            "torch"
        )
        has_onnx_asr = _module_available("onnx_asr")
        has_funasr = _module_available("funasr")
        has_qwen3_asr = _module_available("onnxruntime") and _module_available(
            "tokenizers"
        )
        _LOGGER.debug(
            "Backends available: sherpa=%s transformers=%s onnx_asr=%s funasr=%s"
            " qwen3_asr=%s",
            has_sherpa,
            has_transformers,
            has_onnx_asr,
            has_funasr,
            has_qwen3_asr,
        )

        # Select speech-to-text library
        stt_library = guess_stt_library(
            self.preferred_stt_library,
            self.model,
            language,
            has_transformers=has_transformers,
            has_sherpa=has_sherpa,
            has_onnx_asr=has_onnx_asr,
            has_funasr=has_funasr,
            has_qwen3_asr=has_qwen3_asr,
        )
        self._stt_library[language] = stt_library
        return stt_library

    def should_vad_clip(self, language: Optional[str] = None) -> bool:
        """Report whether leading/trailing silence should be clipped."""
        return vad_clip_enabled(
            self.resolve_stt_library(language),
            self.vad_clip,
            self.vad_clip_libraries,
        )

    async def load_transcriber(self, language: Optional[str] = None) -> Transcriber:
        """Load or get transcriber from cache for a language."""
        language = language or self.preferred_language
        stt_library = self.resolve_stt_library(language)

        # Streaming is only supported by the sherpa backend.
        streaming = self.sherpa_streaming and (stt_library == SttLibrary.SHERPA)

        # Select model
        model = self.model
        if model is None:  # auto
            machine = platform.machine().lower()
            is_arm = ("arm" in machine) or ("aarch" in machine)
            model = guess_model(
                stt_library,
                language,
                is_arm,
                streaming=streaming,
                gpu=is_gpu(self.device),
            )

        _LOGGER.debug(
            "Selected stt-library '%s' with model '%s' (streaming=%s, device=%s)",
            stt_library.value,
            model,
            streaming,
            self.device,
        )

        # Load transcriber
        assert stt_library != SttLibrary.AUTO
        assert model

        key = (stt_library, model, streaming)

        async with self._transcriber_lock[key]:
            transcriber = self._transcriber.get(key)
            if transcriber is not None:
                return transcriber

            if stt_library == SttLibrary.SHERPA:
                if streaming:
                    from .sherpa_handler import SherpaStreamingTranscriber  # noqa: F811

                    transcriber = SherpaStreamingTranscriber(
                        model,
                        self.download_dir,
                        cpu_threads=self.cpu_threads,
                        beam_size=self.beam_size,
                        device=self.device,
                    )
                else:
                    from .sherpa_handler import SherpaTranscriber  # noqa: F811

                    transcriber = SherpaTranscriber(
                        model,
                        self.download_dir,
                        cpu_threads=self.cpu_threads,
                        device=self.device,
                    )
            elif stt_library == SttLibrary.ONNX_ASR:
                from .onnx_asr_handler import OnnxAsrTranscriber  # noqa: F811

                transcriber = OnnxAsrTranscriber(
                    model,
                    cache_dir=self.download_dir,
                    local_files_only=self.local_files_only,
                    device=self.device,
                )
            elif stt_library == SttLibrary.TRANSFORMERS:
                from .transformers_whisper import TransformersTranscriber  # noqa: F811

                transcriber = TransformersTranscriber(
                    model,
                    cache_dir=self.download_dir,
                    local_files_only=self.local_files_only,
                    device=self.device,
                )
            elif stt_library == SttLibrary.QWEN3_ASR:
                from .qwen3_asr_handler import Qwen3AsrTranscriber  # noqa: F811

                transcriber = Qwen3AsrTranscriber(
                    model,
                    cache_dir=self.download_dir,
                    local_files_only=self.local_files_only,
                    cpu_threads=self.cpu_threads,
                    device=self.device,
                )
            elif stt_library == SttLibrary.FUNASR:
                from .funasr_handler import FunASRTranscriber  # noqa: F811

                transcriber = FunASRTranscriber(
                    model,
                    cache_dir=self.download_dir,
                    local_files_only=self.local_files_only,
                    device=self.device,
                )
            else:
                transcriber = FasterWhisperTranscriber(
                    model,
                    cache_dir=self.download_dir,
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=self.cpu_threads,
                    vad_parameters=self.vad_parameters,
                    task=self.whisper_task,
                )

            self._transcriber[key] = transcriber

        return transcriber

    async def transcribe(
        self, wav_path: Union[str, Path], language: Optional[str]
    ) -> str:
        """Transcribe WAV file using appropriate transcriber.

        Assume WAV file is 16Khz 16-bit mono PCM.
        """
        transcriber = await self.load_transcriber(language)
        text = await asyncio.to_thread(
            transcriber.transcribe,
            wav_path,
            language=language,
            beam_size=self.beam_size,
            initial_prompt=self.initial_prompt,
        )
        _LOGGER.debug("Transcribed audio: %s", text)

        return text


def vad_clip_enabled(
    stt_library: SttLibrary,
    vad_clip: bool,
    vad_clip_libraries: Optional[Set[SttLibrary]] = None,
) -> bool:
    """Report whether --vad-clip applies to a resolved speech-to-text library.

    `--vad-clip` with no values enables clipping for every library
    (vad_clip_libraries is None); naming libraries restricts it to those, which is
    useful because the benefit is backend-specific. Clipping is a clear win for
    length-proportional backends (qwen3-asr, sherpa, funasr) and a wash for
    faster-whisper, which pads audio to 30s internally regardless.
    """
    if not vad_clip:
        return False

    if vad_clip_libraries is None:
        return True

    return stt_library in vad_clip_libraries


def is_distil_whisper(model: Optional[str]) -> bool:
    """Report whether a model id names a Distil-Whisper checkpoint.

    Distil-Whisper was distilled without previous-text conditioning, so prompt
    context never helps it. `distil-small.en` is actively damaged: a 52-token
    prompt pushes avg_logprob below -1.0 and yields truncated or looping output
    on audio it transcribes perfectly unprompted. `distil-large-v3` is instead
    inert -- stable at every prompt size, but with no biasing effect either.
    Nothing in the CTranslate2 config identifies a distilled checkpoint, so the
    model id is the only signal available.
    """
    return (model is not None) and ("distil" in model.lower())


def guess_stt_library(
    preferred_stt_library: SttLibrary,
    model: Optional[str],
    language: Optional[str],
    *,
    has_transformers: bool,
    has_sherpa: bool,
    has_onnx_asr: bool,
    has_funasr: bool,
    has_qwen3_asr: bool = False,
) -> SttLibrary:
    """Resolve which speech-to-text library to use.

    When the preferred library is AUTO and no model is forced, pick the best
    available specialized backend for the language; otherwise faster-whisper.
    A non-AUTO library falls back to faster-whisper when its dependency is
    missing.
    """
    if preferred_stt_library == SttLibrary.AUTO:
        if model is None:  # auto-select a per-language backend
            if (language == "ru") and has_onnx_asr:
                # Prefer GigaAM via onnx-asr
                return SttLibrary.ONNX_ASR

            if (language == "en") and has_sherpa:
                # Prefer Parakeet via sherpa for English. The v3 Parakeet model
                # claims to auto detect other languages, but it doesn't work.
                return SttLibrary.SHERPA

            if (
                sense_voice_language(language) in ("zh", "yue", "ja", "ko")
            ) and has_funasr:
                # Prefer SenseVoice via FunASR for Chinese, Cantonese, Japanese,
                # and Korean (incl. locale codes like "zh-CN").
                return SttLibrary.FUNASR

        # Default to faster-whisper
        return SttLibrary.FASTER_WHISPER

    # Explicit library: fall back to faster-whisper if its dependency is absent.
    available = {
        SttLibrary.TRANSFORMERS: has_transformers,
        SttLibrary.SHERPA: has_sherpa,
        SttLibrary.ONNX_ASR: has_onnx_asr,
        SttLibrary.FUNASR: has_funasr,
        SttLibrary.QWEN3_ASR: has_qwen3_asr,
    }
    if not available.get(preferred_stt_library, True):
        _LOGGER.debug("Falling back to faster-whisper (missing dependencies)")
        return SttLibrary.FASTER_WHISPER

    return preferred_stt_library


def guess_model(
    stt_library: SttLibrary,
    language: Optional[str],
    is_arm: bool,
    streaming: bool = False,
    gpu: bool = False,
) -> str:
    """Automatically guess STT model id.

    Only the faster-whisper defaults change on a GPU, where the int8 CTranslate2
    conversions used on the CPU are the wrong trade: they save memory a GPU has
    to spare and give up the float16 throughput it was built for. The other
    backends' default models are published in a single quantization, so there is
    nothing to switch to.
    """
    if stt_library == SttLibrary.SHERPA:
        if streaming:
            # Best available streaming (OnlineRecognizer) model. The Kroko
            # streaming zipformers produce mixed-case, punctuated output with
            # much better accuracy than the older LibriSpeech models. They are
            # per-language, so warn for languages we don't have a default for.
            if language in (None, "en"):
                return "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"

            if language in ("de", "es", "fr"):
                return f"sherpa-onnx-streaming-zipformer-{language}-kroko-2025-08-06"

            _LOGGER.warning(
                "No default streaming sherpa model for language '%s'; pass "
                "--model to choose one. Falling back to English.",
                language,
            )
            return "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"

        if language == "en":
            return "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"

        # Non-English
        return "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"

    if stt_library == SttLibrary.TRANSFORMERS:
        if language == "en":
            if is_arm:
                return "openai/whisper-tiny.en"

            return "openai/whisper-base.en"

        # Non-English
        if is_arm:
            return "openai/whisper-tiny"

        return "openai/whisper-base"

    if stt_library == SttLibrary.ONNX_ASR:
        return "gigaam-v2-rnnt"

    if stt_library == SttLibrary.QWEN3_ASR:
        # Merged decoder: same weights as qwen3-asr-0.6b-onnx-int4, but the
        # biasing prompt's KV cache is reusable and the package is 785 MB rather
        # than 1.4 GB. The split repo still works if given explicitly.
        return "rhasspy/qwen3-asr-0.6b-onnx-int4-merged"

    if stt_library == SttLibrary.FUNASR:
        return "FunAudioLLM/SenseVoiceSmall"

    # faster-whisper
    if language == "lt":
        # Lithuanian is the one language where the generic defaults are not
        # just weaker but unusable. Measured on FLEURS lt (986 utterances,
        # 17,136 reference words, identical normalisation for every model):
        #
        #   paprika-whisper-lt-v3 (this fine-tune)  12.11%
        #   faster-whisper-large-v3                 24.23%
        #   faster-whisper-small   (GPU default)    65.79%
        #   faster-whisper-base    (CPU default)    87.84%
        #
        # At 87.84% the CPU default does not produce a worse transcript, it
        # produces no usable transcript at all — and it fails silently, looking
        # like a working feature.
        #
        # This is a CTranslate2 int8 conversion of kristijonas/paprika-whisper-lt-v3
        # (CC BY 4.0), a whisper-large-v3-turbo fine-tune trained on 3,281 h of
        # the LIEPA-3 corpus by Kristijonas Jakubsonas, who published it openly.
        #
        # Published in a single quantization, like the other backends' defaults,
        # so the GPU and arm branches below do not apply. It is int8, but ~800M
        # parameters — larger than the small the GPU branch would pick — and
        # runs at 0.23 s per utterance on an RTX PRO 4500.
        return "RobertasTa/paprika-whisper-lt-v3-ct2-int8"

    if gpu:
        # float16 weights, and one size up: a GPU can afford it, and small is
        # where Whisper's accuracy starts being worth the download. Larger
        # models (e.g. Systran/faster-whisper-large-v3) are a --model away.
        return "Systran/faster-whisper-small"

    if is_arm:
        return "rhasspy/faster-whisper-tiny-int8"

    return "rhasspy/faster-whisper-base-int8"
