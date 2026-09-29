"""Logic for model selection, loading, and transcription."""

import asyncio
import importlib.util
import logging
import platform
from collections import defaultdict
from pathlib import Path
from typing import AbstractSet, Any, Dict, Optional, Set, Tuple, Union

from .const import (
    GIGAAM_LANGUAGES,
    KROKO_STREAMING_LANGUAGES,
    PARAKEET_LANGUAGES,
    QWEN3_ASR_LANGUAGES,
    SENSE_VOICE_LANGUAGES,
    SttLibrary,
    Transcriber,
    base_language,
    sense_voice_language,
)
from .device import is_gpu, resolve_compute_type
from .faster_whisper_handler import FasterWhisperTranscriber
from .languages import WHISPER_LANGUAGES

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
        self._available: Optional[Dict[str, bool]] = None

    def resolve_stt_library(self, language: Optional[str] = None) -> SttLibrary:
        """Resolve which speech-to-text library will be used for a language.

        Cached per language: backend availability cannot change while running, and
        callers (transcriber loading, the --vad-clip decision) ask repeatedly.
        """
        language = language or self.preferred_language

        cached = self._stt_library.get(language)
        if cached is not None:
            return cached

        # Select speech-to-text library
        stt_library = guess_stt_library(
            self.preferred_stt_library,
            self.model,
            language,
            **self._available_backends(),
        )
        self._stt_library[language] = stt_library
        return stt_library

    def _available_backends(self) -> Dict[str, bool]:
        """Report which backends are installed.

        Detected *without* importing them: importing a backend loads its native
        libraries (sherpa-onnx, torch, onnxruntime, funasr), and some of those
        abort the whole process at import time on certain CPUs (e.g. torch's LSE
        atomics SIGILL on the Raspberry Pi 4's ARMv8.0 Cortex-A72). We only want
        to pay that cost - and take that risk - for the single backend actually
        selected, so probe with find_spec here and defer the real import to the
        branch that instantiates the transcriber.

        Memoized because supported_languages() resolves a backend for every
        candidate language, and find_spec is not free.
        """
        if self._available is not None:
            return self._available

        available = {
            "has_sherpa": _module_available("sherpa_onnx"),
            "has_transformers": _module_available("transformers")
            and _module_available("torch"),
            "has_onnx_asr": _module_available("onnx_asr"),
            "has_funasr": _module_available("funasr"),
            "has_qwen3_asr": _module_available("onnxruntime")
            and _module_available("tokenizers"),
        }
        _LOGGER.debug("Backends available: %s", available)

        self._available = available
        return available

    def supported_languages(self) -> Set[str]:
        """Report the language codes this configuration can actually transcribe.

        Resolved per language rather than per backend, because under --stt-library
        auto the backend is chosen *by* language: "en" may go to sherpa while "de"
        falls through to faster-whisper. So walk every code any backend claims and
        keep the ones whose resolved backend actually supports them. That
        automatically follows the routing in guess_stt_library, including its
        fallback to faster-whisper when an explicitly requested backend is not
        installed.

        These are the sets for each backend's *default* model. An explicit
        --model can be narrower (a whisper ".en" checkpoint is handled below;
        an arbitrary sherpa or ONNX model id cannot be inspected without
        downloading it).
        """
        if is_english_only_model(self.model) and self.resolve_stt_library() in (
            SttLibrary.FASTER_WHISPER,
            SttLibrary.TRANSFORMERS,
        ):
            return {"en"}

        languages: Set[str] = set()
        for language in ALL_LANGUAGES:
            stt_library = self.resolve_stt_library(language)
            streaming = self.sherpa_streaming and (stt_library == SttLibrary.SHERPA)
            if language in library_languages(stt_library, streaming=streaming):
                languages.add(language)

        return languages

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

            transcriber = self._load_cache_first(stt_library, model, streaming)
            self._transcriber[key] = transcriber

        return transcriber

    def _load_cache_first(
        self, stt_library: SttLibrary, model: str, streaming: bool
    ) -> Transcriber:
        """Build a transcriber, preferring the cache over the network.

        The hub check is what breaks when there is no internet, not the model
        load: everything a cached model needs is already on disk. On a network
        that blackholes outbound traffic (a Docker bridge marked `internal`) that
        check stalls for the full TCP timeout on every start rather than failing
        fast, which boot-loops the container. So try the cache first and only
        reach for the network when a file is genuinely missing.

        --local-files-only stays strict: no fallback, so a model that isn't
        cached is an error instead of a surprise download.
        """
        if self.local_files_only:
            return self._build_transcriber(
                stt_library, model, streaming, local_files_only=True
            )

        try:
            return self._build_transcriber(
                stt_library, model, streaming, local_files_only=True
            )
        except OSError as exc:
            # Every backend reports a cache miss as an OSError:
            # huggingface_hub raises LocalEntryNotFoundError (a FileNotFoundError
            # subclass), transformers raises a bare OSError, and the sherpa
            # handler raises FileNotFoundError itself. A corrupt cache lands here
            # too, and the retry below surfaces it as the download failing.
            _LOGGER.debug("Model '%s' is not cached (%s), downloading", model, exc)

        return self._build_transcriber(
            stt_library, model, streaming, local_files_only=False
        )

    def _build_transcriber(
        self,
        stt_library: SttLibrary,
        model: str,
        streaming: bool,
        local_files_only: bool,
    ) -> Transcriber:
        """Construct the transcriber for a backend."""
        if stt_library == SttLibrary.SHERPA:
            if streaming:
                from .sherpa_handler import SherpaStreamingTranscriber  # noqa: F811

                return SherpaStreamingTranscriber(
                    model,
                    self.download_dir,
                    local_files_only=local_files_only,
                    cpu_threads=self.cpu_threads,
                    beam_size=self.beam_size,
                    device=self.device,
                )

            from .sherpa_handler import SherpaTranscriber  # noqa: F811

            return SherpaTranscriber(
                model,
                self.download_dir,
                local_files_only=local_files_only,
                cpu_threads=self.cpu_threads,
                device=self.device,
            )

        if stt_library == SttLibrary.ONNX_ASR:
            from .onnx_asr_handler import OnnxAsrTranscriber  # noqa: F811

            return OnnxAsrTranscriber(
                model,
                cache_dir=self.download_dir,
                local_files_only=local_files_only,
                device=self.device,
            )

        if stt_library == SttLibrary.TRANSFORMERS:
            from .transformers_whisper import TransformersTranscriber  # noqa: F811

            return TransformersTranscriber(
                model,
                cache_dir=self.download_dir,
                local_files_only=local_files_only,
                device=self.device,
            )

        if stt_library == SttLibrary.QWEN3_ASR:
            from .qwen3_asr_handler import Qwen3AsrTranscriber  # noqa: F811

            return Qwen3AsrTranscriber(
                model,
                cache_dir=self.download_dir,
                local_files_only=local_files_only,
                cpu_threads=self.cpu_threads,
                device=self.device,
            )

        if stt_library == SttLibrary.FUNASR:
            from .funasr_handler import FunASRTranscriber  # noqa: F811

            return FunASRTranscriber(
                model,
                cache_dir=self.download_dir,
                local_files_only=local_files_only,
                device=self.device,
            )

        return FasterWhisperTranscriber(
            model,
            cache_dir=self.download_dir,
            local_files_only=local_files_only,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            vad_parameters=self.vad_parameters,
            task=self.whisper_task,
        )

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


# Every code any backend might claim. supported_languages() filters this down to
# the ones the configured backend really handles. The non-Whisper entries are the
# region-qualified and out-of-Whisper codes (zh-HK, fil) that would otherwise
# never be considered.
ALL_LANGUAGES = (
    WHISPER_LANGUAGES
    | PARAKEET_LANGUAGES
    | KROKO_STREAMING_LANGUAGES
    | GIGAAM_LANGUAGES
    | SENSE_VOICE_LANGUAGES
    | QWEN3_ASR_LANGUAGES
)


def library_languages(
    stt_library: SttLibrary, *, streaming: bool = False
) -> AbstractSet[str]:
    """Report the languages a backend's default model can transcribe."""
    if stt_library == SttLibrary.SHERPA:
        return KROKO_STREAMING_LANGUAGES if streaming else PARAKEET_LANGUAGES

    if stt_library == SttLibrary.ONNX_ASR:
        return GIGAAM_LANGUAGES

    if stt_library == SttLibrary.FUNASR:
        return SENSE_VOICE_LANGUAGES

    if stt_library == SttLibrary.QWEN3_ASR:
        return QWEN3_ASR_LANGUAGES

    # faster-whisper and transformers both run Whisper checkpoints. AUTO only
    # reaches here as a fallback, which is faster-whisper.
    return WHISPER_LANGUAGES


def is_english_only_model(model: Optional[str]) -> bool:
    """Report whether a model id names an English-only Whisper checkpoint.

    Whisper publishes ".en" variants (tiny.en, base.en, distil-small.en) whose
    tokenizer has no language tokens at all, so they transcribe English whatever
    language is requested. Reporting the full multilingual list for one would
    let Home Assistant route any pipeline to it.
    """
    if model is None:
        return False

    name = model.lower().split("/")[-1]
    return name.endswith(".en") or (".en-" in name)


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
            # Compared by base language so "en-US" and "EN" route like "en".
            # Home Assistant sends back the bare code we advertised, but
            # --language is typed by hand and other Wyoming clients send
            # whatever they like; sense_voice_language below already normalizes.
            base = base_language(language) if language else None

            if (base == "ru") and has_onnx_asr:
                # Prefer GigaAM via onnx-asr
                return SttLibrary.ONNX_ASR

            if (base == "en") and has_sherpa:
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
    # Selected by base language, matching guess_stt_library above.
    base = base_language(language) if language else None

    if stt_library == SttLibrary.SHERPA:
        if streaming:
            # Best available streaming (OnlineRecognizer) model. The Kroko
            # streaming zipformers produce mixed-case, punctuated output with
            # much better accuracy than the older LibriSpeech models. They are
            # per-language, so warn for languages we don't have a default for.
            if base in (None, "en"):
                return "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"

            if base in ("de", "es", "fr"):
                return f"sherpa-onnx-streaming-zipformer-{base}-kroko-2025-08-06"

            _LOGGER.warning(
                "No default streaming sherpa model for language '%s'; pass "
                "--model to choose one. Falling back to English.",
                language,
            )
            return "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"

        if base == "en":
            return "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"

        # Non-English
        return "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"

    if stt_library == SttLibrary.TRANSFORMERS:
        if base == "en":
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
    if gpu:
        # float16 weights, and one size up: a GPU can afford it, and small is
        # where Whisper's accuracy starts being worth the download. Larger
        # models (e.g. Systran/faster-whisper-large-v3) are a --model away.
        return "Systran/faster-whisper-small"

    if is_arm:
        return "rhasspy/faster-whisper-tiny-int8"

    return "rhasspy/faster-whisper-base-int8"
