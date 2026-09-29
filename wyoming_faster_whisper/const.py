"""Constants."""

import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class SttLibrary(str, Enum):
    """Speech-to-text library."""

    AUTO = "auto"
    FASTER_WHISPER = "faster-whisper"
    TRANSFORMERS = "transformers"
    SHERPA = "sherpa"
    ONNX_ASR = "onnx-asr"
    FUNASR = "funasr"
    QWEN3_ASR = "qwen3-asr"


AUTO_LANGUAGE = "auto"
AUTO_MODEL = "auto"

# Where to look for Home Assistant when biasing toward its names.
HASS_API_URL = "http://homeassistant.local:8123/api"


# Home Assistant's own tag parser splits on either separator, so we do too.
_SEPARATORS = re.compile(r"[-_]")


def normalize_language_tag(code: str) -> str:
    """Lower-case a language tag and settle on "-" as the separator.

    "zh_HK" -> "zh-hk". The tables below are keyed by full tag, so a client using
    the underscore form would otherwise miss the entry and fall back to the base
    language -- which for Hong Kong silently means Mandarin instead of Cantonese.
    """
    return _SEPARATORS.sub("-", code.lower(), count=1)


def base_language(code: str) -> str:
    """Strip the region from a language tag and lower-case it.

    "en-US" -> "en". Used where a backend or model is selected by bare language:
    Home Assistant sends back the tag we advertised, but the --language option is
    typed by hand and other Wyoming clients send whatever they like.
    """
    return _SEPARATORS.split(code.lower(), maxsplit=1)[0]


def canonical_language_code(code: str) -> str:
    """Upper-case the region of a language tag, as Home Assistant writes it.

    "zh-hk" -> "zh-HK". Bare codes are returned unchanged. Script subtags
    (e.g. "sr-Latn") would be mangled, but no backend here reports one.
    """
    language, separator, region = code.partition("-")
    return f"{language}{separator}{region.upper()}"


# ---------------------------------------------------------------------------
# Backend language support
#
# Each set below lists the languages a backend's *default* model family can
# transcribe. build_info reports them to Home Assistant, so they are written in
# the tag form HA's matcher (homeassistant.util.language) compares against:
# bare ISO-639-1 codes, plus a region qualifier only where the region selects
# genuinely different model behaviour.
#
# The region qualifier matters for exactly one case today. HA's matcher treats
# "yue" and "zh" as unrelated languages, so a Cantonese-capable backend that
# only advertises "yue" is invisible to HA's zh-HK pipeline -- home-assistant/
# intents ships zh-CN/zh-HK/zh-TW and no bare "zh" or "yue" at all. Advertising
# "zh-HK" instead scores an exact region match, so a zh-HK pipeline reaches
# Cantonese while zh-CN/zh-TW still fall through to Mandarin via "zh".
# ---------------------------------------------------------------------------

# SenseVoice (FunASR) can be told to decode these languages explicitly;
# otherwise it auto-detects. Maps the locale-style codes that Home Assistant /
# intent-sentences use (e.g. "zh-CN") onto the base SenseVoice language.
_SENSE_VOICE_LANGUAGES = {
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hk": "yue",  # Hong Kong audio is typically Cantonese
    "yue": "yue",
    "ja": "ja",
    "ko": "ko",
    "en": "en",
}


def sense_voice_language(language: Optional[str]) -> Optional[str]:
    """Normalize a language code to a SenseVoice language, or None if unsupported."""
    if not language:
        return None

    return _SENSE_VOICE_LANGUAGES.get(normalize_language_tag(language))


# Advertised form of the keys above (regions upper-cased, as HA writes them).
SENSE_VOICE_LANGUAGES = {
    "en",
    "ja",
    "ko",
    "yue",
    "zh",
    "zh-CN",
    "zh-HK",
    "zh-TW",
}

# Language names Qwen3-ASR was trained to accept in the forced-language suffix.
# These live here rather than next to the decoder in qwen3_asr_handler so that
# build_info can report the supported codes without importing onnxruntime.
_QWEN3_ASR_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
    # Hong Kong audio is typically Cantonese, matching SenseVoice above. Without
    # this entry the locale fallback below strips the region and forces
    # Mandarin, silently losing Cantonese on a model that supports it.
    "zh-hk": "Cantonese",
}


def qwen3_asr_language(language: Optional[str]) -> Optional[str]:
    """Normalize a language code to a Qwen3-ASR language name, or None."""
    if not language:
        return None

    code = normalize_language_tag(language)
    name = _QWEN3_ASR_LANGUAGE_NAMES.get(code)
    if name is None:
        # Accept locale-style codes like "en-US" or "zh-CN".
        name = _QWEN3_ASR_LANGUAGE_NAMES.get(base_language(code))

    return name


# Advertised form of the keys above (regions upper-cased, as HA writes them).
QWEN3_ASR_LANGUAGES = {
    canonical_language_code(code) for code in _QWEN3_ASR_LANGUAGE_NAMES
}

# sherpa's offline default, NeMo Parakeet TDT 0.6b v3. (The v2 model picked for
# English is a subset of this: English only.)
PARAKEET_LANGUAGES = {
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "es",
    "sv",
    "ru",
    "uk",
}

# sherpa with --sherpa-streaming. The Kroko streaming zipformers are
# per-language and only these four are published; guess_model falls back to the
# English one (with a warning) for anything else.
KROKO_STREAMING_LANGUAGES = {"de", "en", "es", "fr"}

# onnx-asr's default model is GigaAM, which is Russian-only.
GIGAAM_LANGUAGES = {"ru"}


class Transcriber(ABC):
    """Base class for transcribers."""

    @abstractmethod
    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        pass

    def count_prompt_tokens(self, text: str) -> Optional[int]:
        """Count the tokens ``text`` would use as an initial_prompt.

        Returns None when this backend has no tokenizer to ask, in which case
        callers fall back to an estimate. Used to fit as many entity names as
        possible into the prompt without crossing the model's context limit.
        """
        return None

    @property
    def supports_streaming(self) -> bool:
        """Whether this transcriber can process audio chunks incrementally.

        When False (the default), callers must buffer the entire utterance and
        use transcribe(). When True, start_stream() returns a StreamingSession
        that transcribes audio as it arrives.
        """
        return False

    def start_stream(
        self,
        language: Optional[str] = None,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> "StreamingSession":
        """Begin a new streaming transcription session.

        The returned session holds all per-utterance state, so a single
        (shared) transcriber can drive multiple concurrent sessions.

        Only valid when supports_streaming is True.
        """
        raise NotImplementedError


class StreamingSession(ABC):
    """A single in-progress streaming transcription.

    Created by Transcriber.start_stream(). Holds per-utterance state so it is
    safe to use one session per client connection even when the underlying
    transcriber is shared.
    """

    @abstractmethod
    def accept_chunk(self, audio_bytes: bytes) -> None:
        """Feed a chunk of audio (16Khz 16-bit mono PCM) to the stream."""

    @abstractmethod
    def finish(self) -> str:
        """Finish the stream and return the final transcript."""
