"""Tests for language-code helpers in const.

These are dependency-free (no STT backend required) so they run everywhere.
"""

import pytest

from wyoming_faster_whisper.const import (
    GIGAAM_LANGUAGES,
    KROKO_STREAMING_LANGUAGES,
    PARAKEET_LANGUAGES,
    QWEN3_ASR_LANGUAGES,
    SENSE_VOICE_LANGUAGES,
    canonical_language_code,
    qwen3_asr_language,
    sense_voice_language,
)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        # Bare SenseVoice codes pass through.
        ("zh", "zh"),
        ("yue", "yue"),
        ("ja", "ja"),
        ("ko", "ko"),
        ("en", "en"),
        # Locale-style codes (e.g. from Home Assistant / intent-sentences).
        ("zh-CN", "zh"),
        ("zh-TW", "zh"),
        ("zh-HK", "yue"),  # Hong Kong audio is typically Cantonese
        # Case-insensitive, and either separator (Home Assistant accepts both).
        ("ZH-cn", "zh"),
        ("zh_HK", "yue"),
        ("ZH_hk", "yue"),
        ("JA", "ja"),
        # Unsupported / empty -> None (caller falls back to auto-detect).
        ("de", None),
        ("es", None),
        ("auto", None),
        ("", None),
        (None, None),
    ],
)
def test_sense_voice_language(code, expected) -> None:
    assert sense_voice_language(code) == expected


# --- Qwen3-ASR language normalization --------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en", "English"),
        ("EN", "English"),
        ("en-US", "English"),  # locale-style
        ("zh", "Chinese"),
        ("zh-CN", "Chinese"),
        ("zh-TW", "Chinese"),
        # Hong Kong is Cantonese, not Mandarin. Without the explicit entry the
        # locale fallback would strip "-HK" and force Chinese.
        ("zh-HK", "Cantonese"),
        ("zh_HK", "Cantonese"),
        ("yue", "Cantonese"),
        ("fil", "Filipino"),
        ("xx", None),
        ("", None),
        (None, None),
    ],
)
def test_qwen3_asr_language(code, expected) -> None:
    assert qwen3_asr_language(code) == expected


# --- Advertised language sets ----------------------------------------------


def test_advertised_sense_voice_codes_all_resolve() -> None:
    """Every code we report for FunASR must be one SenseVoice can be told."""
    for code in SENSE_VOICE_LANGUAGES:
        assert sense_voice_language(code) is not None, code


def test_advertised_qwen3_codes_all_resolve() -> None:
    """Every code we report for Qwen3-ASR must map to a trained language name."""
    for code in QWEN3_ASR_LANGUAGES:
        assert qwen3_asr_language(code) is not None, code


@pytest.mark.parametrize(
    "languages",
    [
        SENSE_VOICE_LANGUAGES,
        QWEN3_ASR_LANGUAGES,
        PARAKEET_LANGUAGES,
        KROKO_STREAMING_LANGUAGES,
        GIGAAM_LANGUAGES,
    ],
)
def test_advertised_codes_are_canonically_cased(languages) -> None:
    """Home Assistant writes regions upper-cased (zh-HK), so we must too."""
    for code in languages:
        assert canonical_language_code(code) == code, code


def test_cantonese_is_advertised_as_zh_hk() -> None:
    """Home Assistant's matcher treats "yue" and "zh" as unrelated languages.

    home-assistant/intents ships zh-CN/zh-HK/zh-TW and no bare "zh" or "yue", so
    a backend that only says "yue" can never be reached by a Cantonese pipeline.
    """
    for languages in (SENSE_VOICE_LANGUAGES, QWEN3_ASR_LANGUAGES):
        assert "zh-HK" in languages
