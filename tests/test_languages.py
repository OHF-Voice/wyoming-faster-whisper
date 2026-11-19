"""Tests for normalizing client language codes onto Whisper's.

Dependency-free: faster-whisper is a hard dependency, so the code list is always
importable.
"""

import pytest

from wyoming_faster_whisper.const import base_language
from wyoming_faster_whisper.languages import WHISPER_LANGUAGES, whisper_language


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en", "en"),
        ("EN", "en"),
        # Home Assistant falls back to the raw pipeline language when its
        # matcher finds nothing or a pipeline has no stt_language stored. Both
        # Whisper backends raise ValueError for these, failing transcription.
        ("pt-BR", "pt"),
        ("zh-CN", "zh"),
        ("de-CH", "de"),
        ("en_US", "en"),
        # Home Assistant languages Whisper has no token for -> auto-detect.
        ("ga", None),
        ("kw", None),
        # Qwen3-ASR has Filipino; Whisper does not.
        ("fil", None),
        ("", None),
        (None, None),
    ],
)
def test_whisper_language(code, expected) -> None:
    assert whisper_language(code) == expected


def test_every_result_is_a_real_whisper_code() -> None:
    """Whatever comes back must be safe to hand to a Whisper tokenizer."""
    for code in ("en", "PT-br", "zh-HK", "no", "ga", "fil", "auto", "xyz"):
        result = whisper_language(code)
        assert (result is None) or (result in WHISPER_LANGUAGES), code


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en", "en"),
        ("EN", "en"),
        ("en-US", "en"),
        ("en_US", "en"),
        ("zh-HK", "zh"),
        ("fil", "fil"),
    ],
)
def test_base_language(code, expected) -> None:
    assert base_language(code) == expected
