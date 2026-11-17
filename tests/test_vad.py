"""Tests for clipping audio to its speech region."""

import wave

import pytest

from wyoming_faster_whisper.vad import (
    RATE,
    StreamingSpeechClipper,
    clip_to_speech,
    speech_span,
)

from . import _DIR


def _speech() -> bytes:
    with wave.open(str(_DIR / "turn-on-the-office-lamp.wav"), "rb") as wav_file:
        assert wav_file.getframerate() == RATE
        return wav_file.readframes(wav_file.getnframes())


def _silence(seconds: float) -> bytes:
    return bytes(int(seconds * RATE) * 2)


def test_speech_span_pads_around_the_speech() -> None:
    probs = [0.0] * 10 + [0.9] * 5 + [0.0] * 10
    # 512-sample chunks are 32 ms, so 64 ms of padding is two chunks a side.
    assert speech_span(probs, 0.5, 64, 512) == (8, 17)


def test_speech_span_stays_inside_the_audio() -> None:
    assert speech_span([0.9, 0.0, 0.9], 0.5, 1000, 512) == (0, 3)


def test_speech_span_without_speech_is_none() -> None:
    assert speech_span([0.1, 0.2], 0.5, 400, 512) is None
    assert speech_span([], 0.5, 400, 512) is None


@pytest.mark.parametrize("piece_bytes", [1024, 1000, 333])
def test_streaming_clip_matches_clip_to_speech(piece_bytes: int) -> None:
    # Pieces that don't line up with VAD windows (or even samples) must not
    # change where the speech is found.
    audio = _silence(1.5) + _speech() + _silence(1.5)
    expected = clip_to_speech(audio)
    assert expected
    assert len(expected) < len(audio)

    clipper = StreamingSpeechClipper()
    for start in range(0, len(audio), piece_bytes):
        clipper.process(audio[start : start + piece_bytes])

    assert clipper.clip(audio) == expected


def test_streaming_clip_of_silence_is_empty() -> None:
    audio = _silence(2.0)
    clipper = StreamingSpeechClipper()
    clipper.process(audio)
    assert clipper.clip(audio) == b""
