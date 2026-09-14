"""Tests for streaming sherpa-onnx sessions.

These use a stand-in recognizer rather than a real model: the behavior under
test is arithmetic about chunk boundaries, and the models that expose it are
hundreds of megabytes.
"""

from typing import List

import numpy as np
import pytest

# Skip the whole module unless sherpa-onnx is installed.
pytest.importorskip("sherpa_onnx")

from wyoming_faster_whisper.sherpa_handler import (  # noqa: E402
    _FALLBACK_TAIL_PADDING_SECONDS,
    _MIN_TAIL_PADDING_SECONDS,
    _RATE,
    SherpaStreamingSession,
    _measure_tail_padding,
)


class FakeStream:
    """Records the audio a session feeds in."""

    def __init__(self) -> None:
        self.chunks: List[np.ndarray] = []
        self.decoded_samples = 0
        self.finished = False

    @property
    def samples(self) -> int:
        return sum(len(chunk) for chunk in self.chunks)

    def accept_waveform(self, rate: int, samples: np.ndarray) -> None:
        assert rate == _RATE
        assert not self.finished, "audio after input_finished()"
        self.chunks.append(np.asarray(samples))

    def input_finished(self) -> None:
        self.finished = True

    def all_samples(self) -> np.ndarray:
        if not self.chunks:
            return np.empty(0, dtype=np.float32)

        return np.concatenate(self.chunks)


class FakeRecognizer:
    """Stand-in for sherpa_onnx.OnlineRecognizer.

    Reproduces the one behavior the tail padding exists for: a stream is only
    ready to decode once a whole chunk of samples is available, so anything
    left over when the audio ends is never seen by the encoder.
    """

    def __init__(self, chunk_samples: int) -> None:
        self.chunk_samples = chunk_samples
        self.streams: List[FakeStream] = []

    def create_stream(self) -> FakeStream:
        stream = FakeStream()
        self.streams.append(stream)
        return stream

    def is_ready(self, stream: FakeStream) -> bool:
        return (stream.samples - stream.decoded_samples) >= self.chunk_samples

    def decode_stream(self, stream: FakeStream) -> None:
        assert self.is_ready(stream)
        stream.decoded_samples += self.chunk_samples

    def get_result(self, stream: FakeStream) -> str:
        return f"decoded {stream.decoded_samples}"


def test_measure_tail_padding_covers_a_whole_chunk() -> None:
    """Padding is enough for the encoder to flush a partially-filled chunk.

    1.41s is the chunk of the Kroko streaming zipformers, for which the 0.66s
    from sherpa-onnx's own examples drops the last word or two.
    """
    chunk_samples = int(1.41 * _RATE)
    padding = _measure_tail_padding(FakeRecognizer(chunk_samples))

    assert padding >= chunk_samples


def test_measure_tail_padding_has_a_floor() -> None:
    """A model with a small chunk still gets sherpa-onnx's 0.66s."""
    padding = _measure_tail_padding(FakeRecognizer(int(0.1 * _RATE)))

    assert padding == int(_MIN_TAIL_PADDING_SECONDS * _RATE)


def test_measure_tail_padding_falls_back() -> None:
    """A recognizer that never reports ready doesn't hang or pad forever."""
    padding = _measure_tail_padding(FakeRecognizer(int(60 * _RATE)))

    assert padding == int(_FALLBACK_TAIL_PADDING_SECONDS * _RATE)


def test_finish_pads_then_decodes_everything() -> None:
    """The last chunk of speech is decoded, not left in the encoder."""
    chunk_samples = int(1.41 * _RATE)
    recognizer = FakeRecognizer(chunk_samples)
    padding = _measure_tail_padding(recognizer)

    session = SherpaStreamingSession(recognizer, padding)
    # One chunk of audio plus a little: the remainder is what gets dropped
    # without padding.
    speech_samples = chunk_samples + 1000
    session.accept_chunk(b"\x01\x00" * speech_samples)
    assert session.stream.decoded_samples == chunk_samples

    session.finish()

    assert session.stream.finished
    assert session.stream.decoded_samples >= speech_samples

    # Only silence was appended, and enough of it.
    tail = session.stream.all_samples()[speech_samples:]
    assert len(tail) >= padding
    assert not np.any(tail)


def test_finish_keeps_dangling_byte() -> None:
    """An odd-length chunk's carried-over byte is not lost at the end."""
    recognizer = FakeRecognizer(int(0.5 * _RATE))
    session = SherpaStreamingSession(recognizer, int(0.66 * _RATE))

    session.accept_chunk(b"\x01\x00" * 100 + b"\x02")
    assert session.stream.samples == 100

    session.finish()
    assert session.stream.samples > 100
