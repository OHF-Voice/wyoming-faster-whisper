"""Tests for optional pySilero end-of-command detection."""

from typing import Iterable, List

from wyoming_faster_whisper import endpointing
from wyoming_faster_whisper.__main__ import build_info


class ProbabilityDetector:
    def __init__(self, probabilities: Iterable[float]) -> None:
        self.probabilities = iter(probabilities)
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def process_samples(self, _samples: List[float]) -> float:
        return next(self.probabilities)


def pcm_chunks(count: int) -> bytes:
    return b"\x00\x00" * endpointing.SAMPLES_PER_VAD_CHUNK * count


def test_silero_endpoint_requires_speech_then_silence() -> None:
    probabilities = [0.0] * 20 + [0.9] * 10 + [0.0] * 22
    detector = ProbabilityDetector(probabilities)
    endpoint = endpointing.SileroEndpointDetector(0.7, detector)
    endpoint.reset()

    assert endpoint.process(pcm_chunks(20))
    assert endpoint.process(pcm_chunks(10))
    assert not endpoint.process(pcm_chunks(22))
    assert detector.resets == 1


def test_speech_resets_silence_countdown() -> None:
    probabilities = [0.9] * 10 + [0.0] * 15 + [0.9] + [0.0] * 22
    endpoint = endpointing.SileroEndpointDetector(
        0.7, ProbabilityDetector(probabilities)
    )
    endpoint.reset()

    assert endpoint.process(pcm_chunks(10))
    assert endpoint.process(pcm_chunks(15))
    assert endpoint.process(pcm_chunks(1))
    assert endpoint.process(pcm_chunks(21))
    assert not endpoint.process(pcm_chunks(1))


def test_server_vad_disables_home_assistant_endpointing() -> None:
    assert build_info("model", ["en"]).asr[0].requires_external_vad
    assert (
        not build_info("model", ["en"], requires_external_vad=False)
        .asr[0]
        .requires_external_vad
    )
