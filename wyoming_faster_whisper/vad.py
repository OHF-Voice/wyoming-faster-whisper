"""Clip audio to its speech region with pysilero-vad before running STT.

Trimming leading/trailing non-speech reduces Whisper hallucinations on silence
and shortens the audio the model has to process. Audio is 16 kHz 16-bit mono PCM.
"""

import logging
import wave
from pathlib import Path
from typing import List, Optional, Tuple, Union

from pysilero_vad import SileroVoiceActivityDetector

_LOGGER = logging.getLogger(__name__)

RATE = 16000
WIDTH = 2  # bytes per sample (16-bit)


def speech_span(
    probs: List[float], threshold: float, pad_ms: int, chunk_samples: int
) -> Optional[Tuple[int, int]]:
    """Return the [start, end) VAD chunk indices to keep, or None if no speech.

    The span runs from the first to the last chunk at or above threshold, widened
    by pad_ms of context on each side.
    """
    speech = [i for i, p in enumerate(probs) if p >= threshold]
    if not speech:
        return None

    pad_chunks = max(0, (pad_ms * RATE // 1000) // chunk_samples)
    start = max(0, speech[0] - pad_chunks)
    end = min(len(probs), speech[-1] + 1 + pad_chunks)
    return start, end


def clip_to_speech(
    audio: bytes,
    threshold: float = 0.5,
    pad_ms: int = 400,
    detector: Optional[SileroVoiceActivityDetector] = None,
) -> bytes:
    """Trim leading/trailing non-speech, keeping pad_ms of context on each side.

    Returns the clipped 16 kHz 16-bit mono PCM. Returns b"" if no speech is
    detected (callers should decide whether to fall back to the full audio).
    A new detector is created per call unless one is supplied; the detector is
    stateful, so a shared instance must not be used across threads.
    """
    if detector is None:
        detector = SileroVoiceActivityDetector()
    else:
        detector.reset()

    chunk_bytes = detector.chunk_bytes()
    chunk_samples = detector.chunk_samples()

    # Not enough audio for even a single VAD window.
    if len(audio) < chunk_bytes:
        return b""

    probs = list(detector.process_chunks(audio))
    span = speech_span(probs, threshold, pad_ms, chunk_samples)
    if span is None:
        return b""

    start, end = span
    return audio[start * chunk_bytes : end * chunk_bytes]


class StreamingSpeechClipper:
    """clip_to_speech() for audio that arrives in pieces.

    process() runs the VAD on each piece as it arrives, so clipping the whole
    utterance at the end is only a slice: no model work is left for after the
    audio stops. clip() gives the same result clip_to_speech() would have given
    for everything passed to process(), with one exception: when the audio is an
    exact multiple of the VAD window long, pysilero-vad's process_chunks() (which
    clip_to_speech() uses) skips the last window and process() does not, so with
    speech near the end the clip can be one window (32 ms) longer. Not
    thread-safe.
    """

    def __init__(self, threshold: float = 0.5, pad_ms: int = 400) -> None:
        self.threshold = threshold
        self.pad_ms = pad_ms
        self._detector = SileroVoiceActivityDetector()
        self._chunk_bytes = self._detector.chunk_bytes()
        self._pending = b""
        self._probs: List[float] = []
        self._last_speech = -1  # index of the last window at/above threshold

    def process(self, audio: bytes) -> None:
        """Run the VAD over the next piece of 16 kHz 16-bit mono PCM."""
        audio = self._pending + audio
        usable = len(audio) - (len(audio) % self._chunk_bytes)

        # One window at a time: process_chunks() raises on less than a window
        # and skips the last one when given an exact multiple, which a stream of
        # pieces would hit on every piece.
        for start in range(0, usable, self._chunk_bytes):
            prob = self._detector.process_chunk(
                audio[start : start + self._chunk_bytes]
            )
            if prob >= self.threshold:
                self._last_speech = len(self._probs)
            self._probs.append(prob)

        self._pending = audio[usable:]

    def clip(self, audio: bytes) -> bytes:
        """Clip ``audio`` (everything passed to process()) to its speech region.

        Returns b"" if no speech was detected, as clip_to_speech() does.
        """
        span = speech_span(
            self._probs, self.threshold, self.pad_ms, self._detector.chunk_samples()
        )
        if span is None:
            return b""

        start, end = span
        return audio[start * self._chunk_bytes : end * self._chunk_bytes]

    def settled_span(self, min_silence_ms: int = 0) -> Optional[Tuple[int, int]]:
        """The [start, end) byte range clip() will keep unless speech resumes.

        None until there has been speech followed by enough silence to cover
        the trailing pad and at least ``min_silence_ms``: before that, more
        audio would still extend the clip.
        """
        if self._last_speech < 0:
            return None

        silence = len(self._probs) - (self._last_speech + 1)
        chunk_samples = self._detector.chunk_samples()
        needed = max(self.pad_ms, min_silence_ms) * RATE // 1000
        if silence * chunk_samples < needed:
            return None

        span = speech_span(self._probs, self.threshold, self.pad_ms, chunk_samples)
        assert span is not None
        start, end = span
        return start * self._chunk_bytes, end * self._chunk_bytes


def clip_wav_to_speech(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    threshold: float = 0.5,
    pad_ms: int = 400,
) -> bool:
    """Clip a 16 kHz 16-bit mono WAV to its speech region, writing to dst_path.

    Returns True if speech was found and dst_path now holds the clipped audio,
    or False if the WAV couldn't be read or no speech was detected (in which
    case dst_path is not written and callers should use the original audio).
    """
    try:
        with wave.open(str(src_path), "rb") as wav_file:
            audio = wav_file.readframes(wav_file.getnframes())
    except (wave.Error, OSError) as err:
        _LOGGER.warning("VAD clip: could not read %s: %s", src_path, err)
        return False

    clipped = clip_to_speech(audio, threshold=threshold, pad_ms=pad_ms)
    if not clipped:
        _LOGGER.debug("VAD clip: no speech detected, using full audio")
        return False

    clipped_file: wave.Wave_write = wave.open(str(dst_path), "wb")
    with clipped_file:
        clipped_file.setframerate(RATE)
        clipped_file.setsampwidth(WIDTH)
        clipped_file.setnchannels(1)
        clipped_file.writeframes(clipped)

    _LOGGER.debug(
        "VAD clip: %d ms -> %d ms",
        len(audio) // WIDTH * 1000 // RATE,
        len(clipped) // WIDTH * 1000 // RATE,
    )
    return True
