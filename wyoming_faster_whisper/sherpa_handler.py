"""Code for transcription using the sherpa-onnx library."""

import logging
import shutil
import tarfile
import urllib.request
import wave
from pathlib import Path
from typing import Optional, Union

import numpy as np
import sherpa_onnx as so

from .const import StreamingSession, Transcriber
from .device import is_gpu, sherpa_provider

_LOGGER = logging.getLogger(__name__)

_RATE = 16000
_URL_FORMAT = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{model_id}.tar.bz2"

# Trailing silence fed to a streaming model before input_finished() so the
# encoder can flush its final chunk. Without this, the last word(s) of an
# utterance are dropped: sherpa-onnx only runs the encoder once a *full* chunk
# of frames is available, and whatever is left over when the audio ends is
# never decoded.
#
# How much is needed is a property of the model rather than a constant. A chunk
# is the encoder's `T` frames, and that varies widely between releases: 45
# frames (0.45s) for the 2023 LibriSpeech zipformers, but 141 frames (1.41s) for
# the 2025 Kroko ones that `--sherpa-streaming` defaults to. sherpa-onnx's own
# decode-files examples hard-code 0.66s, which covers the former and truncates
# the latter, so measure the model instead of trusting a number.
#
# That measurement is a floor rather than the whole answer: it is what the
# encoder needs to see every frame, but a transducer emits its last tokens a
# little after the audio they belong to, so keep sherpa-onnx's 0.66s as a
# minimum. Below it the 2023 models drop a final short word they otherwise get.
_MIN_TAIL_PADDING_SECONDS = 0.66
_TAIL_PADDING_PROBE_SECONDS = 0.02
_MAX_TAIL_PADDING_SECONDS = 5.0
_FALLBACK_TAIL_PADDING_SECONDS = 1.5


def _measure_tail_padding(recognizer: "so.OnlineRecognizer") -> int:
    """Return the samples of trailing silence needed to flush the final chunk.

    A fresh stream reports ready as soon as one full chunk of frames has
    arrived, so the audio it takes to get there from empty is exactly the
    worst case for a partially-filled chunk at the end of an utterance.
    """
    step = int(_TAIL_PADDING_PROBE_SECONDS * _RATE)
    min_samples = int(_MIN_TAIL_PADDING_SECONDS * _RATE)
    max_samples = int(_MAX_TAIL_PADDING_SECONDS * _RATE)
    silence = np.zeros(step, dtype=np.float32)

    stream = recognizer.create_stream()
    samples = 0
    while samples < max_samples:
        stream.accept_waveform(_RATE, silence)
        samples += step
        if recognizer.is_ready(stream):
            samples = max(samples, min_samples)
            _LOGGER.debug("Tail padding: %.2f second(s)", samples / _RATE)
            return samples

    _LOGGER.warning(
        "Could not determine the chunk size of the streaming model; "
        "using %.2f second(s) of tail padding",
        _FALLBACK_TAIL_PADDING_SECONDS,
    )
    return int(_FALLBACK_TAIL_PADDING_SECONDS * _RATE)


def _ensure_model(
    model_id: str, cache_dir: Union[str, Path], local_files_only: bool = False
) -> Path:
    """Download/extract a sherpa-onnx model if needed and return its directory."""
    cache_dir = Path(cache_dir)
    model_dir = cache_dir / model_id
    _LOGGER.debug("Looking for sherpa model: %s", model_dir)

    if not model_dir.exists():
        if local_files_only:
            # FileNotFoundError so ModelLoader can treat a sherpa cache miss the
            # same as huggingface_hub's LocalEntryNotFoundError.
            raise FileNotFoundError(
                f"Sherpa model '{model_id}' is not in {cache_dir} and downloading "
                "is disabled by --local-files-only"
            )

        url = _URL_FORMAT.format(model_id=model_id)
        _LOGGER.info("Downloading %s", url)
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download/extract to cache dir.
            # We assume that the .tar.bz2 contains a directory named after
            # the model id.
            with urllib.request.urlopen(url) as response:
                with tarfile.open(fileobj=response, mode="r|bz2") as tar:
                    for member in tar:
                        tar.extract(member, path=cache_dir)
        except Exception:
            # Delete directory so we'll download again next time
            shutil.rmtree(model_dir, ignore_errors=True)
            raise

    return model_dir


def _find_model_file(model_dir: Path, prefix: str, prefer_int8: bool = True) -> str:
    """Find a model file by prefix, preferring one quantization over the other.

    Streaming zipformer models use versioned file names (e.g.
    encoder-epoch-99-avg-1.int8.onnx), so we glob rather than hard-code.

    int8 is the right default on the CPU. On the GPU it is not: the CUDA
    execution provider has to partition around the quantization nodes it cannot
    run, so an fp32 graph is usually faster there when the model ships one.
    Several models (e.g. the Parakeet TDT releases) are int8-only, so this is a
    preference and not a requirement.
    """
    int8_matches = sorted(model_dir.glob(f"{prefix}*.int8.onnx"))
    other_matches = sorted(
        p
        for p in model_dir.glob(f"{prefix}*.onnx")
        if not p.name.endswith(".int8.onnx")
    )

    ordered = (
        (int8_matches, other_matches) if prefer_int8 else (other_matches, int8_matches)
    )
    for matches in ordered:
        if matches:
            return str(matches[0])

    raise FileNotFoundError(f"No '{prefix}*.onnx' model file found in {model_dir}")


def _resolve_provider(device: str) -> str:
    """Return the provider to use, falling back to the CPU with a warning.

    sherpa-onnx bundles its own onnxruntime rather than using the pip package,
    and the wheel on PyPI is CPU-only: it accepts provider="cuda", logs nothing
    useful, and runs on the CPU anyway. Detect that here so the fallback is
    stated rather than silently discovered as "the GPU image is no faster". A
    CUDA build ships an extra provider library next to the Python module.

    Note that the CUDA build is deliberately *not* used in the GPU image: its
    bundled onnxruntime and the pip onnxruntime-gpu package cannot both
    initialize CUDA in one process without crashing, and `--stt-library auto`
    can load two backends at once. See Dockerfile.gpu.
    """
    if not is_gpu(device):
        return "cpu"

    lib_dir = Path(so.__file__).parent / "lib"
    if not lib_dir.is_dir() or any(lib_dir.glob("libonnxruntime_providers_cuda*")):
        # Either a CUDA build, or an unrecognized layout (e.g. a source build)
        # about which nothing can be concluded - let sherpa-onnx decide.
        return sherpa_provider(device)

    _LOGGER.warning(
        "Device '%s' was requested but this sherpa-onnx build has no CUDA "
        "provider; running on the CPU instead. This is expected in the GPU "
        "image, where sherpa-onnx is installed from PyPI on purpose.",
        device,
    )
    return "cpu"


def _bytes_to_samples(audio_bytes: bytes) -> np.ndarray:
    """Convert 16-bit mono PCM bytes to a float32 sample array."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0


class SherpaTranscriber(Transcriber):
    """Wrapper for sherpa-onnx model."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        local_files_only: bool = False,
        cpu_threads: int = 4,
        device: str = "cpu",
    ) -> None:
        """Initialize model."""
        model_dir = _ensure_model(model_id, cache_dir, local_files_only)
        provider = _resolve_provider(device)
        prefer_int8 = provider == "cpu"

        # Load model. Locate the files by prefix so that both the int8-only
        # releases and model directories that ship an fp32 graph work.
        self.recognizer = so.OfflineRecognizer.from_transducer(
            num_threads=cpu_threads,
            encoder=_find_model_file(model_dir, "encoder", prefer_int8),
            decoder=_find_model_file(model_dir, "decoder", prefer_int8),
            joiner=_find_model_file(model_dir, "joiner", prefer_int8),
            tokens=f"{model_dir}/tokens.txt",
            provider=provider,
            model_type="nemo_transducer",
        )

        # Prime model so that the first transcription will be fast
        stream = self.recognizer.create_stream()
        stream.accept_waveform(_RATE, np.zeros(shape=(128), dtype=np.float32))
        self.recognizer.decode_stream(stream)

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Returns transcription for WAV file.

        WAV file must be 16Khz 16-bit mono audio.
        """
        wav_file: wave.Wave_read = wave.open(str(wav_path), "rb")
        with wav_file:
            assert wav_file.getframerate() == _RATE, "Sample rate must be 16Khz"
            assert wav_file.getsampwidth() == 2, "Width must be 16-bit (2 bytes)"
            assert wav_file.getnchannels() == 1, "Audio must be mono"
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        audio_array = _bytes_to_samples(audio_bytes)
        stream = self.recognizer.create_stream()
        stream.accept_waveform(_RATE, audio_array)
        self.recognizer.decode_stream(stream)
        return stream.result.text


class SherpaStreamingTranscriber(Transcriber):
    """Wrapper for a streaming sherpa-onnx model (OnlineRecognizer).

    Use with natively-streaming models (e.g. streaming zipformer transducers).
    Offline models such as Parakeet TDT are NOT supported here.
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        local_files_only: bool = False,
        cpu_threads: int = 4,
        beam_size: int = 5,
        device: str = "cpu",
    ) -> None:
        """Initialize model."""
        model_dir = _ensure_model(model_id, cache_dir, local_files_only)
        provider = _resolve_provider(device)
        prefer_int8 = provider == "cpu"

        # A beam size > 1 enables beam search, which is more accurate than the
        # default greedy decoding at a small latency cost.
        if beam_size > 1:
            decoding_method = "modified_beam_search"
        else:
            decoding_method = "greedy_search"

        # Load model. Streaming zipformer file names are versioned, so locate
        # them by prefix instead of hard-coding.
        self.recognizer = so.OnlineRecognizer.from_transducer(
            encoder=_find_model_file(model_dir, "encoder", prefer_int8),
            decoder=_find_model_file(model_dir, "decoder", prefer_int8),
            joiner=_find_model_file(model_dir, "joiner", prefer_int8),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=cpu_threads,
            provider=provider,
            decoding_method=decoding_method,
            max_active_paths=max(beam_size, 1),
        )

        self.tail_padding_samples = _measure_tail_padding(self.recognizer)

        # Prime model so that the first transcription will be fast. This needs a
        # whole chunk of audio: with less, the recognizer never reports ready
        # and the encoder is not run at all.
        stream = self.recognizer.create_stream()
        stream.accept_waveform(
            _RATE, np.zeros(shape=self.tail_padding_samples, dtype=np.float32)
        )
        while self.recognizer.is_ready(stream):
            self.recognizer.decode_stream(stream)

    @property
    def supports_streaming(self) -> bool:
        return True

    def start_stream(
        self,
        language: Optional[str] = None,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> StreamingSession:
        return SherpaStreamingSession(self.recognizer, self.tail_padding_samples)

    def transcribe(
        self,
        wav_path: Union[str, Path],
        language: Optional[str],
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
    ) -> str:
        """Returns transcription for WAV file (batch fallback).

        WAV file must be 16Khz 16-bit mono audio.
        """
        wav_file: wave.Wave_read = wave.open(str(wav_path), "rb")
        with wav_file:
            assert wav_file.getframerate() == _RATE, "Sample rate must be 16Khz"
            assert wav_file.getsampwidth() == 2, "Width must be 16-bit (2 bytes)"
            assert wav_file.getnchannels() == 1, "Audio must be mono"
            audio_bytes = wav_file.readframes(wav_file.getnframes())

        session = self.start_stream(language, beam_size, initial_prompt)
        session.accept_chunk(audio_bytes)
        return session.finish()


class SherpaStreamingSession(StreamingSession):
    """A single in-progress streaming transcription for OnlineRecognizer."""

    def __init__(
        self, recognizer: "so.OnlineRecognizer", tail_padding_samples: int
    ) -> None:
        self.recognizer = recognizer
        self.tail_padding_samples = tail_padding_samples
        self.stream = recognizer.create_stream()
        self._leftover: bytes = b""

    def accept_chunk(self, audio_bytes: bytes) -> None:
        audio_bytes = self._leftover + audio_bytes

        if len(audio_bytes) % 2:
            self._leftover = audio_bytes[-1:]
            audio_bytes = audio_bytes[:-1]
        else:
            self._leftover = b""

        if not audio_bytes:
            return

        self.stream.accept_waveform(_RATE, _bytes_to_samples(audio_bytes))
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

    def finish(self) -> str:
        # Carry over leftover bytes from the last chunk, if any. This is needed because
        # the model expects a multiple of 2 bytes (16-bit PCM) for each chunk
        if self._leftover:
            self.stream.accept_waveform(
                _RATE, _bytes_to_samples(self._leftover + b"\x00")
            )
            self._leftover = b""

        # Feed trailing silence so the encoder can flush its final chunk;
        # otherwise the last word(s) of the utterance are cut off.
        tail_padding = np.zeros(self.tail_padding_samples, dtype=np.float32)
        self.stream.accept_waveform(_RATE, tail_padding)

        # Signal end of audio and flush any remaining frames.
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        return self.recognizer.get_result(self.stream)
