"""Code for transcription using the onnx-asr library."""

import wave
from pathlib import Path
from typing import Optional, Union
from unittest.mock import patch

import numpy as np
import onnx_asr
import onnxruntime as ort
from huggingface_hub import snapshot_download

from .const import Transcriber
from .device import onnx_providers, warn_if_no_onnx_gpu

_RATE = 16000


class OnnxAsrTranscriber(Transcriber):
    """Wrapper for onnx-asr model."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        local_files_only: bool,
        device: str = "cpu",
        quantization: Optional[str] = None,
    ) -> None:
        """Initialize model."""
        warn_if_no_onnx_gpu(device, ort.get_available_providers())

        # Force download to our cache dir
        def snapshot_download_with_cache(*args, **kwargs) -> str:
            kwargs["cache_dir"] = str(Path(cache_dir).resolve())
            kwargs["local_files_only"] = local_files_only

            return snapshot_download(*args, **kwargs)

        with patch("huggingface_hub.snapshot_download", snapshot_download_with_cache):
            self.onnx_model = onnx_asr.load_model(
                model_id,
                providers=onnx_providers(device),
                quantization=quantization,
            )

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

        audio_array = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        )

        recognize_kwargs = {}
        if language:
            recognize_kwargs["language"] = language

        text = self.onnx_model.recognize(  # type: ignore[call-overload]
            audio_array, sample_rate=_RATE, **recognize_kwargs
        )
        return text
