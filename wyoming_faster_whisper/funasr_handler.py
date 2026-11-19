"""Code for transcription using the FunASR library."""

import contextlib
import sys
import wave
from pathlib import Path
from typing import Optional, Union

# FunASR is imported at module scope (not lazily inside __init__) so that
# importing this module raises ImportError when FunASR is absent. ModelLoader
# relies on that to detect the backend and fall back to faster-whisper, matching
# onnx_asr_handler/transformers_whisper.
#
# AutoModel comes from its own module rather than the package: recent FunASR
# releases (1.4.27) export it lazily through a module __getattr__, which pylint
# cannot follow.
import numpy as np
from funasr.auto.auto_model import AutoModel
from funasr.download.name_maps_from_hub import name_maps_hf
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from huggingface_hub import snapshot_download

from .const import Transcriber, sense_voice_language
from .device import torch_device

_RATE = 16000


class FunASRTranscriber(Transcriber):
    """Wrapper for a FunASR model (SenseVoice / Paraformer / Fun-ASR-Nano)."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Union[str, Path],
        local_files_only: bool = False,
        device: str = "cpu",
    ) -> None:
        """Initialize model."""
        self._postprocess = rich_transcription_postprocess
        self._is_sense_voice = "SenseVoice" in model_id

        # Resolve the model directory before handing it to FunASR. Left to
        # itself, FunASR calls snapshot_download(model) with no arguments of its
        # own -- so the model lands in the default hub cache rather than
        # cache_dir -- and then swallows any download error ("Download: ...
        # failed!") before failing later with an unrelated "model is not
        # registered". Downloading here keeps the files where they belong and
        # lets a cache miss surface as the LocalEntryNotFoundError that
        # ModelLoader needs to see. Passing a directory skips FunASR's own
        # download path entirely.
        #
        # The environment variables that would otherwise control this (HF_HOME,
        # HF_HUB_OFFLINE) are no help: huggingface_hub reads them into constants
        # at import time, and FunASR has already imported it by way of
        # transformers before we get here.
        model_dir = Path(model_id)
        if not model_dir.is_dir():
            model_dir = Path(
                snapshot_download(
                    # FunASR accepts short aliases ("paraformer-zh") as well as
                    # repo ids; snapshot_download only knows repo ids.
                    name_maps_hf.get(model_id, model_id),
                    cache_dir=str(Path(cache_dir).resolve()),
                    local_files_only=local_files_only,
                )
            )

        # FunASR prints a "funasr version: ..." banner to stdout when a model is
        # built. With the stdio:// transport that line corrupts the Wyoming
        # protocol, so redirect stdout to stderr while loading.
        with contextlib.redirect_stdout(sys.stderr):
            self.model = AutoModel(
                model=str(model_dir),
                hub="hf",
                device=torch_device(device),
                disable_update=True,
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

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        gen_kwargs = {"input": audio, "cache": {}, "use_itn": True, "batch_size_s": 300}
        if self._is_sense_voice:
            gen_kwargs["language"] = sense_voice_language(language) or "auto"

        with contextlib.redirect_stdout(sys.stderr):
            result = self.model.generate(**gen_kwargs)
        text = result[0]["text"] if result else ""
        return self._postprocess(text).strip()
