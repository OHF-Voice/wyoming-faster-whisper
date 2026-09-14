"""Code for Whisper transcription using HuggingFace's transformers library."""

import wave
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .const import Transcriber
from .device import is_gpu, torch_device
from .languages import whisper_language

_RATE = 16000


class TransformersTranscriber(Transcriber):
    """Wrapper for HuggingFace transformers Whisper model."""

    def __init__(
        self,
        model_id: str,
        cache_dir: Optional[Union[str, Path]] = None,
        local_files_only: bool = False,
        device: str = "cpu",
    ) -> None:
        """Initialize Whisper model."""
        self.processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=local_files_only
        )

        # float16 on the GPU: Whisper is trained in fp16 and every GPU worth
        # using has tensor cores for it. Keep float32 on the CPU, where fp16
        # arithmetic is emulated and slower.
        self.torch_dtype = torch.float16 if is_gpu(device) else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            torch_dtype=self.torch_dtype,
        )
        self.model.to(torch_device(device))
        self.model.eval()

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

        audio_tensor = (
            torch.frombuffer(audio_bytes, dtype=torch.int16).float() / 32768.0
        )

        inputs = self.processor(audio_tensor, sampling_rate=_RATE, return_tensors="pt")

        # Move inputs to the model's device, casting the float features (the log
        # mel spectrogram) to its dtype while leaving integer tensors such as
        # attention masks alone.
        inputs = {
            key: (
                value.to(self.model.device, dtype=self.torch_dtype)
                if value.is_floating_point()
                else value.to(self.model.device)
            )
            for key, value in inputs.items()
        }
        generate_args = {**inputs, "num_beams": beam_size}

        if initial_prompt:
            prompt_ids = (
                self.processor.tokenizer(
                    initial_prompt, return_tensors="pt", add_special_tokens=False
                )
                .input_ids[0]
                .to(self.model.device)
            )
            generate_args["prompt_ids"] = prompt_ids

        # set_prefix_tokens raises for a code Whisper does not have a token for,
        # and defers it to generate() below, failing the transcription.
        whisper_code = whisper_language(language)
        if whisper_code:
            self.processor.tokenizer.set_prefix_tokens(
                language=whisper_code, task="transcribe"
            )

        with torch.no_grad():
            # Ignore warning about attention_mask because we're only doing a single utterance.
            generated_ids = self.model.generate(**generate_args)
            transcription = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

        return transcription
