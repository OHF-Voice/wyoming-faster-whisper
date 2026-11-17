"""Tests for faster-whisper."""

import asyncio
import importlib.util
import re
import sys
import wave
from asyncio.subprocess import PIPE

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart, AudioStop, wav_to_chunks
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Describe, Info

from . import _DIR, _LOCAL_DIR, _SAMPLES_PER_CHUNK, _START_TIMEOUT, _TRANSCRIBE_TIMEOUT


def _download_sherpa_model(streaming: bool) -> None:
    from wyoming_faster_whisper.models import SttLibrary, guess_model
    from wyoming_faster_whisper.sherpa_handler import _ensure_model

    model_id = guess_model(SttLibrary.SHERPA, "en", is_arm=False, streaming=streaming)
    _ensure_model(model_id, _LOCAL_DIR)


def _needs(module: str):
    """Skip a param when the backend's package isn't installed (e.g. funasr in CI)."""
    return pytest.mark.skipif(
        importlib.util.find_spec(module) is None,
        reason=f"{module} not installed",
    )


@pytest.mark.parametrize(
    ("stt_library", "model", "extra_args"),
    [
        ("faster-whisper", "base-int8", []),
        ("faster-whisper", "base-int8", ["--vad-clip"]),
        pytest.param(
            "transformers", "openai/whisper-base.en", [], marks=_needs("transformers")
        ),
        pytest.param("sherpa", "auto", [], marks=_needs("sherpa_onnx")),
        # --vad-clip is backend-agnostic (clips the WAV before dispatch), so it
        # must work for non-faster-whisper batch backends too.
        pytest.param("sherpa", "auto", ["--vad-clip"], marks=_needs("sherpa_onnx")),
        pytest.param(
            "sherpa", "auto", ["--sherpa-streaming"], marks=_needs("sherpa_onnx")
        ),
        pytest.param(
            "funasr", "FunAudioLLM/SenseVoiceSmall", [], marks=_needs("funasr")
        ),
        pytest.param(
            "funasr",
            "FunAudioLLM/SenseVoiceSmall",
            ["--vad-clip"],
            marks=_needs("funasr"),
        ),
    ],
)
@pytest.mark.asyncio
async def test_faster_whisper(stt_library: str, model: str, extra_args: list) -> None:
    if stt_library == "sherpa":
        # The server downloads its model before it answers Describe, and the
        # sherpa models are hundreds of megabytes: fetch it first, so that
        # _START_TIMEOUT covers starting up rather than the download.
        await asyncio.to_thread(
            _download_sherpa_model, "--sherpa-streaming" in extra_args
        )

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_faster_whisper",
        "--uri",
        "stdio://",
        "--stt-library",
        stt_library,
        "--model",
        model,
        "--data-dir",
        str(_LOCAL_DIR),
        "--language",
        "en",
        "--vad-filter",
        *extra_args,
        stdin=PIPE,
        stdout=PIPE,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    # Check info
    await async_write_event(Describe().event(), proc.stdin)
    while True:
        event = await asyncio.wait_for(
            async_read_event(proc.stdout), timeout=_START_TIMEOUT
        )
        assert event is not None

        if not Info.is_type(event.type):
            continue

        info = Info.from_event(event)
        assert len(info.asr) == 1, "Expected one asr service"
        break

    await async_write_event(Transcribe().event(), proc.stdin)

    # Test known WAV
    with wave.open(str(_DIR / "turn_on_the_living_room_lamp.wav"), "rb") as example_wav:
        await async_write_event(
            AudioStart(
                rate=example_wav.getframerate(),
                width=example_wav.getsampwidth(),
                channels=example_wav.getnchannels(),
            ).event(),
            proc.stdin,
        )
        for chunk in wav_to_chunks(example_wav, _SAMPLES_PER_CHUNK):
            await async_write_event(chunk.event(), proc.stdin)

        await async_write_event(AudioStop().event(), proc.stdin)

    while True:
        event = await asyncio.wait_for(
            async_read_event(proc.stdout), timeout=_TRANSCRIBE_TIMEOUT
        )
        assert event is not None

        if not Transcript.is_type(event.type):
            continue

        transcript = Transcript.from_event(event)
        text = transcript.text.lower().strip()
        text = re.sub(r"[^a-z ]", "", text)
        assert text == "turn on the living room lamp"
        break

    # Need to close stdin for graceful termination
    proc.stdin.close()
    _, stderr = await proc.communicate()

    assert proc.returncode == 0, stderr.decode()
