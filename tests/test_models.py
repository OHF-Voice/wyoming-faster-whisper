"""Tests for pure model/library selection logic.

These are dependency-free: guess_stt_library takes backend-availability flags
as arguments, so the real STT backends need not be installed.
"""

import sys
from types import SimpleNamespace

import pytest

from wyoming_faster_whisper.const import SttLibrary
from wyoming_faster_whisper.models import (
    ModelLoader,
    guess_model,
    guess_stt_library,
    is_distil_whisper,
    vad_clip_enabled,
)

_ALL_AVAILABLE = dict(
    has_transformers=True,
    has_sherpa=True,
    has_onnx_asr=True,
    has_funasr=True,
    has_qwen3_asr=True,
)


def _guess(preferred, language, model=None, **avail):
    flags = {**_ALL_AVAILABLE, **avail}
    return guess_stt_library(preferred, model, language, **flags)


# --- AUTO: per-language backend selection ---------------------------------


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        # FunASR (SenseVoice) languages, including locale-style codes.
        ("zh", SttLibrary.FUNASR),
        ("zh-CN", SttLibrary.FUNASR),
        ("zh-TW", SttLibrary.FUNASR),
        ("zh-HK", SttLibrary.FUNASR),  # Hong Kong -> Cantonese
        ("yue", SttLibrary.FUNASR),
        ("ja", SttLibrary.FUNASR),
        ("ko", SttLibrary.FUNASR),
        # Other specialized backends are unaffected.
        ("en", SttLibrary.SHERPA),
        ("ru", SttLibrary.ONNX_ASR),
        # Everything else defaults to faster-whisper.
        ("de", SttLibrary.FASTER_WHISPER),
        (None, SttLibrary.FASTER_WHISPER),
    ],
)
def test_auto_selects_per_language_backend(language, expected) -> None:
    assert _guess(SttLibrary.AUTO, language) == expected


def test_auto_funasr_languages_fall_back_when_funasr_missing() -> None:
    # zh would route to FunASR, but it isn't installed -> faster-whisper.
    assert (
        _guess(SttLibrary.AUTO, "zh-CN", has_funasr=False) == SttLibrary.FASTER_WHISPER
    )


def test_auto_with_explicit_model_skips_per_language_selection() -> None:
    # A forced model disables auto backend selection (stays faster-whisper).
    assert (
        _guess(SttLibrary.AUTO, "zh-CN", model="some/model")
        == SttLibrary.FASTER_WHISPER
    )


# --- Explicit library: dependency fallback --------------------------------


def test_explicit_funasr_kept_when_available() -> None:
    assert _guess(SttLibrary.FUNASR, "zh") == SttLibrary.FUNASR


def test_explicit_funasr_falls_back_when_missing() -> None:
    assert (
        _guess(SttLibrary.FUNASR, "zh", has_funasr=False) == SttLibrary.FASTER_WHISPER
    )


def test_explicit_qwen3_asr_kept_when_available() -> None:
    assert _guess(SttLibrary.QWEN3_ASR, "en") == SttLibrary.QWEN3_ASR


def test_explicit_qwen3_asr_falls_back_when_missing() -> None:
    assert (
        _guess(SttLibrary.QWEN3_ASR, "en", has_qwen3_asr=False)
        == SttLibrary.FASTER_WHISPER
    )


def test_auto_never_selects_qwen3_asr() -> None:
    # Qwen3-ASR is opt-in only: it is large and slow relative to the
    # per-language defaults, so AUTO must not route to it.
    for language in ("en", "ru", "zh", "de", None):
        assert _guess(SttLibrary.AUTO, language) != SttLibrary.QWEN3_ASR


def test_explicit_faster_whisper_is_passthrough() -> None:
    assert (
        _guess(SttLibrary.FASTER_WHISPER, "en", has_sherpa=False)
        == SttLibrary.FASTER_WHISPER
    )


# --- --vad-clip library selection -----------------------------------------


def test_vad_clip_off_when_flag_absent() -> None:
    for library in SttLibrary:
        assert not vad_clip_enabled(library, vad_clip=False)


def test_bare_vad_clip_applies_to_every_library() -> None:
    # `--vad-clip` with no values -> vad_clip_libraries is None.
    for library in SttLibrary:
        assert vad_clip_enabled(library, vad_clip=True, vad_clip_libraries=None)


def test_named_library_is_clipped() -> None:
    assert vad_clip_enabled(
        SttLibrary.QWEN3_ASR, vad_clip=True, vad_clip_libraries={SttLibrary.QWEN3_ASR}
    )


def test_unnamed_libraries_are_not_clipped() -> None:
    # `--vad-clip qwen3-asr` must leave faster-whisper alone, where clipping
    # buys nothing (audio is padded to 30s internally regardless).
    assert not vad_clip_enabled(
        SttLibrary.FASTER_WHISPER,
        vad_clip=True,
        vad_clip_libraries={SttLibrary.QWEN3_ASR},
    )


def test_several_libraries_can_be_named() -> None:
    named = {SttLibrary.QWEN3_ASR, SttLibrary.SHERPA}
    assert vad_clip_enabled(SttLibrary.SHERPA, vad_clip=True, vad_clip_libraries=named)
    assert vad_clip_enabled(
        SttLibrary.QWEN3_ASR, vad_clip=True, vad_clip_libraries=named
    )
    assert not vad_clip_enabled(
        SttLibrary.FUNASR, vad_clip=True, vad_clip_libraries=named
    )


def test_named_libraries_are_ignored_when_flag_is_off() -> None:
    # Defensive: a stale library set must not enable clipping on its own.
    assert not vad_clip_enabled(
        SttLibrary.QWEN3_ASR, vad_clip=False, vad_clip_libraries={SttLibrary.QWEN3_ASR}
    )


# --- default models on GPU ------------------------------------------------


def test_faster_whisper_default_is_int8_on_cpu() -> None:
    assert guess_model(SttLibrary.FASTER_WHISPER, "en", is_arm=False).endswith("-int8")
    assert guess_model(SttLibrary.FASTER_WHISPER, "en", is_arm=True).endswith("-int8")


def test_faster_whisper_default_is_float16_on_gpu() -> None:
    # int8 saves memory a GPU has to spare and gives up the float16 throughput
    # it was built for.
    model = guess_model(SttLibrary.FASTER_WHISPER, "en", is_arm=False, gpu=True)
    assert not model.endswith("-int8")
    assert model == "Systran/faster-whisper-small"


def test_faster_whisper_gpu_default_ignores_arm() -> None:
    # A GPU-capable arm64 host (Jetson) has no reason to fall back to tiny.
    assert guess_model(
        SttLibrary.FASTER_WHISPER, "en", is_arm=True, gpu=True
    ) == guess_model(SttLibrary.FASTER_WHISPER, "en", is_arm=False, gpu=True)


@pytest.mark.parametrize(
    "library",
    [
        SttLibrary.SHERPA,
        SttLibrary.TRANSFORMERS,
        SttLibrary.ONNX_ASR,
        SttLibrary.FUNASR,
        SttLibrary.QWEN3_ASR,
    ],
)
def test_other_backends_keep_the_same_default_model_on_gpu(library) -> None:
    # Their default models are published in one quantization only, so there is
    # nothing to switch to; the device only changes the execution provider.
    for language in ("en", "de", "ru", "zh", None):
        assert guess_model(library, language, is_arm=False, gpu=True) == guess_model(
            library, language, is_arm=False
        )


# --- Distil-Whisper detection ---------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "Systran/faster-distil-whisper-small.en",
        "Systran/faster-distil-whisper-large-v3",
        "distil-small.en",
        "distil-whisper/distil-large-v3",
        "DISTIL-SMALL.EN",  # case-insensitive
    ],
)
def test_distil_models_are_detected(model) -> None:
    assert is_distil_whisper(model)


@pytest.mark.parametrize(
    "model",
    [
        None,  # --model auto never resolves to a distil checkpoint
        "tiny-int8",
        "small.en",
        "Systran/faster-whisper-large-v3",
        "openai/whisper-tiny.en",
    ],
)
def test_standard_models_are_not_detected(model) -> None:
    assert not is_distil_whisper(model)


def test_every_default_model_takes_a_prompt() -> None:
    # A guessed model must never be one that biasing cannot work with.
    for library in SttLibrary:
        if library == SttLibrary.AUTO:
            continue

        for language in ("en", "de", "ru", "zh", None):
            for is_arm in (False, True):
                for gpu in (False, True):
                    assert not is_distil_whisper(
                        guess_model(library, language, is_arm=is_arm, gpu=gpu)
                    )


# --- cache-first loading --------------------------------------------------


def _loader(**kwargs) -> ModelLoader:
    """Build a ModelLoader with the arguments the cache-first path cares about."""
    return ModelLoader(
        preferred_stt_library=SttLibrary.FASTER_WHISPER,
        preferred_language="en",
        download_dir="/data",
        local_files_only=kwargs.pop("local_files_only", False),
        model="Systran/faster-whisper-base",
        compute_type="default",
        onnx_quantization=kwargs.pop("onnx_quantization", None),
        device="cpu",
        beam_size=5,
        cpu_threads=4,
        initial_prompt=None,
        vad_parameters=None,
        **kwargs,
    )


def test_onnx_quantization_is_forwarded(monkeypatch) -> None:
    calls = []

    def transcriber(*args, **kwargs):
        calls.append((args, kwargs))
        return "transcriber"

    module = SimpleNamespace(OnnxAsrTranscriber=transcriber)
    monkeypatch.setitem(sys.modules, "wyoming_faster_whisper.onnx_asr_handler", module)

    loader = _loader(onnx_quantization="int8")
    result = loader._build_transcriber(
        SttLibrary.ONNX_ASR,
        "nemo-parakeet-tdt-0.6b-v2",
        streaming=False,
        local_files_only=True,
    )

    assert result == "transcriber"
    assert calls[0][1]["quantization"] == "int8"


def _record_attempts(loader: ModelLoader, fail_cached: bool) -> list:
    """Replace backend construction with a recorder of local_files_only values."""
    attempts: list = []

    def build(stt_library, model, streaming, local_files_only):
        attempts.append(local_files_only)
        if local_files_only and fail_cached:
            # What huggingface_hub raises for a cache miss.
            raise FileNotFoundError("not cached")

        return f"transcriber(local_files_only={local_files_only})"

    loader._build_transcriber = build  # type: ignore[method-assign]
    return attempts


@pytest.mark.asyncio
async def test_cached_model_is_loaded_without_the_hub() -> None:
    # The common case: the model is on disk, so nothing should reach the network
    # even though --local-files-only was not passed.
    loader = _loader()
    attempts = _record_attempts(loader, fail_cached=False)

    assert await loader.load_transcriber() == "transcriber(local_files_only=True)"
    assert attempts == [True]


@pytest.mark.asyncio
async def test_missing_model_falls_back_to_downloading() -> None:
    loader = _loader()
    attempts = _record_attempts(loader, fail_cached=True)

    assert await loader.load_transcriber() == "transcriber(local_files_only=False)"
    assert attempts == [True, False]


@pytest.mark.asyncio
async def test_local_files_only_does_not_fall_back() -> None:
    # Explicitly asking for offline means a missing model is an error, not a
    # download.
    loader = _loader(local_files_only=True)
    attempts = _record_attempts(loader, fail_cached=True)

    with pytest.raises(FileNotFoundError):
        await loader.load_transcriber()

    assert attempts == [True]


@pytest.mark.asyncio
async def test_transcriber_is_only_built_once() -> None:
    loader = _loader()
    attempts = _record_attempts(loader, fail_cached=True)

    first = await loader.load_transcriber()
    assert await loader.load_transcriber() is first
    assert attempts == [True, False]
