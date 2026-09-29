"""Tests for pure model/library selection logic.

These are dependency-free: guess_stt_library takes backend-availability flags
as arguments, so the real STT backends need not be installed.
"""

import logging

import pytest

from wyoming_faster_whisper.__main__ import _warn_if_language_unsupported
from wyoming_faster_whisper.const import (
    GIGAAM_LANGUAGES,
    KROKO_STREAMING_LANGUAGES,
    PARAKEET_LANGUAGES,
    QWEN3_ASR_LANGUAGES,
    SENSE_VOICE_LANGUAGES,
    SttLibrary,
)
from wyoming_faster_whisper.models import (
    WHISPER_LANGUAGES,
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
        device="cpu",
        beam_size=5,
        cpu_threads=4,
        initial_prompt=None,
        vad_parameters=None,
        **kwargs,
    )


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


# --- reported languages ----------------------------------------------------


def _languages(
    preferred_stt_library=SttLibrary.AUTO,
    model=None,
    *,
    sherpa_streaming=False,
    **avail,
) -> set:
    """Report what a configuration would advertise, with backends faked."""
    loader = ModelLoader(
        preferred_stt_library=preferred_stt_library,
        preferred_language=None,
        download_dir="/data",
        local_files_only=False,
        model=model,
        compute_type="default",
        device="cpu",
        beam_size=5,
        cpu_threads=4,
        initial_prompt=None,
        vad_parameters=None,
        sherpa_streaming=sherpa_streaming,
    )
    # _available_backends is memoized, so seeding the cache stands in for
    # whichever extras happen to be installed in the test environment.
    loader._available = {**_ALL_AVAILABLE, **avail}  # noqa: SLF001
    return loader.supported_languages()


def test_auto_reports_every_whisper_language() -> None:
    # faster-whisper is a hard dependency and the fallback for anything the
    # specialized backends don't claim, so auto really does cover all of them.
    assert WHISPER_LANGUAGES <= _languages()


def test_auto_adds_cantonese_when_funasr_is_installed() -> None:
    # zh-HK routes to SenseVoice, which decodes it as Cantonese. Whisper has no
    # such code, so it is only reportable because FunASR is there.
    assert "zh-HK" in _languages(has_funasr=True)
    assert "zh-HK" not in _languages(has_funasr=False)


def test_auto_does_not_report_qwen3_only_languages() -> None:
    # guess_stt_library never auto-selects qwen3-asr, so "fil" is unreachable
    # even with the extra installed.
    assert "fil" not in _languages(has_qwen3_asr=True)


@pytest.mark.parametrize(
    ("library", "expected"),
    [
        (SttLibrary.ONNX_ASR, GIGAAM_LANGUAGES),
        (SttLibrary.FUNASR, SENSE_VOICE_LANGUAGES),
        (SttLibrary.SHERPA, PARAKEET_LANGUAGES),
        (SttLibrary.QWEN3_ASR, QWEN3_ASR_LANGUAGES),
    ],
)
def test_explicit_backend_reports_only_its_own_languages(library, expected) -> None:
    assert _languages(library) == expected


def test_explicit_qwen3_reports_filipino_and_cantonese() -> None:
    languages = _languages(SttLibrary.QWEN3_ASR)
    # Neither has a Whisper token, so the old fixed list could never report them.
    assert {"fil", "zh-HK"} <= languages


def test_streaming_sherpa_reports_only_the_published_kroko_languages() -> None:
    assert _languages(SttLibrary.SHERPA, sherpa_streaming=True) == (
        KROKO_STREAMING_LANGUAGES
    )
    # Streaming only narrows sherpa; it is ignored for every other backend.
    assert _languages(SttLibrary.FUNASR, sherpa_streaming=True) == (
        SENSE_VOICE_LANGUAGES
    )


def test_missing_backend_falls_back_to_the_whisper_list() -> None:
    # guess_stt_library falls back to faster-whisper when the extra is absent,
    # and the reported languages have to follow it.
    assert _languages(SttLibrary.ONNX_ASR, has_onnx_asr=False) == WHISPER_LANGUAGES


def test_english_only_whisper_model_reports_only_english() -> None:
    # A ".en" checkpoint has no language tokens at all: it transcribes English
    # whatever is requested.
    assert _languages(
        SttLibrary.FASTER_WHISPER, model="Systran/faster-whisper-base.en"
    ) == {"en"}
    assert _languages(SttLibrary.FASTER_WHISPER, model="distil-small.en") == {"en"}
    assert (
        _languages(SttLibrary.FASTER_WHISPER, model="Systran/faster-whisper-base")
        == WHISPER_LANGUAGES
    )


def test_auto_with_explicit_model_reports_the_whisper_list() -> None:
    # An explicit --model turns off per-language backend selection, so
    # faster-whisper handles everything.
    assert _languages(model="Systran/faster-whisper-base") == WHISPER_LANGUAGES


# --- backend selection is by base language ---------------------------------


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        # Home Assistant echoes back the bare code we advertised, but --language
        # is typed by hand and other Wyoming clients send what they like.
        ("en-US", SttLibrary.SHERPA),
        ("EN", SttLibrary.SHERPA),
        ("en_GB", SttLibrary.SHERPA),
        ("ru-RU", SttLibrary.ONNX_ASR),
        # Cantonese must not be flattened to Mandarin on the way through.
        ("zh-HK", SttLibrary.FUNASR),
        ("de-CH", SttLibrary.FASTER_WHISPER),
    ],
)
def test_locale_codes_route_like_their_base_language(language, expected) -> None:
    assert _guess(SttLibrary.AUTO, language) == expected


@pytest.mark.parametrize(
    ("library", "language", "expected"),
    [
        (SttLibrary.SHERPA, "en-US", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"),
        (SttLibrary.SHERPA, "de-CH", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"),
        (SttLibrary.TRANSFORMERS, "en-US", "openai/whisper-base.en"),
        (SttLibrary.TRANSFORMERS, "de-CH", "openai/whisper-base"),
    ],
)
def test_default_model_follows_the_base_language(library, language, expected) -> None:
    assert guess_model(library, language, is_arm=False) == expected


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("en-US", "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06"),
        ("fr_FR", "sherpa-onnx-streaming-zipformer-fr-kroko-2025-08-06"),
        # The region must not leak into the model id.
        ("de-CH", "sherpa-onnx-streaming-zipformer-de-kroko-2025-08-06"),
    ],
)
def test_streaming_model_follows_the_base_language(language, expected) -> None:
    assert (
        guess_model(SttLibrary.SHERPA, language, is_arm=False, streaming=True)
        == expected
    )


# --- --language validation -------------------------------------------------


@pytest.mark.parametrize(
    "language",
    [
        None,  # not set
        "ru",  # exactly what the backend reports
        "ru-RU",  # region qualifier: the backend normalizes it itself
        "RU",
    ],
)
def test_supported_language_does_not_warn(language, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        _warn_if_language_unsupported(language, ["ru"])

    assert not caplog.records


@pytest.mark.parametrize("language", ["de", "xyz", "en-US"])
def test_unsupported_language_warns(language, caplog) -> None:
    # GigaAM is Russian-only: without this, a typo or a wrong backend only shows
    # up as every transcription being auto-detected.
    with caplog.at_level(logging.WARNING):
        _warn_if_language_unsupported(language, ["ru"])

    assert len(caplog.records) == 1
    assert language in caplog.records[0].getMessage()
