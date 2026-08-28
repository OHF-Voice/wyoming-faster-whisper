#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import platform
import re
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import faster_whisper
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer, AsyncTcpServer

from . import __version__
from .const import (
    AUTO_LANGUAGE,
    AUTO_MODEL,
    HASS_API_URL,
    PARAKEET_LANGUAGES,
    SttLibrary,
)
from .dispatch_handler import DispatchEventHandler
from .models import ModelLoader
from .vocabulary import DEFAULT_PROMPT_MAX_TOKENS

if TYPE_CHECKING:
    # Imported for typing only: aiohttp is an optional extra.
    from .name_cache import HassNameCache

_LOGGER = logging.getLogger(__name__)


def _resolve_secret(value: Optional[str], env_var: str) -> Optional[str]:
    """Resolve a CLI value from a matching env var or its ``_FILE`` variant.

    Precedence: explicit CLI flag > ``<env_var>_FILE`` (docker secrets) >
    ``<env_var>``. The CLI flag always wins so an explicit argument can never
    be silently shadowed by a stale env var left over in the container.
    Compose/Swarm secrets are mounted as files, not env vars, so the token
    and API URL need the same ``_FILE``-suffix convention as speech-to-phrase
    rather than requiring the plaintext value in the environment.
    """
    if value is not None:
        return value

    file_path = os.environ.get(f"{env_var}_FILE")
    if file_path:
        try:
            return Path(file_path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise SystemExit(f"Could not read {env_var}_FILE ({file_path}): {exc}")

    return os.environ.get(env_var)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    #
    parser.add_argument(
        "--zeroconf",
        nargs="?",
        const="faster-whisper",
        help="Enable discovery over zeroconf with optional name (default: faster-whisper)",
    )
    #
    parser.add_argument(
        "--model", default=AUTO_MODEL, help=f"Name of model to use (or {AUTO_MODEL})"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference (default: cpu)",
    )
    parser.add_argument(
        "--language",
        default=AUTO_LANGUAGE,
        help=f"Default language to set for transcription (default: {AUTO_LANGUAGE})",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="Compute type (float16, int8, etc.)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=0,
        help="Size of beam during decoding (0 for auto)",
    )
    parser.add_argument(
        "--cpu-threads",
        default=4,
        type=int,
        help="Number of CPU threads to use for inference (default: 4, faster-whisper, sherpa-onnx, and qwen3-asr)",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt for the first window "
        "(faster-whisper); used as biasing context for entity names (qwen3-asr)",
    )
    parser.add_argument(
        "--whisper-task",
        default="transcribe",
        help="Task for whisper to do (default: transcribe)",
        choices=["transcribe", "translate"],
    )
    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Enable Silero VAD to filter out non-speech which can reduce hallucinations (default: false, faster-whisper only)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD speech probability threshold (default: 0.5, faster-whisper only)",
    )
    parser.add_argument(
        "--vad-min-speech-ms",
        type=int,
        default=250,
        help="VAD minimum speech duration in ms (default: 250, faster-whisper only)",
    )
    parser.add_argument(
        "--vad-min-silence-ms",
        type=int,
        default=2000,
        help="VAD minimum silence duration in ms to split (default: 2000, faster-whisper only)",
    )
    parser.add_argument(
        "--vad-clip",
        nargs="*",
        metavar="STT_LIBRARY",
        choices=[lib.value for lib in SttLibrary if lib != SttLibrary.AUTO],
        help="Use pysilero-vad to clip leading/trailing silence before "
        "transcription (default: disabled). Pass --vad-clip for every library, or "
        "name the libraries it applies to, e.g. --vad-clip qwen3-asr",
    )
    parser.add_argument(
        "--vad-clip-threshold",
        type=float,
        default=0.5,
        help="Speech probability threshold for --vad-clip (default: 0.5)",
    )
    parser.add_argument(
        "--vad-clip-pad-ms",
        type=int,
        default=400,
        help="Context kept around speech for --vad-clip in ms (default: 400)",
    )
    parser.add_argument(
        "--stt-library",
        choices=[lib.value for lib in SttLibrary],
        default=SttLibrary.AUTO,
        help="Set library to use for speech-to-text (may require extra dependencies)",
    )
    parser.add_argument(
        "--sherpa-streaming",
        action="store_true",
        help="Use sherpa-onnx OnlineRecognizer for true streaming transcription "
        "(sherpa only; requires a streaming model, e.g. a streaming zipformer)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Don't check HuggingFace hub for updates every time",
    )
    # Home Assistant name biasing (extra: hass)
    parser.add_argument(
        "--hass-token",
        default=None,
        help="Long-lived access token for Home Assistant. Enables biasing toward "
        "the names of exposed entities, areas, and floors (extra: hass). Falls "
        "back to the HASS_TOKEN env var, or HASS_TOKEN_FILE for a mounted secret.",
    )
    parser.add_argument(
        "--hass-api",
        default=None,
        help=f"URL of the Home Assistant API (default: {HASS_API_URL}). Falls back "
        "to the HASS_API env var, or HASS_API_FILE for a mounted secret.",
    )
    parser.add_argument(
        "--hass-refresh-seconds",
        type=float,
        default=0.0,
        help="Minimum seconds between name refreshes (default: 0, refresh on "
        "every utterance)",
    )
    parser.add_argument(
        "--hass-prompt-max-tokens",
        type=int,
        default=DEFAULT_PROMPT_MAX_TOKENS,
        help="Token budget for names in the prompt "
        f"(default: {DEFAULT_PROMPT_MAX_TOKENS}, max useful: 223 for Whisper)",
    )
    parser.add_argument(
        "--hass-prompt-timeout",
        type=float,
        default=1.0,
        help="Seconds to wait for an unfinished name refresh before transcribing "
        "with the previous names (default: 1.0)",
    )
    #
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    args.hass_token = _resolve_secret(args.hass_token, "HASS_TOKEN")
    args.hass_api = _resolve_secret(args.hass_api, "HASS_API") or HASS_API_URL

    if not args.download_dir:
        # Download to first data dir by default
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    args.stt_library = SttLibrary(args.stt_library)

    # argparse gives None when --vad-clip is absent, [] when it is passed with no
    # values (every library), and the named libraries otherwise.
    vad_clip = args.vad_clip is not None
    vad_clip_libraries = (
        {SttLibrary(value) for value in args.vad_clip} if args.vad_clip else None
    )

    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch" in machine)

    if args.beam_size <= 0:
        args.beam_size = 1 if is_arm else 5
        _LOGGER.debug("Beam size automatically selected: %s", args.beam_size)

    # Resolve model name
    model_name = args.model
    model_match = re.match(r"^(tiny|base|small|medium)[.-]int8$", args.model)
    if model_match:
        # Original models re-uploaded to huggingface
        model_size = model_match.group(1)
        model_name = f"{model_size}-int8"
        args.model = f"rhasspy/faster-whisper-{model_name}"

    if args.language == AUTO_LANGUAGE:
        # Whisper does not understand auto
        args.language = None

    if args.model == AUTO_MODEL:
        args.model = None

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="faster-whisper",
                description="Faster Whisper transcription with CTranslate2",
                attribution=Attribution(
                    name="Guillaume Klein",
                    url="https://github.com/guillaumekln/faster-whisper/",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="Systran",
                            url="https://huggingface.co/Systran",
                        ),
                        installed=True,
                        languages=sorted(
                            list(
                                # pylint: disable=protected-access
                                set(faster_whisper.tokenizer._LANGUAGE_CODES).union(
                                    PARAKEET_LANGUAGES
                                )
                            )
                        ),
                        version=faster_whisper.__version__,
                    )
                ],
            )
        ],
    )

    vad_parameters: Optional[Dict[str, Any]] = None
    if args.vad_filter:
        vad_parameters = {
            "threshold": args.vad_threshold,
            "min_speech_duration_ms": args.vad_min_speech_ms,
            "min_silence_duration_ms": args.vad_min_silence_ms,
        }

    loader = ModelLoader(
        preferred_stt_library=args.stt_library,
        preferred_language=args.language,
        download_dir=args.download_dir,
        local_files_only=args.local_files_only,
        model=args.model,
        compute_type=args.compute_type,
        device=args.device,
        beam_size=args.beam_size,
        cpu_threads=args.cpu_threads,
        initial_prompt=args.initial_prompt,
        vad_parameters=vad_parameters,
        whisper_task=args.whisper_task,
        sherpa_streaming=args.sherpa_streaming,
        vad_clip=vad_clip,
        vad_clip_threshold=args.vad_clip_threshold,
        vad_clip_pad_ms=args.vad_clip_pad_ms,
        vad_clip_libraries=vad_clip_libraries,
    )

    # Load model
    _LOGGER.debug("Pre-loading transcriber")
    await loader.load_transcriber()

    names = await _load_names(args)

    server = AsyncServer.from_uri(args.uri)

    if args.zeroconf:
        if not isinstance(server, AsyncTcpServer):
            raise ValueError("Zeroconf requires tcp:// uri")

        from wyoming.zeroconf import HomeAssistantZeroconf

        tcp_server: AsyncTcpServer = server
        hass_zeroconf = HomeAssistantZeroconf(
            name=args.zeroconf, port=tcp_server.port, host=tcp_server.host
        )
        await hass_zeroconf.register_server()
        _LOGGER.debug("Zeroconf discovery enabled")

    _LOGGER.info("Ready")
    await server.run(
        partial(
            DispatchEventHandler,
            wyoming_info,
            loader,
            names,
        )
    )


# -----------------------------------------------------------------------------


async def _load_names(args: argparse.Namespace) -> Optional["HassNameCache"]:
    """Set up biasing toward Home Assistant's names, if a token was given."""
    if not args.hass_token:
        return None

    try:
        from .hass_api import HomeAssistant
        from .name_cache import HassNameCache
    except ImportError as exc:
        raise ImportError(
            "--hass-token requires the 'hass' extra: "
            "pip install 'wyoming-faster-whisper[hass]'"
        ) from exc

    names = HassNameCache(
        HomeAssistant(args.hass_token, api_url=args.hass_api),
        prefix=args.initial_prompt,
        max_tokens=args.hass_prompt_max_tokens,
        refresh_seconds=args.hass_refresh_seconds,
        wait_timeout=args.hass_prompt_timeout,
    )

    # Fetch once now so the first utterance is biased too. Best effort: Home
    # Assistant may still be starting up, and that must not stop this server
    # from serving. Every utterance retries.
    if await names.refresh():
        _LOGGER.info("Biasing toward names from %s", args.hass_api)
    else:
        _LOGGER.warning(
            "No names loaded from %s yet; will retry on each utterance",
            args.hass_api,
        )

    return names


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
