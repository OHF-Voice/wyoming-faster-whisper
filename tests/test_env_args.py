"""Tests for environment variable defaults."""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pytest

from wyoming_faster_whisper.__main__ import get_parser
from wyoming_faster_whisper.const import SttLibrary
from wyoming_faster_whisper.env_args import parse_args

# Enough to satisfy the required options when the test is about something else.
BASE_ARGS = ["--uri", "tcp://0.0.0.0:10300", "--data-dir", "/data"]


def run(argv: Optional[Sequence[str]] = None, **environ: str) -> argparse.Namespace:
    """Parse arguments with only the given environment."""
    return parse_args(
        get_parser(), BASE_ARGS if argv is None else argv, environ=environ
    )


def test_no_env_vars() -> None:
    """Defaults are untouched when nothing is set."""
    args = run()
    assert args.uri == "tcp://0.0.0.0:10300"
    assert args.data_dir == ["/data"]
    assert args.device == "cpu"
    assert args.onnx_quantization is None
    assert args.beam_size == 0
    assert not args.debug


def test_required_arg_from_env() -> None:
    """A required option no longer has to be on the command line."""
    args = run([], WYO_WHISPER_URI="tcp://0.0.0.0:10301", WYO_WHISPER_DATA_DIR="/data")
    assert args.uri == "tcp://0.0.0.0:10301"
    assert args.data_dir == ["/data"]


def test_required_arg_still_required(capsys: pytest.CaptureFixture) -> None:
    """A required option with no variable set still fails."""
    with pytest.raises(SystemExit):
        run([])

    assert "--uri" in capsys.readouterr().err


def test_command_line_wins() -> None:
    """An explicit argument is never shadowed by the environment."""
    args = run(BASE_ARGS + ["--device", "cuda"], WYO_WHISPER_DEVICE="cpu")
    assert args.device == "cuda"


def test_onnx_quantization_from_env() -> None:
    args = run(WYO_WHISPER_ONNX_QUANTIZATION="int8")
    assert args.onnx_quantization == "int8"


def test_type_conversion() -> None:
    """Values go through the option's own type."""
    args = run(
        WYO_WHISPER_BEAM_SIZE="3",
        WYO_WHISPER_VAD_THRESHOLD="0.25",
        WYO_WHISPER_VAD_ENDPOINTING="0.7",
    )
    assert args.beam_size == 3
    assert args.vad_threshold == 0.25
    assert args.vad_endpointing == 0.7


def test_invalid_type(capsys: pytest.CaptureFixture) -> None:
    """A value the option's type rejects is an error, not a crash."""
    with pytest.raises(SystemExit):
        run(WYO_WHISPER_BEAM_SIZE="large")

    assert "WYO_WHISPER_BEAM_SIZE" in capsys.readouterr().err


def test_vad_endpointing_must_be_positive(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        run(WYO_WHISPER_VAD_ENDPOINTING="0")

    assert "WYO_WHISPER_VAD_ENDPOINTING" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_flag_true(value: str) -> None:
    """A flag can be turned on from the environment."""
    assert run(WYO_WHISPER_DEBUG=value).debug


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_flag_false(value: str) -> None:
    """A flag set to a false value stays off, unlike a plain truth test."""
    assert not run(WYO_WHISPER_DEBUG=value).debug


def test_flag_invalid(capsys: pytest.CaptureFixture) -> None:
    """A flag set to something that is neither is an error."""
    with pytest.raises(SystemExit):
        run(WYO_WHISPER_VAD_FILTER="maybe")

    assert "WYO_WHISPER_VAD_FILTER" in capsys.readouterr().err


def test_append_arg() -> None:
    """Repeatable options split on the path separator."""
    data_dirs = os.pathsep.join(["/data", "/media/models"])
    args = run(
        [], WYO_WHISPER_URI="tcp://0.0.0.0:10300", WYO_WHISPER_DATA_DIR=data_dirs
    )
    assert args.data_dir == ["/data", "/media/models"]


def test_append_arg_command_line_wins() -> None:
    """Command-line values replace the environment's instead of adding to it."""
    args = run(
        ["--uri", "tcp://0.0.0.0:10300", "--data-dir", "/custom"],
        WYO_WHISPER_DATA_DIR="/data",
    )
    assert args.data_dir == ["/custom"]


def test_empty_required_var_is_unset(capsys: pytest.CaptureFixture) -> None:
    """An empty variable does not satisfy a required option.

    A blank value in a compose file or a .env has to mean the same as not
    setting it, so argparse reports the missing option instead of the server
    starting with an empty string and failing further in.
    """
    with pytest.raises(SystemExit):
        run([], WYO_WHISPER_URI="", WYO_WHISPER_DATA_DIR="/data")

    assert "--uri" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        run([], WYO_WHISPER_URI="tcp://0.0.0.0:10300", WYO_WHISPER_DATA_DIR="")

    assert "--data-dir" in capsys.readouterr().err


def test_empty_optional_var_keeps_default() -> None:
    """An empty variable leaves an optional option on its default."""
    args = run(WYO_WHISPER_DEVICE="", WYO_WHISPER_MODEL="", WYO_WHISPER_HASS_TOKEN="")
    assert args.device == "cpu"
    assert args.model == "auto"
    assert args.hass_token is None


def test_whitespace_only_var_is_unset() -> None:
    """A variable holding only whitespace is empty too."""
    assert run(WYO_WHISPER_DEVICE="   ").device == "cpu"


def test_empty_secret_file_is_unset(tmp_path: Path) -> None:
    """An empty secret file does not enable the feature it would configure."""
    token_path = tmp_path / "hass_token"
    token_path.write_text("\n", encoding="utf-8")

    assert run(WYO_WHISPER_HASS_TOKEN_FILE=str(token_path)).hass_token is None


def test_list_arg() -> None:
    """An option taking several values splits on commas or spaces."""
    args = run(WYO_WHISPER_VAD_CLIP="qwen3-asr, sherpa")
    assert args.vad_clip == ["qwen3-asr", "sherpa"]


def test_list_arg_empty() -> None:
    """An empty value means the flag with no values (every library)."""
    args = run(WYO_WHISPER_VAD_CLIP="")
    assert args.vad_clip == []


def test_optional_value_arg() -> None:
    """An option with an optional value takes its constant when empty."""
    assert run(WYO_WHISPER_ZEROCONF="").zeroconf == "faster-whisper"
    assert run(WYO_WHISPER_ZEROCONF="kitchen").zeroconf == "kitchen"


def test_choices_checked(capsys: pytest.CaptureFixture) -> None:
    """A value outside the option's choices is an error."""
    with pytest.raises(SystemExit):
        run(WYO_WHISPER_STT_LIBRARY="not-a-library")

    assert "WYO_WHISPER_STT_LIBRARY" in capsys.readouterr().err

    args = run(WYO_WHISPER_STT_LIBRARY="sherpa")
    assert SttLibrary(args.stt_library) == SttLibrary.SHERPA


def test_secret_file(tmp_path: Path) -> None:
    """A _FILE variable names a file to read the value out of."""
    token_path = tmp_path / "hass_token"
    token_path.write_text("s3cret\n", encoding="utf-8")

    args = run(WYO_WHISPER_HASS_TOKEN_FILE=str(token_path))
    assert args.hass_token == "s3cret"


def test_secret_file_wins_over_plain_var(tmp_path: Path) -> None:
    """The file is preferred: a stale plain variable cannot shadow the secret."""
    token_path = tmp_path / "hass_token"
    token_path.write_text("from-file", encoding="utf-8")

    args = run(
        WYO_WHISPER_HASS_TOKEN_FILE=str(token_path), WYO_WHISPER_HASS_TOKEN="from-env"
    )
    assert args.hass_token == "from-file"


def test_secret_file_missing(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """An unreadable secret file fails loudly instead of starting unconfigured."""
    with pytest.raises(SystemExit):
        run(WYO_WHISPER_HASS_TOKEN_FILE=str(tmp_path / "missing"))

    assert "WYO_WHISPER_HASS_TOKEN_FILE" in capsys.readouterr().err


def test_unknown_var_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A misspelled variable is reported instead of silently ignored."""
    run(WYO_WHISPER_MODEL_NAME="tiny-int8")
    assert "WYO_WHISPER_MODEL_NAME" in caplog.text


def test_help_and_version_ignored() -> None:
    """--help and --version take no value, so they get no variable."""
    args = run(WYO_WHISPER_HELP="1", WYO_WHISPER_VERSION="1")
    assert args.uri == "tcp://0.0.0.0:10300"
