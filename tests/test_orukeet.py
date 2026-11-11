"""Integrity, offline use, and atomic installation of Orukeet exports."""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from wyoming_faster_whisper import orukeet


def release(tmp_path, monkeypatch, *, link=False):
    """Build a tiny release exercising the real installer without model weights."""
    payload = b"model bytes"
    archive = tmp_path / "model.tar.bz2"
    with tarfile.open(archive, "w:bz2") as bundle:
        member = tarfile.TarInfo(orukeet.MODEL_ID + "/encoder.int8.onnx")
        member.size = len(payload)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "../outside"
            bundle.addfile(member)
        else:
            bundle.addfile(member, io.BytesIO(payload))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "archive": archive.name,
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "files": [
                    {
                        "path": "encoder.int8.onnx",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ).ljust(1867),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        orukeet, "_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    download = Mock(
        side_effect=lambda filename, **_: str(
            manifest if filename.endswith(".json") else archive
        )
    )
    monkeypatch.setattr(orukeet, "hf_hub_download", download)
    return manifest, archive, download


def test_installed_model_loads_offline_without_hub(tmp_path, monkeypatch):
    _, _, download = release(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    installed = orukeet.ensure_model(cache)
    assert (installed / "encoder.int8.onnx").read_bytes() == b"model bytes"
    assert download.call_count == 2
    assert download.call_args_list[0].kwargs["filename"] == "onnx/manifest.json"
    assert all(
        call.kwargs["revision"] == orukeet._REVISION for call in download.call_args_list
    )
    download.side_effect = AssertionError("unexpected network")
    assert orukeet.ensure_model(cache, local_files_only=True) == installed
    assert not list(cache.glob(".orukeet-*"))


def test_hub_cache_can_install_offline(tmp_path, monkeypatch):
    _, _, download = release(tmp_path, monkeypatch)
    orukeet.ensure_model(tmp_path / "cache", local_files_only=True)
    assert all(call.kwargs["local_files_only"] for call in download.call_args_list)


@pytest.mark.parametrize("which", ["manifest", "archive"])
def test_corrupt_download_does_not_create_model(tmp_path, monkeypatch, which):
    manifest, archive, _ = release(tmp_path, monkeypatch)
    path = manifest if which == "manifest" else archive
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        orukeet.ensure_model(cache)
    assert not (cache / orukeet.MODEL_ID).exists()


def test_corrupt_installed_model_is_rejected_without_network(tmp_path, monkeypatch):
    _, _, download = release(tmp_path, monkeypatch)
    cache = tmp_path / "cache"
    installed = orukeet.ensure_model(cache)
    (installed / "encoder.int8.onnx").write_bytes(b"wrong bytes")
    download.side_effect = AssertionError("unexpected network")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        orukeet.ensure_model(cache)


def test_archive_links_are_rejected_and_staging_removed(tmp_path, monkeypatch):
    release(tmp_path, monkeypatch, link=True)
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="not a file"):
        orukeet.ensure_model(cache)
    assert not (cache / orukeet.MODEL_ID).exists()
    assert not list(cache.glob(".orukeet-*"))


def test_interrupted_download_has_no_installed_directory(tmp_path, monkeypatch):
    _, _, download = release(tmp_path, monkeypatch)
    download.side_effect = OSError("connection interrupted")
    cache = tmp_path / "cache"
    with pytest.raises(OSError, match="interrupted"):
        orukeet.ensure_model(cache)
    assert not (cache / orukeet.MODEL_ID).exists()


def test_streaming_rejected_before_download(monkeypatch):
    pytest.importorskip("sherpa_onnx")
    from wyoming_faster_whisper import sherpa_handler

    download = Mock(side_effect=AssertionError("unexpected download"))
    monkeypatch.setattr(sherpa_handler, "_ensure_model", download)
    with pytest.raises(ValueError, match="offline model"):
        sherpa_handler.SherpaStreamingTranscriber(orukeet.MODEL_ID, Path("unused"))
    download.assert_not_called()
