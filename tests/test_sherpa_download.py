"""Tests for downloading sherpa-onnx models into the cache."""

import io
import random
import tarfile

import pytest

# Skip the whole module unless sherpa-onnx is installed.
pytest.importorskip("sherpa_onnx")

from wyoming_faster_whisper import sherpa_handler  # noqa: E402

_MODEL_ID = "sherpa-onnx-test-model"


def _archive() -> bytes:
    # Incompressible and several 100k bz2 blocks long, so that a download cut
    # off half-way has already written part of the model.
    noise = random.Random(0).randbytes(400_000)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2", compresslevel=1) as tar:
        for name in ("encoder.int8.onnx", "tokens.txt"):
            data = noise if name.endswith(".onnx") else name.encode() * 1000
            info = tarfile.TarInfo(f"{_MODEL_ID}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    return buffer.getvalue()


class _Interrupted(io.BytesIO):
    """A response that dies part-way, as a killed download does."""

    def __init__(self, data: bytes, limit: int, error: BaseException) -> None:
        super().__init__(data)
        self.limit = limit
        self.error = error

    def read(self, size=-1):  # type: ignore[no-untyped-def]
        if self.tell() >= self.limit:
            raise self.error

        if (size is None) or (size < 0):
            size = self.limit - self.tell()

        return super().read(min(size, self.limit - self.tell()))


def _serve(monkeypatch, response) -> None:
    monkeypatch.setattr(sherpa_handler.urllib.request, "urlopen", lambda url: response)


def test_download_extracts_the_model(tmp_path, monkeypatch) -> None:
    _serve(monkeypatch, io.BytesIO(_archive()))
    model_dir = sherpa_handler._ensure_model(_MODEL_ID, tmp_path)

    assert model_dir == tmp_path / _MODEL_ID
    assert (model_dir / "tokens.txt").read_bytes() == b"tokens.txt" * 1000
    # Nothing but the model is left in the cache.
    assert [path.name for path in tmp_path.iterdir()] == [_MODEL_ID]


@pytest.mark.parametrize("error", [OSError("connection reset"), KeyboardInterrupt()])
def test_interrupted_download_leaves_no_model(tmp_path, monkeypatch, error) -> None:
    # KeyboardInterrupt stands in for anything that isn't an Exception: the
    # partial model must not survive to be mistaken for a complete one.
    archive = _archive()
    _serve(monkeypatch, _Interrupted(archive, len(archive) // 2, error))
    with pytest.raises(type(error)):
        sherpa_handler._ensure_model(_MODEL_ID, tmp_path)

    assert not list(tmp_path.iterdir())

    # The next attempt downloads it again.
    _serve(monkeypatch, io.BytesIO(archive))
    assert (sherpa_handler._ensure_model(_MODEL_ID, tmp_path) / "tokens.txt").exists()


def test_a_download_finished_elsewhere_is_kept(tmp_path, monkeypatch) -> None:
    other = tmp_path / _MODEL_ID

    class _Racing(io.BytesIO):
        def close(self) -> None:
            # Another process moves its copy into place first.
            other.mkdir(exist_ok=True)
            (other / "tokens.txt").write_text("theirs")
            super().close()

    _serve(monkeypatch, _Racing(_archive()))
    model_dir = sherpa_handler._ensure_model(_MODEL_ID, tmp_path)
    assert (model_dir / "tokens.txt").read_text() == "theirs"
    assert [path.name for path in tmp_path.iterdir()] == [_MODEL_ID]
