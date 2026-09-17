"""Install the pinned Orukeet sherpa-onnx export from Hugging Face."""

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

MODEL_ID = "sherpa-onnx-orukeet-v0.1.0-int8"
_REVISION = "55a984d46f68323301837194ce647c702f55facc"
_MANIFEST_SHA256 = "7e80f93f0e9b923c392424b0f85d28a717feee0a4d2a6aa9bfa723693868e727"


def _verify(path: Path, sha256: str, size: int) -> None:
    """Reject incomplete or altered files before native inference loads them."""
    if path.stat().st_size != size:
        raise ValueError(f"Orukeet size mismatch: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise ValueError(f"Orukeet SHA-256 mismatch: {path.name}")


def ensure_model(cache_dir: Path, local_files_only: bool = False) -> Path:
    """Install atomically, and use an installed model without network access.

    The real JSON release manifest verifies the model download and participates
    in Hugging Face's normal download accounting. Recognition sends no requests.
    """
    model_dir = cache_dir / MODEL_ID
    manifest_path = model_dir / "manifest.json"
    if manifest_path.is_file():
        _verify(manifest_path, _MANIFEST_SHA256, 1867)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["files"]:
            _verify(model_dir / record["path"], record["sha256"], record["bytes"])
        return model_dir

    # Keep the hub cache on the same persistent volume as the server's models.
    def download(filename: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id="oruk/orukeet",
                filename=filename,
                revision=_REVISION,
                cache_dir=str(cache_dir / ".huggingface"),
                local_files_only=local_files_only,
            )
        )

    manifest_path = download("onnx/manifest.json")
    _verify(manifest_path, _MANIFEST_SHA256, 1867)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive_path = download("onnx/" + manifest["archive"])
    _verify(archive_path, manifest["archive_sha256"], manifest["archive_bytes"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".orukeet-", dir=cache_dir) as staging:
        stage = Path(staging)
        # Extract only the regular files named by the verified manifest. Never
        # follow archive links, paths, or metadata supplied by a tar member.
        with tarfile.open(archive_path, "r:bz2") as archive:
            for record in manifest["files"]:
                member = archive.getmember(MODEL_ID + "/" + record["path"])
                if not member.isfile():
                    raise ValueError(
                        f"Orukeet archive member is not a file: {member.name}"
                    )
                target = stage / record["path"]
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Missing Orukeet archive member: {member.name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                _verify(target, record["sha256"], record["bytes"])
        shutil.copyfile(manifest_path, stage / "manifest.json")
        # No partially extracted directory is ever visible to the recognizer.
        stage.rename(model_dir)
    return model_dir
