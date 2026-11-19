"""Tests for the onnx-asr backend."""

import importlib
import sys
from types import ModuleType
from unittest.mock import ANY, Mock


def test_quantization_is_passed_to_onnx_asr(monkeypatch, tmp_path) -> None:
    fake_onnx_asr = ModuleType("onnx_asr")
    fake_onnx_asr.load_model = Mock()  # type: ignore[attr-defined]
    fake_onnxruntime = ModuleType("onnxruntime")
    fake_onnxruntime.get_available_providers = Mock(  # type: ignore[attr-defined]
        return_value=["CPUExecutionProvider"]
    )
    fake_onnxruntime.SessionOptions = Mock  # type: ignore[attr-defined]
    fake_huggingface_hub = ModuleType("huggingface_hub")
    fake_huggingface_hub.snapshot_download = Mock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_onnx_asr)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)
    monkeypatch.delitem(
        sys.modules, "wyoming_faster_whisper.onnx_asr_handler", raising=False
    )

    handler = importlib.import_module("wyoming_faster_whisper.onnx_asr_handler")
    handler.OnnxAsrTranscriber(
        "nemo-parakeet-tdt-0.6b-v2",
        cache_dir=tmp_path,
        local_files_only=True,
        quantization="int8",
    )

    fake_onnx_asr.load_model.assert_called_once_with(  # type: ignore[attr-defined]
        "nemo-parakeet-tdt-0.6b-v2",
        sess_options=ANY,
        providers=["CPUExecutionProvider"],
        quantization="int8",
    )


def test_revision_reaches_snapshot_download(monkeypatch, tmp_path) -> None:
    fake_huggingface_hub = ModuleType("huggingface_hub")
    fake_huggingface_hub.snapshot_download = Mock(  # type: ignore[attr-defined]
        return_value=str(tmp_path)
    )

    def load_model(model_id, **_kwargs):
        # onnx-asr downloads through huggingface_hub.snapshot_download, looked
        # up at call time, which is what the handler patches.
        sys.modules["huggingface_hub"].snapshot_download(
            model_id, local_dir=None, local_files_only=False, allow_patterns=["*"]
        )

    fake_onnx_asr = ModuleType("onnx_asr")
    fake_onnx_asr.load_model = load_model  # type: ignore[attr-defined]
    fake_onnxruntime = ModuleType("onnxruntime")
    fake_onnxruntime.get_available_providers = Mock(  # type: ignore[attr-defined]
        return_value=["CPUExecutionProvider"]
    )
    fake_onnxruntime.SessionOptions = Mock  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx_asr", fake_onnx_asr)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_onnxruntime)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)
    monkeypatch.delitem(
        sys.modules, "wyoming_faster_whisper.onnx_asr_handler", raising=False
    )

    handler = importlib.import_module("wyoming_faster_whisper.onnx_asr_handler")
    handler.OnnxAsrTranscriber(
        "RobertasTa/lt-ausys",
        cache_dir=tmp_path,
        local_files_only=True,
        revision="caad00f",
    )

    fake_huggingface_hub.snapshot_download.assert_called_once_with(  # type: ignore[attr-defined]
        "RobertasTa/lt-ausys",
        local_dir=None,
        local_files_only=True,
        allow_patterns=["*"],
        cache_dir=str(tmp_path.resolve()),
        revision="caad00f",
    )
