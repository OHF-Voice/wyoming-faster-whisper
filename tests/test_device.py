"""Tests for --device translation.

Dependency-free: the point of wyoming_faster_whisper.device is that the rules can
be checked without torch, onnxruntime, or sherpa-onnx installed.
"""

from types import SimpleNamespace

import pytest

from wyoming_faster_whisper.device import (
    ctranslate2_device,
    device_index,
    disable_onnx_spinning,
    is_gpu,
    is_vulkan,
    onnx_providers,
    onnx_session_args,
    resolve_compute_type,
    sherpa_provider,
    torch_device,
    warn_if_no_onnx_gpu,
)

# --- is_gpu ----------------------------------------------------------------


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:3", "CUDA", " cuda "])
def test_cuda_is_gpu(device) -> None:
    assert is_gpu(device)


@pytest.mark.parametrize("device", ["cpu", "CPU", "auto", "mps", "xpu", ""])
def test_everything_else_is_not_gpu(device) -> None:
    # Only CUDA counts: it is the one family every backend here can target.
    # "auto" deliberately reads as non-GPU so the onnxruntime backends get a
    # provider list that works rather than one that might not.
    assert not is_gpu(device)


# --- device index ----------------------------------------------------------


def test_index_is_none_when_unspecified() -> None:
    assert device_index("cuda") is None
    assert device_index("cpu") is None


def test_index_is_parsed() -> None:
    assert device_index("cuda:0") == 0
    assert device_index("cuda:2") == 2


def test_unparsable_index_is_ignored() -> None:
    assert device_index("cuda:first") is None


# --- per-backend translation ----------------------------------------------


def test_torch_device_passes_through() -> None:
    # torch understands both forms natively.
    assert torch_device("cuda") == "cuda"
    assert torch_device("cuda:1") == "cuda:1"
    assert torch_device(" cpu ") == "cpu"


def test_ctranslate2_splits_the_ordinal_out() -> None:
    # CTranslate2 takes device_index as a separate argument.
    assert ctranslate2_device("cuda:1") == ("cuda", 1)


def test_ctranslate2_leaves_bare_devices_alone() -> None:
    assert ctranslate2_device("cuda") == ("cuda", None)
    assert ctranslate2_device("cpu") == ("cpu", None)
    # "auto" is CTranslate2's own value and must survive untouched.
    assert ctranslate2_device("auto") == ("auto", None)


def test_sherpa_provider() -> None:
    assert sherpa_provider("cuda") == "cuda"
    assert sherpa_provider("cuda:1") == "cuda"
    assert sherpa_provider("cpu") == "cpu"


def test_onnx_providers_on_cpu() -> None:
    assert onnx_providers("cpu") == ["CPUExecutionProvider"]


def test_onnx_providers_keep_cpu_fallback() -> None:
    # A container without a usable driver must degrade to CPU rather than fail
    # to load the model at all.
    providers = onnx_providers("cuda")
    assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_onnx_providers_carry_the_device_id() -> None:
    providers = onnx_providers("cuda:2")
    assert providers == [
        ("CUDAExecutionProvider", {"device_id": 2}),
        "CPUExecutionProvider",
    ]


# --- compute type ---------------------------------------------------------


def test_compute_type_default_becomes_float16_on_gpu() -> None:
    assert resolve_compute_type("default", "cuda") == "float16"
    assert resolve_compute_type("default", "cuda:1") == "float16"


def test_compute_type_default_is_left_alone_on_cpu() -> None:
    assert resolve_compute_type("default", "cpu") == "default"


def test_explicit_compute_type_is_never_overridden() -> None:
    # Someone asking for int8_float16 on a GPU means it.
    assert resolve_compute_type("int8_float16", "cuda") == "int8_float16"
    assert resolve_compute_type("int8", "cuda") == "int8"


# --- silent-CPU-fallback warning ------------------------------------------


def test_no_warning_when_cpu_was_asked_for(caplog) -> None:
    assert not warn_if_no_onnx_gpu("cpu", ["CPUExecutionProvider"])
    assert not caplog.records


def test_gpu_available_is_not_warned_about(caplog) -> None:
    assert warn_if_no_onnx_gpu(
        "cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    assert not caplog.records


def test_missing_cuda_provider_warns(caplog) -> None:
    # The whole point: onnxruntime loads the model on the CPU provider without
    # complaint, so the only symptom would be no speedup. This fires both when
    # the CPU package is installed and when a session falls back because the
    # CUDA provider library could not be loaded.
    assert not warn_if_no_onnx_gpu("cuda", ["CPUExecutionProvider"])
    assert len(caplog.records) == 1
    assert "onnxruntime-gpu" in caplog.records[0].getMessage()


# --- Vulkan (WebGPU) --------------------------------------------------------


@pytest.mark.parametrize("device", ["vulkan", "vulkan:1", "VULKAN", " vulkan "])
def test_vulkan_is_vulkan(device) -> None:
    assert is_vulkan(device)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "auto", "webgpu", ""])
def test_others_are_not_vulkan(device) -> None:
    assert not is_vulkan(device)


def test_vulkan_is_not_cuda() -> None:
    # is_gpu() gates CUDA-only choices (float16, GPU model defaults, sherpa's
    # CUDA provider); none of them apply to Vulkan.
    assert not is_gpu("vulkan")
    assert resolve_compute_type("default", "vulkan") == "default"
    assert sherpa_provider("vulkan") == "cpu"


def test_vulkan_index_is_parsed() -> None:
    assert device_index("vulkan:1") == 1


def test_non_onnx_backends_fall_back_to_cpu_on_vulkan() -> None:
    # CTranslate2 rejects device names it does not know, and torch has no
    # Vulkan device that runs these models.
    assert ctranslate2_device("vulkan") == ("cpu", None)
    assert ctranslate2_device("vulkan:1") == ("cpu", None)
    assert torch_device("vulkan") == "cpu"
    assert onnx_providers("vulkan") == ["CPUExecutionProvider"]


def test_vulkan_is_not_a_cuda_warning(caplog) -> None:
    # Vulkan has its own warnings (webgpu_ep_device, the qwen3-asr handler's
    # fallback, ModelLoader for other backends); this one is CUDA's alone, so
    # a CPU fallback is reported once rather than twice.
    assert not warn_if_no_onnx_gpu("vulkan", ["CPUExecutionProvider"])
    assert not caplog.records


class _FakeEpDevice:
    ep_name = "WebGpuExecutionProvider"


class _FakeOrt:
    """Stand-in for the onnxruntime module's plugin-EP API."""

    def __init__(self, devices) -> None:
        self.devices = devices
        self.registered: list = []

    def register_execution_provider_library(self, name: str, path: str) -> None:
        self.registered.append((name, path))

    def get_ep_devices(self):
        return self.devices


class _RecordingProviderOptions:
    def __init__(self) -> None:
        self.added: list = []
        self.log_severity_level = 2

    def add_provider_for_devices(self, devices, options) -> None:
        self.added.append((devices, options))


def test_session_args_without_vulkan_use_the_providers_list() -> None:
    options = _RecordingProviderOptions()
    args, webgpu_device = onnx_session_args(_FakeOrt([]), options, "cuda:1")

    assert args["providers"] == onnx_providers("cuda:1")
    assert args["sess_options"] is options
    assert webgpu_device is None
    assert not options.added


def test_session_args_attach_the_webgpu_device(monkeypatch) -> None:
    pytest.importorskip("onnxruntime_ep_webgpu")
    monkeypatch.setattr(
        "wyoming_faster_whisper.device._webgpu_registered", False, raising=True
    )
    ep_device = _FakeEpDevice()
    cpu_device = SimpleNamespace(ep_name="CPUExecutionProvider")
    fake_ort = _FakeOrt([cpu_device, ep_device])
    options = _RecordingProviderOptions()

    args, webgpu_device = onnx_session_args(fake_ort, options, "vulkan:1")

    assert args == {"sess_options": options}
    assert webgpu_device is ep_device
    assert options.added == [([ep_device], {"adapterIndex": "1"})]
    assert len(fake_ort.registered) == 1


def test_session_args_fall_back_to_cpu_without_an_adapter(monkeypatch) -> None:
    pytest.importorskip("onnxruntime_ep_webgpu")
    monkeypatch.setattr(
        "wyoming_faster_whisper.device._webgpu_registered", True, raising=True
    )
    options = _RecordingProviderOptions()

    args, webgpu_device = onnx_session_args(_FakeOrt([]), options, "vulkan")

    assert args["providers"] == ["CPUExecutionProvider"]
    assert webgpu_device is None
    assert not options.added


# --- disable_onnx_spinning --------------------------------------------------


class _RecordingSessionOptions:
    """Stand-in for onnxruntime.SessionOptions that records config entries."""

    def __init__(self) -> None:
        self.entries: dict = {}

    def add_session_config_entry(self, key: str, value: str) -> None:
        self.entries[key] = value


def test_disable_onnx_spinning_turns_off_both_pools() -> None:
    options = _RecordingSessionOptions()

    assert disable_onnx_spinning(options) is options
    assert options.entries == {
        "session.intra_op.allow_spinning": "0",
        "session.inter_op.allow_spinning": "0",
    }
