"""Translation of the --device option into each backend's own vocabulary.

There is one user-facing device string, but the backends disagree on how to name
hardware: CTranslate2 and sherpa-onnx take "cpu"/"cuda", torch takes
"cpu"/"cuda"/"cuda:1", and onnxruntime takes an ordered list of execution
providers. Resolving that in one place keeps the string-sniffing out of the five
handlers, and keeps the rules testable without any backend installed.

Accepted values are "cpu", "cuda", and "cuda:N". Anything else is passed through
unchanged to the torch- and CTranslate2-based backends (so "auto", "mps", or
"xpu" still reach the library that understands them) and treated as CPU for the
onnxruntime-based ones, which need an explicit provider list.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_LOGGER = logging.getLogger(__name__)

# Execution provider entries are either a name or (name, options).
OnnxProvider = Union[str, Tuple[str, Dict[str, Any]]]

_CPU_PROVIDER = "CPUExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"


def is_gpu(device: str) -> bool:
    """Report whether a device string names a CUDA GPU.

    Only CUDA is recognized: it is the one backend family every library in this
    project can target (CTranslate2 has no ROCm or XPU backend at all).
    """
    return device.strip().lower().startswith("cuda")


def device_index(device: str) -> Optional[int]:
    """Extract the GPU ordinal from "cuda:N", or None if unspecified."""
    _, _, index = device.strip().lower().partition(":")
    if not index:
        return None

    try:
        return int(index)
    except ValueError:
        _LOGGER.warning("Ignoring unparsable device index in '%s'", device)
        return None


def torch_device(device: str) -> str:
    """Return the device string for a torch-based backend (transformers, FunASR).

    torch understands "cuda" and "cuda:N" directly, so this is a passthrough
    that exists to normalize whitespace and to give the handlers a single named
    entry point alongside the other translations here.
    """
    return device.strip()


def ctranslate2_device(device: str) -> Tuple[str, Optional[int]]:
    """Return (device, device_index) for faster-whisper.

    CTranslate2 takes the ordinal as a separate ``device_index`` argument rather
    than as part of the device string, so "cuda:1" has to be split apart.
    """
    index = device_index(device)
    if index is None:
        return device.strip(), None

    base, _, _ = device.strip().partition(":")
    return base, index


def sherpa_provider(device: str) -> str:
    """Return the sherpa-onnx provider name.

    Requires a CUDA-enabled sherpa-onnx build. The CPU wheel published on PyPI
    accepts provider="cuda" and silently runs on the CPU anyway, so a GPU image
    must install the "+cuda" wheel from the k2-fsa index.
    """
    return "cuda" if is_gpu(device) else "cpu"


def onnx_providers(device: str) -> List[OnnxProvider]:
    """Return the onnxruntime execution providers to try, in order.

    The CPU provider is always kept as the last entry so a container without a
    usable driver degrades to CPU inference instead of failing to load a model.
    """
    if not is_gpu(device):
        return [_CPU_PROVIDER]

    index = device_index(device)
    if index is None:
        return [_CUDA_PROVIDER, _CPU_PROVIDER]

    return [(_CUDA_PROVIDER, {"device_id": index}), _CPU_PROVIDER]


def warn_if_no_onnx_gpu(device: str, providers: Sequence[str]) -> bool:
    """Warn when a GPU was asked for but onnxruntime is not using CUDA.

    Takes the provider list as an argument rather than importing onnxruntime, so
    this module stays importable (and testable) without it. Returns whether the
    GPU is in use.

    Prefer passing a live session's ``get_providers()`` over
    ``ort.get_available_providers()``: the latter lists the CUDA provider as
    available even when its shared library cannot be loaded (the usual cause
    being an onnxruntime-gpu built for a different CUDA major version than the
    one installed), in which case sessions silently fall back to the CPU. Either
    way the failure is quiet - every model still loads and transcribes, just not
    on the GPU - so it is worth a warning rather than leaving it to be noticed as
    "the GPU image is no faster".
    """
    if not is_gpu(device):
        return False

    if _CUDA_PROVIDER in providers:
        return True

    _LOGGER.warning(
        "Device '%s' was requested but onnxruntime is not using %s "
        "(in effect: %s), so inference will run on the CPU. Either "
        "onnxruntime-gpu is not installed, or its CUDA/cuDNN requirements are "
        "not met here - check the onnxruntime errors logged above.",
        device,
        _CUDA_PROVIDER,
        ", ".join(providers) or "none",
    )
    return False


def resolve_compute_type(compute_type: str, device: str) -> str:
    """Pick the CTranslate2 compute type, filling in a GPU-appropriate default.

    CTranslate2's own "default" means "whatever the model was converted to",
    which for the int8 models this project downloads by default would mean int8
    on hardware that is much faster at float16.
    """
    if (compute_type == "default") and is_gpu(device):
        return "float16"

    return compute_type


def disable_onnx_spinning(options: Any) -> Any:
    """Stop onnxruntime's thread pools from busy-waiting after each run.

    By default the intra-op (and inter-op) pools keep spinning for a while after
    a call returns, so the threads stay busy for seconds after the transcript is
    sent. In a voice pipeline the LLM starts on the same cores at exactly that
    moment, and the spinning slows it down (issue #135). Transcription itself is
    not affected: the spin only shortens the wake-up between consecutive runs.

    Takes an ``onnxruntime.SessionOptions`` and returns it, so this module still
    imports without onnxruntime installed.
    """
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options
