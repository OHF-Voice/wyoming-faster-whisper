"""Tests for the Qwen3-ASR prompt/front-end helpers.

The model files are multi-GB, so these cover the pure logic only: the chat
template the decoder is prompted with, and the mel front-end. Skipped when
onnxruntime is not installed (the handler imports it at module scope).
"""

import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

qwen3_asr_handler = pytest.importorskip(
    "wyoming_faster_whisper.qwen3_asr_handler",
    reason="qwen3-asr extra not installed",
)

_build_prompt_ids = qwen3_asr_handler.Qwen3AsrTranscriber._build_prompt_ids
_causal_mask = qwen3_asr_handler._causal_mask
_log_mel_spectrogram = qwen3_asr_handler._log_mel_spectrogram
_mel_filterbank = qwen3_asr_handler._mel_filterbank
_system_ids = qwen3_asr_handler._system_ids
qwen3_asr_language = qwen3_asr_handler.qwen3_asr_language

_AUDIO_PAD = qwen3_asr_handler._AUDIO_PAD
_BLOCKED = qwen3_asr_handler._MASK_BLOCKED


# --- language normalization ------------------------------------------------


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en", "English"),
        ("EN", "English"),
        ("en-US", "English"),  # locale-style
        ("zh-CN", "Chinese"),
        ("yue", "Cantonese"),
        ("xx", None),  # unsupported
        (None, None),
        ("", None),
    ],
)
def test_language_normalization(code, expected) -> None:
    assert qwen3_asr_language(code) == expected


# --- prompt construction ---------------------------------------------------


def test_prompt_has_one_audio_pad_per_encoder_frame() -> None:
    ids = _build_prompt_ids(None, 7, [], [])
    assert ids.count(_AUDIO_PAD) == 7


def test_prompt_matches_chat_template_layout() -> None:
    ids = _build_prompt_ids(None, 1, [], [])
    # <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    assert ids[:9] == [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]
    # ...<|audio_end|><|im_end|>\n<|im_start|>assistant\n
    assert ids[10:] == [151670, 151645, 198, 151644, 77091, 198]


def test_context_is_placed_in_the_system_turn() -> None:
    context = [1111, 2222]
    ids = _build_prompt_ids(None, 1, context, [])
    # Context sits after "<|im_start|>system\n" and before "<|im_end|>".
    assert ids[:3] == [151644, 8948, 198]
    assert ids[3:5] == context
    assert ids[5] == 151645


def test_forced_language_suffix_goes_after_the_assistant_turn() -> None:
    language_ids = [3333, qwen3_asr_handler._ASR_TEXT]
    ids = _build_prompt_ids(None, 1, [], language_ids)
    assert ids[-2:] == language_ids
    assert ids[-3] == 198  # newline closing "<|im_start|>assistant\n"


# --- merged decoder: the cached span ---------------------------------------


def test_system_turn_is_the_prompts_prefix() -> None:
    """The cached span must be a literal prefix of the prompt, or reusing its KV
    would silently attend to the wrong positions."""
    context = [1111, 2222]
    ids = _build_prompt_ids(None, 3, context, [])
    system = _system_ids(context)
    assert ids[: len(system)] == system


def test_system_turn_holds_the_context_and_no_audio() -> None:
    system = _system_ids([1111, 2222])
    assert system == [151644, 8948, 198, 1111, 2222, 151645, 198]
    # Nothing about the audio may be inside the cached span.
    assert _AUDIO_PAD not in system


def test_system_turn_length_tracks_the_context() -> None:
    assert len(_system_ids([])) == 5
    assert len(_system_ids([1, 2, 3])) == 8


# --- merged decoder: attention mask ----------------------------------------


def test_mask_is_causal_when_there_is_no_cache() -> None:
    mask = _causal_mask(3, 0)
    assert mask.shape == (1, 1, 3, 3)
    # Row i may attend to columns <= i.
    assert (mask[0, 0] == 0).tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, True],
    ]


def test_mask_allows_every_cached_position() -> None:
    mask = _causal_mask(2, 4)
    assert mask.shape == (1, 1, 2, 6)
    # The four cached positions are open to both new tokens...
    assert (mask[0, 0, :, :4] == 0).all()
    # ...and the new tokens stay causal among themselves.
    assert mask[0, 0, 0, 4] == 0
    assert mask[0, 0, 0, 5] == _BLOCKED
    assert mask[0, 0, 1, 4] == 0
    assert mask[0, 0, 1, 5] == 0


def test_single_token_step_attends_to_everything() -> None:
    mask = _causal_mask(1, 7)
    assert mask.shape == (1, 1, 1, 8)
    assert (mask == 0).all()


def test_mask_blocks_with_a_large_negative_and_is_float32() -> None:
    mask = _causal_mask(2, 0)
    assert mask.dtype == np.float32
    assert mask[0, 0, 0, 1] == _BLOCKED
    assert _BLOCKED < -1e30


# --- model file selection --------------------------------------------------


class _FakeSession:
    """Enough of an ORT session to get through the constructor."""

    def __init__(self, path, **kwargs) -> None:
        self.path = str(path)
        self.providers = list(kwargs.get("providers") or ["CPUExecutionProvider"])

    def get_providers(self):
        # The real session reports what it actually ended up using, which is how
        # the handler detects a CUDA provider that failed to load.
        return self.providers

    def get_inputs(self):
        return [SimpleNamespace(name="mel")]

    def get_outputs(self):
        # logits, then present_keys/present_values shaped
        # [num_layers, batch, kv_heads, seq, head_dim] with the batch and
        # sequence dims symbolic, as the real export has them.
        kv = SimpleNamespace(name="present_keys", shape=[4, "batch", 2, "seq", 8])
        return [SimpleNamespace(name="logits", shape=["batch", "q", 99]), kv, kv]


def _fake_model_dir(tmp_path, merged: bool):
    """A model directory with the right filenames and no real weights."""
    path = tmp_path / ("merged" if merged else "split")
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"decoder": {"hidden_size": 8}}))
    (path / "tokenizer.json").write_text("{}")
    (path / "embed_tokens.bin").write_bytes(b"")
    names = ["encoder.int4.onnx"]
    names += (
        ["decoder_merged.int4.onnx"]
        if merged
        else ["decoder_init.int4.onnx", "decoder_step.int4.onnx"]
    )
    for name in names:
        (path / name).write_bytes(b"")
    return path


@pytest.fixture(name="stub_runtime")
def stub_runtime_fixture(monkeypatch):
    monkeypatch.setattr(qwen3_asr_handler.ort, "InferenceSession", _FakeSession)
    monkeypatch.setattr(
        qwen3_asr_handler, "Tokenizer", SimpleNamespace(from_file=lambda path: object())
    )
    monkeypatch.setattr(
        qwen3_asr_handler.np,
        "fromfile",
        lambda *args, **kwargs: np.zeros(16, dtype=np.float16),
    )


def test_merged_decoder_is_preferred_when_present(tmp_path, stub_runtime) -> None:
    model = qwen3_asr_handler.Qwen3AsrTranscriber(
        str(_fake_model_dir(tmp_path, merged=True)), str(tmp_path)
    )
    assert model._merged is not None
    assert model._decoder_init is None
    assert model._decoder_step is None


def test_split_decoder_is_used_when_merged_is_absent(tmp_path, stub_runtime) -> None:
    model = qwen3_asr_handler.Qwen3AsrTranscriber(
        str(_fake_model_dir(tmp_path, merged=False)), str(tmp_path)
    )
    assert model._merged is None
    assert model._decoder_init is not None
    assert model._decoder_step is not None


def test_empty_kv_is_shaped_from_the_graph(tmp_path, stub_runtime) -> None:
    # Taken from present_keys so the cache does not depend on config.json.
    model = qwen3_asr_handler.Qwen3AsrTranscriber(
        str(_fake_model_dir(tmp_path, merged=True)), str(tmp_path)
    )
    assert model._empty_kv.shape == (4, 1, 2, 0, 8)
    assert model._empty_kv.dtype == np.float32


def test_prefix_cache_starts_empty(tmp_path, stub_runtime) -> None:
    model = qwen3_asr_handler.Qwen3AsrTranscriber(
        str(_fake_model_dir(tmp_path, merged=True)), str(tmp_path)
    )
    assert model._prefix_key is None
    assert model._prefix_kv is None


def test_allow_patterns_cover_both_decoder_layouts() -> None:
    patterns = set(qwen3_asr_handler._ALLOW_PATTERNS)
    assert qwen3_asr_handler._MERGED_DECODER in patterns
    assert "decoder_merged.int4.onnx.data" in patterns
    # The split layout must still be downloadable.
    assert {"decoder_init.int4.onnx", "decoder_step.int4.onnx"} <= patterns
    assert "decoder_weights.int4.data" in patterns


# --- mel front-end ---------------------------------------------------------


def test_mel_filterbank_shape_and_normalization() -> None:
    filters = _mel_filterbank()
    assert filters.shape == (128, (400 // 2) + 1)
    assert filters.dtype == np.float32
    assert (filters >= 0).all()
    # Every filter carries some weight (no empty bands).
    assert (filters.sum(axis=1) > 0).all()


def test_log_mel_frame_count_and_range() -> None:
    # 1 second of 16kHz audio -> 100 frames (10ms hop), minus the dropped frame.
    audio = np.zeros(16000, dtype=np.float32)
    mel = _log_mel_spectrogram(audio, _mel_filterbank())
    assert mel.shape == (1, 128, 100)
    assert mel.dtype == np.float32
    # The log scale is clamped to an 8-decade window then rescaled.
    assert mel.max() - mel.min() <= 2.0 + 1e-6


# --- forced prefix ---------------------------------------------------------


class _PrefillRecorder:
    """Enough of a transcriber for decode_tokens: records the prompt prefilled."""

    def __init__(self, generated) -> None:
        self.generated = list(generated)
        self.prompts: list = []
        self._mel_filters = _mel_filterbank()
        self._encoder_inputs = ["mel"]
        self._encoder = SimpleNamespace(
            run=lambda names, inputs, run_options=None: [np.zeros((1, 2, 4), dtype=np.float32)]
        )
        self._merged = object()

    def _context_ids(self, initial_prompt, language):
        return [1111], [3333, qwen3_asr_handler._ASR_TEXT]

    def _build_prompt_ids(self, n_audio_tokens, context_ids, language_ids):
        return _build_prompt_ids(None, n_audio_tokens, context_ids, language_ids)

    def _generate_merged(self, audio_features, prompt_ids, n_system, run_options):
        self.prompts.append(prompt_ids)
        return list(self.generated)


decode_tokens = qwen3_asr_handler.Qwen3AsrTranscriber.decode_tokens


def test_forced_prefix_follows_the_language_suffix() -> None:
    model = _PrefillRecorder([7, 8, 151645])
    audio = np.zeros(1600, dtype=np.float32)
    tokens = decode_tokens(model, audio, "en", None, prefix=[5, 6])

    prompt = model.prompts[0]
    assert prompt[-4:] == [3333, qwen3_asr_handler._ASR_TEXT, 5, 6]
    # The prefix is part of the result, ahead of what was generated after it.
    assert tokens == [5, 6, 7, 8, 151645]


def test_no_prefix_leaves_the_prompt_unchanged() -> None:
    model = _PrefillRecorder([7, 151645])
    audio = np.zeros(1600, dtype=np.float32)
    assert decode_tokens(model, audio, "en", None) == [7, 151645]
    assert model.prompts[0][-1] == qwen3_asr_handler._ASR_TEXT


# --- streaming session -----------------------------------------------------


_RATE = qwen3_asr_handler._RATE
_UNFIXED_CHUNKS = qwen3_asr_handler._UNFIXED_CHUNKS
_UNFIXED_TOKENS = qwen3_asr_handler._UNFIXED_TOKENS


class _FakeStreamingModel:
    """Records every decode; each returns one token per 100 samples."""

    def __init__(self, fail_passes: int = 0) -> None:
        self.calls: list = []
        self.gate = threading.Event()
        self.gate.set()
        self.fail_passes = fail_passes
        self.aborted = 0

    def decode_tokens(
        self, audio, language, initial_prompt, prefix=(), run_options=None
    ):
        # Background passes (the ones with run_options) block like a long model
        # run until the gate opens, and terminate aborts them as ORT would.
        while (run_options is not None) and not self.gate.wait(timeout=0.01):
            if run_options.terminate:
                self.aborted += 1
                raise RuntimeError("Exiting due to terminate flag being set to true.")

        self.calls.append((len(audio), list(prefix)))
        if self.fail_passes:
            self.fail_passes -= 1
            raise RuntimeError("boom")

        return list(range(len(audio) // 100))

    def tokens_to_text(self, tokens):
        return f"{len(tokens)} tokens"


def _session(model, chunk_seconds: float = 0.1):
    return qwen3_asr_handler.Qwen3AsrStreamingSession(
        model, "en", None, chunk_seconds=chunk_seconds
    )


def _silence(seconds: float) -> bytes:
    return bytes(int(seconds * _RATE) * 2)


def _drain(session) -> None:
    if session._worker is not None:
        session._worker.join(timeout=5)


def test_no_pass_before_a_chunk_of_audio() -> None:
    model = _FakeStreamingModel()
    session = _session(model, chunk_seconds=1.0)
    session.accept_chunk(_silence(0.5))
    _drain(session)
    assert not model.calls

    # Short commands get exactly one decode, the same as the batch path.
    assert session.finish() == "80 tokens"
    assert model.calls == [(8000, [])]


def test_a_pass_runs_once_a_chunk_has_arrived() -> None:
    model = _FakeStreamingModel()
    session = _session(model)
    session.accept_chunk(_silence(0.1))
    _drain(session)
    assert model.calls == [(1600, [])]


def test_passes_do_not_queue_behind_a_running_one() -> None:
    model = _FakeStreamingModel()
    model.gate.clear()
    session = _session(model)

    for _ in range(5):
        session.accept_chunk(_silence(0.1))

    model.gate.set()
    _drain(session)
    # Only the first pass ran; the rest arrived while it was busy.
    assert len(model.calls) == 1

    # The next chunk catches up on everything buffered meanwhile.
    session.accept_chunk(_silence(0.1))
    _drain(session)
    assert model.calls[-1][0] == 6 * 1600


def test_prefix_is_withheld_for_the_first_passes_then_rolled_back() -> None:
    model = _FakeStreamingModel()
    session = _session(model)
    for _ in range(_UNFIXED_CHUNKS + 1):
        session.accept_chunk(_silence(0.1))
        _drain(session)

    prefixes = [prefix for _, prefix in model.calls]
    assert prefixes[:_UNFIXED_CHUNKS] == [[]] * _UNFIXED_CHUNKS
    # The previous result had 32 tokens; the last few are not forced.
    previous = list(range((_UNFIXED_CHUNKS * 1600) // 100))
    assert prefixes[_UNFIXED_CHUNKS] == previous[:-_UNFIXED_TOKENS]


def test_finish_aborts_a_running_pass_and_decodes_all_audio() -> None:
    model = _FakeStreamingModel()
    session = _session(model)
    for _ in range(_UNFIXED_CHUNKS):
        session.accept_chunk(_silence(0.1))
        _drain(session)

    # This pass is still running when the audio stops.
    model.gate.clear()
    session.accept_chunk(_silence(0.1))
    session.accept_chunk(_silence(0.05))

    started = time.monotonic()
    text = session.finish()
    assert time.monotonic() - started < 1.0
    assert model.aborted == 1
    assert text == "56 tokens"

    # The final pass read everything and used the last *finished* pass's result.
    length, prefix = model.calls[-1]
    assert length == 5600
    assert prefix == list(range(32))[:-_UNFIXED_TOKENS]


def test_odd_byte_chunks_are_reassembled() -> None:
    model = _FakeStreamingModel()
    session = _session(model, chunk_seconds=10)
    audio = _silence(0.1)
    session.accept_chunk(audio[:1001])
    session.accept_chunk(audio[1001:])
    session.finish()
    assert model.calls == [(1600, [])]


def test_trailing_odd_byte_is_padded_into_a_sample() -> None:
    model = _FakeStreamingModel()
    session = _session(model, chunk_seconds=10)
    session.accept_chunk(b"\x00" * 201)
    session.finish()
    assert model.calls == [(101, [])]


def test_a_failed_pass_only_loses_the_hint() -> None:
    model = _FakeStreamingModel(fail_passes=1)
    session = _session(model)
    session.accept_chunk(_silence(0.1))
    _drain(session)
    assert session._passes == 0

    assert session.finish() == "16 tokens"


def test_finish_with_no_audio_is_empty() -> None:
    model = _FakeStreamingModel()
    assert _session(model).finish() == ""
    assert not model.calls


def test_streaming_transcriber_starts_streaming_sessions(
    tmp_path, stub_runtime
) -> None:
    model = qwen3_asr_handler.Qwen3AsrStreamingTranscriber(
        str(_fake_model_dir(tmp_path, merged=True)), str(tmp_path)
    )
    assert model.supports_streaming
    session = model.start_stream("en", initial_prompt="Vocabulary: Ecobee.")
    assert isinstance(session, qwen3_asr_handler.Qwen3AsrStreamingSession)
    assert session.initial_prompt == "Vocabulary: Ecobee."
