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


# --- draft verification ----------------------------------------------------


_EOS = 151645
_VOCAB = 16
_FAKE_EOS = 13  # end of text in the fake decoder's 16-token vocabulary


def _one_hot(tokens) -> np.ndarray:
    rows = np.zeros((len(tokens), _VOCAB), dtype=np.float32)
    for row, token in enumerate(tokens):
        rows[row, token % _VOCAB] = 1.0
    return rows


class _FakeDecoder:
    """A 'model' whose greedy output is always ``target``, with a fake KV cache.

    Mirrors one prefill over the prompt plus a draft: row j of the prefill
    logits picks target[j], whatever the draft says, and the cache gains one
    position per token fed. Steps check the cache never holds a rejected token.
    """

    def __init__(self, target, n_prompt: int = 4) -> None:
        self.target = list(target)
        self.n_prompt = n_prompt
        self.steps = 0

    def prefill(self, draft):
        logits = _one_hot(self.target[: len(draft) + 1])
        if len(logits) < len(draft) + 1:  # target ended inside the draft
            logits = np.concatenate(
                [logits, _one_hot([0] * (len(draft) + 1 - len(logits)))]
            )
        fed = self.n_prompt + len(draft)
        cache = np.zeros((1, 1, 1, fed, 1), dtype=np.float32)
        # Tag each draft position: 1 if it matches the target there, else 2.
        for j, token in enumerate(draft):
            ok = j < len(self.target) and token == self.target[j]
            cache[0, 0, 0, self.n_prompt + j, 0] = 1.0 if ok else 2.0
        return logits, cache

    def _target(self, index):
        return self.target[index] if index < len(self.target) else 0

    def step(self, tokens, position, keys, values):
        self.steps += 1
        assert keys.shape[3] == position, "cache length must match the position"
        assert (keys != 2.0).all(), "a rejected draft token is still in the cache"
        index = position - self.n_prompt
        assert tokens[0] == self.target[index], "fed a token greedy would not pick"

        # The first token is the confirmed one; any after it are guesses, tagged
        # by whether they are right so a stale wrong one is caught next step.
        tags = [1.0] + [
            1.0 if token == self._target(index + j) else 2.0
            for j, token in enumerate(tokens[1:], start=1)
        ]
        new = np.array(tags, np.float32).reshape(1, 1, 1, -1, 1)
        keys = np.concatenate([keys, new], axis=3)
        rows = [self._target(index + j + 1) for j in range(len(tokens))]
        return _one_hot(rows)[np.newaxis], keys, keys


def _decode(target, draft, multi_token_steps=True):
    decoder = _FakeDecoder(target)
    logits, cache = decoder.prefill(draft)
    tokens = qwen3_asr_handler._continue_greedy(
        logits,
        draft,
        decoder.n_prompt,
        cache,
        cache,
        decoder.step,
        multi_token_steps=multi_token_steps,
    )
    return tokens, decoder.steps


_TARGET = [3, 5, 7, 9, 11, _FAKE_EOS]


@pytest.fixture(name="small_eos")
def small_eos_fixture(monkeypatch):
    # Let the fake vocabulary end text the way the real one does.
    monkeypatch.setattr(qwen3_asr_handler, "_EOS_IDS", frozenset({_FAKE_EOS}))


@pytest.mark.parametrize(
    "draft",
    [
        [],  # no draft: plain greedy decoding
        _TARGET,  # entirely right
        _TARGET[:3],  # right, but short
        [3, 5, 8, 9],  # wrong from the third token
        [4, 4, 4],  # wrong from the start
        _TARGET + [1, 2, 3],  # right, then runs past the end of text
    ],
)
def test_any_draft_gives_the_greedy_result(small_eos, draft) -> None:
    tokens, _ = _decode(_TARGET, draft)
    assert tokens == _TARGET


_LONG = [3, 5, 7, 9, 2, 4, 6, 8, 10, 12, 14, 1, _FAKE_EOS]


@pytest.mark.parametrize("multi_token_steps", [True, False])
@pytest.mark.parametrize(
    "draft",
    [
        [3, 5, 7, 15, 2, 4, 6, 8, 10, 12, 14, 1, _FAKE_EOS],  # one word differs
        [3, 5, 7, 9, 4, 6, 8, 10, 12, 14, 1, _FAKE_EOS],  # one word missing
        [3, 5, 7, 9, 2, 0, 4, 6, 8, 10, 12, 14, 1, _FAKE_EOS],  # one word extra
        [3, 5, 2, 4, 2, 4, 2, 4],  # repeats that resemble the target
    ],
)
def test_draft_is_found_again_after_a_difference(
    small_eos, draft, multi_token_steps
) -> None:
    tokens, _ = _decode(_LONG, draft, multi_token_steps)
    assert tokens == _LONG


def test_finding_the_draft_again_saves_the_rest_of_it(small_eos) -> None:
    draft = [3, 5, 7, 15, 2, 4, 6, 8, 10, 12, 14, 1, _FAKE_EOS]
    _, single = _decode(_LONG, draft, multi_token_steps=False)
    _, multi = _decode(_LONG, draft, multi_token_steps=True)
    # One-token steps must generate everything after the difference; with
    # multi-token steps the draft resumes once two tokens match it again.
    assert single == len(_LONG) - 4
    assert multi <= 3


def test_an_accepted_draft_skips_its_decoder_steps(small_eos) -> None:
    _, no_draft = _decode(_TARGET, [])
    _, right = _decode(_TARGET, _TARGET)
    _, half = _decode(_TARGET, _TARGET[:3])
    assert no_draft == len(_TARGET) - 1
    assert right == 0
    assert half == len(_TARGET) - 4


def test_accept_draft_stops_at_the_first_mismatch() -> None:
    logits = _one_hot([3, 5, 7, 9])
    assert qwen3_asr_handler._accept_draft(logits, [3, 5, 6]) == [3, 5, 7]
    assert qwen3_asr_handler._accept_draft(logits, [3, 5, 7]) == [3, 5, 7, 9]
    assert qwen3_asr_handler._accept_draft(logits[:1], []) == [3]


class _PrefillRecorder:
    """Enough of a transcriber for decode_tokens: records what is prefilled."""

    def __init__(self, generated) -> None:
        self.generated = list(generated)
        self.calls: list = []
        self._mel_filters = _mel_filterbank()
        self._encoder_inputs = ["mel"]
        self._encoder = SimpleNamespace(
            run=lambda names, inputs, run_options=None: [
                np.zeros((1, 2, 4), dtype=np.float32)
            ]
        )
        self._merged = object()

    def _context_ids(self, initial_prompt, language):
        return [1111], [3333, qwen3_asr_handler._ASR_TEXT]

    def _build_prompt_ids(self, n_audio_tokens, context_ids, language_ids):
        return _build_prompt_ids(None, n_audio_tokens, context_ids, language_ids)

    def _generate_merged(
        self, audio_features, prompt_ids, n_system, draft, run_options
    ):
        self.calls.append((prompt_ids, draft))
        return list(self.generated)


decode_tokens = qwen3_asr_handler.Qwen3AsrTranscriber.decode_tokens


def test_draft_reaches_the_decoder_separately_from_the_prompt() -> None:
    model = _PrefillRecorder([7, 8, _EOS])
    audio = np.zeros(1600, dtype=np.float32)
    assert decode_tokens(model, audio, "en", None, draft=[5, 6]) == [7, 8, _EOS]

    prompt, draft = model.calls[0]
    assert prompt[-2:] == [3333, qwen3_asr_handler._ASR_TEXT]
    assert draft == [5, 6]


def test_no_draft_by_default() -> None:
    model = _PrefillRecorder([7, _EOS])
    audio = np.zeros(1600, dtype=np.float32)
    assert decode_tokens(model, audio, "en", None) == [7, _EOS]
    assert model.calls[0][1] == []


# --- streaming session -----------------------------------------------------


_RATE = qwen3_asr_handler._RATE


class _FakeStreamingModel:
    """Records every decode; each returns one token per 100 samples."""

    def __init__(self, fail_passes: int = 0) -> None:
        self.calls: list = []
        self.gate = threading.Event()
        self.gate.set()
        self.fail_passes = fail_passes
        self.aborted = 0

    def decode_tokens(
        self, audio, language, initial_prompt, draft=(), run_options=None
    ):
        # Background passes (the ones with run_options) block like a long model
        # run until the gate opens, and terminate aborts them as ORT would.
        while (run_options is not None) and not self.gate.wait(timeout=0.01):
            if run_options.terminate:
                self.aborted += 1
                raise RuntimeError("Exiting due to terminate flag being set to true.")

        self.calls.append((len(audio), list(draft)))
        if self.fail_passes:
            self.fail_passes -= 1
            raise RuntimeError("boom")

        return list(range(len(audio) // 100))

    def tokens_to_text(self, tokens):
        return f"{len(tokens)} tokens"


def _session(model, chunk_seconds: float = 0.1, clipper=None):
    return qwen3_asr_handler.Qwen3AsrStreamingSession(
        model, "en", None, chunk_seconds=chunk_seconds, clipper=clipper
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


def test_each_pass_drafts_from_the_last_one() -> None:
    model = _FakeStreamingModel()
    session = _session(model)
    for _ in range(3):
        session.accept_chunk(_silence(0.1))
        _drain(session)

    drafts = [draft for _, draft in model.calls]
    # The first pass has nothing to go on; each later one gets the previous
    # pass's whole result (16 and 32 tokens here).
    assert drafts == [[], list(range(16)), list(range(32))]


def test_finish_aborts_a_running_pass_and_decodes_all_audio() -> None:
    model = _FakeStreamingModel()
    session = _session(model)
    for _ in range(2):
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
    length, draft = model.calls[-1]
    assert length == 5600
    assert draft == list(range(32))


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


def test_a_failed_pass_only_loses_the_draft() -> None:
    model = _FakeStreamingModel(fail_passes=1)
    session = _session(model)
    session.accept_chunk(_silence(0.1))
    _drain(session)

    assert session.finish() == "16 tokens"
    assert model.calls[-1] == (1600, [])


def test_finish_with_no_audio_is_empty() -> None:
    model = _FakeStreamingModel()
    assert _session(model).finish() == ""
    assert not model.calls


class _FakeClipper:
    """Keeps the middle half of the audio, or finds no speech at all."""

    def __init__(self, speech: bool = True) -> None:
        self.speech = speech
        self.processed = 0

    def process(self, audio: bytes) -> None:
        self.processed += len(audio)

    def clip(self, audio: bytes) -> bytes:
        if not self.speech:
            return b""

        quarter = (len(audio) // 8) * 2
        return audio[quarter : len(audio) - quarter]


def test_vad_clip_shortens_only_the_final_pass() -> None:
    model = _FakeStreamingModel()
    clipper = _FakeClipper()
    session = _session(model, clipper=clipper)
    session.accept_chunk(_silence(0.1))
    _drain(session)
    session.accept_chunk(_silence(0.1)[:1001])
    session.accept_chunk(_silence(0.1)[1001:])
    session.finish()

    # The VAD saw every (whole-sample) byte as it arrived...
    assert clipper.processed == 2 * 3200
    # ...the intermediate pass read everything, and the final pass the middle.
    assert model.calls[0][0] == 1600
    assert model.calls[-1][0] == 1600


def test_vad_clip_without_speech_uses_all_audio() -> None:
    model = _FakeStreamingModel()
    session = _session(model, chunk_seconds=10, clipper=_FakeClipper(speech=False))
    session.accept_chunk(_silence(0.1))
    session.finish()
    assert model.calls == [(1600, [])]


def test_streaming_transcriber_clips_only_when_asked(tmp_path, stub_runtime) -> None:
    model_dir = str(_fake_model_dir(tmp_path, merged=True))
    plain = qwen3_asr_handler.Qwen3AsrStreamingTranscriber(model_dir, str(tmp_path))
    assert plain.start_stream("en").clipper is None

    clipping = qwen3_asr_handler.Qwen3AsrStreamingTranscriber(
        model_dir, str(tmp_path), vad_clip=True, vad_clip_pad_ms=200
    )
    clipper = clipping.start_stream("en").clipper
    assert clipper is not None
    assert clipper.pad_ms == 200


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
