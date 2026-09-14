# Changelog

## 3.8.1

- Streaming sherpa-onnx models no longer cut off the last word or two of an
  utterance. The trailing silence fed before `input_finished()` is now measured
  from the model rather than fixed at the 0.66s in sherpa-onnx's own examples:
  the encoder only runs on whole chunks, and a chunk is 1.41s for the Kroko
  streaming zipformers that `--sherpa-streaming` defaults to, so the tail of
  every utterance went undecoded. 0.66s stays the minimum for models whose
  chunk is shorter
  - Priming those models at startup was a no-op for the same reason — it fed
    128 samples, far less than one chunk — so the first transcription paid for
    the warm-up

## 3.8.0

- Add optional `--vad-endpointing SECONDS` server-side command endpointing, based
  on speech-to-phrase's Silero VAD segmenter. It sends the transcript after the
  configured post-speech silence instead of waiting for `audio-stop`, advertises
  that external VAD is not required while enabled, and remains independent of
  the transcription-time `--vad-filter` and `--vad-clip` options

- Models that are already downloaded now load without contacting Hugging Face, so a server with no route to the internet starts instead of boot-looping (#93 by @schuylermartin45, #92)
  - Loading is cache-first by default: each backend is built with `local_files_only` and only retried as a download when a file is genuinely missing. The hub check is what fails without internet, not the model load, and on a network that drops outbound traffic rather than refusing it — a Docker bridge marked `internal` — it stalled for the full TCP timeout on every start
  - `--local-files-only` still means what it says: no fallback, so a model that isn't cached is an error rather than a surprise download. It had no effect at all on faster-whisper, the default backend, which never received the flag
  - FunASR ignored `--download-dir` entirely, putting its ~900 MB of models in `~/.cache/huggingface` instead, and reported a failed download as `model ... is not registered` several steps later. Its model directory is now resolved before FunASR is handed the model
  - Sherpa models, which come from GitHub releases rather than the hub, are covered too
- Both Docker images set `HF_HOME`, `XDG_CACHE_HOME`, and `MODELSCOPE_CACHE` to `/data` so the caches that no command-line option reaches also land on the volume
  - The Xet chunk cache used during hub downloads (`$HF_HOME/xet`) is the big one: several GB, previously written to the container's filesystem
  - It also fixes the unwritable `/.cache` that a container running as a uid with no passwd entry would hit, since `$HOME` is unset there

- Add a Docker health check to both images, matching the one in wyoming-piper 2.4.3 (#119 by @netsho)
  - It sends the server a `Describe` and requires an `Info` with an ASR program back, rather than only opening a socket: the port is bound by the OS, so a connect-only check stays green even when the event loop is wedged, while a round trip proves the accept loop and the event handler are both still running
  - `--start-period` is 5 minutes, since the server only starts listening once the model is loaded — which means downloading it on first run — and it takes 3 consecutive failures to turn the container unhealthy, because transcription runs on the event loop and a check can time out behind a long request
  - The URI to check follows `WYO_WHISPER_URI` (or its `_FILE` form) when the container sets it, with a listen-everywhere host like `0.0.0.0` rewritten to loopback. A `--uri` passed to `docker run` instead is invisible to it, so pass the same one to `python3 -m wyoming_faster_whisper.health_check --uri ...`

- Warn at startup when `--initial-prompt` or `--hass-token` is used with a Distil-Whisper model, and document that these models are not compatible with prompting (#118 by @jfoxwoosh)
  - Distil-Whisper was distilled without previous-text conditioning, so a prompt never helps it. `distil-small.en` is actively damaged: from about 52 prompt tokens on, `avg_logprob` drops below faster-whisper's -1.0 threshold, every temperature fails, and output that was correct unprompted comes back truncated (`Start a timer for 25 minutes` → `Start a timer.`) or looping (`Add Hot Dog, Hot Dog, Hot Dog, …`). Whether the transcript ends up wrong or empty then depends on `no_speech_prob` for that utterance
  - `distil-large-v3` is inert rather than broken — stable at every prompt size, but with no biasing effect either, still returning `EcoBe` with `Ecobee` in the prompt
  - The prompt is still sent, since the damage is size-dependent and a short `--initial-prompt` may be harmless. Standard Whisper models are unaffected: `small.en` is steady through a 97-token prompt and biasing works as intended there

## 3.7.0

- Every command-line option can now also be set from the environment: `--some-option` reads `WYO_WHISPER_SOME_OPTION`, so a Compose stack can configure the container without rewriting its `command:` (#64 by @ffeliziani-tpmc, #112 by @DennisGaida)
  - Values can also come from a file named by `WYO_WHISPER_SOME_OPTION_FILE`, the Docker Compose/Swarm secrets convention, which keeps a long-lived Home Assistant token out of both the command line and the environment
  - Precedence is command line, then `_FILE`, then the plain variable: an explicit argument is never silently overridden by a stale variable in a container
  - Flags take `true`/`false` rather than being on whenever the variable exists, `--data-dir` splits on `:`, `--vad-clip` splits on commas or spaces, and values are checked against the option's type and choices with the variable named in the error
  - A `WYO_WHISPER_` variable matching no option is warned about at startup instead of being ignored, and one left empty means the same as not setting it — except for the three that give empty a meaning of their own (a flag is off, `--vad-clip` is every library, `--zeroconf` is the default name)
  - The Docker entrypoint drops its baked-in `--uri`, `--data-dir`, and `--device` defaults when the matching variable is set, since a command-line argument would otherwise always win over it

- Home Assistant names now fill the prompt budget in four priority tiers: areas/floors that hold an exposed entity, then entity names in the domains people say out loud (`PRIORITY_DOMAINS`: light, switch, fan, media_player, climate, scene, todo), then the remaining areas/floors, then the remaining entity names
  - Aliases are no longer ranked below every name. An alias is what the speaker says *instead of* the entity's name, so each one now sits directly behind the name it belongs to and shares its tier — previously a home with enough areas and entities to fill the budget lost every alias
  - An entity's area is resolved through its device when it has no area of its own, so the device registry is read too — only when an exposed entity actually needs it
  - In a home that overruns the budget, this stops exposed-by-the-hundred sensors and empty areas from crowding out the lights, scenes and media players a command names
- Add `Dockerfile.gpu`, a CUDA variant of the image that runs faster-whisper, transformers, onnx-asr, and qwen3-asr on an NVIDIA GPU: CUDA torch plus `onnxruntime-gpu` on a CUDA 12.8 / cuDNN 9 base. `--device cuda` is the default there; `docker run --gpus all` and the NVIDIA Container Toolkit are required (it carries the same extras as the CPU image, so FunASR is still not included in either) (#76 by @lmoe)
  - **Build it yourself** — this one is not published to Docker Hub. It comes out around 10.7 GB, mostly the CUDA torch wheel, against ~1.6 GB for the CPU image, and Home Assistant OS offers no GPU passthrough, so everyone who can use it is already running Docker directly. `docker build -f Dockerfile.gpu -t wyoming-whisper:gpu .` — see the README
  - NVIDIA and amd64 only. CTranslate2 (faster-whisper) has no ROCm or XPU backend and no arm64 CUDA wheel, which is what bounds the image; the torch-based backends alone would work elsewhere
  - `--stt-library sherpa` stays on the CPU there. sherpa-onnx bundles its own onnxruntime, and two CUDA-enabled onnxruntime builds in one process segfault — reachable via `--stt-library auto`, which routes English to sherpa and Russian to onnx-asr. sherpa's default models are int8, which the CUDA provider gains little on, so the CPU wheel is the better half of the trade
- `--device` now reaches every backend, not just faster-whisper and FunASR. `transformers` moves the model to the device and runs float16 there, `sherpa` selects the CUDA provider, and `onnx-asr`/`qwen3-asr` select the CUDA execution provider with a CPU fallback. `cuda:N` selects a specific GPU (#87 by @zackify)
  - Previously `--device cuda` was silently ignored by four of the six backends
- On CUDA, `--compute-type` defaults to `float16` rather than the model's own type, and the auto-selected faster-whisper model is `Systran/faster-whisper-small` (float16) rather than an int8 conversion
- `sherpa` now locates its model files by prefix instead of hard-coding `*.int8.onnx`, and prefers an fp32 graph over int8 when running on the GPU with a model that ships both
- The server now warns instead of staying silent when `--device cuda` cannot actually be used — a CPU-only onnxruntime or sherpa-onnx build, or a CUDA provider that fails to load — since every backend still loads and transcribes on the CPU, and the only other symptom is the absence of a speedup

## 3.6.0

- Add `--hass-token` (extra: `hass`) to bias transcription toward the names in Home Assistant: conversation-exposed entity names and aliases, plus area and floor names, read over the websocket API and passed to the model as a prompt (fixes e.g. "What's the temperature of the incubi?" → "What's the temperature of the Ecobee?")
  - Names are refreshed in the background starting at `AudioStart`, so the fetch finishes while the speaker is still talking and adds no latency; a slow or unreachable Home Assistant falls back to the previous names and never fails a transcript
  - Names are added to the prompt in priority order (areas, floors, entity names, aliases) up to `--hass-prompt-max-tokens` (default 200, Whisper's hard cap is 223); `--initial-prompt` is kept at the front
  - Add `--hass-api`, `--hass-refresh-seconds`, `--hass-prompt-max-tokens`, and `--hass-prompt-timeout`

- Add support for [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) via `--stt-library qwen3-asr` (extra: `qwen3_asr`), defaulting to [`rhasspy/qwen3-asr-0.6b-onnx-int4-merged`](https://huggingface.co/rhasspy/qwen3-asr-0.6b-onnx-int4-merged)
  - `--initial-prompt` now also biases the Qwen3-ASR backend: it is passed as the model's context prompt, which corrects entity names (e.g. `Vocabulary: Ecobee.` turns "incubator" into "Ecobee")
  - Qwen3-ASR is opt-in only (`auto` never selects it): the model is 785 MB and needs ~1.6 GB of RAM, and it is slower than the per-language defaults

- Qwen3-ASR: support a merged decoder export (`decoder_merged.int4.onnx`) that takes a KV cache and a dynamic sequence length, so the biasing prompt's KV is computed once and reused instead of being re-prefilled every utterance. On a Pi 5 with a 50-name prompt, a 3.2s command goes from 3.42s to 2.20s (1.56x), peak RSS from 2.25 GB to 1.55 GB, and the package from 1407 MB to 785 MB
  - The merged layout is selected automatically when `decoder_merged.int4.onnx` is present; model directories with `decoder_init`/`decoder_step` keep working unchanged
  - The speedup applies to short commands. Long-form audio sees ~1.04x, since the cached prompt is a small share of the work — but the memory saving grows with length (4.16 GB → 2.87 GB on a 30s clip)
  - The default Qwen3-ASR model is now the merged export. On LibriSpeech test-other (n=200) it transcribes byte-identically to the split export with no prompt (5.35% WER for both), and scores 5.33% vs 5.43% with a 50-name prompt — a difference well inside sampling noise. Pass `--model rhasspy/qwen3-asr-0.6b-onnx-int4` for the split export, which stays published

- `--vad-clip` now takes optional speech-to-text libraries, so clipping can be enabled only where it pays off: `--vad-clip qwen3-asr` clips for that backend alone, while a bare `--vad-clip` still applies to every library. Qwen3-ASR costs ~235 ms per second of audio on a Pi 5 (encoder, prefill, *and* per-token decode all scale with input length), so trimming silence off a typical command saves ~20%; faster-whisper pads to 30s internally and gains nothing

- The Docker image now includes the `qwen3-asr` and `hass` extras

## 3.5.0

- Bump torch to avoid regression: https://github.com/pytorch/pytorch/issues/146792
- Use `find_spec` to avoid importing modules for backend check

## 3.4.1

- Use `pysilero-vad>=3.4.0`

## 3.4.0

- Disable VAD by default (use `--vad-clip` to enable)
- Apply `--vad-clip` to all batch backends (not just faster-whisper); clipping happens on the WAV before dispatch. Mainly a latency win for length-proportional backends like sherpa/FunASR on silence-heavy audio; streaming backends are unaffected
- Bump `pysilero-vad` to use GGML version

## 3.3.1

- Ensure zh/yue/ja/ko default to FunASR

## 3.3.0

- Add FunASR speech-to-text backend (`--stt-library funasr`) defaulting to `FunAudioLLM/SenseVoiceSmall` (`@LauraGPT`)
  - Non-autoregressive and notably faster than Whisper; supports English, Chinese, Cantonese, Japanese, and Korean well
  - Install with the `funasr` extra (`pip install '.[funasr]'`)

## 3.2.1

- Fix streaming sherpa cutting off the end of utterances (add tail padding before flushing)
- Default streaming sherpa to the Kroko 2025 zipformer models (mixed-case, punctuated, much better accuracy than the old LibriSpeech model); adds `de`/`es`/`fr` defaults
- Use `--beam-size` for streaming sherpa decoding (beam search when > 1, greedy otherwise)

## 3.2.0

- Fix transformers language
- Add initial prompt to transformers
- Add `--whisper-task` which can be set to "translate" instead of "transcribe" (`@M4TH1EU`)
- Add `--sherpa-streaming` to prefer streaming models (`@pkrahmer`)
- Bump `onnx-asr` to 0.11.0 (supports `istupakov/canary-1b-v2-onnx`)

## 3.1.0

- Refactor to dynamically load models
- Only prefer Parakeet for English (other languages don't detect reliably)
- Add `--vad-filter`, `--vad-threshold`, `--vad-min-speech-ms`, `--vad-min-silence-ms` (thanks @lmoe)
- Add `zeroconf` to Docker image

## 3.0.2

- Set `--data-dir /data` in Docker run script

## 3.0.1

- Fix model auto selection logic

## 3.0.0

- Add support for `sherpa-onnx` and Nvidia's parakeet model
- Add support for [GigaAM](https://github.com/salute-developers/GigaAM) for Russian via [`onnx-asr`](https://github.com/istupakov/onnx-asr)
- Add `--stt-library` to select speech-to-text library (deprecate `--use-transformers`)
- Default `--model` to "auto" (prefer parakeet)
- Add Docker build here
- Default `--language` to "auto"
- Add `--cpu-threads` for faster-whisper (@Zerwin)

## 2.5.0

- Add support for HuggingFace transformers Whisper models (--use-transformers)

## 2.4.0

- Add "auto" for model and beam size (0) to select values based on CPU

## 2.3.0

- Bump faster-whisper package to 1.1.0
- Supports model `turbo` for faster processing

## 2.2.0

- Bump faster-whisper package to 1.0.3

## 2.1.0

- Added `--initial-prompt` (see https://github.com/openai/whisper/discussions/963)

## 2.0.0

- Use faster-whisper PyPI package
- `--model` can now be a HuggingFace model like `Systran/faster-distil-whisper-small.en`

## 1.1.0

- Fix enum use for Python 3.11+
- Add tests and Github actions
- Bump tokenizers to 0.15
- Bump wyoming to 1.5.2

## 1.0.0

- Initial release
