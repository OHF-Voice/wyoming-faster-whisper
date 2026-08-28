# Wyoming Faster Whisper

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [faster-whisper](https://github.com/guillaumekln/faster-whisper/) speech to text system.

## Home Assistant Add-on

[![Show add-on](https://my.home-assistant.io/badges/supervisor_addon.svg)](https://my.home-assistant.io/redirect/supervisor_addon/?addon=core_whisper)

[Source](https://github.com/home-assistant/addons/tree/master/whisper)

## Local Install

Clone the repository and set up Python virtual environment:

``` sh
git clone https://github.com/rhasspy/wyoming-faster-whisper.git
cd wyoming-faster-whisper
script/setup
```

Run a server anyone can connect to:

```sh
script/run --model tiny-int8 --language en --uri 'tcp://0.0.0.0:10300' --data-dir /data --download-dir /data
```

The `--model` can also be a HuggingFace model like `Systran/faster-distil-whisper-small.en`

**NOTE**: Models are downloaded to the first `--data-dir` directory.

## Biasing Toward Your Home Assistant Names

Whisper has never heard of your thermostat. "What's the temperature of the Ecobee?"
comes back as "What's the temperature of the incubi?" — the acoustics were fine,
the model just has no reason to think that word exists.

Given a long-lived access token, the server reads the names in your home over the
Home Assistant websocket API and feeds them to the model as a prompt, which fixes
exactly that class of error:

```sh
script/run --uri 'tcp://0.0.0.0:10300' --data-dir /data \
    --hass-token "$TOKEN" --hass-api 'http://homeassistant.local:8123/api'
```

Requires the `hass` extra:

```sh
pip install 'wyoming-faster-whisper[hass]'
```

It collects the names of **conversation-exposed** entities and their aliases, plus
your area and floor names — the names a speaker can actually say. Nothing else is
read, and no service is ever called.

The fetch is free in latency terms: it starts when the audio starts, while the
speaker is still talking, and the names are ready by the time the audio stops.
Home Assistant being slow or unreachable only costs freshness — the previous names
are used, or none at all, and the transcript still comes back.

| Option | Default | Purpose |
| --- | --- | --- |
| `--hass-token` | | Long-lived access token. Enables everything above. Falls back to `HASS_TOKEN`, or `HASS_TOKEN_FILE` to read it from a file (docker secrets). |
| `--hass-api` | `http://homeassistant.local:8123/api` | Where to find Home Assistant. Falls back to `HASS_API`, or `HASS_API_FILE` to read it from a file (docker secrets). |
| `--hass-refresh-seconds` | `0` | Minimum seconds between refreshes. `0` refreshes every utterance, so a rename takes effect immediately. |
| `--hass-prompt-max-tokens` | `200` | Token budget for names. Whisper's hard cap is 223 and quality falls off before it. |
| `--hass-prompt-timeout` | `1.0` | How long to wait on an unfinished refresh before transcribing with the names already on hand. |

`--hass-token` and `--hass-api` also fall back to environment variables —
`HASS_TOKEN` / `HASS_API` — so the token doesn't have to live in a command-line
argument. Each also has a `_FILE` variant (`HASS_TOKEN_FILE`, `HASS_API_FILE`)
that names a file to read the value from instead, the convention Docker Compose
and Swarm secrets use: a secret is mounted as a file under `/run/secrets/`, not
delivered as an env var, so the env var instead points at the file. Precedence
is CLI flag, then `_FILE`, then the plain env var — an explicit flag always
wins.

```yaml
services:
  whisper:
    image: rhasspy/wyoming-whisper
    command: ["--model", "tiny-int8", "--language", "en"]
    ports:
      - "10300:10300"
    volumes:
      - ./data:/data
    environment:
      HASS_TOKEN_FILE: /run/secrets/hass_token
      HASS_API_FILE: /run/secrets/hass_api
    secrets:
      - hass_token
      - hass_api

secrets:
  hass_token:
    file: ./secrets/hass_token.txt
  hass_api:
    file: ./secrets/hass_api.txt
```

A large home has more names than the budget holds. They are added in priority
order — areas, floors, entity names, then aliases — and cut off when the budget
runs out; run with `--debug` to see how many were dropped and the exact prompt
used. `--initial-prompt` still works and is kept at the front of the prompt, ahead
of anything discovered from Home Assistant.

This biases `faster-whisper` and `qwen3-asr`, the backends that take a prompt.
Others ignore it.

### Prompt cost on qwen3-asr

For `qwen3-asr` the prompt is not free: the model has to read it before it starts
decoding, at roughly 2.8ms per token, so a 50-name list can double the time for a
short command.

The default model avoids this. The prompt sits ahead of the audio in the chat
template, so its state depends only on the prompt and is computed once, then
reused for every later utterance. The layout is chosen from the files present, so
older model directories with `decoder_init`/`decoder_step` keep working as before
— pass `--model rhasspy/qwen3-asr-0.6b-onnx-int4` to use one.

Measured on a Pi 5 (4 threads, 3.2s command, 50 names):

| | split | merged (default) |
| --- | --- | --- |
| latency | 3.42s | 2.20s |
| peak RSS | 2.25 GB | 1.55 GB |
| on disk | 1407 MB | 785 MB |

The latency win is for short commands. Long-form audio gains little (~1.04x on a
30s clip), because the cached prompt is a small share of that work — though the
memory saving grows with length.

Accuracy is unchanged: on LibriSpeech test-other (n=200) the two produce
byte-identical transcripts with no prompt (5.35% WER for both), and 5.33% vs
5.43% with a 50-name prompt.

## Docker Image

``` sh
docker run -it -p 10300:10300 -v /path/to/local/data:/data rhasspy/wyoming-whisper \
    --model tiny-int8 --language en
```

**NOTE**: Models are downloaded to `/data`, so make sure this points to a Docker volume.

[Source](https://github.com/rhasspy/wyoming-addons/tree/master/whisper)
