#!/usr/bin/env bash
cd /usr/src

# Defaults for the image. Each is dropped when the matching WYO_WHISPER_* env
# var is set, so `docker run -e WYO_WHISPER_URI=...` is not fighting an argument
# that would always win over it.
#
# STT_DEVICE is set to "cuda" by Dockerfile.gpu and unset in the CPU image.
defaults=()
[ -n "${WYO_WHISPER_URI}${WYO_WHISPER_URI_FILE}" ] ||
    defaults+=(--uri 'tcp://0.0.0.0:10300')
[ -n "${WYO_WHISPER_DATA_DIR}${WYO_WHISPER_DATA_DIR_FILE}" ] ||
    defaults+=(--data-dir '/data')
[ -n "${WYO_WHISPER_DEVICE}${WYO_WHISPER_DEVICE_FILE}" ] ||
    defaults+=(--device "${STT_DEVICE:-cpu}")

# "$@" comes last so an explicit argument from the command line overrides both.
exec .venv/bin/python3 -m wyoming_faster_whisper "${defaults[@]}" "$@"
