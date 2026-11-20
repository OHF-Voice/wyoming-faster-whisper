# Ubuntu 26.10 for its Mesa 26.2. qwen3-asr on --device vulkan runs its encoder
# and prompt prefill twice as fast on Mesa 26.2 as on 25.2 (Ubuntu 24.04) or
# 26.0 (Ubuntu 26.04), about 1.5x per command. 26.10 is not an LTS release (it
# is supported until July 2027): move on to 27.04 and then 28.04 LTS, and
# check that the Mesa version does not go backwards when you do.
FROM ubuntu:26.10
ARG TARGETARCH
ARG TARGETVARIANT

# Install faster-whisper
WORKDIR /usr/src

COPY ./pyproject.toml ./
# libvulkan1 and mesa-vulkan-drivers let qwen3-asr run on an AMD or Intel GPU
# with `--device vulkan` and `docker run --device /dev/dri` (see README). Mesa's
# lavapipe driver is removed: it is Vulkan on the CPU, and without a GPU passed
# in it would be picked as the "GPU" and run slower than the native CPU path.
# With it gone, a missing /dev/dri logs a warning and falls back to the CPU.
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        libvulkan1 \
        mesa-vulkan-drivers \
    && rm -f /usr/share/vulkan/icd.d/lvp_icd*.json \
    \
    && python3 -m venv .venv \
    && .venv/bin/pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && .venv/bin/pip3 install --no-cache-dir \
        --extra-index-url 'https://download.pytorch.org/whl/cpu' \
        'torch==2.15.0' \
    \
    && .venv/bin/pip3 install --no-cache-dir \
        --extra-index-url https://www.piwheels.org/simple \
        -e '.[zeroconf,transformers,sherpa,onnx-asr,qwen3-asr-vulkan,hass]' \
    \
    && rm -rf /var/lib/apt/lists/*

COPY ./ ./

# Keep every cache on the /data volume instead of the container's filesystem.
# Model files already go there (--download-dir defaults to the first --data-dir),
# but the libraries also write caches that no per-call argument reaches: the Xet
# chunk cache used during hub downloads is $HF_HOME/xet and can run to several
# GB, and torch/others fall back to $XDG_CACHE_HOME. Unset, all of that lands in
# $HOME/.cache - which is thrown away on every container recreate, and is an
# unwritable /.cache when the container runs as a uid with no passwd entry.
#
# These have to be environment variables: huggingface_hub reads them into
# constants at import time, so setting them from Python would be too late.
ENV HF_HOME=/data \
    XDG_CACHE_HOME=/data \
    MODELSCOPE_CACHE=/data/modelscope

EXPOSE 10300

# The server only starts listening once the model is loaded, which means
# downloading it on first run -- hence the long start period, during which
# failures don't count against --retries. --retries covers the other direction:
# transcription runs on the event loop, so a check can time out behind a long
# request without the server being unhealthy.
HEALTHCHECK --interval=30s --timeout=20s --start-period=5m --retries=3 \
    CMD ["/usr/src/.venv/bin/python3", "-m", "wyoming_faster_whisper.health_check"]

ENTRYPOINT ["bash", "docker_run.sh"]
