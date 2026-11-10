FROM debian:bookworm-slim
ARG TARGETARCH
ARG TARGETVARIANT

# Install faster-whisper
WORKDIR /usr/src

COPY ./pyproject.toml ./
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
    \
    && python3 -m venv .venv \
    && .venv/bin/pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && .venv/bin/pip3 install --no-cache-dir \
        --extra-index-url 'https://download.pytorch.org/whl/cpu' \
        'torch==2.13.0' \
    \
    && .venv/bin/pip3 install --no-cache-dir \
        --extra-index-url https://www.piwheels.org/simple \
        -e '.[zeroconf,transformers,sherpa,onnx-asr,qwen3-asr,hass]' \
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
