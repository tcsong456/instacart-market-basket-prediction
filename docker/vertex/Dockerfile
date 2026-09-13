# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/app \
    XDG_CACHE_HOME=/home/app/.cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "${APP_GID}" app \
    && useradd \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        --create-home \
        --shell /usr/sbin/nologin \
        app \
    && mkdir -p /home/app/.cache \
    && chown -R app:app /home/app

COPY --chown=root:root pyproject.toml README.md /app/
COPY --chown=root:root src/ /app/src/

RUN python -m pip install --upgrade "pip<27" \
    && python -m pip install .

COPY --chown=root:root docker/vertex/entrypoint.sh \
    /usr/local/bin/vertex-entrypoint

RUN chmod 0555 /usr/local/bin/vertex-entrypoint

USER app

ENTRYPOINT ["/usr/local/bin/vertex-entrypoint"]