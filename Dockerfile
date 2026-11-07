# syntax=docker/dockerfile:1
#
# text2icd's HTTP service and web UI (task 10.4), with the release's graph, L2's vectors and the
# model they were embedded with baked in, so `docker run` needs no network.
#
#   docker build -t text2icd .
#   docker run --rm -p 127.0.0.1:8000:8000 text2icd          # http://127.0.0.1:8000
#   docker run --rm text2icd code "type 2 DM with diabetic CKD stage 3a"
#
# .github/workflows/docker.yml pushes it to ghcr.io/a2psilva/text2icd on version tags. The build
# context holds the files .dockerignore allows and nothing else: never data/ or .env.

ARG PYTHON=3.12
ARG UV=0.13.7
ARG RELEASE=fy2027

FROM ghcr.io/astral-sh/uv:${UV} AS uv
FROM --platform=$BUILDPLATFORM ghcr.io/astral-sh/uv:${UV} AS build-uv

# -- The environment: text2icd with its `api` and `embeddings` extras, for the target platform.
FROM python:${PYTHON}-slim-trixie AS app
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/text2icd/venv
WORKDIR /src
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project --extra api --extra embeddings
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra api --extra embeddings

# -- The data, made once on the build machine: it is the same for every platform. The graph and
# vectors are downloaded when this text2icd lists the release's data release; otherwise the graph
# is built from the CMS files (fetched from cms.gov and verified) and the vectors embedded here.
FROM --platform=$BUILDPLATFORM python:${PYTHON}-slim-trixie AS data
ARG RELEASE
COPY --from=build-uv /uv /bin/uv
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/text2icd/venv \
    PATH=/opt/text2icd/venv/bin:$PATH \
    TEXT2ICD_DATA_DIR=/opt/text2icd/data \
    TEXT2ICD_RELEASE=${RELEASE} \
    HF_HOME=/opt/text2icd/hf \
    ORT_DISABLE_TELEMETRY=1
WORKDIR /src
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project --extra embeddings
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra embeddings
RUN text2icd setup --l2 \
    && if [ ! -f "$TEXT2ICD_DATA_DIR/processed/text2icd-graph-$RELEASE.vectors.npy" ]; then \
           text2icd graph embed && text2icd setup --l2; \
       fi \
    && text2icd data where

# -- The image: the environment, the graph and vectors, the model; a user without privileges.
FROM python:${PYTHON}-slim-trixie
ARG RELEASE
LABEL org.opencontainers.image.title="text2icd" \
      org.opencontainers.image.description="ICD-10-CM codes for diagnosis phrases, with the Index path to each code" \
      org.opencontainers.image.source="https://github.com/a2psilva/text2icd" \
      org.opencontainers.image.licenses="MIT"
RUN useradd --system --create-home --uid 10001 text2icd
COPY --from=app /opt/text2icd/venv /opt/text2icd/venv
COPY --from=data /opt/text2icd/data/processed /opt/text2icd/data/processed
COPY --from=data /opt/text2icd/hf /opt/text2icd/hf
ENV PATH=/opt/text2icd/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TEXT2ICD_DATA_DIR=/opt/text2icd/data \
    TEXT2ICD_RELEASE=${RELEASE} \
    HF_HOME=/opt/text2icd/hf \
    HF_HUB_OFFLINE=1 \
    ORT_DISABLE_TELEMETRY=1
USER text2icd
WORKDIR /home/text2icd
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]
ENTRYPOINT ["text2icd"]
# 0.0.0.0 inside the container; publish the port on 127.0.0.1 to keep it on this machine.
CMD ["serve", "--host", "0.0.0.0"]
