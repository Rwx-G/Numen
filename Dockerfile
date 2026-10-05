# syntax=docker/dockerfile:1
# Single GPU image for Numen. By default it runs the plain kernel and
# self-improvement is DORMANT: no supervisor, NUMEN_SRC_DIR unset, normal seccomp.
# To enable self-improvement, run this same image under the supervisor entrypoint
# with NUMEN_SRC_DIR set and seccomp unconfined (docker-compose.selfimprove.yml).
# It therefore carries the full toolchain (CUDA devel + Rust), a writable source
# checkout, the supervisor, and bubblewrap, on top of GPU inference. Deliberately
# large - this is a single-user, GPU-host deployment, not a portable runtime.

# --- Node: node + npm for the `claude -p` CLI. Digest-pinned 2026-10-05. ---
FROM node:24-bookworm-slim@sha256:0e0ff40c39bc087845bfb27465a0df4ea419520094bc35842ff83dd8cbe6f9b6 AS node

# --- Builder: CUDA 13.3 devel + Rust, build the kernel (cuda) and the supervisor.
# No cache mount, so the compiled target and cargo home persist into the image for
# an incremental self-rebuild at runtime. CUDA_COMPUTE_CAP=120 forces candle to
# compile sm_120 (Blackwell / RTX 5090) kernels with no GPU present at build time.
# The CUDA minor is capped by mistralrs-quant, whose build.rs refuses any toolkit
# outside its SUPPORTED_CUDA_TOOLKIT_VERSIONS list (13.3 is the newest in v0.9.4),
# and by the host driver (nvidia-smi "CUDA UMD Version" must be at least as high).
# Digest-pinned 2026-10-05; re-pin deliberately and re-verify on a base update. ---
FROM nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87 AS builder
ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:/usr/local/cuda/bin:${PATH} \
    CUDA_COMPUTE_CAP=120
# hadolint ignore=DL3008,DL4006  # unpinned toolchain apt + the official rustup installer pipe, taken from the base image
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      curl \
      git \
      pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
       | sh -s -- -y --default-toolchain 1.99.0 --profile minimal \
    && rustup component add clippy
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY src ./src
RUN cargo build --release --bin numen --features cuda \
    && cargo build --release -p numen-supervisor \
    && cp target/release/numen /numen \
    && cp target/release/numen-supervisor /numen-supervisor

# --- Final: CUDA 13.3 devel (keeps nvcc so a self-deploy can rebuild the kernel
# with the cuda feature) + the Rust toolchain + node/claude + bubblewrap. ---
FROM nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87
ARG IMAGE_REVISION="dev"
ARG IMAGE_CREATED="unknown"
LABEL org.opencontainers.image.title="Numen" \
      org.opencontainers.image.description="Self-hosted causal-reasoning AI agent (GPU, self-improvement-capable)" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.revision="${IMAGE_REVISION}" \
      org.opencontainers.image.created="${IMAGE_CREATED}" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:/usr/local/cuda/bin:/usr/local/bin:${PATH} \
    CUDA_COMPUTE_CAP=120 \
    HOME=/home/numen \
    NODE_ENV=production \
    NUMEN_DB_PATH=/data/numen.db \
    NUMEN_BIND=0.0.0.0:8080 \
    NUMEN_KERNEL_BIN=/app/numen \
    NUMEN_BASE_BRANCH=main \
    NUMEN_DEPLOY_MARKER=/app/deploy.request \
    NUMEN_WAKEUP_MARKER=/app/wakeup \
    NUMEN_KILL_SWITCH=/app/STOP

# build-essential/git/pkg-config let the kernel rebuild itself; bubblewrap
# sandboxes that build gate (no network, read-only root).
# hadolint ignore=DL3008  # unpinned: take the base image's versions, matching the toolchain apt convention
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      bubblewrap \
      ca-certificates \
      git \
      pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Node 24 + npm + the Claude CLI for `claude -p`, authenticated at runtime via
# CLAUDE_CODE_OAUTH_TOKEN. Pinned 2026-10-05; re-pin after reviewing the changelog.
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \
    && npm install -g @anthropic-ai/claude-code@2.1.289 \
    && npm cache clean --force

# The Rust toolchain and cargo home (registry + git deps) from the builder, owned
# by the runtime user so a self-rebuild can write the registry and target.
COPY --from=builder --chown=10001:10001 /opt/cargo /opt/cargo
COPY --from=builder /opt/rustup /opt/rustup

RUN groupadd -r -g 10001 numen \
    && useradd -r -u 10001 -g numen -d /home/numen numen \
    && mkdir -p /home/numen /data /app \
    && chown -R numen:numen /home/numen /data /app

# The kernel + supervisor binaries, and a writable source checkout the supervisor
# rebuilds from (only wired when NUMEN_SRC_DIR is set, i.e. in self-improve mode).
COPY --from=builder --chown=numen:numen /numen /app/numen
COPY --from=builder --chown=numen:numen /numen-supervisor /app/numen-supervisor
COPY --chown=numen:numen Cargo.toml Cargo.lock /app/src/
COPY --chown=numen:numen crates /app/src/crates
COPY --chown=numen:numen src /app/src/src
COPY --from=builder --chown=numen:numen /build/target /app/src/target

USER 10001
WORKDIR /app

# Git-init the checkout so the workshop can branch and commit (the build context
# excludes .git, so this is a fresh history). target/ is ignored so the multi-GB
# build cache is never committed.
RUN printf 'target/\n' > /app/src/.gitignore \
    && git config --global user.email "numen@local" \
    && git config --global user.name "Numen" \
    && git config --global --add safe.directory /app/src \
    && git -C /app/src init -q -b main \
    && git -C /app/src add -A \
    && git -C /app/src commit -q -m "self-improvement baseline"

EXPOSE 8080
# Long start period: loading a multi-GB model on startup precedes serving.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD node -e "require('http').get('http://127.0.0.1:8080/health',r=>process.exit(r.statusCode===200?0:1)).on('error',()=>process.exit(1))"

# Default: the plain kernel. Self-improvement is dormant (no supervisor watching
# the deploy marker, NUMEN_SRC_DIR unset). The self-improve compose overrides the
# entrypoint to /app/numen-supervisor and sets NUMEN_SRC_DIR + NUMEN_BUILD_FEATURES.
ENTRYPOINT ["/app/numen"]
