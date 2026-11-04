# Lightweight image: vLLM OpenAI base + orbitkv built from source (release).
FROM vllm/vllm-openai:latest AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies for orbitkv (Rust toolchain, proto, Python headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    curl \
    git \
    libcli11-dev \
    liburing-dev \
    libssl-dev \
    pkg-config \
    protobuf-compiler \
    python3-dev \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain and maturin for wheel build
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable \
 && . "$HOME/.cargo/env" \
 && rustup component add clippy rustfmt \
 && python3 -m pip install --no-cache-dir maturin

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace/orbitkv
COPY . .

# Build release wheel and stage it for the runtime image
RUN ./scripts/build-wheel.sh --release \
 && cp target/wheels/orbitkv-*.whl /tmp/


FROM vllm/vllm-openai:latest

COPY --from=builder /tmp/orbitkv-*.whl /tmp/
RUN python3 -m pip install --no-cache-dir /tmp/orbitkv-*.whl \
 && rm /tmp/orbitkv-*.whl

WORKDIR /workspace
