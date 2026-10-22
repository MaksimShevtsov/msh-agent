FROM ghcr.io/astral-sh/uv:0.9.5-python3.13-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 PYDANTIC_AI_NO_BANNER=1 \
    BRAIN_STATE_DIR=/app/state

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runs as root on purpose: state/ is a host bind mount (Docker Desktop shows it as
# root-owned), and the overflow store refuses a root it does not own. The agent
# registers no shell or file tools, so the model gets no in-container access.
VOLUME /app/state

ENTRYPOINT ["uv", "run", "--no-sync", "brain-agent"]
CMD ["loop"]
