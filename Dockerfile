# syntax=docker/dockerfile:1
# --- build stage: has uv and compilers' worth of tooling; none of it reaches the final image ---
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# 1) dependencies first: this layer is rebuilt only when pyproject.toml or uv.lock change, not on every code edit
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) then our code
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
RUN uv sync --frozen --no-dev

# --- runtime stage ---
FROM python:3.12-slim
# non-root: a compromised process cannot write to the image or bind privileged ports
RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=build --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2).status == 200 else 1)"
# exec form: uvicorn is PID 1 and receives SIGTERM directly, so it shuts down gracefully
CMD ["uvicorn", "docuask.main:app", "--host", "0.0.0.0", "--port", "8000"]