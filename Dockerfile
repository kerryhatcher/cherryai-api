# syntax=docker/dockerfile:1
FROM python:3.13-slim-bookworm AS base

COPY --from=ghcr.io/astral-sh/uv:0.11.29 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first so this layer is cached across code-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY . .
RUN uv sync --locked --no-dev

EXPOSE 8000

# The migration job (see .do/app.yaml) overrides this with
# `uv run alembic upgrade head` using the same image.
CMD ["uv", "run", "cherryai", "serve", "--host", "0.0.0.0", "--port", "8000"]
# force cache invalidation Sun Jul 26 05:30:33 PM EDT 2026
