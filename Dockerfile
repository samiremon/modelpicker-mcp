FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
WORKDIR /app
RUN useradd --create-home --uid 1000 modelpicker
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --extra local --no-dev --no-install-project
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --extra local --no-dev \
    && install -d -o modelpicker -g modelpicker -m 700 /home/modelpicker/.modelpicker
USER modelpicker
ENV MODELPICKER_HOME=/home/modelpicker/.modelpicker \
    TRANSPORT=streamable-http HOST=0.0.0.0 PORT=7860 PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=3)"
CMD ["modelpicker", "serve"]
