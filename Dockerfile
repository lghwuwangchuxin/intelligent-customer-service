# 🔴 3.12：生态里多个包封顶 <3.14，3.13 上装不全
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

WORKDIR /app

# 先装依赖再拷代码，改代码不会让依赖层失效
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY prompts ./prompts
COPY agents ./agents
COPY deploy ./deploy
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# 🔴 开发机 Intel Mac（torch 封顶 2.2.2）与生产 Linux（2.14）跑不出一致结果。
#    需要本地重排模型时用 --extra local-models，并注意这个差异。
