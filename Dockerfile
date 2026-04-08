FROM python:3.11-slim

# Install uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app/bess_rl

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking
ENV UV_LINK_MODE=copy

# System build tools + curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better caching
# We copy only the files needed for dependency resolution
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy entire project
COPY . .

# Sync the project (installs the current package if applicable)
RUN uv sync --frozen --no-dev

# Ensure the app can find the bess_rl package and the venv
ENV PYTHONPATH=/app
ENV PATH="/app/bess_rl/.venv/bin:$PATH"

EXPOSE 8000

# Run the backend main script
# Since WORKDIR is /app/bess_rl, we run backend/main.py
CMD ["python", "backend/main.py"]
