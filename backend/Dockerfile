FROM python:3.11-slim

WORKDIR /app

# System build tools + curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU torch via extra index)
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy entire project into a nested 'bess_rl' folder
# This ensures that 'from bess_rl.xxx' imports work from /app
COPY . /app/bess_rl

# Make /app importable
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "bess_rl/backend/main.py"]
