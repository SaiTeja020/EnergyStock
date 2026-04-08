FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set PYTHONPATH so all internal imports resolve correctly
# The project root is /app, so 'import openenv...' etc. resolves
ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

# Run the unified FastAPI server directly
CMD ["python", "backend/main.py"]
