# Use Python 3.11-slim as the base for a clean, lean environment
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for UV and build-essential
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for high-speed dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Copy the unified project files
COPY . .

# Install the project and dependencies via uv
RUN uv sync

# Expose the official OpenEnv port
EXPOSE 7860

# Launch the unified forensic server
CMD ["uv", "run", "--python", "3.11", "server"]
