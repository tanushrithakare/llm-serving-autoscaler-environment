FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# HuggingFace Spaces uses port 7860; local Docker uses 8000
# The PORT env var controls which port the server listens on
ENV PORT=7860
EXPOSE 7860

# Run the server (reads PORT env var via our main entry point)
CMD ["python", "-m", "server.app"]
