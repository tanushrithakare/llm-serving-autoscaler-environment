# Use Python 3.11-slim for a lean, standard execution environment
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies globally (system-wide) 
# This ensures that 'python inference.py' or 'python -m uvicorn' works 
# regardless of how the validator invokes them.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the standard port
ENV PORT=7860
EXPOSE 7860

# Launch the unified forensic server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
