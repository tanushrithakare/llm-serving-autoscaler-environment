FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860
EXPOSE 7860
CMD python -m uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}
