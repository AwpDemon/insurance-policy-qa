FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ipqa ./ipqa
COPY app ./app
COPY data/processed ./data/processed
# Pre-download the reranker weights at build time so the container starts offline.
RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder as T; T('Xenova/ms-marco-MiniLM-L-12-v2')"

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
