FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/app/.fastembed

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the ONNX embedder into the image so cold starts never download it.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

# Data bundle (built by data_build/build_data.py, published as a GitHub release).
ARG DATA_URL=https://github.com/tyxgx/streampulse-demo/releases/download/data-v1
RUN mkdir data && python -c "import urllib.request as u; b='$DATA_URL'; [u.urlretrieve(f'{b}/{f}', f'data/{f}') for f in 'chunks_q8.npy norms_sq.npy meta.json chunks.sqlite country.parquet artist.parquet label.parquet song.parquet'.split()]"

COPY server.py retrieval.py sql_layer.py llm.py ./
COPY static static

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
