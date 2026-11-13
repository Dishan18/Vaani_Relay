# One image for the API, the worker and the stand-in model service.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/hf FASTEMBED_CACHE_PATH=/app/.cache/fastembed
WORKDIR /app
COPY requirements.txt requirements-service.txt ./
RUN pip install -r requirements-service.txt
COPY . .
# Bake the embedding model and the library's vectors into the image: no download or embedding at start-up,
# and every replica has the identical index (same version string -> same cache keys).
RUN python -c "from relay.pipeline import load_production_config as c; from relay.dense import build_retriever as b; r = b(c()); r.search('warm up', 3)"
# Nothing is fetched at run time: the model files are in the image, so a replica can't stall on the Hub.
ENV HF_HUB_OFFLINE=1
RUN useradd --create-home --uid 10001 relay && chown -R relay /app
USER relay
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)" || exit 1
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
