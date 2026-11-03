FROM python:3.12-slim AS builder

WORKDIR /build

COPY packaging/requirements.txt ./

RUN pip install --no-cache-dir --target=/deps -r requirements.txt


FROM python:3.12-slim AS runtime

RUN groupadd --gid 10001 ml \
    && useradd --uid 10001 --gid ml \
       --no-create-home --shell /usr/sbin/nologin ml

WORKDIR /app

COPY --from=builder /deps /deps

ENV PYTHONPATH=/deps \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY dataset.yaml ./
COPY src/__init__.py src/dataset_config.py src/
COPY packaging/app/__init__.py packaging/app/serve.py packaging/app/
COPY packaging/model/model.onnx packaging/model/metadata.json packaging/model/

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)"]

ARG VERSION=1.0.0
ARG MODEL_VERSION=unknown

LABEL org.opencontainers.image.title="scoring-api" \
      org.opencontainers.image.version="${VERSION}" \
      mlflow.model_name="scoring-model" \
      mlflow.model_alias="champion" \
      mlflow.model_version="${MODEL_VERSION}"

CMD ["python", "-m", "uvicorn", "app.serve:app", "--app-dir", "packaging", "--host", "0.0.0.0", "--port", "8000"]
