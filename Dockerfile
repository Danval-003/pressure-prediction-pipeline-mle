# syntax=docker/dockerfile:1

########################
# Etapa 1: Builder
########################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=1

WORKDIR /app

# Copiamos TODO solo en el builder
COPY . /app

# Instalar solo lo necesario para compilar dependencias
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Crear (opcional) un venv para aislar dependencias
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Instalar paquetes locales y dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        ./packages/data_preparation \
        "./packages/data_understanding[app]" \
        ./packages/modeling \
        ./packages/evaluating \
        ./packages/deployment \
        fastapi \
        "uvicorn[standard]==0.30.0"

########################
# Etapa 2: Runtime
########################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=1 \
    SERVICE=pipeline \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Solo copiamos el entorno de Python ya listo desde el builder
COPY --from=builder /opt/venv /opt/venv

# Ahora sí copiamos SOLO el código que necesitamos para correr
# (Idealmente limitar esto con .dockerignore)
COPY . /app

RUN chmod +x docker/entrypoint.sh

VOLUME ["/data", "/models", "/artifacts"]

EXPOSE 8501

CMD ["uvicorn", "docker.serve_api:app", "--host", "0.0.0.0", "--port", "8501"]
