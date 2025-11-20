# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=1 \
    SERVICE=pipeline

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        ./packages/data_preparation \
        "./packages/data_understanding[app]" \
        ./packages/modeling \
        ./packages/evaluating \
        ./packages/deployment && \
    pip cache purge

RUN chmod +x docker/entrypoint.sh

VOLUME ["/data", "/models", "/artifacts"]

EXPOSE 8501

ENTRYPOINT ["bash", "docker/entrypoint.sh"]
