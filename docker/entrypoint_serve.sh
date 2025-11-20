#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

exec uvicorn docker.serve_api:app --host "${HOST}" --port "${PORT}"
