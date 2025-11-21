#!/usr/bin/env bash
set -euo pipefail

SERVICE="${SERVICE:-pipeline}"

case "$SERVICE" in
  pipeline)
    PIPE_ARGS=()
    if [[ -n "${DATA_PATH:-}" ]]; then
      PIPE_ARGS+=("--data-path" "${DATA_PATH}")
    fi
    exec python docker/run_crispdm_pipeline.py "${PIPE_ARGS[@]}" "$@"
    ;;
  streamlit)
    PORT="${PORT:-8501}"
    exec uvicorn docker.serve_api:app --host 0.0.0.0 --port "${PORT}" "$@"
    ;;
  bash)
    exec bash "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
