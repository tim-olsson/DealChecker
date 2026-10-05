#!/usr/bin/env bash
# Starts the ADK web UI with the same configuration as the CLI.
# Usage:  bash scripts/web.sh              # local UI at http://127.0.0.1:8000
#         bash scripts/web.sh --port 8080  # extra args go to `adk web`
# Set TRACE_TO_CLOUD=true in .env to also export traces/metrics (--otel_to_cloud).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

args=()
[[ -n "${SESSION_SERVICE_URI:-}" ]] && args+=(--session_service_uri "${SESSION_SERVICE_URI}")
[[ -n "${MEMORY_SERVICE_URI:-}" ]] && args+=(--memory_service_uri "${MEMORY_SERVICE_URI}")
[[ "${TRACE_TO_CLOUD:-false}" == "true" ]] && args+=(--otel_to_cloud)

exec uv run adk web "${args[@]}" "$@" src
