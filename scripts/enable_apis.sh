#!/usr/bin/env bash
# Enables every Google Cloud API DealChecker uses (idempotent; already-enabled
# APIs are a no-op). Run once per project before the other scripts.
#
#   scripts/enable_apis.sh [PROJECT_ID]     (default: $GOOGLE_CLOUD_PROJECT)
#
# Enabling APIs is free; usage is billed per API.
set -euo pipefail

PROJECT="${1:-${GOOGLE_CLOUD_PROJECT:?Pass a project id or set GOOGLE_CLOUD_PROJECT}}"

APIS=(
  aiplatform.googleapis.com     # Agent Engine: sessions + Memory Bank
  modelarmor.googleapis.com     # Model Armor guardrail templates
  logging.googleapis.com        # Cloud Logging (structured logs)
  telemetry.googleapis.com      # OpenTelemetry ingestion (trace export endpoint)
  cloudtrace.googleapis.com     # Cloud Trace (viewing traces)
  observability.googleapis.com  # Observability buckets; without it spans are dropped ("_Trace bucket not found")
)

export GOOGLE_API_USE_CLIENT_CERTIFICATE="${GOOGLE_API_USE_CLIENT_CERTIFICATE:-false}"

echo "Enabling APIs in project ${PROJECT}:"
printf '  %s\n' "${APIS[@]}"
gcloud services enable "${APIS[@]}" --project "${PROJECT}"
echo "Done."
