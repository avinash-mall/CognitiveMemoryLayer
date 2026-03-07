#!/usr/bin/env sh
# EXAMPLE_META: {"name":"api_curl_examples","kind":"shell","summary":"Direct curl examples for health, write, read, turn, forget, stats, and sessions.","requires_api":true,"requires_api_key":true,"requires_base_url":true,"requires_admin_key":false,"requires_embedded":false,"requires_openai":false,"requires_anthropic":false,"interactive":false,"timeout_sec":60}

set -eu

if [ -f ".env" ]; then
  tmp_env="$(mktemp)"
  trap 'rm -f "$tmp_env"' EXIT HUP INT TERM
  tr -d '\r' < .env > "$tmp_env"
  set -a
  . "$tmp_env"
  set +a
fi

if [ -z "${CML_BASE_URL:-}" ]; then
  echo "Set CML_BASE_URL in the repo root .env before running this example."
  exit 1
fi

if [ -z "${CML_API_KEY:-}" ]; then
  echo "Set CML_API_KEY in the repo root .env before running this example."
  exit 1
fi

BASE_URL="${CML_BASE_URL%/}"
case "${BASE_URL}" in
  */api/v1) API_URL="${BASE_URL}" ;;
  *) API_URL="${BASE_URL}/api/v1" ;;
esac

echo "# Health Check"
curl -s "${API_URL}/health" -H "X-API-Key: ${CML_API_KEY}"
printf "\n\n"

echo "# Write Memory"
curl -s -X POST "${API_URL}/memory/write" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${CML_API_KEY}" \
  -d '{"content":"User likes tea and works in product engineering.","session_id":"curl-demo","context_tags":["preferences","career"]}'
printf "\n\n"

echo "# Read Memory (packet)"
curl -s -X POST "${API_URL}/memory/read" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${CML_API_KEY}" \
  -d '{"query":"user preferences","max_results":5,"format":"packet"}'
printf "\n\n"

echo "# Read Memory (llm_context)"
curl -s -X POST "${API_URL}/memory/read" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${CML_API_KEY}" \
  -d '{"query":"summarize the user","format":"llm_context"}'
printf "\n\n"

echo "# Process Turn"
curl -s -X POST "${API_URL}/memory/turn" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${CML_API_KEY}" \
  -d '{"user_message":"What drink do I prefer?","session_id":"curl-demo","max_context_tokens":500}'
printf "\n\n"

echo "# Forget Memory"
curl -s -X POST "${API_URL}/memory/forget" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${CML_API_KEY}" \
  -d '{"query":"tea","action":"archive"}'
printf "\n\n"

echo "# Memory Stats"
curl -s "${API_URL}/memory/stats" -H "X-API-Key: ${CML_API_KEY}"
printf "\n\n"

echo "# Create Session"
curl -s -X POST "${API_URL}/session/create" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${CML_API_KEY}" \
  -d '{"ttl_hours":24}'
printf "\n"
