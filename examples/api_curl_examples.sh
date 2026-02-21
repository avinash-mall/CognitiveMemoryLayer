#!/bin/bash
# Cognitive Memory Layer — curl examples (direct API, no py-cml)
# Set AUTH__API_KEY and optionally CML_BASE_URL before running.
# Prerequisites: API server running (docker compose -f docker/docker-compose.yml up api)
# Optional: pipe output through jq for pretty-print

BASE_URL="${CML_BASE_URL:-http://localhost:8000}"
API_URL="${BASE_URL%/}/api/v1"
KEY="${AUTH__API_KEY:?Set AUTH__API_KEY}"

echo "# Health Check"
curl -s "$API_URL/health" -H "X-API-Key: $KEY"
echo -e "\n"

echo "# Write Memory"
curl -s -X POST "$API_URL/memory/write" \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"content": "User likes pizza", "session_id": "curl-demo", "context_tags": ["preferences"]}'
echo -e "\n"

echo "# Read Memory (packet)"
curl -s -X POST "$API_URL/memory/read" \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"query": "food preferences", "max_results": 5, "format": "packet"}'
echo -e "\n"

echo "# Read Memory (llm_context)"
curl -s -X POST "$API_URL/memory/read" \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"query": "food preferences", "format": "llm_context"}'
echo -e "\n"

echo "# Process Turn"
curl -s -X POST "$API_URL/memory/turn" \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"user_message": "What do I like?", "session_id": "curl-demo"}'
echo -e "\n"

echo "# Forget Memory"
curl -s -X POST "$API_URL/memory/forget" \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"query": "pizza", "action": "archive"}'
echo -e "\n"

echo "# Memory Stats"
curl -s "$API_URL/memory/stats" -H "X-API-Key: $KEY"
echo -e "\n"

echo "# Create Session"
curl -s -X POST "$API_URL/session/create" \
  -H "Content-Type: application/json" -H "X-API-Key: $KEY" \
  -d '{"ttl_hours": 24}'
