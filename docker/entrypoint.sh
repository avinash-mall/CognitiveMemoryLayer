#!/usr/bin/env bash
# CML Docker entrypoint — hands off to the container CMD (e.g. uvicorn, pytest).
set -e
exec "$@"
