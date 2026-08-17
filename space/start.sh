#!/usr/bin/env bash
set -euo pipefail

# JupyterLab serves under /lab so the reverse proxy in server.py can forward to
# it without rewriting asset paths. Token auth is off because the Space is public
# and unauthenticated anyway; XSRF is disabled for the same reason (the proxy
# would otherwise have to forward the cookie/header pair).
jupyter lab \
  --ip=127.0.0.1 --port="${LAB_PORT:-8888}" --no-browser \
  --ServerApp.base_url=/lab \
  --ServerApp.token= --ServerApp.password= \
  --ServerApp.disable_check_xsrf=True \
  --ServerApp.allow_origin='*' \
  --ServerApp.root_dir="${TORCHLEET_ROOT:-/torchleet}" \
  --allow-root &

# Wait for the lab to answer before serving, so the first click on
# "Solve in JupyterLab" does not land on a connection error.
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${LAB_PORT:-8888}/lab/api/status" >/dev/null && break
  sleep 1
done

exec uvicorn server:api --host 0.0.0.0 --port "${PORT:-7860}"
