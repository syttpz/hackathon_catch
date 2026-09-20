#!/bin/sh
set -eu
cd /opt/viam/trajectory-local
CONFIG=$(ls -1 /root/.viam/cached_cloud_config_*.json 2>/dev/null | head -1)
exec /opt/viam/trajectory-local-venv/bin/python -m old_files.red_ball.catch_throw --machine-config "$CONFIG" "$@"
