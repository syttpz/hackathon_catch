#!/bin/sh
set -eu
cd /opt/viam/trajectory-local
CONFIG=$(ls -1 /root/.viam/cached_cloud_config_*.json 2>/dev/null | head -1)
PY=/opt/viam/trajectory-local-venv/bin/python
if [ "${1:-}" = capture ]; then
  shift
  exec "$PY" -m motion.calibrate_cam2 capture --machine-config "$CONFIG" "$@"
fi
exec "$PY" -m motion.calibrate_cam2 "$@"
