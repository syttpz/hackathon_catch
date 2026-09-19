#!/bin/sh
set -eu
cd /opt/viam/trajectory-local
CONFIG=$(ls -1 /root/.viam/cached_cloud_config_*.json 2>/dev/null | head -1)
PY=/opt/viam/trajectory-local-venv/bin/python
if [ "${1:-}" = collect ]; then
  shift
  exec "$PY" -m motion.calibrate_cam2_pnp collect --machine-config "$CONFIG" "$@"
fi
exec "$PY" -m motion.calibrate_cam2_pnp "$@"
