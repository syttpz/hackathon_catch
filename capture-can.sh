#!/bin/sh
set -eu
cd /opt/viam/trajectory-local
exec /opt/viam/trajectory-local-venv/bin/python -m motion.capture_can --machine-config /root/.viam/cached_cloud_config_49d63d4f-4191-4d94-9e43-e379f846dd0e.json "$@"
