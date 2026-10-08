#!/bin/bash
# Retry an on-demand L40S (g6e.4xlarge, then 2xlarge) every 5 minutes for up to 8 hours; stop on first success.
HERE=$(cd "$(dirname "$0")" && pwd)
export MOEFY_BOX_NAME=${MOEFY_BOX_NAME:-jev-l40s} MOEFY_SSH_ALIAS=${MOEFY_SSH_ALIAS:-jevl}
unset MOEFY_SPOT
for i in $(seq 1 96); do
  for t in g6e.4xlarge g6e.2xlarge; do
    if "$HERE/gpu.sh" up $t 2>&1 | tail -1 | grep -q "^instance"; then
      "$HERE/gpu.sh" ssh-config >/dev/null; echo "$(date -u +%FT%TZ) LAUNCHED $t"; exit 0
    fi
  done
  echo "$(date -u +%FT%TZ) attempt $i: no capacity"
  sleep 300
done
echo "gave up"
