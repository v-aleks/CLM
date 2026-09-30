#!/usr/bin/env bash
# usage: wait_for_url.sh URL TIMEOUT_SECS [INTERVAL_SECS]
# Polls `URL` with curl until it returns 2xx, or fails after TIMEOUT_SECS.
# Exit code: 0 on success, 1 on timeout.
set -eu

url="${1:?usage: wait_for_url.sh URL TIMEOUT_SECS [INTERVAL_SECS]}"
timeout="${2:-180}"
interval="${3:-2}"

if ! [[ "$timeout" =~ ^[0-9]+$ ]] || ! [[ "$interval" =~ ^[0-9]+$ ]]; then
  echo "wait_for_url: TIMEOUT_SECS and INTERVAL_SECS must be integers" >&2
  exit 2
fi

deadline=$(( SECONDS + timeout ))
attempt=0
while (( SECONDS < deadline )); do
  attempt=$((attempt + 1))
  if curl -fsS -o /dev/null --max-time 5 "$url" 2>/dev/null; then
    echo "wait_for_url: $url ready (after ${attempt} attempt(s))"
    exit 0
  fi
  sleep "$interval"
done

echo "wait_for_url: $url not ready after ${timeout}s (${attempt} attempt(s))" >&2
exit 1
