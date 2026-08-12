#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 VIDEO OUTPUT" >&2
  exit 2
fi

video="$1"
output="$2"
mkdir -p "$(dirname "${output}")"
ffmpeg -hide_banner -loglevel error -y \
  -i "${video}" \
  -vf "select='eq(n,0)+eq(n,80)+eq(n,140)+eq(n,180)+eq(n,210)+eq(n,230)',scale=640:360,tile=3x2" \
  -frames:v 1 "${output}"
