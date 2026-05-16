#!/usr/bin/env bash
# screenshot-app: capture a deterministic PNG of a locally-running web app via headless Chrome.
# usage: screenshot.sh [url] [output] [width] [height]
set -euo pipefail

URL="${1:-http://127.0.0.1:8000}"
OUTPUT="${2:-docs/screenshot.png}"
WIDTH="${3:-1200}"
HEIGHT="${4:-1600}"

CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ ! -x "$CHROME" ]; then
  echo "screenshot.sh: Chrome not found at $CHROME" >&2
  echo "Install Google Chrome from https://www.google.com/chrome/ or adjust this path." >&2
  exit 1
fi

if ! curl -s -o /dev/null --max-time 2 "$URL"; then
  echo "screenshot.sh: cannot reach $URL (is the server running?)" >&2
  exit 2
fi

mkdir -p "$(dirname "$OUTPUT")"
PROFILE_DIR="$(mktemp -d -t chrome-headless-screenshot)"
trap 'rm -rf "$PROFILE_DIR"' EXIT

echo "screenshot.sh: capturing $URL at ${WIDTH}x${HEIGHT} -> $OUTPUT"
"$CHROME" \
  --headless=new \
  --hide-scrollbars \
  --no-first-run \
  --no-default-browser-check \
  --user-data-dir="$PROFILE_DIR" \
  --window-size="${WIDTH},${HEIGHT}" \
  --virtual-time-budget=3000 \
  --screenshot="$OUTPUT" \
  "$URL" 2>/dev/null

if [ ! -s "$OUTPUT" ]; then
  echo "screenshot.sh: capture produced no output file" >&2
  exit 3
fi

echo "screenshot.sh: wrote $(ls -la "$OUTPUT" | awk '{print $5}') bytes to $OUTPUT"
