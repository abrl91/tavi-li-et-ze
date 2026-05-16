---
name: screenshot-app
description: Capture a deterministic PNG screenshot of a locally-running web application via headless Chrome. Use when generating or refreshing README screenshots, verifying that a UI change rendered correctly, or producing before/after design comparison images. Requires Google Chrome installed at the standard macOS path. For motion captures (GIF / video), see the screen-recording section in this file; ffmpeg is required for GIF conversion.
---

# screenshot-app

Capture a PNG screenshot of a locally-running web app using headless Chrome. Deterministic, no GUI interaction, no extra Python deps.

## When to use

- README needs an updated screenshot of the app
- Verifying that a UI change rendered correctly without opening a browser
- Producing comparison images before and after a design tweak
- Any case where a deterministic still capture of a web page is needed

NOT for: capturing native desktop apps (use macOS `screencapture` directly), capturing video (see Screen recording below), or screenshotting external sites (just open them in a browser).

## Prerequisites

- Google Chrome installed at `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` (default macOS path)
- The target web app must be running and reachable at the URL passed in

## Usage

```bash
./.claude/skills/screenshot-app/screenshot.sh <url> [output] [width] [height]
```

Defaults if omitted: `http://127.0.0.1:8000` to `docs/screenshot.png` at `1200x1600`.

## Full workflow with auto server start

```bash
# 1. start the app server in background
make api > /tmp/screenshot-server.log 2>&1 &
SERVER_PID=$!

# 2. wait for it to come up
until curl -s http://127.0.0.1:8000 >/dev/null 2>&1; do sleep 0.2; done

# 3. capture (sleep briefly to let fonts render)
sleep 1
./.claude/skills/screenshot-app/screenshot.sh \
  http://127.0.0.1:8000/briefs/2026-05-15 \
  docs/screenshot.png \
  1200 1600

# 4. clean up
kill $SERVER_PID 2>/dev/null
```

## Flags inside screenshot.sh

| Chrome flag | Purpose |
|---|---|
| `--headless=new` | Modern headless mode (Chrome 109+) |
| `--hide-scrollbars` | Suppress the scrollbar artifact on the right edge |
| `--virtual-time-budget=3000` | Give the page 3 simulated seconds to settle (font load, JS) |
| `--user-data-dir=/tmp/...` | Isolated profile so headless does not conflict with the user's running Chrome |
| `--no-first-run` | Skip Chrome's first-run wizard |
| `--window-size=W,H` | Viewport dimensions |
| `--screenshot=path` | Write screenshot to path |

## Screen recording (manual upgrade)

`screenshot.sh` only does stills. For motion, two paths:

**MP4 inline on GitHub** (preferred):
1. `screencapture -V 8 -R x,y,w,h docs/demo.mov` records 8 seconds of the rectangle (find coords via `screencapture -i` first to learn them, or just record the full screen and crop later)
2. Drag the .mov or convert to .mp4 then drag into a GitHub issue/PR draft. GitHub uploads it and gives back a `https://github.com/user-attachments/...` URL that plays inline in markdown.

**GIF inline** (works in any markdown viewer):
1. Record as above
2. `ffmpeg -i demo.mov -vf "fps=15,scale=1000:-1:flags=lanczos" -loop 0 docs/demo.gif`
3. Reference in README: `![demo](docs/demo.gif)`

ffmpeg is not installed on this machine. Install via `brew install ffmpeg` if GIF conversion is needed.
