#!/usr/bin/env bash
# Send the local commits that the Linux box does not have yet, without pushing to GitHub.
# (A git bundle carries commits as a file, so the GitHub pre-push account prompt is never involved.)
# Usage: scripts/sync-box.sh [ssh-host] [directory-on-the-box]
set -euo pipefail
HOST="${1:-pricing-server}"
DIR="${2:-tech-challenge-04}"
BASE="$(ssh "$HOST" "cd $DIR && git rev-parse HEAD")"
git bundle create box.bundle "$BASE..main" >/dev/null
scp -q box.bundle "$HOST:$DIR/box.bundle"
rm -f box.bundle
ssh "$HOST" "cd $DIR && git fetch -q box.bundle main && git reset -q --hard FETCH_HEAD && rm -f box.bundle && git log --oneline | head -1"
