#!/usr/bin/env bash
# setup_thirdparty.sh — initialize ComfyUI-WanNodeExperiments study material.
#
# Clones the two foundational Wan 2.2 wrappers into third_party/ for STUDY ONLY.
# Nothing in this pack imports from third_party/ (see ABSOLUTE RULE #2); the
# directory is gitignored. Credits: Kijai & wuwukaka.
#
# Usage:
#   bash setup_thirdparty.sh           # shallow clone (default)
#   FULL=1 bash setup_thirdparty.sh    # full history
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TP="$HERE/third_party"
mkdir -p "$TP"

DEPTH="--depth 1"
[ "${FULL:-0}" = "1" ] && DEPTH=""

clone() {
  local url="$1" dst="$2"
  if [ -d "$TP/$dst/.git" ]; then
    echo "==> $dst already present, pulling latest"
    git -C "$TP/$dst" pull --ff-only || true
  else
    echo "==> cloning $dst"
    git clone $DEPTH "$url" "$TP/$dst"
  fi
}

clone "https://github.com/kijai/ComfyUI-WanVideoWrapper.git"   "ComfyUI-WanVideoWrapper"
clone "https://github.com/wuwukaka/ComfyUI-WanAnimatePlus.git" "ComfyUI-WanAnimatePlus"

echo ""
echo "Study clones ready in: $TP"
echo "Reminder: these are reference-only. The pack imports nothing from them."
