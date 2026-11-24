#!/usr/bin/env bash
#
# setup.sh — one-time environment setup for the qc-video skill.
#
# 1. Checks the REQUIRED system binaries (ffmpeg, ffprobe).
# 2. Creates a Python venv at the skill root and installs the OPTIONAL
#    enhanced-detector packages (numpy, opencv-python).
#
# The core scanner works with ffmpeg alone; the venv only adds the blur and
# flash-frame detectors. Safe to re-run.

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$SKILL_DIR/.venv"

echo "== qc-video setup =="
echo "skill dir: $SKILL_DIR"

# --- required system binaries ------------------------------------------------
missing=0
for bin in ffmpeg ffprobe; do
  if command -v "$bin" >/dev/null 2>&1; then
    echo "  [ok]  $bin -> $(command -v "$bin")"
  else
    echo "  [MISSING] $bin  — install with: brew install ffmpeg"
    missing=1
  fi
done

# --- optional: QCTools CLI ---------------------------------------------------
if command -v qcli >/dev/null 2>&1; then
  echo "  [ok]  qcli (optional) -> $(command -v qcli)"
else
  echo "  [--]  qcli not found (optional; only needed for the QCTools GUI route)"
fi

if [[ "$missing" -eq 1 ]]; then
  echo "ERROR: install the missing REQUIRED binaries above, then re-run." >&2
  exit 1
fi

# --- python venv + optional packages -----------------------------------------
if [[ ! -d "$VENV" ]]; then
  echo "creating venv at $VENV"
  python3 -m venv "$VENV"
fi
echo "installing optional detector packages (numpy, opencv-python)..."
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r "$SKILL_DIR/requirements.txt"

echo
"$VENV/bin/python" - <<'PY'
import numpy, cv2
print(f"  numpy  {numpy.__version__}")
print(f"  opencv {cv2.__version__}")
PY

echo
echo "Done. Run scans with the venv interpreter so the enhanced detectors are active:"
echo "  $VENV/bin/python $SKILL_DIR/scripts/qc-scan.py VIDEO [VIDEO ...]"
