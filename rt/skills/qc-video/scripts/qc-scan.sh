#!/usr/bin/env bash
#
# qc-scan.sh — one-pass QC sweep for a video/audio file.
# Detection only: reports problems with timestamps, never modifies the source.
#
# Usage: bash qc-scan.sh "/path/to/video.mp4"
#
# Thresholds are sane defaults. For unusual sources (dark footage, film pulldown,
# full-range video), see references/detection-rules.md and run individual
# detectors with tuned flags instead.

set -uo pipefail

IN="${1:-}"
if [[ -z "$IN" ]]; then
  echo "Usage: bash qc-scan.sh \"/path/to/video.mp4\"" >&2
  exit 2
fi
if [[ ! -f "$IN" ]]; then
  echo "File not found: $IN" >&2
  exit 2
fi
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "ffmpeg/ffprobe not found on PATH." >&2
  exit 3
fi

hr()  { printf '\n========== %s ==========\n' "$1"; }
note(){ printf '  %s\n' "$1"; }

echo "QC scan: $IN"

# ---- 0. Probe ----------------------------------------------------------------
hr "SOURCE INFO"
ffprobe -v error -show_entries \
  "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,sample_rate,channels" \
  -of default=noprint_wrappers=1 "$IN"

HAS_AUDIO=$(ffprobe -v error -select_streams a -show_entries stream=index -of csv=p=0 "$IN" | head -1)

# ---- 1. Container / decode errors -------------------------------------------
hr "DECODE ERRORS (empty = clean)"
ffmpeg -hide_banner -v error -i "$IN" -f null - 2>&1 | head -40 || true

# ---- 2. Black segments -------------------------------------------------------
hr "BLACK SEGMENTS"
ffmpeg -hide_banner -i "$IN" -vf "blackdetect=d=0.1:pic_th=0.98:pix_th=0.10" \
  -an -f null - 2>&1 | grep blackdetect || note "none detected"

# ---- 3. Freeze / stuck frames -----------------------------------------------
hr "FREEZE / STUCK FRAMES"
ffmpeg -hide_banner -i "$IN" -vf "freezedetect=n=0.003:d=0.5" \
  -map 0:v -f null - 2>&1 | grep -i 'lavfi.freezedetect' || note "none detected"

# ---- 4. Frame count vs. expected --------------------------------------------
hr "FRAME COUNT (expected vs. actual)"
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$IN")
RFR=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$IN")
ACTUAL=$(ffprobe -v error -count_frames -select_streams v:0 \
          -show_entries stream=nb_read_frames -of csv=p=0 "$IN")
note "duration=${DUR}s  r_frame_rate=${RFR}  actual_frames=${ACTUAL}"
if [[ -n "${DUR:-}" && "${RFR:-}" == */* ]]; then
  EXPECTED=$(awk -F/ -v d="$DUR" '{ if ($2>0) printf "%.0f", ($1/$2)*d }' <<<"$RFR")
  note "expected_frames≈${EXPECTED}  (large gap ⇒ drops or VFR; telecine expected on film)"
fi

# ---- 5. Interlacing ----------------------------------------------------------
hr "INTERLACING (idet)"
ffmpeg -hide_banner -i "$IN" -vf idet -frames:v 2000 -an -f null - 2>&1 \
  | grep -Ei 'Single frame|Multi frame' || note "idet produced no summary"

# ---- 6. Letterbox / crop -----------------------------------------------------
hr "CROP / LETTERBOX (most common detected crop first)"
ffmpeg -hide_banner -i "$IN" -vf cropdetect=round=2 -frames:v 500 -an -f null - 2>&1 \
  | grep -o 'crop=[0-9:]*' | sort | uniq -c | sort -rn | head -5 \
  || note "cropdetect produced no output"

# ---- 7. Audio checks ---------------------------------------------------------
if [[ -n "$HAS_AUDIO" ]]; then
  hr "AUDIO SILENCE"
  ffmpeg -hide_banner -i "$IN" -af "silencedetect=n=-50dB:d=0.5" \
    -f null - 2>&1 | grep silence || note "none detected"

  hr "AUDIO CLIPPING / PEAKS"
  ffmpeg -hide_banner -i "$IN" -af astats -f null - 2>&1 \
    | grep -Ei 'Peak level|Flat factor|Peak count' | head -12 || note "no astats output"

  hr "LOUDNESS (EBU R128 — compare to your delivery target)"
  ffmpeg -hide_banner -i "$IN" -af ebur128=peak=true -f null - 2>&1 | tail -18 || true
else
  hr "AUDIO"
  note "no audio stream present"
fi

hr "DONE"
echo "Detection only — nothing was modified. See references/detection-rules.md to tune thresholds."
