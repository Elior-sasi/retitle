---
name: qc-video
description: >-
  Detect quality problems across one or more video/audio files — black frames,
  freeze/stuck frames, dropped or duplicate frames, interlacing, illegal
  broadcast levels, letterboxing, silence, clipping, loudness violations, and
  A/V sync issues — and write a report listing, per finding, which video, which
  frame, which timecode, and what the issue is (Markdown by default; csv/json/
  txt on request). Use this skill whenever the user wants to QC, batch-check,
  inspect, analyze, validate, or "check the quality of" media files, asks why a
  video looks or sounds wrong, wants a QC report, or mentions QCTools / qcli /
  ffmpeg quality checks — even if they don't name a specific defect. Detection
  only: it reports problems with timestamps; it does not silently "fix" them.
---

# QC Video

Detect and report common video and audio quality defects across one or more
media files, and hand the user a report that says — for every issue — **which
video, which frame, which timecode, and what the issue is**.

## Input & output contract

**Input:** one or more video files the user specifies (a batch). Accept any
number of paths; scan each independently.

**Output:** a written report where every finding carries the video name, the
absolute frame number, the timecode (`HH:MM:SS:FF`), the severity, the issue,
and a short detail. The default format is a **Markdown (`.md`) file**. If the
user names a different format, honor it — the scanner supports `md`, `csv`,
`json`, and `txt` via `--format`, and a custom path via `--output`.

## The one rule that matters most

**Detection is not repair.** This skill finds problems and tells the user
*where* they are and *how confident* the detector is. It never edits, re-encodes,
or overwrites the source. If the user then wants to mitigate a specific defect
(trim black, dedupe frozen frames, deinterlace), that's a separate, explicit
step the user approves — because most "quality loss" (compression, real dropped
content) cannot be recovered, and pretending otherwise misleads them.

This skill runs on **macOS, Linux, and Windows**. `scripts/qc-scan.py` is pure,
cross-platform Python (stdlib + subprocess); only the two setup helpers and the
optional quick-look script are OS-specific.

## Requirements

**Required — system binaries** (these are the real analysis engine; pip cannot
install them):

| Tool | Purpose | Install (macOS) | Install (Windows) | Install (Linux) |
|------|---------|-----------------|-------------------|-----------------|
| `ffmpeg` | runs all detection filters | `brew install ffmpeg` | `winget install Gyan.FFmpeg` | `apt install ffmpeg` |
| `ffprobe` | reads framerate/duration/streams (ships with ffmpeg) | (bundled) | (bundled) | (bundled) |

The core scanner (`scripts/qc-scan.py`) needs **nothing else** — it uses only the
Python standard library. `ffmpeg`/`ffprobe` emit plain-text, timestamped results
that the scanner parses directly.

**Optional — Python venv for enhanced detectors.** Two checks that ffmpeg can't
do well are powered by Python libraries installed into a venv at the skill root:

| Package | Enables |
|---------|---------|
| `numpy` | flash-frame / flicker detection (luma time-series analysis) |
| `opencv-python` | blur / out-of-focus detection (variance of Laplacian) |

If the venv is absent, `qc-scan.py` still runs and simply **skips** those two
checks, printing a note so coverage is never silently reduced.

**Optional — `qcli`** (the QCTools CLI): produces a richer graph/thumbnail
report, but it needs the QCTools desktop GUI to view. Reserve it for when the
user wants visual waveform/vectorscope inspection rather than a text report.

### First-time setup

Run once to check the required binaries and build the venv. Use the helper for
your OS:

```bash
# macOS / Linux
bash <SKILL_DIR>/scripts/setup.sh
```
```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy Bypass -File <SKILL_DIR>\scripts\setup.ps1
```

Both verify `ffmpeg`/`ffprobe`, create `.venv/` at the skill root, and install
`requirements.txt` (numpy, opencv-python).

## Workflow

The bundled batch scanner does the heavy lifting — probing each file, running
every detector, converting hits to frame + timecode, and writing the report in
the requested format. Prefer it over hand-running filters.

1. **Collect the file list and output preference.** Gather every video path the
   user gave. Note whether they specified a format/path; if not, default to a
   Markdown file.
2. **Run the scanner** (see below). It probes framerate/duration per file —
   which is what makes frame numbers and timecodes correct — so you don't need a
   separate probe step.
3. **Read the report back to the user**, leading with the highest-severity
   findings. Don't just point at the file; summarize what was found where.
4. **Tune only if a source needs it.** If the user has unusual footage
   (legitimately dark, film pulldown, full-range video), read
   `references/detection-rules.md` and run the specific detector with adjusted
   thresholds, because the wrong threshold produces either noise or false
   silence. The reference has the exact flags and the false-positive traps.

## Running the batch scan

Run with the **venv interpreter** (`<SKILL_DIR>/.venv/bin/python` on macOS/Linux,
`<SKILL_DIR>\.venv\Scripts\python.exe` on Windows) so the flash and blur
detectors are active. Fall back to the system Python (`python3` on macOS/Linux,
`py` or `python` on Windows) only if the venv isn't set up — the two optional
checks are skipped, everything else still runs.

```bash
# macOS / Linux, enhanced (recommended):
<SKILL_DIR>/.venv/bin/python <SKILL_DIR>/scripts/qc-scan.py VIDEO [VIDEO ...] \
  [--format md|csv|json|txt] [--output PATH] [--jobs N]
```
```powershell
# Windows, enhanced (recommended):
<SKILL_DIR>\.venv\Scripts\python.exe <SKILL_DIR>\scripts\qc-scan.py VIDEO [VIDEO ...] `
  [--format md|csv|json|txt] [--output PATH] [--jobs N]
```

- Multiple files: pass them all; each is scanned and reported independently.
- **Parallelism:** files are scanned concurrently — `--jobs N` sets how many at
  once (default: half the CPU cores). For a big batch this is a large speedup;
  report order always follows input order regardless of finish order. Since each
  file's own detectors are already multi-threaded, going much past half the cores
  oversubscribes the CPU and stops helping.
- Default `--format md`; default output `qc-report.<ext>` in the current
  directory. Honor any format/path the user requested.
- Detection only — the script never modifies the source files.
- If `.venv/` is missing, run the setup helper for your OS first (see Requirements).

Every finding row carries: **video · frame · timecode (HH:MM:SS:FF) · severity ·
issue · detail**. Segment defects (black, freeze, silence) are reported at their
start, with duration in the detail column.

For a quick single-file console look without writing a report, `scripts/qc-scan.sh
"<file>"` prints raw detector output to the terminal (macOS/Linux only — it's a
bash convenience wrapper; on Windows use `qc-scan.py` directly).

## Severity guide

**High** = missing/broken content or hard compliance fail (black gaps,
corruption, silence where there should be audio). **Medium** = visible/audible
defect that may be acceptable depending on intent (short freeze, mild
interlacing, clipping). **Low** = borderline or spec-nitpick (minor letterbox).
**Info** = a measured value the user must judge against their own delivery spec
(loudness). If a file is clean, say so and name what was checked — "clean" only
means something when the coverage is stated.

## When the user asks to fix something

Surface what's *mitigable* vs. *unrecoverable*, then let the user choose. See the
"Mitigation (only when asked)" section of `references/detection-rules.md` for the
safe ffmpeg recipes (trim, mpdecimate, yadif, crop, loudnorm). Always write to a
**new** file, never over the source, and tell the user exactly what the operation
does and does not restore.
