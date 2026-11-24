#!/usr/bin/env python3
"""
qc-scan.py — batch video/audio QC scanner (detection only).

Takes one or more video files, runs the standard detector suite on each, and
reports every finding as: which video, which frame, which timecode, and what
the issue is. Never modifies the source files.

Usage:
    python3 qc-scan.py VIDEO [VIDEO ...] [--format md|csv|json|txt] [--output PATH]

Defaults:
    --format md      (Markdown report)
    --output         qc-report.<ext> in the current directory

Notes:
    * Timecode is HH:MM:SS:FF (frame-accurate, non-drop) derived from the video's
      frame rate. `frame` is the absolute 0-based frame index.
    * Segment defects (black, freeze, silence) are reported at their START, with
      duration in the detail column.
    * Thresholds are sane defaults; see references/detection-rules.md to tune for
      unusual sources (dark footage, film pulldown, full-range video).
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Optional enhanced detectors. The core scan runs on ffmpeg + stdlib alone; these
# libraries (installed into the skill's .venv via scripts/setup.sh) only power the
# flash-frame and blur checks. Absence is not an error — those checks are skipped.
try:
    import numpy as _np
except ImportError:
    _np = None
try:
    import cv2 as _cv2
except ImportError:
    _cv2 = None


# ----------------------------- ffmpeg helpers --------------------------------

def run(cmd):
    """Run a command, return combined stdout+stderr as text (never raises)."""
    # Force UTF-8 decoding with replacement: ffmpeg output can contain non-ASCII
    # (filenames, metadata), and Windows' default locale codec (cp1252) would
    # otherwise raise or mangle it.
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace")
    return p.stdout or ""


def probe(path):
    """Return dict with fps, duration, width, height, codec, has_audio."""
    out = run([
        "ffprobe", "-v", "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate",
        "-of", "json", path,
    ])
    info = {"fps": 25.0, "duration": 0.0, "width": 0, "height": 0,
            "vcodec": "?", "acodec": None, "has_audio": False}
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return info
    info["duration"] = float(data.get("format", {}).get("duration", 0) or 0)
    for s in data.get("streams", []):
        if s.get("codec_type") == "video" and info["width"] == 0:
            info["width"] = s.get("width", 0)
            info["height"] = s.get("height", 0)
            info["vcodec"] = s.get("codec_name", "?")
            info["fps"] = parse_fps(s.get("r_frame_rate", "25/1"))
        elif s.get("codec_type") == "audio":
            info["has_audio"] = True
            info["acodec"] = s.get("codec_name", "?")
    return info


def parse_fps(rate):
    try:
        num, den = rate.split("/")
        den = float(den)
        return float(num) / den if den else 25.0
    except (ValueError, ZeroDivisionError):
        return 25.0


# ----------------------------- time / frame ----------------------------------

def to_frame(seconds, fps):
    return int(round(seconds * fps))


def to_tc(seconds, fps):
    """HH:MM:SS:FF non-drop timecode."""
    if seconds < 0:
        seconds = 0
    fps_int = max(1, int(round(fps)))
    total = int(round(seconds * fps_int))
    ff = total % fps_int
    total_s = total // fps_int
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d}:{ff:02d}"


def fmt_duration(seconds):
    """Human-readable H:MM:SS (or M:SS under an hour)."""
    s = int(round(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def finding(video, seconds, fps, severity, issue, detail):
    return {
        "video": video,
        "frame": to_frame(seconds, fps) if seconds is not None else None,
        "timecode": to_tc(seconds, fps) if seconds is not None else "—",
        "seconds": round(seconds, 3) if seconds is not None else None,
        "severity": severity,       # High | Medium | Low | Info
        "issue": issue,
        "detail": detail,
    }


# ----------------------------- detectors -------------------------------------
#
# Video detection is organised around ONE downscaled decode pass that carries
# several filters at once (video_pass), because a full-res decode is by far the
# dominant cost on HD footage. Cheap or field-sensitive checks (idet, crop) and
# the audio checks run as their own small passes.

def video_pass(path):
    """Single downscaled decode carrying signalstats + black + freeze detection.

    Returns raw combined stderr/stdout. Downscaling to 360p slashes decode cost
    while preserving the luma-average, black, and freeze signals these filters
    rely on. Run at default log level so decode errors surface here too.
    """
    return run([
        "ffmpeg", "-hide_banner", "-i", path, "-vf",
        "scale=-2:360,signalstats,metadata=print:key=lavfi.signalstats.YAVG,"
        "blackdetect=d=0.1:pic_th=0.98:pix_th=0.10,"
        "freezedetect=n=0.003:d=0.5",
        "-an", "-f", "null", "-",
    ])


def parse_decode_errors(out, path, fps):
    findings = []
    for line in out.splitlines():
        low = line.lower()
        if ("lavfi" in low or line.startswith("frame:") or "black_" in low
                or "freeze" in low or not line.strip()):
            continue
        if "error" in low or "corrupt" in low or "invalid data" in low:
            findings.append(finding(path, None, fps, "High", "Decode error", line.strip()[:160]))
    return findings[:20]


def parse_black(out, path, fps):
    findings = []
    for m in re.finditer(r"black_start:([0-9.]+) black_end:([0-9.]+) black_duration:([0-9.]+)", out):
        start, end, dur = map(float, m.groups())
        findings.append(finding(path, start, fps, "High", "Black segment",
                                f"{dur:.2f}s black (to {to_tc(end, fps)})"))
    return findings


def parse_freeze(out, path, fps):
    findings = []
    starts = re.findall(r"freeze_start: ([0-9.]+)", out)
    durs = re.findall(r"freeze_duration: ([0-9.]+)", out)
    for i, start in enumerate(starts):
        dur = durs[i] if i < len(durs) else "?"
        d = f"{float(dur):.2f}s held" if dur != "?" else "frame held"
        findings.append(finding(path, float(start), fps, "Medium", "Freeze / stuck frame", d))
    return findings


def detect_framecount(path, fps, duration):
    # count_packets reads the container index without decoding — near-instant,
    # and packet count equals frame count for a normal video stream.
    out = run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
               "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path])
    try:
        actual = int(out.strip().splitlines()[0])
    except (ValueError, IndexError):
        return []
    expected = int(round(fps * duration))
    gap = actual - expected
    # tolerate ~1s of slack; bigger gaps mean drops or VFR
    if abs(gap) > max(2, fps):
        sign = "missing" if gap < 0 else "extra"
        return [finding(path, None, fps, "Medium", "Frame count mismatch",
                        f"{abs(gap)} {sign} vs expected {expected} "
                        f"(actual {actual}); check for drops/VFR/telecine")]
    return []


def detect_interlace(path, fps):
    # interlacing is a global property, so a few hundred frames classify it.
    out = run(["ffmpeg", "-hide_banner", "-i", path, "-vf", "idet",
               "-frames:v", "500", "-an", "-f", "null", "-"])
    m = re.findall(r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*"
                   r"Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)", out)
    if not m:
        return []
    tff, bff, prog, und = map(int, m[-1])
    interlaced = tff + bff
    total = interlaced + prog + und
    if total == 0:
        return []
    if interlaced > prog and interlaced > 0.2 * total:
        order = "TFF" if tff >= bff else "BFF"
        mixed = " (mixed field order!)" if tff > 0 and bff > 0 and min(tff, bff) > 0.2 * max(tff, bff) else ""
        return [finding(path, None, fps, "Medium", "Interlacing",
                        f"{order} dominant: TFF={tff} BFF={bff} Prog={prog}{mixed}")]
    return []


def detect_crop(path, fps, width, height):
    out = run(["ffmpeg", "-hide_banner", "-i", path, "-vf", "cropdetect=round=2",
               "-frames:v", "500", "-an", "-f", "null", "-"])
    crops = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", out)
    if not crops:
        return []
    counts = {}
    for c in crops:
        counts[c] = counts.get(c, 0) + 1
    (cw, ch, cx, cy), _ = max(counts.items(), key=lambda kv: kv[1])
    cw, ch = int(cw), int(ch)
    if width and height and (cw < width - 4 or ch < height - 4):
        kind = "Letterbox" if ch < height - 4 else "Pillarbox"
        return [finding(path, None, fps, "Low", f"{kind} / cropped picture",
                        f"active area {cw}x{ch} inside {width}x{height}")]
    return []


def audio_pass(path):
    """Single audio decode carrying silence + clipping (astats) + loudness."""
    return run(["ffmpeg", "-hide_banner", "-i", path, "-af",
                "silencedetect=n=-50dB:d=0.5,astats,ebur128=peak=true",
                "-f", "null", "-"])


def parse_silence(out, path, fps):
    findings = []
    starts = re.findall(r"silence_start: (-?[0-9.]+)", out)
    durs = re.findall(r"silence_duration: ([0-9.]+)", out)
    for i, start in enumerate(starts):
        dur = durs[i] if i < len(durs) else "?"
        d = f"{float(dur):.2f}s silent" if dur != "?" else "silent"
        findings.append(finding(path, max(0.0, float(start)), fps, "Medium",
                                "Audio silence", d))
    return findings


def parse_clipping(out, path, fps):
    peak = re.findall(r"Peak level dB: (-?[0-9.]+|inf)", out)
    flat = re.findall(r"Flat factor: ([0-9.]+)", out)
    if not peak:
        return []
    def fnum(x):
        return 0.0 if x == "inf" else float(x)
    peak_max = max(fnum(p) for p in peak)
    flat_max = max((float(f) for f in flat), default=0.0)
    if peak_max >= -0.1 and flat_max > 0:
        return [finding(path, None, fps, "Medium", "Audio clipping",
                        f"peak {peak_max:.2f} dBFS, flat factor {flat_max:.1f}")]
    return []


def parse_loudness(out, path, fps):
    # ebur128 prints a running "I: … LUFS" every frame; the integrated value is
    # the LAST one (the Summary block), not the first (the startup floor).
    lufs_all = re.findall(r"I:\s*(-?[0-9.]+|-?inf)\s*LUFS", out)
    tp_all = re.findall(r"Peak:\s*(-?[0-9.]+|-?inf)\s*dBFS", out)
    if not lufs_all:
        return []
    lufs = lufs_all[-1]
    tp = tp_all[-1] if tp_all else "?"
    return [finding(path, None, fps, "Info", "Loudness (verify vs. delivery spec)",
                    f"integrated {lufs} LUFS, true peak {tp} dBFS")]


def detect_flash(out, path, fps):
    """Flash frames / flicker from the YAVG series in the video pass (numpy)."""
    if _np is None:
        return []
    yavg = [float(x) for x in re.findall(r"lavfi\.signalstats\.YAVG=([0-9.]+)", out)]
    if len(yavg) < 5:
        return []
    y = _np.array(yavg)
    # a flash frame jumps far from BOTH neighbours; measure the local spike
    prev, nxt = y[:-2], y[2:]
    cur = y[1:-1]
    spike = cur - _np.maximum(prev, nxt)          # how far above both sides
    dip = _np.minimum(prev, nxt) - cur            # how far below both sides
    dev = _np.maximum(spike, dip)
    thresh = max(18.0, 4.0 * _np.median(_np.abs(_np.diff(y))))  # adaptive
    findings = []
    for i, d in enumerate(dev):
        if d > thresh:
            sec = (i + 1) / fps
            kind = "bright flash" if spike[i] >= dip[i] else "dark dip"
            findings.append(finding(path, sec, fps, "Medium", "Flash frame / flicker",
                                    f"{kind}, luma jump {d:.0f} (thr {thresh:.0f})"))
    # collapse runs of adjacent flagged frames into the first one
    collapsed, last = [], -10
    for f in findings:
        if f["frame"] - last > 2:
            collapsed.append(f)
        last = f["frame"]
    return collapsed[:50]


def detect_blur(path, fps, duration):
    """Out-of-focus / soft frames via variance of Laplacian (needs opencv)."""
    if _cv2 is None or duration <= 0:
        return []
    cap = _cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    # sample at most ~60 frames across the file — each seek forces a keyframe
    # decode, so the sample count, not the video length, drives the cost.
    n_samples = min(60, max(4, int(duration)))
    step = duration / n_samples
    scores, times = [], []
    t = 0.0
    while t < duration:
        cap.set(_cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        # downscale before the Laplacian: sharpness ranking is preserved and the
        # variance is computed on far fewer pixels.
        h = frame.shape[0]
        if h > 480:
            scale = 480.0 / h
            frame = _cv2.resize(frame, None, fx=scale, fy=scale, interpolation=_cv2.INTER_AREA)
        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
        scores.append(_cv2.Laplacian(gray, _cv2.CV_64F).var())
        times.append(t)
        t += step
    cap.release()
    if len(scores) < 4:
        return []
    arr = _np.array(scores) if _np is not None else scores
    med = (float(_np.median(arr)) if _np is not None
           else sorted(scores)[len(scores) // 2])
    # Flag only frames FAR softer than the file's own norm and below an absolute
    # floor. Drama footage has lots of motion blur and shallow-focus shots, so a
    # loose threshold is pure noise; keep this conservative and advisory (Low).
    findings = []
    for sc, tm in zip(scores, times):
        if sc < 0.15 * med and sc < 40:
            findings.append(finding(path, tm, fps, "Low", "Soft / out of focus",
                                    f"sharpness {sc:.0f} vs median {med:.0f}"))
    return findings[:50]


# ----------------------------- per-file driver -------------------------------

def scan_file(path):
    info = probe(path)
    fps, dur = info["fps"], info["duration"]
    findings = []

    # one downscaled video decode carries black, freeze, flash-stats, and errors
    vout = video_pass(path)
    findings += parse_decode_errors(vout, path, fps)
    findings += parse_black(vout, path, fps)
    findings += parse_freeze(vout, path, fps)
    findings += detect_flash(vout, path, fps)

    # cheap / field-sensitive checks get their own small passes
    findings += detect_framecount(path, fps, dur)
    findings += detect_interlace(path, fps)
    findings += detect_crop(path, fps, info["width"], info["height"])
    findings += detect_blur(path, fps, dur)

    # one audio decode carries silence, clipping, and loudness
    if info["has_audio"]:
        aout = audio_pass(path)
        findings += parse_silence(aout, path, fps)
        findings += parse_clipping(aout, path, fps)
        findings += parse_loudness(aout, path, fps)

    # sort within a file: real problems by time, file-wide (no frame) last
    findings.sort(key=lambda f: (f["seconds"] is None, f["seconds"] or 0))
    return info, findings


# ----------------------------- report writers --------------------------------

SEV_ICON = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "Info": "ℹ️"}


def write_md(results, out_path):
    lines = ["# QC Report", ""]
    total = sum(len(f) for _, f, _ in results)
    lines.append(f"Scanned **{len(results)}** file(s) — **{total}** finding(s) total. "
                 "Detection only; no files were modified.")
    lines.append("")
    for path, findings, info in results:
        lines.append(f"## {Path(path).name}")
        lines.append("")
        audio = f"{info['acodec']}" if info['has_audio'] else "none"
        lines.append(
            f"- **Resolution:** {info['width']}×{info['height']}\n"
            f"- **Frame rate:** {info['fps']:.3f} fps\n"
            f"- **Duration:** {fmt_duration(info['duration'])} "
            f"({info['duration']:.2f}s)\n"
            f"- **Video codec:** {info['vcodec']}  ·  **Audio:** {audio}"
        )
        lines.append("")
        if not findings:
            lines.append("_No issues detected._")
            lines.append("")
            continue
        lines.append("| Frame | Timecode | Severity | Issue | Detail |")
        lines.append("|------:|----------|----------|-------|--------|")
        for f in findings:
            frame = f["frame"] if f["frame"] is not None else "—"
            lines.append(f"| {frame} | {f['timecode']} | "
                         f"{SEV_ICON.get(f['severity'],'')} {f['severity']} | "
                         f"{f['issue']} | {f['detail']} |")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(results, out_path):
    import csv
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "frame", "timecode", "seconds", "severity", "issue", "detail"])
        for path, findings, _ in results:
            for f in findings:
                w.writerow([Path(path).name, f["frame"], f["timecode"],
                            f["seconds"], f["severity"], f["issue"], f["detail"]])


def write_json(results, out_path):
    payload = []
    for path, findings, info in results:
        payload.append({"video": path, "source": info, "findings": findings})
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")


def write_txt(results, out_path):
    lines = []
    for path, findings, info in results:
        lines.append(f"### {Path(path).name}  "
                     f"({info['width']}x{info['height']} {info['fps']:.3f}fps, {info['duration']:.2f}s)")
        if not findings:
            lines.append("  no issues detected")
        for f in findings:
            frame = f["frame"] if f["frame"] is not None else "—"
            lines.append(f"  [{f['severity']:<6}] {f['timecode']} (frame {frame})  "
                         f"{f['issue']} — {f['detail']}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


WRITERS = {"md": write_md, "csv": write_csv, "json": write_json, "txt": write_txt}


# ----------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch video/audio QC scanner (detection only).")
    ap.add_argument("videos", nargs="+", help="one or more video files to scan")
    ap.add_argument("--format", choices=list(WRITERS), default="md",
                    help="output format (default: md)")
    ap.add_argument("--output", help="output file path (default: qc-report.<ext>)")
    ap.add_argument("--jobs", "-j", type=int, default=0,
                    help="scan this many files in parallel (default: half the CPU "
                         "cores). Each file still runs its detectors internally; "
                         "this parallelises across files.")
    args = ap.parse_args()

    missing = [v for v in args.videos if not Path(v).is_file()]
    if missing:
        print("File(s) not found:\n  " + "\n  ".join(missing), file=sys.stderr)
        sys.exit(2)

    # report optional-detector coverage up front so "clean" is never misleading
    skipped = []
    if _np is None:
        skipped.append("flash/flicker (needs numpy)")
    if _cv2 is None:
        skipped.append("blur (needs opencv-python)")
    if skipped:
        print("note: optional detectors skipped - " + "; ".join(skipped)
              + "\n      run the setup script and use the venv interpreter to enable them.",
              file=sys.stderr)

    # Files are independent, so scan several at once. Each worker still runs its
    # own ffmpeg passes (which are themselves multi-threaded), so the default
    # caps at half the cores to avoid oversubscribing the CPU.
    jobs = args.jobs if args.jobs > 0 else max(1, (os.cpu_count() or 2) // 2)
    jobs = min(jobs, len(args.videos))
    n = len(args.videos)
    results = [None] * n

    if jobs <= 1:
        for i, v in enumerate(args.videos):
            print(f"[{i + 1}/{n}] scanning: {Path(v).name}", file=sys.stderr)
            info, findings = scan_file(v)
            results[i] = (v, findings, info)
    else:
        print(f"scanning {n} files, {jobs} in parallel...", file=sys.stderr)
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(scan_file, v): (i, v) for i, v in enumerate(args.videos)}
            done = 0
            for fut in as_completed(futs):
                i, v = futs[fut]
                done += 1
                try:
                    info, findings = fut.result()
                except Exception as e:  # keep the batch alive if one file fails
                    info = {"fps": 0.0, "duration": 0.0, "width": 0, "height": 0,
                            "vcodec": "?", "acodec": None, "has_audio": False}
                    findings = [finding(v, None, 0.0, "High", "Scan error", str(e)[:160])]
                results[i] = (v, findings, info)
                print(f"[{done}/{n}] done: {Path(v).name}", file=sys.stderr)

    out_path = Path(args.output) if args.output else Path(f"qc-report.{args.format}")
    WRITERS[args.format](results, out_path)

    total = sum(len(f) for _, f, _ in results)
    print(f"\nDone - {total} finding(s) across {len(results)} file(s).", file=sys.stderr)
    print(f"Report written to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
