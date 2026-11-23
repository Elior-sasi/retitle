"""Video I/O through ffmpeg pipes - frame exact, no seeking guesswork."""
import json, subprocess, os
from fractions import Fraction
import numpy as np

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")
# no console windows flashing for ffmpeg/tesseract when running as a windowed Windows app
NOWIN = {"creationflags": 0x08000000} if os.name == "nt" else {}


def probe(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-print_format", "json", "-show_streams",
                          "-show_format", path], capture_output=True, text=True, check=True, **NOWIN).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    a = [s for s in d["streams"] if s["codec_type"] == "audio"]
    fps = Fraction(v.get("avg_frame_rate") or v["r_frame_rate"])
    if fps == 0:
        fps = Fraction(v["r_frame_rate"])
    dur = float(d["format"].get("duration", 0))
    return {"width": int(v["width"]), "height": int(v["height"]),
            "fps_num": fps.numerator, "fps_den": fps.denominator, "fps": float(fps),
            "duration": dur, "has_audio": bool(a), "pix_fmt": v.get("pix_fmt"),
            "codec": v.get("codec_name")}


def frames(path, w, h, scale=None, first=None, last=None):
    """Yield every decoded frame (BGR uint8) in order. scale=(sw,sh) to downscale.
    first/last (inclusive): only that stretch is converted and sent (counted by decoded frames: exact)."""
    flt = []
    if first is not None or last is not None:
        flt.append("trim=" + ":".join(x for x in (f"start_frame={first}" if first else "",
                                                     f"end_frame={last + 1}" if last is not None else "") if x))
    if scale:
        flt.append(f"scale={scale[0]}:{scale[1]}:flags=area")
    vf = ["-vf", ",".join(flt)] if flt else []
    ow, oh = scale if scale else (w, h)
    cmd = [FFMPEG, "-v", "error", "-i", path, "-map", "0:v:0", "-fps_mode", "passthrough",
           *vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=ow * oh * 3 * 4, **NOWIN)
    n = ow * oh * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
    finally:
        p.stdout.close()
        p.wait()


class Writer:
    """Encode BGR frames, copy audio from src."""

    def __init__(self, out, src, info, crf=14, preset="veryfast", scale=None):
        fps = f"{info['fps_num']}/{info['fps_den']}"
        vf = ["-vf", f"scale={scale}:-2:flags=area"] if scale else []
        cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{info['width']}x{info['height']}", "-r", fps, "-i", "-",
               "-i", src, "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
               *vf, "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
        self.out = out
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, **NOWIN)

    def write(self, f):
        self.p.stdin.write(np.ascontiguousarray(f).tobytes())

    def abort(self):
        try:
            self.p.kill(); self.p.wait()
        except Exception:
            pass
        try:
            os.remove(self.out)
        except OSError:
            pass

    def close(self):
        self.p.stdin.close()
        self.p.wait()
        if self.p.returncode:
            raise RuntimeError("ffmpeg encode failed")


def load_json(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_json(p, d):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)


def tc(frame, fps):
    s = frame / fps
    return f"{int(s // 60)}:{s % 60:05.2f}"


def grab(path, f, info):
    """Decode exactly frame f (accurate seek: decode from previous keyframe, keep first pts >= t)."""
    w, h, fps = info["width"], info["height"], info["fps"]
    t = max(0.0, (f - 0.3) / fps)
    cmd = [FFMPEG, "-v", "error", "-ss", f"{t:.6f}", "-i", path, "-map", "0:v:0", "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    r = subprocess.run(cmd, capture_output=True, **NOWIN)
    if len(r.stdout) < w * h * 3:
        return None
    return np.frombuffer(r.stdout[:w * h * 3], np.uint8).reshape(h, w, 3).copy()


def count_frames(path):
    """Number of video frames without decoding (packet count)."""
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", path], capture_output=True, text=True, **NOWIN).stdout
    try:
        return int(out.strip().split(",")[0])
    except ValueError:
        return 0


def frames_at(path, w, h, ranges):
    """Only the frames in `ranges` [(a, b) inclusive], as (index, frame). One decode pass; the frames
    in between are decoded by ffmpeg but never converted or sent (fast on long videos)."""
    rs = []
    for a, b in sorted(ranges):
        if rs and a <= rs[-1][1] + 1:
            rs[-1][1] = max(rs[-1][1], b)
        else:
            rs.append([a, b])
    if not rs:
        return
    idx = [i for a, b in rs for i in range(a, b + 1)]
    expr = "+".join(f"between(n\\,{a}\\,{b})" for a, b in rs)
    cmd = [FFMPEG, "-v", "error", "-i", path, "-map", "0:v:0", "-vf", f"select={expr}", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=w * h * 3 * 4, **NOWIN)
    n = w * h * 3
    try:
        for i in idx:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        p.kill()
        p.wait()
