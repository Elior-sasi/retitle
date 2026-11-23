"""Pass 6: write the final video. Untouched frames pass through; title frames get the clean
background crop + the Hebrew layer. Audio is copied as is."""
import os
import cv2, numpy as np
from . import vio, render
from .extract import groups


def _overlay(frame, layers):
    if not layers:
        return frame
    out = frame.astype(np.float32) / 255.0
    for L in layers:
        out = L[..., :3] + out * (1 - L[..., 3:4])
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def frame_result(proj, f, frame, T, meta, cache, W, H):
    out = frame.copy()
    for gid, lines in groups(T).items():
        act = [t for t in lines if t.get("translate", True) and "region" in t]
        if not act:
            continue
        p = os.path.join(proj, "clean", gid, f"{f:06d}.png")
        if os.path.exists(p):
            x0, y0, x1, y1 = act[0]["region"]
            c = cv2.imread(p)
            if c is not None and c.shape[:2] == out[y0:y1, x0:x1].shape[:2]:   # stale until the card is rebuilt
                out[y0:y1, x0:x1] = c
    lays = []
    for t in T["titles"]:
        lays += render.title_layers(proj, t, meta, cache, f, W, H)
    return _overlay(out, lays)


def run(proj, out_name=None, crf=14, preset="veryfast", scale=None, progress=None):
    P = vio.load_json(os.path.join(proj, "project.json"))
    T = vio.load_json(os.path.join(proj, "titles.json"))
    meta = vio.load_json(os.path.join(proj, "layers", "meta.json"), {})
    # sequences need a meta entry too
    for t in T["titles"]:
        if t.get("layer", {}).get("type") == "sequence":
            meta.setdefault(t["id"], {})
    info = P["info"]; W, H = info["width"], info["height"]
    os.makedirs(os.path.join(proj, "out"), exist_ok=True)
    base = os.path.splitext(os.path.basename(P["video"]))[0]
    out = os.path.join(proj, "out", out_name or f"{base}_HEB.mp4")
    wr = vio.Writer(out, P["video"], info, crf=crf, preset=preset, scale=scale)
    busy = set()
    for t in T["titles"]:
        if t.get("translate", True):
            busy.update(range(min(t["in"], t.get("cache", [t["in"]])[0]), max(t["out"], t.get("cache", [0, t["out"]])[1]) + 1))
    cache = {}
    n = P.get("n_frames") or 1
    try:
        for f, fr in enumerate(vio.frames(P["video"], W, H)):
            wr.write(frame_result(proj, f, fr, T, meta, cache, W, H) if f in busy else fr)
            if progress and f % 48 == 0:
                progress(f / n)
    except BaseException:
        wr.abort()
        raise
    wr.close()
    return out


def overlay_frame(proj, f, T, meta, cache, W, H):
    """Hebrew layers only, premultiplied BGRA float, or None when nothing is on screen."""
    out = None
    for t in T["titles"]:
        if t.get("mode") == "remove":
            continue
        for L in render.title_layers(proj, t, meta, cache, f, W, H):
            out = L if out is None else L + out * (1 - L[..., 3:4])
    return out


def run_overlay(proj, progress=None):
    """Transparent video with only the Hebrew titles (ProRes 4444 with alpha, .mov), frame-exact with the
    source, to lay over another cut of the same video (e.g. a textless version) in Premiere/Resolve."""
    import subprocess
    P = vio.load_json(os.path.join(proj, "project.json"))
    T = vio.load_json(os.path.join(proj, "titles.json"))
    meta = vio.load_json(os.path.join(proj, "layers", "meta.json"), {})
    for t in T["titles"]:
        if t.get("layer", {}).get("type") == "sequence":
            meta.setdefault(t["id"], {})
    info = P["info"]; W, H = info["width"], info["height"]
    n = P.get("n_frames") or int(round(info["duration"] * info["fps"]))
    os.makedirs(os.path.join(proj, "out"), exist_ok=True)
    base = os.path.splitext(os.path.basename(P["video"]))[0]
    out = os.path.join(proj, "out", f"{base}_HEB_titles_alpha.mov")
    cmd = [vio.FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{W}x{H}",
           "-r", f"{info['fps_num']}/{info['fps_den']}", "-i", "-",
           "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le", "-vendor", "apl0",
           "-movflags", "+write_colr", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, **vio.NOWIN)
    empty = np.zeros((H, W, 4), np.uint8).tobytes()
    cache = {}
    try:
        for f in range(n):
            L = overlay_frame(proj, f, T, meta, cache, W, H)
            if L is None:
                p.stdin.write(empty)
            else:
                a = L[..., 3:4]
                rgb = np.where(a > 1e-4, L[..., :3] / np.maximum(a, 1e-4), 0)   # straight alpha, as editors expect
                p.stdin.write(np.clip(np.concatenate([rgb, a], 2) * 255 + 0.5, 0, 255).astype(np.uint8).tobytes())
            if progress and f % 24 == 0:
                progress(f / n)
        p.stdin.close(); p.wait()
    except BaseException:
        try:
            p.kill(); p.wait(); os.remove(out)
        except Exception:
            pass
        raise
    if p.returncode:
        raise RuntimeError("ffmpeg ProRes encode failed")
    return out
