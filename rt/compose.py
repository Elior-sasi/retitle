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
            out[y0:y1, x0:x1] = cv2.imread(p)
    lays = []
    for t in T["titles"]:
        if not t.get("translate", True) or t["id"] not in meta:
            continue
        lay = t.get("layer", {"type": "text"})
        if lay["type"] == "sequence":
            p = os.path.join(proj, lay["dir"], f"{f:06d}.png")
            if os.path.exists(p):
                lays.append(render.load_layer(p))
            continue
        if not (t["in"] <= f <= t["out"]):
            continue
        if t["id"] not in cache:
            cache[t["id"]] = render.load_layer(meta[t["id"]]["file"])
        anim = dict(t.get("anim", {"mode": "static"}))
        anim.setdefault("in", t["in"]); anim.setdefault("out", t["out"])
        L = render.place(cache[t["id"]], meta[t["id"]], anim, f, W, H)
        if L is not None:
            lays.append(L)
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
    for f, fr in enumerate(vio.frames(P["video"], W, H)):
        wr.write(frame_result(proj, f, fr, T, meta, cache, W, H) if f in busy else fr)
        if progress and f % 48 == 0:
            progress(f / n)
    wr.close()
    return out
