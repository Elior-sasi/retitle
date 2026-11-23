"""Pass 3: cache full-res region crops per card (group), estimate mask/style, refine exact in/out."""
import os, glob, shutil
import cv2, numpy as np
from . import vio, masks
from .shots import shot_of


def groups(T):
    g = {}
    for t in T["titles"]:
        g.setdefault(t["group"], []).append(t)
    return g


def _region(lines, W, H):
    b = np.array([t["box"] for t in lines])
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    h = max(t["box"][3] - t["box"][1] for t in lines)
    m = int(max(80, 1.2 * h))
    return [int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(W, x1 + m)), int(min(H, y1 + m))]


def run(proj, pad=48):
    P = vio.load_json(os.path.join(proj, "project.json"))
    S = vio.load_json(os.path.join(proj, "shots.json"))
    T = vio.load_json(os.path.join(proj, "titles.json"))
    W, H, n = P["info"]["width"], P["info"]["height"], P["n_frames"]
    every = S["every"]
    jobs = {}
    for gid, lines in groups(T).items():
        if not any(t.get("translate", True) for t in lines):
            continue
        reg = lines[0].get("region") or _region(lines, W, H)
        a = min(t.get("in_override", t["sample_in"]) for t in lines) - every - 2
        b = max(t.get("out_override", t["sample_out"]) for t in lines) + every + 2
        a, b = max(0, a), min(n - 1, b)
        sa, _ = shot_of(S["cuts"], n, a)
        _, eb = shot_of(S["cuts"], n, b)
        lo, hi = max(sa, a - pad), min(eb - 1, b + pad)
        for t in lines:
            t["region"] = reg
        jobs[gid] = (lo, hi, reg)
        d = os.path.join(proj, "cache", gid)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    if not jobs:
        return
    first = min(j[0] for j in jobs.values()); last = max(j[1] for j in jobs.values())
    for i, f in enumerate(vio.frames(P["video"], W, H)):
        if i > last:
            break
        if i < first:
            continue
        for gid, (lo, hi, (x0, y0, x1, y1)) in jobs.items():
            if lo <= i <= hi:
                cv2.imwrite(os.path.join(proj, "cache", gid, f"{i:06d}.png"), f[y0:y1, x0:x1])
    # estimate mask + style per line at its reference frame, then refine in/out
    for gid, lines in groups(T).items():
        if gid not in jobs:
            continue
        lo, hi, reg = jobs[gid]
        for t in lines:
            t["cache"] = [lo, hi]
            if not t.get("translate", True):
                continue
            ref = cv2.imread(os.path.join(proj, "cache", gid, f"{t['ref']:06d}.png"))
            words = [[w[0] - reg[0], w[1] - reg[1], w[2] - reg[0], w[3] - reg[1]] for w in t["words"]]
            if "mask" not in t or t.get("auto_mask", True):
                est = masks.estimate(ref, words)
                if est is None:
                    t["warn"] = "mask estimate failed - set mask manually"
                    continue
                t["mask"] = est["mask"]
                t.setdefault("style", {}).update({k: v for k, v in est["style"].items()
                                                   if k not in t.get("style_locked", [])})
            refine(proj, t, gid, S, n)
    vio.save_json(os.path.join(proj, "titles.json"), T)


def _grad(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def _presence(proj, t, gid, frames):
    """Per frame: how much of this line is on screen (1 = like the reference frame).
    max of (a) colour-model pixel count and (b) correlation of edge structure with the
    reference text - (b) still sees faint text during fades."""
    reg = t["region"]
    bx = t["box"]
    h = bx[3] - bx[1]
    box = [bx[0] - reg[0] - h, bx[1] - reg[1] - h, bx[2] - reg[0] + h, bx[3] - reg[1] + h]
    refimg = cv2.imread(os.path.join(proj, "cache", gid, f"{t['ref']:06d}.png"))
    core = masks.make(refimg, {**t["mask"], "grow": 0, "shadow": None, "extra": []}, box) > 0
    band = cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    gref = _grad(refimg)[band]
    gref = (gref - gref.mean()) / (gref.std() + 1e-6)
    cnt, cor = {}, {}
    for f in frames:
        img = cv2.imread(os.path.join(proj, "cache", gid, f"{f:06d}.png"))
        if img is None:
            continue
        m = masks.make(img, {**t["mask"], "grow": 0, "shadow": None, "extra": []}, box)
        cnt[f] = float((m > 0).sum())
        g = _grad(img)[band]
        g = (g - g.mean()) / (g.std() + 1e-6)
        cor[f] = float((g * gref).mean())
    return cnt, cor


def refine(proj, t, gid, S, n):
    every = S["every"]
    lo, hi = t["cache"]
    fr = list(range(lo, hi + 1))
    pres, cor = _presence(proj, t, gid, fr)
    refc = pres.get(t["ref"], 0) or max(pres.values())
    far = lambda f: f < t["sample_in"] - every or f > t["sample_out"] + every
    outside = [v for f, v in pres.items() if far(f)]
    base = float(np.median(outside)) if outside else 0.0
    cbase = [v for f, v in cor.items() if far(f)]
    cb = float(np.percentile(cbase, 90)) if cbase else 0.1
    rel = {}
    for f, v in pres.items():
        a = (v - base) / max(1.0, refc - base)
        c = (cor[f] - cb) / max(0.05, 1 - cb)
        rel[f] = max(a, c)
    t["presence"] = {str(f): round(v, 3) for f, v in rel.items()}
    if "in_override" in t:
        t["in"] = t["in_override"]
    else:
        f = t["sample_in"]
        while f - 1 >= lo and rel.get(f - 1, 0) > 0.06:
            f -= 1
        t["in"] = f
    if "out_override" in t:
        t["out"] = t["out_override"]
    else:
        f = t["sample_out"]
        while f + 1 <= hi and rel.get(f + 1, 0) > 0.06:
            f += 1
        t["out"] = f
    # snap to cuts within 1 frame
    for c in S["cuts"]:
        if abs(t["in"] - c) <= 1:
            t["in"] = c
        if abs(t["out"] + 1 - c) <= 1:
            t["out"] = c - 1
