"""Pass 3: cache full-res region crops per card (group), estimate mask/style, refine exact in/out."""
import os, glob, shutil
import cv2, numpy as np
PNG_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 1]   # still lossless, faster to write
from . import vio, masks
from .shots import shot_of


def groups(T):
    g = {}
    for t in T["titles"]:
        g.setdefault(t["group"], []).append(t)
    return g


def _region(lines, W, H):
    b = np.array([t.get("text_box", t["box"]) for t in lines])
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    h = max(t["box"][3] - t["box"][1] for t in lines)
    m = int(max(80, 1.2 * h))
    if any(t.get("logo_full") for t in lines):          # symbols / decorations around the text: look further
        m = int(max(160, 3.0 * h))
    return [int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(W, x1 + m)), int(min(H, y1 + m))]


def cache_size_ok(d, reg):
    """The saved frames really are crops of this region (Claude or the user may have changed the region
    without the frames being taken again - then erasing and placing use the wrong picture)."""
    f = sorted(glob.glob(os.path.join(d, "*.png")))
    if not f:
        return False
    im = cv2.imread(f[0], cv2.IMREAD_UNCHANGED)
    return im is not None and im.shape[1] == reg[2] - reg[0] and im.shape[0] == reg[3] - reg[1]


def _cache_ok(lines, lo, hi, reg, d):
    """The card's saved frames still cover what it needs, at the same area."""
    if not os.path.isdir(d) or not cache_size_ok(d, reg):
        return False
    for t in lines:
        if not t.get("translate", True):
            continue
        if "mask" not in t or "cache" not in t or t.get("reextract") or t.get("warn"):
            return False
        if list(t.get("region") or []) != list(reg):
            return False
        c0, c1 = t["cache"]
        if lo < c0 or hi > c1:
            return False
    return any(t.get("translate", True) for t in lines)


def run(proj, pad=48, progress=None):
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
        if any(t.get("logo_full") and not t.get("logo_bg_done") for t in lines):
            reg = _region(lines, W, H)
        a = min(min(t.get("in_override", t["sample_in"]), t.get("prev_sample", t["sample_in"] - every)) for t in lines) - 2
        b = max(max(t.get("out_override", t["sample_out"]), t.get("next_sample", t["sample_out"] + every)) for t in lines) + 2
        a, b = max(0, a), min(n - 1, b)
        sa, _ = shot_of(S["cuts"], n, a)
        _, eb = shot_of(S["cuts"], n, b)
        lo, hi = max(sa, a - pad), min(eb - 1, b + pad)
        # slow camera moves: reach further (up to 10 s) until the picture has moved ~40 px,
        # so the background behind the letters shows up in some frame
        mo = S.get("motion")
        if mo:
            lim = int(10 * P["info"]["fps"])
            dist = lambda x, y: abs(mo[x][0] - mo[y][0]) + abs(mo[x][1] - mo[y][1])
            while lo > sa and a - lo < lim and dist(lo, a) < 40:
                lo -= 1
            while hi < eb - 1 and hi - b < lim and dist(hi, b) < 40:
                hi += 1
        d = os.path.join(proj, "cache", gid)
        if _cache_ok(lines, lo, hi, reg, d):
            continue                                  # nothing changed for this card: keep its frames
        for t in lines:
            t["region"] = reg
            t.pop("sam", None)                        # SAM 2's masks were for the old frames
        shutil.rmtree(os.path.join(proj, "sam", gid), ignore_errors=True)
        jobs[gid] = (lo, hi, reg)
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    if not jobs:
        return
    first = min(j[0] for j in jobs.values()); last = max(j[1] for j in jobs.values())
    want = sum(hi - lo + 1 for lo, hi, _ in jobs.values())
    for k, (i, f) in enumerate(vio.frames_at(P["video"], W, H, [(lo, hi) for lo, hi, _ in jobs.values()])):
        if progress and k % 12 == 0:
            progress(0.6 * k / max(1, want))
        for gid, (lo, hi, (x0, y0, x1, y1)) in jobs.items():
            if lo <= i <= hi:
                cv2.imwrite(os.path.join(proj, "cache", gid, f"{i:06d}.png"), f[y0:y1, x0:x1], PNG_FAST)
    # estimate mask + style per line at its reference frame, then refine in/out
    nlines = max(1, sum(len(l) for g, l in groups(T).items() if g in jobs))
    k = 0
    for gid, lines in groups(T).items():
        if gid not in jobs:
            continue
        lo, hi, reg = jobs[gid]
        for t in lines:
            k += 1
            if progress:
                progress(0.6 + 0.4 * k / nlines)
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
            if t.get("logo_full"):
                logo_area(proj, t, gid, S, reg)
    for t in T["titles"]:
        t.pop("reextract", None)
    mark_logos(T, (P.get("info") or {}).get("height") or 1080)
    vio.save_json(os.path.join(proj, "titles.json"), T)


def mark_logos(T, H):
    """A title logo, without asking Claude: a card in colour whose letters are very large, or whose
    lines are of very different sizes (THE small over LOST TIGER big). Marked "styled" so it is designed
    as a logo, not set as plain text. A decision made by Claude's review or by the user stays."""
    for gid, lines in groups(T).items():
        def lh(t):            # measured letter height, else (right after analysis) from the text box
            v = (t.get("style") or {}).get("letter_h")
            b = t.get("ref_box") or t.get("box")
            return v or ((b[3] - b[1]) * 0.75 if b else 0)
        act = [t for t in lines if t.get("translate", True) and lh(t)]
        if not act or any("styled" in t and not t.get("styled_auto") for t in act):
            continue
        hs = [lh(t) for t in act]
        def chroma(t):
            fl = (t.get("mask") or {}).get("fill_lab")
            return ((fl[1] - 128) ** 2 + (fl[2] - 128) ** 2) ** 0.5 if fl else None
        cs = [c for c in (chroma(t) for t in act) if c is not None]
        coloured = (max(cs) > 20) if cs else None          # unknown before the first build
        big = max(hs) >= 0.22 * H
        mixed = len(act) >= 2 and max(hs) >= 2.5 * min(hs) and max(hs) >= 0.12 * H
        logo = ((coloured and (big or mixed)) or (coloured is None and mixed)) and \
            all(t.get("kind", "text") != "sub" for t in act)
        for t in act:
            if logo:
                t["styled"] = True; t["styled_auto"] = True
            elif t.get("styled_auto"):
                t.pop("styled", None); t.pop("styled_auto", None)


def _bg_plate(proj, t, gid, S, f):
    """The background before the title appears (or after it ends), moved with the camera to frame f.
    Returns (crop, frame index) or (None, None) when the title spans its whole cached range."""
    lo, hi = t["cache"]
    pb = None
    if t["in"] - 4 >= lo:
        pb = max(lo, t["in"] - 4)
    elif t["out"] + 4 <= hi:
        pb = min(hi, t["out"] + 4)
    if pb is None:
        return None, None
    bg = cv2.imread(os.path.join(proj, "cache", gid, f"{pb:06d}.png"))
    if bg is None:
        return None, None
    mo = S.get("motion")
    if mo and f < len(mo) and pb < len(mo):
        dx, dy = mo[f][0] - mo[pb][0], mo[f][1] - mo[pb][1]
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            bg = cv2.warpAffine(bg, np.float32([[1, 0, dx], [0, 1, dy]]), (bg.shape[1], bg.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    return bg, pb


def logo_diff(img, bg, thr=28):
    d = np.abs(img.astype(np.int16) - bg.astype(np.int16)).max(2)
    m = (d > thr).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def logo_area(proj, t, gid, S, reg):
    """Whole logo: the graphics that appear together with the text (symbols, frames, ornaments) are found by
    comparing the reference frame with the background before the title. Extends box/ref_box to them."""
    ref = cv2.imread(os.path.join(proj, "cache", gid, f"{t['ref']:06d}.png"))
    bg, pb = _bg_plate(proj, t, gid, S, t["ref"])
    if ref is None or bg is None:
        t["warn"] = "אין רקע בלי הכותרת באותו שוט - אי אפשר לזהות את כל הלוגו"
        return
    tb = t.setdefault("text_box", t["box"]); trb = t.setdefault("text_ref_box", t["ref_box"])
    h = trb[3] - trb[1]
    m = cv2.morphologyEx(logo_diff(ref, bg), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    near = [trb[0] - reg[0] - 1.5 * h, trb[1] - reg[1] - 1.5 * h, trb[2] - reg[0] + 1.5 * h, trb[3] - reg[1] + 1.5 * h]
    box = [trb[0] - reg[0], trb[1] - reg[1], trb[2] - reg[0], trb[3] - reg[1]]
    for i in range(1, n):
        x, y, w, hh, a = st[i]
        if a < 30 or x > near[2] or x + w < near[0] or y > near[3] or y + hh < near[1]:
            continue
        box = [min(box[0], x), min(box[1], y), max(box[2], x + w), max(box[3], y + hh)]
    lb = [int(box[0] + reg[0]), int(box[1] + reg[1]), int(box[2] + reg[0]), int(box[3] + reg[1])]
    t["logo_bg"] = pb
    t["ref_box"] = lb
    t["box"] = [min(tb[0], lb[0]), min(tb[1], lb[1]), max(tb[2], lb[2]), max(tb[3], lb[3])]
    t["logo_bg_done"] = True


def _grad(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def _presence(proj, t, gid, frames):
    """Per frame: how much of this line is on screen (1 = like the reference frame).
    max of (a) colour-model pixel count and (b) correlation of edge structure with the
    reference text - (b) still sees faint text during fades.
    Works on the line's own neighbourhood only (not the whole card), which is what makes long titles fast."""
    reg = t["region"]
    bx = t.get("text_box", t["box"])
    h = bx[3] - bx[1]
    box = [bx[0] - reg[0] - h, bx[1] - reg[1] - h, bx[2] - reg[0] + h, bx[3] - reg[1] + h]
    refimg = cv2.imread(os.path.join(proj, "cache", gid, f"{t['ref']:06d}.png"))
    H, W = refimg.shape[:2]
    m = 20                                             # margin for the 31 px top-hat inside masks.make
    X0, Y0 = max(0, box[0] - m), max(0, box[1] - m)
    X1, Y1 = min(W, box[2] + m), min(H, box[3] + m)
    sub = lambda im: im[Y0:Y1, X0:X1]
    sbox = [box[0] - X0, box[1] - Y0, box[2] - X0, box[3] - Y0]
    params = {**t["mask"], "grow": 0, "shadow": None, "extra": []}
    core = masks.make(sub(refimg), params, sbox) > 0
    band = cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    gref = _grad(sub(refimg))[band]
    gref = (gref - gref.mean()) / (gref.std() + 1e-6)
    cnt, cor = {}, {}
    for f in frames:
        img = cv2.imread(os.path.join(proj, "cache", gid, f"{f:06d}.png"))
        if img is None:
            continue
        si = sub(img)
        mk = masks.make(si, params, sbox)
        cnt[f] = float((mk > 0).sum())
        g = _grad(si)[band]
        g = (g - g.mean()) / (g.std() + 1e-6)
        cor[f] = float((g * gref).mean())
    return cnt, cor


def refine(proj, t, gid, S, n):
    every = S["every"]
    lo, hi = t["cache"]
    fr = list(range(lo, hi + 1))
    pres, cor = _presence(proj, t, gid, fr)
    refc = pres.get(t["ref"], 0) or max(pres.values())
    far = lambda f: f < t.get("prev_sample", t["sample_in"] - every) or f > t.get("next_sample", t["sample_out"] + every)
    outside = [v for f, v in pres.items() if far(f)]
    base = float(np.median(outside)) if outside else 0.0
    cbase = [v for f, v in cor.items() if far(f)]
    cb = float(np.percentile(cbase, 90)) if cbase else 0.1
    # another title at the same place (consecutive subtitles / cards): colour alone cannot tell
    # them apart, so the boundary is decided by the shape of this text only
    from .detect import iou
    T = vio.load_json(os.path.join(proj, "titles.json"))
    shared = any(o["id"] != t["id"] and iou(o["box"], t["box"]) > 0.3 and lo <= o["sample_in"] <= hi + 1 or
                 (o["id"] != t["id"] and iou(o["box"], t["box"]) > 0.3 and lo - 1 <= o["sample_out"] <= hi)
                 for o in T["titles"])
    rel = {}
    for f, v in pres.items():
        a = (v - base) / max(1.0, refc - base)
        c = (cor[f] - cb) / max(0.05, 1 - cb)
        rel[f] = c if shared else max(a, c)
    t["presence"] = {str(f): round(v, 3) for f, v in rel.items()}
    # the walk stops at the shot edges around the samples: text seen in the next shot would have
    # been sampled there (a lookalike - orange rocks after an orange logo - must not stretch the title)
    from .shots import shot_of
    sa, _ = shot_of(S["cuts"], n, t["sample_in"])
    _, eb = shot_of(S["cuts"], n, t["sample_out"])
    lo, hi = max(lo, sa), min(hi, eb - 1)
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
