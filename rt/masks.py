"""Text mask model: which pixels belong to the burned-in text (fill + outline + shadow)."""
import cv2, numpy as np


def _shift(m, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST)


CHROMA_W = 0.35  # video is 4:2:0 - thin strokes lose saturation; weigh lightness over colour


def _wdist(lab, ref):
    d = lab - np.asarray(ref, np.float32)
    return np.sqrt(d[..., 0] ** 2 + CHROMA_W * (d[..., 1] ** 2 + d[..., 2] ** 2))


def estimate(crop, words, stroke_hint=None):
    """crop: BGR image of the region at the reference frame.
    words: word boxes in crop coords. Returns mask params + style guesses."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    core = np.zeros(gray.shape, np.uint8)
    for x0, y0, x1, y1 in words:
        h = y1 - y0
        p = max(4, int(0.25 * h))
        X0, Y0, X1, Y1 = max(0, int(x0) - p), max(0, int(y0) - p), min(W, int(x1) + p), min(H, int(y1) + p)
        if X1 - X0 < 8 or Y1 - Y0 < 8:
            continue
        L = lab[Y0:Y1, X0:X1].reshape(-1, 3)
        k = 4
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, lbl, cen = cv2.kmeans(L, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lbl = lbl.reshape(Y1 - Y0, X1 - X0)
        inner = np.zeros_like(lbl, bool)
        inner[int(y0) - Y0:int(y1) - Y0, int(x0) - X0:int(x1) - X0] = True
        ring = np.zeros_like(lbl, bool)
        ring[:2] = ring[-2:] = True; ring[:, :2] = ring[:, -2:] = True
        best = None
        for c in range(k):
            sel = lbl == c
            fi = (sel & inner).sum() / max(1, inner.sum())
            fb = (sel & ring).sum() / max(1, ring.sum())
            sc = fi - 3 * fb
            if fi > 0.08 and (best is None or sc > best[0]):
                best = (sc, c)
        if best is None:
            continue
        tm = (lbl == best[1]) & inner
        core[Y0:Y1, X0:X1] |= tm.astype(np.uint8)
    if core.sum() < 20:
        return None
    # erode a little so the colour sample is the fill, not the antialias edge
    inner = cv2.erode(core, np.ones((3, 3), np.uint8)) if core.sum() > 400 else core
    fill_lab = np.median(lab[inner > 0], axis=0)
    dist = _wdist(lab[inner > 0], fill_lab)
    tol = float(np.clip(np.percentile(dist, 90) * 1.6 + 12, 22, 70))
    fill_bgr = np.median(crop[inner > 0], axis=0)
    hgt = float(np.median([y1 - y0 for x0, y0, x1, y1 in words]))
    # stroke width via distance transform of the core
    dt = cv2.distanceTransform(core, cv2.DIST_L2, 3)
    stroke = float(np.percentile(dt[core > 0], 90) * 2) if core.any() else 3
    # outline: ring just outside the fill with a consistent colour different from fill
    outline = None
    d1 = cv2.dilate(core, np.ones((3, 3), np.uint8), iterations=max(1, int(stroke * 0.25)))
    ring = (d1 > 0) & (core == 0)
    if ring.sum() > 50:
        rl = lab[ring]
        med = np.median(rl, axis=0)
        spread = np.median(np.linalg.norm(rl - med, axis=1))
        if spread < 18 and np.linalg.norm(med - fill_lab) > 40:
            oc = med
            wdt = 0
            prev = core.copy()
            for r in range(1, 16):
                dr = cv2.dilate(core, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
                rr = (dr > 0) & (prev == 0)
                if rr.sum() < 10 or np.median(np.linalg.norm(lab[rr] - oc, axis=1)) > 30:
                    break
                wdt = r; prev = dr
            outline = {"color": [int(v) for v in np.median(crop[ring], axis=0)], "width": max(1, wdt)}
    # drop shadow: offset that darkens pixels most
    best = None
    ow = outline["width"] if outline else 0
    base = (cv2.dilate(core, np.ones((5 + 2 * ow, 5 + 2 * ow), np.uint8)) > 0)
    bgref = gray[~cv2.dilate(core, np.ones((15, 15), np.uint8)).astype(bool)]
    bgmean = float(bgref.mean()) if bgref.size else 128
    lim = int(max(4, min(20, hgt * 0.15)))
    for dy in range(-lim, lim + 1, 1):
        for dx in range(-lim, lim + 1, 1):
            if dx == 0 and dy == 0:
                continue
            if max(abs(dx), abs(dy)) <= ow + 1:
                continue
            sm = _shift(core, dx, dy).astype(bool) & ~base
            if sm.sum() < 30:
                continue
            dark = bgmean - float(gray[sm].mean())
            if best is None or dark > best[0]:
                best = (dark, dx, dy)
    shadow = None
    if best and best[0] > 25:
        _, dx, dy = best
        sm = _shift(core, dx, dy).astype(bool) & ~base
        shadow = {"dx": dx, "dy": dy, "color": [int(v) for v in np.median(crop[sm], axis=0)],
                  "opacity": float(np.clip(best[0] / max(bgmean, 1), 0.2, 0.95)), "blur": 2}
    return {
        "mask": {"fill_lab": [float(v) for v in fill_lab], "tol": tol,
                 "grow": int(max(3, round(stroke * 0.35), round(hgt * 0.05))) + (outline["width"] if outline else 0),
                 "shadow": [shadow["dx"], shadow["dy"]] if shadow else None,
                 "shadow_grow": int(max(2, round(stroke * 0.4)))},
        "style": {"fill": [int(v) for v in fill_bgr], "outline": outline, "shadow": shadow,
                  "letter_h": hgt, "stroke": stroke},
    }


def make(crop, params, box=None, structure=True):
    """Binary mask (uint8 0/255) of text pixels to remove, in crop coords.
    box limits the search (crop coords, already expanded by caller)."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = _wdist(lab, params["fill_lab"])
    m = (d < params["tol"]).astype(np.uint8)
    if structure:
        # keep only thin, text-like structures: remove big flat blobs of the same colour
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
        blk = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
        contrast = cv2.max(top, blk)
        contrast = cv2.dilate(contrast, np.ones((5, 5), np.uint8))
        m &= (contrast > 18).astype(np.uint8)
    if box is not None:
        lim = np.zeros_like(m)
        x0, y0, x1, y1 = [int(v) for v in box]
        lim[max(0, y0):y1, max(0, x0):x1] = 1
        m &= lim
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    out = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                     (2 * params["grow"] + 1,) * 2))
    if params.get("shadow"):
        dx, dy = params["shadow"]
        sg = params.get("shadow_grow", 3)
        sh = cv2.dilate(_shift(m, dx, dy), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * sg + 1,) * 2))
        # cover the whole streak between text and shadow (long / 3D shadows)
        steps = int(max(abs(dx), abs(dy)))
        for s in range(1, steps):
            sh |= _shift(out, dx * s / steps, dy * s / steps)
        out |= sh
    for extra in params.get("extra", []):  # manual additions [x0,y0,x1,y1] in crop coords
        x0, y0, x1, y1 = extra
        out[y0:y1, x0:x1] = 1
    return (out > 0).astype(np.uint8) * 255
