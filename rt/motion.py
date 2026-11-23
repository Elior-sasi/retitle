"""Full motion of the original title, carried over to the Hebrew layer:

  rotation / perspective  per frame homography (ECC on the text map, seeded by the scale+position match)
  reveal / matte          which part of the original is visible in each frame (wipes, masks, letters appearing,
                          passing behind objects) as a smooth spatial field, applied to the Hebrew layer's alpha
  light / colour          light the original gains over its reference colour, per pixel (flashes, light
                          sweeps, colour changes), applied to the Hebrew layer
  layered logos           a logo given as layers (PSD, or PNGs on one canvas): every layer follows the part of
                          the original under it, with its own motion

Everything is measured on the |source - cleaned| map, so the background does not matter.
Results: t["anim"]["H"] (frame coords), t["anim"]["parts"], and motion/<id>/m<f>.png (matte), g<f>.png (gain).
"""
import os, shutil
import cv2, numpy as np
PNG_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 1]   # still lossless, faster to write
from PIL import Image
from . import vio

MS = 0.5                                   # matte / gain maps are stored at half the region resolution


# ------------------------------------------------------------------ layered logos
def part_files(proj, t):
    lay = t.get("layer") or {}
    return [os.path.join(proj, p) for p in lay.get("files", [])]


def parts_layout(proj, t):
    """Parts of a layered logo: [(PIL RGBA cropped, [x0, y0, x1, y1] in frame coords)].
    All files share one canvas; the canvas is fitted to the width of the original text block."""
    files = part_files(proj, t)
    if not files:
        return []
    ims = [Image.open(p).convert("RGBA") for p in files]
    cw, ch = ims[0].size
    rb = t.get("ref_box", t["box"]); st = t.get("style", {})
    # fit the union of everything that is drawn, like a single PNG logo
    boxes = [im.getbbox() for im in ims]
    ub = [min(b[0] for b in boxes if b), min(b[1] for b in boxes if b), max(b[2] for b in boxes if b), max(b[3] for b in boxes if b)]
    k = max(8.0, (rb[2] - rb[0]) * st.get("size_ratio", 1.0)) / max(1, ub[2] - ub[0])
    cx = (rb[0] + rb[2]) / 2 + st.get("nudge_x", 0); cy = (rb[1] + rb[3]) / 2 + st.get("nudge_y", 0)
    ox = cx - (ub[0] + ub[2]) / 2 * k; oy = cy - (ub[1] + ub[3]) / 2 * k
    out = []
    for im, b in zip(ims, boxes):
        if not b:
            continue
        out.append((im.crop(b), [ox + b[0] * k, oy + b[1] * k, ox + b[2] * k, oy + b[3] * k]))
    return out


def import_layers(proj, tid, paths):
    """PSD (its pixel layers) or several PNGs on one canvas -> logos/<tid>/NN.png. Returns relative paths."""
    d = os.path.join(proj, "logos", tid)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    out = []
    psd = [p for p in paths if p.lower().endswith(".psd")]
    if psd:
        from psd_tools import PSDImage
        doc = PSDImage.open(psd[0])
        W, H = doc.width, doc.height
        layers = [l for l in doc.descendants() if l.kind in ("pixel", "type", "shape", "smartobject") and l.is_visible()]
        for i, l in enumerate(layers):
            try:
                im = l.topil()
            except Exception:
                im = None
            if im is None:
                continue
            can = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            can.alpha_composite(im.convert("RGBA"), (max(0, l.left), max(0, l.top)))
            p = os.path.join(d, f"{i:02d}.png"); can.save(p)
            out.append(os.path.relpath(p, proj).replace("\\", "/"))
        if not out:                                      # flattened PSD: the picture as one layer
            im = doc.topil() or doc.composite()
            p = os.path.join(d, "00.png"); im.convert("RGBA").save(p)
            out.append(os.path.relpath(p, proj).replace("\\", "/"))
    else:
        size = None
        for i, p in enumerate(paths):
            im = Image.open(p).convert("RGBA")
            if size and im.size != size:
                raise ValueError("כל קבצי ה-PNG צריכים להיות באותו גודל קנבס (כמו ייצוא שכבות מפוטושופ)")
            size = im.size
            q = os.path.join(d, f"{i:02d}.png"); im.save(q)
            out.append(os.path.relpath(q, proj).replace("\\", "/"))
    if not out:
        raise ValueError("לא נמצאו שכבות עם תמונה בקובץ")
    return out


# ------------------------------------------------------------------ measuring
def _tmap(proj, gid, f, lines, t, reg):
    o = cv2.imread(os.path.join(proj, "cache", gid, f"{f:06d}.png"))
    c = cv2.imread(os.path.join(proj, "clean", gid, f"{f:06d}.png"))
    if o is None or c is None:
        return None, None
    D = np.abs(o.astype(np.int16) - c.astype(np.int16)).max(2).astype(np.float32)
    D[D < 10] = 0
    for o2 in lines:                                       # other lines of the same card follow their own motion
        if o2 is t or not o2.get("translate", True):
            continue
        b2 = o2["ref_box"]; p2 = int((b2[3] - b2[1]) * 0.3)
        D[max(0, b2[1] - reg[1] - p2):max(0, b2[3] - reg[1] + p2), max(0, b2[0] - reg[0] - p2):max(0, b2[2] - reg[0] + p2)] = 0
    return D, o


def _ecc(tpl, img, init, kind):
    """ECC on the template's own area only (a layer's template is mostly empty)."""
    ys, xs = np.nonzero(tpl > 0.05)
    if len(xs) < 10:
        return None, -1
    m = 6
    x0, y0 = max(0, xs.min() - m), max(0, ys.min() - m)
    x1, y1 = min(tpl.shape[1], xs.max() + m + 1), min(tpl.shape[0], ys.max() + m + 1)
    O = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
    w, cc = _ecc_raw(np.ascontiguousarray(tpl[y0:y1, x0:x1]), img, init @ O, kind)
    return (None, cc) if w is None else (w @ np.linalg.inv(O), cc)


def _ecc_raw(tpl, img, init, kind):
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4)
    w = init.astype(np.float32).copy() if kind == cv2.MOTION_HOMOGRAPHY else init[:2].astype(np.float32).copy()
    try:
        cc, w = cv2.findTransformECC(tpl, img, w, kind, crit, None, 5)
    except cv2.error:
        return None, -1
    if kind != cv2.MOTION_HOMOGRAPHY:
        w = np.vstack([w, [0, 0, 1]])
    return w.astype(np.float64), cc


def _sane(Hm, prev=None, h=40):
    a = Hm[:2, :2] / max(1e-6, Hm[2, 2])
    det = np.linalg.det(a)
    if not (0.04 < det < 25):
        return False
    if abs(Hm[2, 0]) > 0.002 or abs(Hm[2, 1]) > 0.002:   # extreme perspective: not a title move
        return False
    sv = np.linalg.svd(a, compute_uv=False)
    if sv[0] / max(1e-6, sv[1]) > 1.6:                     # titles keep their proportions (roughly)
        return False
    if prev is not None:                                   # continuity with the neighbouring frame
        pa = prev[:2, :2]; pd = abs(np.linalg.det(pa))
        if abs(det / max(1e-6, pd) - 1) > 0.25 or np.abs(Hm[:2, 2] - prev[:2, 2]).max() > max(40, 2.5 * h):
            return False
    return True


def _score(tpl, img, Hm):
    w = cv2.warpPerspective(tpl, Hm, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    m = (w > 0.05) | (img > 0.05)
    if m.sum() < 20:
        return 0.0
    a = w[m] - w[m].mean(); b = img[m] - img[m].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def _scale_h(Hm, s):
    S = np.diag([s, s, 1.0]); Si = np.diag([1 / s, 1 / s, 1.0])
    return S @ Hm @ Si


def _field(num, den, sigma):
    """Normalised convolution: smooth num/den, extrapolated where den is empty. Computed on a reduced grid
    (the field is smooth by design) and returned at MS of the input size."""
    H0, W0 = num.shape[:2]
    q = min(1.0, 3.0 / max(1.0, sigma))
    sz = (max(4, int(W0 * q)), max(4, int(H0 * q)))
    n = cv2.resize(num, sz, interpolation=cv2.INTER_AREA); d = cv2.resize(den, sz, interpolation=cv2.INTER_AREA)
    sg = sigma * q
    n1 = cv2.GaussianBlur(n, (0, 0), sg); d1 = cv2.GaussianBlur(d, (0, 0), sg)
    n2 = cv2.GaussianBlur(n, (0, 0), sg * 3); d2 = cv2.GaussianBlur(d, (0, 0), sg * 3)
    v = np.where(d1 > 0.02, n1 / np.maximum(d1, 1e-6), np.where(d2 > 0.005, n2 / np.maximum(d2, 1e-6), np.nan))
    return cv2.resize(v.astype(np.float32), (int(W0 * MS), int(H0 * MS)), interpolation=cv2.INTER_LINEAR)


def refine(proj, t, gid, lines, parts_rects=None):
    """Adds H / parts / matte / gain to t["anim"] (which already holds [dx, dy, sc, alpha] per frame)."""
    reg = t["region"]; bx = t["ref_box"]; h = bx[3] - bx[1]
    Dref, Oref = _tmap(proj, gid, t["ref"], lines, t, reg)
    if Dref is None:
        return
    RH, RW = Dref.shape
    pad = int(h * 0.4)
    win = np.zeros_like(Dref)
    win[max(0, bx[1] - reg[1] - pad):bx[3] - reg[1] + pad, max(0, bx[0] - reg[0] - pad):bx[2] - reg[0] + pad] = 1
    R = Dref * win
    if not (R > 0).any():
        return
    p_ref = float(np.percentile(R[R > 0], 90))
    norm = lambda D: np.clip(D / max(1e-3, p_ref * 0.5), 0, 1)
    tpl_full = norm(R)
    # templates: the whole title, and one per layer of a layered logo (the part of the original under it)
    tpls = [("all", tpl_full)]
    for i, rc in enumerate(parts_rects or []):
        r = np.zeros(tpl_full.shape, np.uint8)
        x0, y0, x1, y1 = [int(round(v)) for v in (rc[0] - reg[0], rc[1] - reg[1], rc[2] - reg[0], rc[3] - reg[1])]
        r[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1
        # whole shapes of the original under this layer (a shape cut at the layer's edge would distort the fit)
        blobs = cv2.dilate((tpl_full > 0.3).astype(np.uint8), np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(blobs, 8)
        keep = np.zeros(n, bool)
        for j in range(1, n):
            inside = int(((lab == j) & (r > 0)).sum())
            keep[j] = inside > 0.3 * st[j, cv2.CC_STAT_AREA]
        tp = tpl_full * keep[lab]
        tpls.append((i, tp if (tp > 0.5).sum() > 30 else None))      # None: no original under it -> whole motion
    s = float(np.clip(160.0 / max(1, h), 0.15, 0.5))          # ECC on a small map: letters ~ 160 px -> fast
    small = lambda a: cv2.GaussianBlur(cv2.resize(a, (int(RW * s), int(RH * s)), interpolation=cv2.INTER_AREA), (0, 0), 1.2)
    tsm = {k: small(v) if v is not None else None for k, v in tpls}
    anim = t.setdefault("anim", {"mode": "static"})
    frames = anim.get("frames", {})
    rcx = (bx[0] + bx[2]) / 2 - reg[0]; rcy = (bx[1] + bx[3]) / 2 - reg[1]
    Hs = {k: {} for k, _ in tpls}
    prevH = {k: None for k, _ in tpls}
    mdir = os.path.join(proj, "motion", t["id"])
    shutil.rmtree(mdir, ignore_errors=True)
    mattes, gains = [], []
    sigma = max(3.0, h * 0.45)
    ref_rgb = Oref.astype(np.float32)
    ref_fill = (tpl_full > 0.6)
    # from the reference frame (fully visible) outwards, so each frame starts from its neighbour's motion -
    # reveals and exits at the edges then start from a good guess
    ref = min(max(t["ref"], t["in"]), t["out"])
    order = list(range(ref, t["out"] + 1)) + list(range(ref - 1, t["in"] - 1, -1))
    for f in order:
        if f == ref - 1:
            prevH = {k: Hs[k].get(ref) for k, _ in tpls}
        D, O = _tmap(proj, gid, f, lines, t, reg)
        if D is None:
            continue
        S = norm(D)
        Ssm = small(S)
        v = frames.get(str(f), [0, 0, 1, 1])
        dx, dy, sc = v[0], v[1], v[2]
        H0 = np.array([[sc, 0, rcx * (1 - sc) + dx], [0, sc, rcy * (1 - sc) + dy], [0, 0, 1]], np.float64)
        warped_ref = np.zeros_like(S)
        refw = np.zeros((RH, RW, 3), np.float32); fillw = np.zeros((RH, RW), np.float32)
        use_parts = len(tpls) > 1

        def acc(k, src, Hm):
            nonlocal warped_ref, refw, fillw
            if use_parts and k == "all":
                return                                # a layered logo: the parts carry their own motion
            if use_parts and src is tpl_full:          # a layer with nothing of the original under it
                return
            sel = (src > 0.05).astype(np.float32)
            sel = cv2.dilate(sel, np.ones((int(h * 0.3) | 1,) * 2, np.uint8))
            warped_ref = np.maximum(warped_ref, cv2.warpPerspective(src, Hm, (RW, RH)))
            refw += cv2.warpPerspective(ref_rgb * sel[..., None], Hm, (RW, RH))
            fillw = np.maximum(fillw, cv2.warpPerspective((ref_fill * sel).astype(np.float32), Hm, (RW, RH)))
        for k, _ in tpls:
            tp = tsm[k]
            if tp is None or (v[3] if len(v) > 3 else 1) < 0.15:
                Hs[k][f] = None; continue
            full_src = tpl_full if k == "all" else tpls[[x for x, _ in tpls].index(k)][1]
            if prevH[k] is not None:
                # partly hidden (reveal, wipe, passing behind something): geometry does not change, hold it -
                # the matte below carries the visibility
                # mass ratio, insensitive to small misalignment: visible ink vs the ink the moved template should have
                pm = cv2.warpPerspective((full_src > 0.5).astype(np.float32), prevH[k], (RW, RH))
                near = cv2.dilate((pm > 0.5).astype(np.uint8), np.ones((int(h * 0.5) | 1, int(h * 0.5) | 1), np.uint8)) > 0
                seen = float(((S > 0.5) & near).sum()) / max(1.0, float((pm > 0.5).sum()))
                if seen < 0.8:
                    Hs[k][f] = prevH[k]
                    acc(k, full_src, prevH[k])
                    continue
            cands = ([prevH[k]] if prevH[k] is not None else []) + [H0]
            best = (cands[0], _score(tp, Ssm, _scale_h(cands[0], s)))
            # simplest model first; a richer one only when it clearly fits better (flat shapes make rich models drift)
            for init in cands:
                for kind, need in ((cv2.MOTION_AFFINE, 0.01), (cv2.MOTION_HOMOGRAPHY, 0.03)):
                    w, cc = _ecc(tp, Ssm, _scale_h(init, s), kind)
                    if w is None:
                        continue
                    Hf = _scale_h(w, 1 / s)
                    if not _sane(Hf, prevH[k], h):
                        continue
                    sco = _score(tp, Ssm, _scale_h(Hf, s))
                    if sco > best[1] + need:
                        best = (Hf, sco)
                if best[1] > 0.92:
                    break
            Hs[k][f] = best[0]; prevH[k] = best[0]
            acc(k, full_src, best[0])
        if Hs["all"].get(f) is None:
            continue
        # reveal matte: how much of the (moved) original is actually visible here
        Mref = (warped_ref > 0.3).astype(np.float32)
        if Mref.sum() > 20:
            V = _field(np.minimum(S, 1) * Mref, Mref, sigma)
            V = np.nan_to_num(V, nan=float(np.nanmean(V)) if np.isfinite(V).any() else 1.0)
            V = np.clip(V, 0, 1)
            core = cv2.resize(Mref, (V.shape[1], V.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
            if core.any() and (float(V[core].std()) > 0.06 or float(np.percentile(V[core], 5)) < 0.55):
                os.makedirs(mdir, exist_ok=True)
                cv2.imwrite(os.path.join(mdir, f"m{f:06d}.png"), (V * 255).astype(np.uint8), PNG_FAST)
                mattes.append(f)
            # light / colour: the original's colour against its reference colour, on visible fill pixels
            fillw = cv2.erode((fillw > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0     # letter interiors only
            vis = (fillw & (S > 0.8) & (refw.max(2) > 40)).astype(np.float32)
            if vis.sum() > max(30, 0.25 * float(ref_fill.sum())):     # enough of the letters to judge their colour
                # additive light (flash, sweep, glow, colour shift): what the original gained over its reference colour
                G = np.stack([_field((O[..., c].astype(np.float32) - refw[..., c]) * vis, vis, sigma) for c in range(3)], 2)
                G = np.clip(np.nan_to_num(G, nan=0.0), -127, 127)
                vs = cv2.resize(vis, (G.shape[1], G.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
                dev = np.abs(G[vs]).max(1) if vs.any() else np.zeros(1)
                if float(np.median(dev)) > 14:
                    os.makedirs(mdir, exist_ok=True)
                    cv2.imwrite(os.path.join(mdir, f"g{f:06d}.png"), np.clip(G + 128, 0, 255).astype(np.uint8), PNG_FAST)
                    gains.append(f)
    # to frame coordinates
    T = np.array([[1, 0, reg[0]], [0, 1, reg[1]], [0, 0, 1]], np.float64); Ti = np.linalg.inv(T)
    to_frame = lambda Hm: [round(float(x), 6) for x in (T @ Hm @ Ti).ravel()]
    def moving(v):
        m = np.array(v).reshape(3, 3)
        return abs(m[0, 2] - reg[0] * (m[0, 0] - 1) - reg[1] * m[0, 1]) > 0 and (
            np.abs(m[:2, :2] - np.eye(2)).max() > 0.02 or abs(m[2, 0]) > 1e-5 or abs(m[2, 1]) > 1e-5)
    moved = False
    for k, hs in Hs.items():
        out = {str(f): to_frame(Hm) for f, Hm in hs.items() if Hm is not None}
        # rotation / skew / perspective beyond what [dx, dy, scale] already describes
        if any(abs(v[1]) > 0.01 or abs(v[3]) > 0.01 or abs(v[0] - v[4]) > 0.02 or abs(v[6]) > 1e-5 or abs(v[7]) > 1e-5
               for v in out.values()):
            moved = True
        if k == "all":
            anim["H"] = out
        else:
            anim.setdefault("parts", {})[str(k)] = {"H": out} if tpls[k + 1][1] is not None else {"follow": True}
        if k != "all" and out:
            moved = True
    if not parts_rects:
        anim.pop("parts", None)
    anim["matte"] = mattes; anim["gain"] = gains
    anim["center"] = [(bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2]
    anim["mirror"] = mirror_default(t) if t.get("reveal_dir", "auto") == "auto" else t["reveal_dir"] == "mirror"
    anim["mref"] = {"region": reg[:2], "scale": MS}
    if moved or mattes or gains:
        anim["mode"] = "match"
        anim.setdefault("frames", {str(f): [0, 0, 1, 1.0] for f in range(t["in"], t["out"] + 1)})


def mirror_default(t):
    """Reveals of Latin text run left to right; for the Hebrew they are mirrored (right to left)."""
    import re
    src = t.get("src") or ""
    return bool(re.search(r"[A-Za-z]", src)) and not re.search(r"[\u0590-\u05FF]", src)


# ------------------------------------------------------------------ applying
def _nearest(d, f):
    v = d.get(str(f))
    if v is not None:
        return v
    if not d:
        return None
    keys = sorted(int(k) for k in d)
    k = keys[0] if f < keys[0] else keys[-1] if f > keys[-1] else min(keys, key=lambda x: abs(x - f))
    return d[str(k)]


def homography(anim, f, part=None):
    if part is not None:
        p = (anim.get("parts") or {}).get(str(part)) or {}
        if p.get("H"):
            v = _nearest(p["H"], f)
            return np.array(v, np.float64).reshape(3, 3) if v else None
    if anim.get("H"):
        v = _nearest(anim["H"], f)
        return np.array(v, np.float64).reshape(3, 3) if v else None
    return None


_MAP = {}


def _load_map(path, color):
    st = os.path.getmtime(path) if os.path.exists(path) else None
    k = (path, st)
    if k in _MAP:
        return _MAP[k]
    if st is None:
        return None
    im = cv2.imread(path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if len(_MAP) > 400:
        _MAP.clear()
    _MAP[k] = im
    return im


def _paste(full, m, ox, oy):
    """Put map m into full at (ox, oy), clipped at every edge (the offset can be negative)."""
    H, W = full.shape[:2]; h, w = m.shape[:2]
    x0, y0 = max(0, ox), max(0, oy); x1, y1 = min(W, ox + w), min(H, oy + h)
    if x1 > x0 and y1 > y0:
        full[y0:y1, x0:x1] = m[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return full


def apply_maps(L, proj, tid, anim, f, W, H, shift=(0, 0)):
    """Matte (alpha) and light/colour gain on a premultiplied full-frame layer. Returns (layer, matte_used).
    shift: the manual offset of the Hebrew - the maps go with it, so a moved title is revealed the same way."""
    if not proj or not (anim.get("matte") or anim.get("gain")):
        return L, False
    mr = anim.get("mref") or {}
    ox, oy = mr.get("region", [0, 0]); s = mr.get("scale", MS)
    ox, oy = int(round(ox + shift[0])), int(round(oy + shift[1]))
    used = False
    d = os.path.join(proj, "motion", tid)
    if f in set(anim.get("matte", [])):
        m = _load_map(os.path.join(d, f"m{f:06d}.png"), False)
        if m is not None:
            full = np.ones((H, W), np.float32)
            mm = cv2.resize(m, (int(round(m.shape[1] / s)), int(round(m.shape[0] / s))), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
            _paste(full, mm, ox, oy)
            if anim.get("mirror") and anim.get("center"):
                # the original was revealed in its reading direction (left to right): Hebrew goes right to left
                c = np.array([*anim["center"], 1.0])
                Hm = homography(anim, f)
                xc = ((Hm @ c)[0] / (Hm @ c)[2] if Hm is not None else c[0]) + shift[0]
                idx = np.clip(np.round(2 * xc - np.arange(W)).astype(int), 0, W - 1)
                full = full[:, idx]
            L = L * full[..., None]
            used = True
    if f in set(anim.get("gain", [])):
        g = _load_map(os.path.join(d, f"g{f:06d}.png"), True)
        if g is not None:
            gg = (cv2.resize(g, (int(round(g.shape[1] / s)), int(round(g.shape[0] / s))), interpolation=cv2.INTER_LINEAR).astype(np.float32) - 128) / 255
            full = _paste(np.zeros((H, W, 3), np.float32), gg, ox, oy)
            a = L[..., 3:4]
            rgb = L[..., :3] + full * a                                          # light added where the letters are
            L = np.concatenate([np.clip(rgb, 0, a), a], 2)                       # premultiplied: 0 <= colour <= alpha
    return L, used
