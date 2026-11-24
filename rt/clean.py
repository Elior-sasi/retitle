"""Pass 4: remove the source text. Background is rebuilt from neighbouring frames of the same shot
(aligned with ECC, brightness matched, median) - real pixels, no blur. Only what is never visible
in any neighbour is synthesised (FSR inpainting)."""
import os, shutil, warnings, time
warnings.filterwarnings("ignore", category=RuntimeWarning)
import cv2, numpy as np
PNG_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 1]   # still lossless, faster to write
from . import vio, masks, live
from .shots import shot_of
from .extract import groups


_IMG, _MSK = {}, {}


def _load(proj, gid, f):
    k = (proj, gid, f)
    if k not in _IMG:
        _IMG[k] = cv2.imread(os.path.join(proj, "cache", gid, f"{f:06d}.png"))
    return _IMG[k]


_BG = {}
_CTX = {}


def line_mask(img, t, reg, f=None, proj=None, gid=None):
    bx = t.get("text_box", t["box"])
    h = bx[3] - bx[1]
    s = t.get("search", h)
    box = [bx[0] - reg[0] - s, bx[1] - reg[1] - s, bx[2] - reg[0] + s, bx[3] - reg[1] + s]
    m = masks.make(img, t["mask"], box)
    fl = (t.get("mask") or {}).get("fill_lab")
    chroma = fl is not None and ((fl[1] - 128) ** 2 + (fl[2] - 128) ** 2) ** 0.5 > 20
    if t.get("styled") or t.get("design") or t.get("design_member") or (chroma and t.get("kind") != "sub"):
        # coloured / designed lettering: shading and gradients go beyond one fill colour
        m = m | styled_mask(img, t, reg, box)
        logo = bool(t.get("design") or t.get("design_member"))
        if f is not None and (logo or t.get("kind") != "sub"):
            # letters that pop, zoom or slide in leave their final place: a logo is looked for in the
            # whole card, other coloured titles in a wide zone around them
            bw = box[2] - box[0]
            k = 2.0 if t.get("strong") else 1.0          # "strong": Claude saw leftovers - look further
            zone = True if logo else [box[0] - int(0.35 * k * bw), box[1] - int(h * k), box[2] + int(0.35 * k * bw), box[3] + int(h * k)]
            w = styled_mask(img, t, reg, box, wide=zone)
            # far from the logo only big pieces count (the logo zoomed or moved); small specks
            # there are the scene's own colours (sparks, reflections) and stay
            lh = int(max(8, (t.get("style") or {}).get("letter_h") or h))
            near = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lh | 1, lh | 1))) > 0
            far = ((w > 0) & ~near).astype(np.uint8)
            if far.any():
                n_, lab_, st_, _ = cv2.connectedComponentsWithStats(far)
                small = np.zeros(n_, bool); small[1:] = st_[1:, 4] < 0.5 * lh * lh
                w[small[lab_] & (far > 0)] = 0
            # far more "letters" than the title has: the scene itself is in that colour here (a fade from
            # another shot, a fire, a sunset) - keep to the title's own box on this frame
            bwh = (bx[2] - bx[0]) * (bx[3] - bx[1])
            if (w > 0).sum() <= max((8.0 if logo else 3.0) * bwh, 1.5 * (m > 0).sum()):
                m = m | w
    if t.get("logo_bg") is not None and proj and f is not None:
        # whole logo: everything that differs from the background without the title, around the logo box
        from .extract import _bg_plate, logo_diff
        k = (proj, gid, t["id"], f)
        if k not in _BG:
            if len(_BG) > 64:
                _BG.clear()
            S = vio.load_json(os.path.join(proj, "shots.json"), {})
            _BG[k] = _bg_plate(proj, t, gid, S, f)[0]
        bg = _BG[k]
        if bg is not None and bg.shape == img.shape:
            lb = t["box"]; hh = max(8, (t.get("text_ref_box", lb)[3] - t.get("text_ref_box", lb)[1]) // 2)
            lim = np.zeros(img.shape[:2], np.uint8)
            lim[max(0, lb[1] - reg[1] - hh):max(0, lb[3] - reg[1] + hh), max(0, lb[0] - reg[0] - hh):max(0, lb[2] - reg[0] + hh)] = 1
            g = logo_diff(img, bg) & lim
            g = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t["mask"].get("grow", 3) + 1,) * 2))
            m = m | (g * 255)
    return m


_PAL = {}


def _palette(img, t, reg, x0, y0, x1, y1):
    """Letter and background colour centres, learnt on the title's reference frame (where it is
    surely on screen and in place), not on whatever frame happens to come first."""
    proj, gid = _CTX.get("proj"), t.get("group", t["id"])
    if proj and t.get("ref") is not None and os.path.exists(os.path.join(proj, "cache", gid, f"{t['ref']:06d}.png")):
        r = _load(proj, gid, t["ref"])
        if r is not None and r.shape == img.shape:
            img = r
    bx = t.get("text_box", t["box"])
    ix0, iy0, ix1, iy1 = bx[0] - reg[0], bx[1] - reg[1], bx[2] - reg[0], bx[3] - reg[1]
    inner = np.zeros(img.shape[:2], bool)
    for w in t.get("words") or [bx]:
        inner[max(0, w[1] - reg[1]):max(0, w[3] - reg[1]), max(0, w[0] - reg[0]):max(0, w[2] - reg[0])] = True
    core = inner & (masks.make(img, t["mask"], [ix0, iy0, ix1, iy1]) > 0)
    ring = np.zeros(img.shape[:2], bool); ring[y0:y1, x0:x1] = True; ring[max(0, iy0):iy1, max(0, ix0):ix1] = False
    fl = (t.get("mask") or {}).get("fill_lab")
    design = bool(t.get("design") or t.get("design_member")) or bool(
        fl is not None and ((fl[1] - 128) ** 2 + (fl[2] - 128) ** 2) ** 0.5 > 20 and t.get("kind") != "sub")
    if design and _CTX.get("T"):
        # a logo's other lines are not background
        for o in _CTX["T"]["titles"]:
            if o.get("group", o["id"]) == t.get("group", t["id"]):
                ob = o.get("text_box", o["box"])
                ring[max(0, ob[1] - reg[1]):max(0, ob[3] - reg[1]), max(0, ob[0] - reg[0]):max(0, ob[2] - reg[0])] = False
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    def km(px, k):
        if len(px) < 30:
            return None
        px = px[np.random.default_rng(0).choice(len(px), min(len(px), 20000), replace=False)]
        _, _, c = cv2.kmeans(px, min(k, len(px) // 10), None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0), 2, cv2.KMEANS_PP_CENTERS)
        return c
    # letters: the fill and what is drawn inside the word boxes but differs from the ring
    lc = km(lab[inner & ~ring].reshape(-1, 3), 5)
    bc = km(lab[ring].reshape(-1, 3), 6)
    wide_ok = False
    if core.sum() > 30 and lc is not None and bc is not None:
        d = np.min(np.linalg.norm(lc[:, None] - bc[None], axis=2), axis=1)
        lc = lc[d > 12] if (d > 12).any() else lc
        if design:
            # the whole card is searched only with letter colours far from every background colour
            far = np.min(np.linalg.norm(lc[:, None] - bc[None], axis=2), axis=1) > 30
            if far.any():
                wide_ok = True
                # logo parts outside the word boxes (strokes, underline, darker shading) land in the
                # ring: background centres of the letters' own hue would shield those letters
                def hue(c):
                    return np.degrees(np.arctan2(c[:, 2] - 128, c[:, 1] - 128))
                lh = hue(lc[far]); bh = hue(bc)
                bch = np.hypot(bc[:, 1] - 128, bc[:, 2] - 128)
                dh = np.abs((bh[:, None] - lh[None] + 180) % 360 - 180).min(1)
                keep = ~((bch > 20) & (dh < 35))
                if keep.any() and not keep.all():
                    rp = lab[ring].reshape(-1, 3)
                    rp = rp[np.random.default_rng(1).choice(len(rp), min(len(rp), 20000), replace=False)]
                    near_c = np.argmin(((rp[:, None] - bc[None]) ** 2).sum(2), axis=1)
                    if np.isin(near_c, np.where(~keep)[0]).mean() < 0.12:
                        bc = bc[keep]          # a few logo bits, not a background of the logo's colour
                    else:
                        wide_ok = False        # the scene itself has the letters' hue: stay near the title
    return lc, bc, wide_ok


def styled_mask(img, t, reg, box, wide=False):
    """Designed lettering (painted, textured, 3D, stripes, outlines) is more than one fill colour.
    Every pixel near the title whose colour is closer to the letters' colours than to the background's
    is taken, then closed and grown - so no orange bits of the old logo are left at the edges."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])
    if x1 - x0 < 8 or y1 - y0 < 8:
        return np.zeros(img.shape[:2], np.uint8)
    key = t["id"]
    if key not in _PAL:
        _PAL[key] = _palette(img, t, reg, x0, y0, x1, y1)
        if len(_PAL) > 256:
            _PAL.clear()
    lc, bc, pal_ok = _PAL[key]
    if wide is not False and pal_ok:
        # far from the title only clearly coloured letter centres search (white highlights of the
        # letters would also find every glint and edge of the scene)
        far_ = np.min(np.linalg.norm(lc[:, None] - bc[None], axis=2), axis=1) > 30
        chro_ = np.hypot(lc[:, 1] - 128, lc[:, 2] - 128) > 25
        lc = lc[far_ & chro_]
        if not len(lc):
            return np.zeros(img.shape[:2], np.uint8)
        # a logo can zoom, slide or swing in: its letters leave the reference box, so its clearly
        # coloured letters are looked for in the whole card (or the given zone) while it is on screen
        if wide is True:
            x0, y0, x1, y1 = 0, 0, W, H
        else:
            x0, y0, x1, y1 = max(0, wide[0]), max(0, wide[1]), min(W, wide[2]), min(H, wide[3])
    elif wide is not False:
        return np.zeros(img.shape[:2], np.uint8)
    out = np.zeros(img.shape[:2], np.uint8)
    if lc is None or bc is None or not len(lc):
        return out
    # classified at reduced size (it is grown afterwards anyway): this runs for every candidate frame
    crop = img[y0:y1, x0:x1]
    sc = min(1.0, (250000.0 / max(1, crop.shape[0] * crop.shape[1])) ** 0.5)
    small = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else crop
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    def near(c):
        d = None
        for cc in c:
            x = ((lab - cc) ** 2).sum(1)
            d = x if d is None else np.minimum(d, x)
        return np.sqrt(d)
    dl, db = near(lc), near(bc)
    lim = 30 if (t.get("design") or t.get("design_member")) else 22
    sel = ((dl < db * 0.7) & (dl < lim)).reshape(small.shape[0], small.shape[1]).astype(np.uint8)
    if sc < 1:
        sel = cv2.resize(sel, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    lh = max(6, (t.get("text_box", t["box"])[3] - t.get("text_box", t["box"])[1]) // 3)
    k = max(3, lh // 6) | 1
    sel = cv2.morphologyEx(sel, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    sel = cv2.morphologyEx(sel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    sel = cv2.dilate(sel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    out[y0:y1, x0:x1] = sel * 255
    return out


def frame_mask(img, f, lines, reg, margin=0):
    k = (id(img), f, margin, tuple(t["id"] for t in lines))
    if k in _MSK and _MSK[k][0] is img:
        return _MSK[k][1].copy()
    m = _frame_mask(img, f, lines, reg, margin)
    _MSK[k] = (img, m)
    return m.copy()


def _frame_mask(img, f, lines, reg, margin=0):
    m = np.zeros(img.shape[:2], np.uint8)
    for t in lines:
        if t.get("translate", True) and "mask" in t and t["in"] - margin <= f <= t["out"] + margin:
            m |= line_mask(img, t, reg, f, _CTX.get("proj"), _CTX.get("gid"))
    return m


def _align(tgt_g, src_g, valid, scale=0.5, init=None):
    """Affine warp mapping src -> tgt. Returns (warp 2x3 at full res, residual) or (None, inf).
    init: (dx, dy) from the camera-motion track. When given it is trusted: ECC may only refine it."""
    ts = cv2.resize(tgt_g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ss = cv2.resize(src_g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    vm = cv2.resize(valid, (ts.shape[1], ts.shape[0]), interpolation=cv2.INTER_NEAREST)
    warp = np.eye(2, 3, dtype=np.float32)
    if init is not None:
        warp[0, 2], warp[1, 2] = init[0] * scale, init[1] * scale
        w0 = warp.copy()
        try:
            _, warp = cv2.findTransformECC(ts, ss, warp, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-5), vm, 5)
            lin = np.abs(warp[:, :2] - np.eye(2)).max()
            if np.abs(warp[:, 2] - w0[:, 2]).max() > max(3.0, 0.15 * np.abs(w0[:, 2]).max()) or lin > 0.05:
                warp = w0   # ECC wandered off (flat area / graphics): keep the track
        except cv2.error:
            warp = w0
        warp[:, 2] /= scale
        return warp, 0.0
    # coarse translation first (camera pans can be 100+ px between distant frames), ECC refines
    try:
        win = cv2.createHanningWindow((ts.shape[1], ts.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(ts, ss, win)
        if resp > 0.05:
            if abs(dx) < 0.15 and abs(dy) < 0.15 and resp > 0.3:
                return np.eye(2, 3, dtype=np.float32), 0.0   # locked-off camera: no warp needed
            warp[0, 2], warp[1, 2] = dx, dy
    except cv2.error:
        pass
    try:
        warp = _ecc_pyr(ts, ss, warp, vm)
    except cv2.error:
        return None, float("inf")
    warp[:, 2] /= scale
    return warp, 0.0


def _ecc_pyr(ts, ss, warp, vm, it=60):
    """ECC coarse-to-fine: most iterations on a quarter of the pixels, then a short polish at
    full working size - the same fit, several times faster."""
    if min(ts.shape[:2]) >= 160:
        t2 = cv2.resize(ts, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        s2 = cv2.resize(ss, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        v2 = cv2.resize(vm, (t2.shape[1], t2.shape[0]), interpolation=cv2.INTER_NEAREST)
        w2 = warp.copy(); w2[:, 2] *= 0.5
        try:
            _, w2 = cv2.findTransformECC(t2, s2, w2, cv2.MOTION_AFFINE,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, it, 1e-5), v2, 3)
            warp = w2.copy(); warp[:, 2] *= 2.0
            it = 15
        except cv2.error:
            pass
    _, warp = cv2.findTransformECC(ts, ss, warp, cv2.MOTION_AFFINE,
                                   (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, it, 1e-5), vm, 5)
    return warp


def _grad(im):
    g = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def _fit(img, ci, Mc, M, warp, W, H):
    """Warp candidate ci onto img, colour-match it on a thin ring around the text.
    Returns (residual, warped+matched image, warped candidate mask, inside, gain, offset) or None."""
    IM = cv2.WARP_INVERSE_MAP
    wi = cv2.warpAffine(ci, warp, (W, H), flags=cv2.INTER_LINEAR | IM, borderMode=cv2.BORDER_REFLECT)
    inside = cv2.warpAffine(np.full((H, W), 255, np.uint8), warp, (W, H), flags=cv2.INTER_NEAREST | IM)
    wm = cv2.warpAffine(Mc, warp, (W, H), flags=cv2.INTER_NEAREST | IM)
    ring = (cv2.dilate(M, np.ones((11, 11), np.uint8)) > 0) & (M == 0) & (wm == 0) & (inside > 0)
    if ring.sum() < 50:
        return None
    a = img[ring].astype(np.float32); b = wi[ring].astype(np.float32)
    # per-channel affine colour match (gain*x+offset), robust: exposure / flicker, and a
    # semi-transparent box or vignette present here but not in the candidate. Ring pixels that
    # do not follow the same mapping (outside the box edge) are rejected and the fit repeated.
    inl = np.ones(len(a), bool)
    G = np.ones(3, np.float32); O = np.zeros(3, np.float32)
    for it in range(3):
        for ch in range(3):
            x = b[inl, ch]; y = a[inl, ch]
            vx = x.var()
            G[ch] = float(np.clip(((x - x.mean()) * (y - y.mean())).mean() / vx, 0.05, 3)) if vx > 4 else 1.0
            O[ch] = float(y.mean() - G[ch] * x.mean())
        r = np.abs(b * G + O - a).mean(1)
        inl = r < max(6.0, 2.5 * float(np.median(r)))
    if inl.mean() < 0.5:
        return None
    wi = np.clip(wi.astype(np.float32) * G + O, 0, 255)
    res = float(np.abs(wi[ring][inl] - a[inl]).mean())
    return res, wi, wm, inside, G, O


_XT = {}


def _cross_index(proj, gid, reg):
    """Frames of the OTHER title cards of this video that overlap this card's area: (gid2, frame,
    thumb of our area (NaN where not covered), lines2, region2). In trailers the cards often share one
    animated background: what is behind this card's letters is visible in another card."""
    key = (proj, gid)
    if key in _XT:
        return _XT[key]
    T = _CTX.get("T") or vio.load_json(os.path.join(proj, "titles.json"))
    _CTX["T"] = T
    by = {}
    for t in T["titles"]:
        if t.get("translate", True) and "mask" in t and "region" in t:
            by.setdefault(t["group"], []).append(t)
    out = []
    for g2, ls in by.items():
        if g2 == gid:
            continue
        r2 = ls[0]["region"]
        ix0, iy0, ix1, iy1 = max(r2[0], reg[0]), max(r2[1], reg[1]), min(r2[2], reg[2]), min(r2[3], reg[3])
        if (ix1 - ix0) * (iy1 - iy0) < 0.3 * (reg[2] - reg[0]) * (reg[3] - reg[1]):
            continue
        d = os.path.join(proj, "cache", g2)
        try:
            fs = sorted(int(x[:-4]) for x in os.listdir(d) if x.endswith(".png") and x[:-4].isdigit())
        except OSError:
            continue
        for c in fs[::3]:
            im = cv2.imread(os.path.join(d, f"{c:06d}.png"))
            if im is None:
                continue
            sub, cov = _into(im, r2, reg)
            th = cv2.resize(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
            cv_ = cv2.resize(cov, (96, 54), interpolation=cv2.INTER_NEAREST) > 0
            th[~cv_] = np.nan
            out.append((g2, c, th, ls, r2))
    _XT.clear(); _XT[key] = out
    return out


def _into(im, r2, reg):
    """Image of region r2 placed in region reg's coordinates; coverage mask (1 where r2 has pixels)."""
    H, W = reg[3] - reg[1], reg[2] - reg[0]
    out = np.zeros((H, W, 3), np.uint8); cov = np.zeros((H, W), np.uint8)
    x0, y0 = max(r2[0], reg[0]), max(r2[1], reg[1]); x1, y1 = min(r2[2], reg[2]), min(r2[3], reg[3])
    if x1 > x0 and y1 > y0:
        out[y0 - reg[1]:y1 - reg[1], x0 - reg[0]:x1 - reg[0]] = im[y0 - r2[1]:y1 - r2[1], x0 - r2[0]:x1 - r2[0]]
        cov[y0 - reg[1]:y1 - reg[1], x0 - reg[0]:x1 - reg[0]] = 1
    return out, cov


def _letter_like(im, lines, grow=7):
    """Pixels coloured like this card's letters (cards of one video share their title colour): such
    pixels in another card are its letters, not background."""
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = np.zeros(im.shape[:2], np.uint8)
    for t in lines:
        mp = t.get("mask") or {}
        if "fill_lab" in mp:
            m |= (masks._wdist(lab, mp["fill_lab"]) < mp.get("tol", 30) * 1.4).astype(np.uint8)
    if m.any():
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return m * 255


def _cross_candidates(proj, gid, img, M, reg, k=8, lines=()):
    """The k frames of other cards whose background looks most like this frame's (outside the text)."""
    idx = _cross_index(proj, gid, reg)
    if not idx:
        return []
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
    free = cv2.resize((cv2.dilate(M, np.ones((15, 15), np.uint8)) == 0).astype(np.uint8), (96, 54), interpolation=cv2.INTER_NEAREST) > 0
    hole = cv2.resize((M > 0).astype(np.uint8), (96, 54), interpolation=cv2.INTER_NEAREST) > 0
    sc = []
    for g2, c, th, ls, r2 in idx:
        ok = free & ~np.isnan(th)
        if ok.sum() < 150 or (~np.isnan(th) & hole).sum() < 0.5 * max(1, hole.sum()):
            continue                                       # must see most of our hole
        sc.append((float(np.abs(th[ok] - g[ok]).mean()), g2, c, ls, r2))
    sc.sort(key=lambda x: x[0])
    res = []
    for e, g2, c, ls, r2 in sc[:k]:
        if e > 18:
            break
        im = cv2.imread(os.path.join(proj, "cache", g2, f"{c:06d}.png"))
        if im is None:
            continue
        sub, cov = _into(im, r2, reg)
        # the other card's own letters, in our coordinates
        m_full = np.zeros(im.shape[:2], np.uint8)
        for t in ls:
            if t["in"] - 2 <= c <= t["out"] + 2:
                m_full |= line_mask(im, t, r2)
        m2, _ = _into(np.dstack([m_full] * 3), r2, reg)
        m2 = m2[..., 0] | ((1 - cov) * 255).astype(np.uint8)
        hh = max(9, int(min(t["box"][3] - t["box"][1] for t in ls) * 0.25)) | 1
        m2 = cv2.dilate(m2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hh, hh)))
        res.append((sub, m2 | _letter_like(sub, list(lines) + list(ls))))
    return res


def _others_mask(c, others, reg, shape, ci=None):
    """Text of other titles (other cards at the same place, e.g. the previous subtitle) on screen
    at frame c - it must not leak into this card's rebuilt background. Letter-exact when the
    other title has a colour model (the gaps between its letters stay usable), else its box."""
    m = np.zeros(shape, np.uint8)
    for o in others:
        if o["in"] - 2 <= c <= o["out"] + 2:
            if ci is not None and "mask" in o:
                m |= cv2.dilate(line_mask(ci, o, reg), np.ones((5, 5), np.uint8))
                continue
            x0, y0, x1, y1 = o["box"]
            p = int((y1 - y0) * 0.45) + 6
            X0, Y0 = max(0, x0 - reg[0] - p), max(0, y0 - reg[1] - p)
            X1, Y1 = min(shape[1], x1 - reg[0] + p), min(shape[0], y1 - reg[1] + p)
            if X1 > X0 and Y1 > Y0:
                m[Y0:Y1, X0:X1] = 255
    return m


def clean_frame(proj, gid, f, lines, reg, cuts, n, lo, hi, K=14, thr_res=14.0, second=True, others=(), motion=None):
    img = _load(proj, gid, f)
    H, W = img.shape[:2]
    M0 = frame_mask(img, f, lines, reg)
    env = M0.copy()
    if second:
        # fading / moving text the colour model misses: rebuild a wider envelope (text area of
        # neighbouring frames), keep only pixels there that differ from the rebuilt background
        for g in (f - 8, f + 8, f - 3, f + 3, f - 1, f + 1):
            if lo <= g <= hi:
                gi = _load(proj, gid, g)
                env |= frame_mask(gi, g, lines, reg, margin=0) if any(
                    t["in"] <= g <= t["out"] for t in lines) else 0
        env = cv2.dilate(env, np.ones((15, 15), np.uint8))
    M = M0 | env
    if not M.any():
        return img, 0.0, M
    s0, s1 = shot_of(cuts, n, f)
    cands = [c for c in range(max(lo, s0), min(hi, s1 - 1) + 1) if c != f]
    # prefer clean frames (text absent) then nearest
    def key(c):
        busy = any(t.get("translate", True) and t["in"] <= c <= t["out"] for t in lines) or \
            any(o["in"] - 2 <= c <= o["out"] + 2 for o in others)
        return (busy, abs(c - f))
    cands.sort(key=key)
    tg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # each line's mask at its reference frame: in busy candidate frames the text is usually at the
    # same place, and during fades the colour model misses it - exclude this area too
    refm = {}
    for t in lines:
        if t.get("translate", True) and "mask" in t:
            ri = _load(proj, gid, t["ref"])
            if ri is not None:
                refm[t["id"]] = cv2.dilate(line_mask(ri, t, reg, t["ref"], proj, gid), np.ones((5, 5), np.uint8))
    stack, okm = [], []
    used = 0
    # nearest clean frames first, but never more than 2K of them: when they all fail the colour
    # match (e.g. a graphic box appears with the text) fall through to frames with the text
    clean_c = [c for c in cands if not key(c)[0]][:2 * K]
    busy_all = [c for c in cands if key(c)[0]]
    # frames with text: a few nearest, then spread over the whole range - on a moving
    # background the far ones show what is behind the letters, the near ones do not
    near = busy_all[:4]
    rest = sorted(busy_all[4:])
    if len(rest) > 2 * K - 4:
        idx = np.linspace(0, len(rest) - 1, 2 * K - 4).round().astype(int)
        rest = [rest[i] for i in sorted(set(idx))]
    rest.sort(key=lambda c: -abs(c - f))
    busy_c = near + rest
    tries = 0
    for c in clean_c + busy_c:
        if used >= K or tries >= 3 * K:
            break
        tries += 1
        ci = _load(proj, gid, c)
        if ci is None:
            continue
        Mc = frame_mask(ci, c, lines, reg, margin=2) | _others_mask(c, others, reg, ci.shape[:2], ci)
        for t in lines:
            if t["id"] in refm and t["in"] - 2 <= c <= t["out"] + 2:
                Mc |= refm[t["id"]]
        valid_t = ((cv2.dilate(M | Mc, np.ones((9, 9), np.uint8)) == 0)).astype(np.uint8)
        init = None
        if motion is not None:
            init = (motion[c][0] - motion[f][0], motion[c][1] - motion[f][1])
        warp, _ = _align(tg, cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY).astype(np.float32), valid_t, init=init)
        if init is not None:
            tries_w = [warp]   # the track knows the camera: no identity rival (static text fools it)
        else:
            tries_w = [np.eye(2, 3, dtype=np.float32)]
            if warp is not None and np.abs(warp - tries_w[0]).max() > 1e-3:
                tries_w.insert(0, warp)
        best = None
        for wp in tries_w:   # ECC can be pulled by graphics that appear with the text: keep identity as a rival
            r = _fit(img, ci, Mc, M, wp, W, H)
            if r is not None and (best is None or r[0] < best[0]):
                best = r
        if best is None:
            continue
        # a strong colour remap (semi-transparent box) is only trusted when it fits very well;
        # on flat areas a wrong warp can otherwise be "repaired" by the colour fit
        gdev = float(np.abs(np.log(np.maximum(best[4], 1e-3))).max())
        if best[0] > (thr_res if gdev < 0.3 else 6.0):
            continue
        res, wi, wm, inside, G, O = best
        stack.append(wi)
        okm.append((wm == 0) & (inside > 0))
        used += 1
    # not enough background found in this card's own frames: look in the other cards
    cover = np.zeros(M.shape, bool)
    for o in okm:
        cover |= o
    if (M0 > 0).sum() > 0 and float((cover & (M0 > 0)).sum()) / float((M0 > 0).sum()) < 0.7:
        for ci, Mc in _cross_candidates(proj, gid, img, M, reg, lines=[t for t in lines if t.get("translate", True)]):
            if used >= K + 6:
                break
            valid_t = ((cv2.dilate(M | Mc, np.ones((9, 9), np.uint8)) == 0)).astype(np.uint8)
            warp, _ = _align(tg, cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY).astype(np.float32), valid_t)
            best = None
            for wp in ([warp] if warp is not None else []) + [np.eye(2, 3, dtype=np.float32)]:
                r = _fit(img, ci, Mc, M, wp, W, H)
                if r is not None and (best is None or r[0] < best[0]):
                    best = r
            if best is None or best[0] > thr_res:
                continue
            stack.append(best[1]); okm.append((best[2] == 0) & (best[3] > 0)); used += 1
    plate = img.astype(np.float32).copy()
    hole = M > 0
    hole_left = np.zeros_like(hole)
    leak = np.zeros_like(hole)
    if stack:
        S = np.stack(stack)             # k,H,W,3
        V = np.stack(okm)[..., None]    # k,H,W,1
        Sh = S[:, hole].astype(np.float32); Vh = V[:, hole, 0]          # k,n,3 / k,n
        med = np.full(Sh.shape[1:], np.nan, np.float32)
        allv = Vh.all(0)
        if allv.any():                                   # plain median where every frame is usable (fast)
            med[allv] = np.median(Sh[:, allv], axis=0)
        part = ~allv & Vh.any(0)
        if part.any():
            with np.errstate(all="ignore"):
                med[part] = np.nanmedian(np.where(Vh[:, part, None], Sh[:, part], np.nan), axis=0)
        good = ~np.isnan(med).any(1)
        idx = np.argwhere(hole)
        plate[idx[good, 0], idx[good, 1]] = med[good]
        hole_left[idx[~good, 0], idx[~good, 1]] = True
        # what was rebuilt but still looks like the letters (another card's words leaking in): synthesise
        ll = _letter_like(plate.astype(np.uint8), [t for t in lines if t.get("translate", True)], grow=3) > 0
        leak = ll & hole
        hole_left |= leak
    else:
        hole_left = hole.copy()
    plate = plate.astype(np.uint8)
    # synthesise only around the letters themselves; the wider fade envelope is used only
    # where real background was found
    core = (cv2.dilate(M0, np.ones((7, 7), np.uint8)) > 0) | leak
    need = hole_left & core
    if need.any():
        plate = synth_cached(gid, f, plate, need, motion)
    unknown = hole_left & ~core
    res = _logo_left(plate, lines, M0)
    if res is not None:
        # bits of a logo's letters still in the rebuilt picture (taken from a frame where the
        # logo stood elsewhere): synthesised too
        plate = synth(plate, res)
        hole_left |= res
    frac = float((hole_left & (M0 > 0)).sum()) / max(1, (M0 > 0).sum())
    if second:
        d = np.abs(img.astype(np.int16) - plate.astype(np.int16)).max(2)
        extra = (env > 0) & (d > 14) & (M0 == 0) & ~unknown
        extra = cv2.morphologyEx(extra.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        M = M0 | ((cv2.dilate(extra, np.ones((5, 5), np.uint8)) > 0) & ~unknown).astype(np.uint8) * 255
    else:
        M = M0
    # antialias / compression halo around the letters: replace a little beyond the mask
    M = M | (cv2.dilate(M0, np.ones((5, 5), np.uint8)) & ~(unknown.astype(np.uint8) * 255))
    a = cv2.GaussianBlur((M > 0).astype(np.float32), (0, 0), 1.2)[..., None]
    a = np.maximum(a, (M > 0)[..., None].astype(np.float32))
    out = (img.astype(np.float32) * (1 - a) + plate.astype(np.float32) * a).astype(np.uint8)
    # self check: the old letters' colours still standing near the title (missed by the mask while
    # the letters moved, or left at the edges) are found on the result and painted over too
    left = _logo_left(out, lines, M0, reach=121 if any(t.get("strong") for t in lines) else 61)
    LEFT[f] = 0
    if left is not None:
        LEFT[f] = int(left.sum())
        out = synth(out, left)
        M = M | (left.astype(np.uint8) * 255)
    return out, frac, M


LEFT = {}


def synth(img, hole):
    """Fill pixels never seen in any other frame. Structure from a smooth fill (Navier-Stokes),
    texture and grain from exemplar patches of the surrounding picture - no soft blobs."""
    hole = hole.astype(bool)
    if hole.sum() < 150:
        return cv2.inpaint(img, hole.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    from . import lama
    if lama.available():   # learned inpainting (LaMa, local): continues shapes and textures
        try:
            return lama.inpaint(img, hole)
        except Exception:
            pass
    from .inpaint import exemplar
    ns = cv2.inpaint(img, hole.astype(np.uint8) * 255, 5, cv2.INPAINT_NS).astype(np.float32)
    ex = exemplar(img, hole, patch=9).astype(np.float32)
    hf = ex - cv2.GaussianBlur(ex, (0, 0), 2)
    # grain only as strong as the real picture around the hole: copied patch seams stay invisible
    ring = (cv2.dilate(hole.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & ~hole
    if ring.any():
        imf = img.astype(np.float32)
        sig = float((imf - cv2.GaussianBlur(imf, (0, 0), 2))[ring].std())
        hf = np.clip(hf, -2.5 * sig, 2.5 * sig)
    mix = cv2.GaussianBlur(ns, (0, 0), 2) + hf
    out = img.copy()
    out[hole] = np.clip(mix[hole], 0, 255).astype(np.uint8)
    return out


SYNTH = {}   # per card, for a locked-off camera: fill once, reuse (no flicker, much faster)


def _logo_left(plate, lines, M0, reach=31):
    des = [t for t in lines if t.get("translate", True) and t.get("kind") != "sub"
           and t["id"] in _PAL and _PAL[t["id"]][2]]
    if not des or not M0.any():
        return None
    zone = cv2.dilate(M0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))) > 0
    lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB).astype(np.float32)
    px = lab[zone]

    def near(c):
        d = None
        for cc in c:
            x = ((px - cc) ** 2).sum(1)
            d = x if d is None else np.minimum(d, x)
        return np.sqrt(d)
    sel = np.zeros(len(px), bool)
    for t in des:
        lc, bc, _ = _PAL[t["id"]]
        lc = lc[(np.min(np.linalg.norm(lc[:, None] - bc[None], axis=2), axis=1) > 30) & (np.hypot(lc[:, 1] - 128, lc[:, 2] - 128) > 25)]
        if len(lc):
            dl, db = near(lc), near(bc)
            sel |= (dl < db * 0.7) & (dl < 30)
    if sel.sum() < 4:
        return None
    res = np.zeros(plate.shape[:2], np.uint8)
    res[zone] = sel
    res = cv2.morphologyEx(res, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    if not res.any():
        return None
    return cv2.dilate(res, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0


def synth_cached(gid, f, img, hole, motion):
    c = SYNTH.get(gid)
    if c is not None and motion is not None:
        cf, cmask, cimg = c
        still = abs(motion[f][0] - motion[cf][0]) < 0.5 and abs(motion[f][1] - motion[cf][1]) < 0.5
        if still and (hole & ~cmask).sum() <= 0.02 * max(1, hole.sum()):
            out = img.copy()
            out[hole] = cimg[hole]
            rest = hole & ~cmask
            if rest.any():
                out = cv2.inpaint(out, rest.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
            return out
    out = synth(img, hole)
    if motion is not None:
        SYNTH[gid] = (f, hole.copy(), out.copy())
    return out


def _card(args):
    """Clean one card (all frames of one title group). Runs in a worker process."""
    proj, gid = args
    S = vio.load_json(os.path.join(proj, "shots.json"))
    T = vio.load_json(os.path.join(proj, "titles.json"))
    n = vio.load_json(os.path.join(proj, "project.json"))["n_frames"]
    lines = groups(T)[gid]
    act = [t for t in lines if t.get("translate", True) and "mask" in t]
    reg = act[0]["region"]; lo, hi = act[0]["cache"]
    d = os.path.join(proj, "clean", gid)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    _IMG.clear(); _MSK.clear(); SYNTH.clear(); LEFT.clear()
    _CTX.update(proj=proj, gid=gid, T=T)
    a = min(t["in"] for t in act); b = max(t["out"] for t in act)
    ids = {t["id"] for t in lines}
    others = [o for o in T["titles"] if o["id"] not in ids and
              not (o["box"][2] < reg[0] or o["box"][0] > reg[2] or o["box"][3] < reg[1] or o["box"][1] > reg[3])]
    fr = []
    for f in range(a, b + 1):
        out, frac, M = clean_frame(proj, gid, f, lines, reg, S["cuts"], n, lo, hi, others=others,
                                   motion=S.get("motion"))
        cv2.imwrite(os.path.join(d, f"{f:06d}.png"), out, PNG_FAST)
        cv2.imwrite(os.path.join(d, f"m{f:06d}.png"), M, PNG_FAST)
        src = cv2.imread(os.path.join(proj, "cache", gid, f"{f:06d}.png"))
        if src is not None and src.shape == out.shape:
            live.show(proj, f"מוחק את {', '.join(t['src'] for t in act)[:40]} - פריים {f}", out, src)   # RTL: source on the right
        fr.append(frac)
    lf = [f for f in range(a, b + 1) if LEFT.get(f, 0) > 0]
    st = {"frames": [a, b], "synth_mean": round(float(np.mean(fr)), 3), "synth_max": round(float(np.max(fr)), 3),
          "left_frames": len(lf)}
    built = {}
    for t in lines:
        built[t["id"]] = [t["in"], t["out"], bool(t.get("translate", True))]
        built[t["id"] + "#reg"] = list(t.get("region") or [])
    print(gid, st)
    return gid, st, built


def _worker_init(threads):
    os.environ["RT_THREADS"] = str(threads)
    cv2.setNumThreads(threads)


def workers(jobs):
    """Cards run side by side on a multi-core PC; each gets a share of the cores."""
    cpu = os.cpu_count() or 2
    w = int(os.environ.get("RT_WORKERS") or 0) or min(jobs, max(1, cpu // 3), 6)
    return max(1, min(w, jobs))


def clamp_spans(proj, T):
    """A title never runs past the shots it was seen in: older projects (and hand-moved edges)
    could reach into the next shot, which then got erased and painted over for nothing."""
    from .shots import shot_of
    S = vio.load_json(os.path.join(proj, "shots.json"), {})
    n = vio.load_json(os.path.join(proj, "project.json"), {}).get("n_frames")
    cuts = S.get("cuts")
    if not cuts or not n:
        return False
    ch = False
    for t in T["titles"]:
        si, so = t.get("sample_in", t["in"]), t.get("sample_out", t["out"])
        a, _ = shot_of(cuts, n, min(si, so))
        _, b = shot_of(cuts, n, max(si, so))
        i, o = max(t["in"], a), min(t["out"], b - 1)
        if o >= i and (i, o) != (t["in"], t["out"]):
            t["in"], t["out"] = i, o; ch = True
    if ch:
        vio.save_json(os.path.join(proj, "titles.json"), T)
    return ch


def extend_logos(proj, T):
    """A logo that flies in fast (zoom blur) is often found by the text reader only once it slows down,
    and the blur can even look like a cut. Its first frames are found here by the letters' own colours:
    going back from the first known frame while about as much of that colour is there."""
    ch = False
    for gid, lines in groups(T).items():
        for t in lines:
            if not (t.get("design") or t.get("design_member") or t.get("styled")) or "mask" not in t or not t.get("cache"):
                continue
            lo = t["cache"][0]
            if t["in"] <= lo:
                continue
            _CTX.update(proj=proj, gid=gid, T=T)
            reg = t["region"]; bx = t.get("text_box", t["box"]); h = bx[3] - bx[1]
            box = [bx[0] - reg[0] - h, bx[1] - reg[1] - h, bx[2] - reg[0] + h, bx[3] - reg[1] + h]

            def cover(f):
                im = _load(proj, gid, f)
                return None if im is None else int((styled_mask(im, t, reg, box, wide=True) > 0).sum())
            c0 = cover(t["in"])
            if not c0:
                continue
            f = t["in"] - 1
            while f >= lo:
                c = cover(f)
                if c is None or not (0.4 * c0 <= c <= 3.0 * c0):
                    break
                f -= 1
            if f + 1 < t["in"]:
                t["in"] = f + 1; ch = True
    _IMG.clear()
    if ch:
        vio.save_json(os.path.join(proj, "titles.json"), T)
    return ch


def run(proj, only=None, progress=None):
    T = vio.load_json(os.path.join(proj, "titles.json"))
    clamp_spans(proj, T)
    try:
        extend_logos(proj, T)
    except Exception as e:
        print("extend_logos", e)
    stats = vio.load_json(os.path.join(proj, "clean_stats.json"), {})
    built = {}
    todo = {}
    for gid, lines in groups(T).items():
        act = [t for t in lines if t.get("translate", True) and "mask" in t]
        if act and not (only and gid not in only):
            todo[gid] = max(t["out"] for t in act) - min(t["in"] for t in act) + 1
    total = max(1, sum(todo.values()))
    order = sorted(todo, key=lambda g: -todo[g])          # longest first: better packing
    w = workers(len(order))

    def counted():
        c = 0
        for g in order:
            try:
                c += sum(1 for x in os.listdir(os.path.join(proj, "clean", g)) if not x.startswith("m"))
            except OSError:
                pass
        return min(c, total)

    if w <= 1:
        for g in order:
            gid, st, b = _card((proj, g))
            stats[gid] = st; built.update(b)
            if progress:
                progress(counted() / total)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(w, initializer=_worker_init, initargs=(max(1, (os.cpu_count() or 2) // w),))
        try:
            res = [pool.apply_async(_card, ((proj, g),)) for g in order]
            while not all(r.ready() for r in res):
                if progress:
                    progress(counted() / total)     # raises when the user cancels
                time.sleep(0.5)
            for r in res:
                gid, st, b = r.get()
                stats[gid] = st; built.update(b)
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()
    vio.save_json(os.path.join(proj, "clean_stats.json"), stats)
    T2 = vio.load_json(os.path.join(proj, "titles.json"))  # re-read: the editor may have saved meanwhile
    for t in T2["titles"]:
        if t["id"] in built:
            t["built"] = built[t["id"]]
            t["built_region"] = built.get(t["id"] + "#reg")
    vio.save_json(os.path.join(proj, "titles.json"), T2)
