"""Pass 4: remove the source text. Background is rebuilt from neighbouring frames of the same shot
(aligned with ECC, brightness matched, median) - real pixels, no blur. Only what is never visible
in any neighbour is synthesised (FSR inpainting)."""
import os, shutil, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import cv2, numpy as np
from . import vio, masks
from .shots import shot_of
from .extract import groups


_IMG, _MSK = {}, {}


def _load(proj, gid, f):
    k = (proj, gid, f)
    if k not in _IMG:
        _IMG[k] = cv2.imread(os.path.join(proj, "cache", gid, f"{f:06d}.png"))
    return _IMG[k]


def line_mask(img, t, reg):
    bx = t["box"]
    h = bx[3] - bx[1]
    s = t.get("search", h)
    box = [bx[0] - reg[0] - s, bx[1] - reg[1] - s, bx[2] - reg[0] + s, bx[3] - reg[1] + s]
    return masks.make(img, t["mask"], box)


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
            m |= line_mask(img, t, reg)
    return m


def _align(tgt_g, src_g, valid, scale=0.5):
    """Affine warp mapping src -> tgt. Returns (warp 2x3 at full res, residual) or (None, inf)."""
    ts = cv2.resize(tgt_g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ss = cv2.resize(src_g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    vm = cv2.resize(valid, (ts.shape[1], ts.shape[0]), interpolation=cv2.INTER_NEAREST)
    warp = np.eye(2, 3, dtype=np.float32)
    # coarse translation first (camera pans can be 100+ px between distant frames), ECC refines
    try:
        win = cv2.createHanningWindow((ts.shape[1], ts.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(ts, ss, win)
        if resp > 0.05:
            warp[0, 2], warp[1, 2] = dx, dy
    except cv2.error:
        pass
    try:
        _, warp = cv2.findTransformECC(ts, ss, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
                                       vm, 5)
    except cv2.error:
        return None, float("inf")
    warp[:, 2] /= scale
    return warp, 0.0


def clean_frame(proj, gid, f, lines, reg, cuts, n, lo, hi, K=14, thr_res=14.0, second=True):
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
        env = cv2.dilate(env, np.ones((7, 7), np.uint8))
    M = M0 | env
    if not M.any():
        return img, 0.0, M
    s0, s1 = shot_of(cuts, n, f)
    cands = [c for c in range(max(lo, s0), min(hi, s1 - 1) + 1) if c != f]
    # prefer clean frames (text absent) then nearest
    def key(c):
        busy = any(t.get("translate", True) and t["in"] <= c <= t["out"] for t in lines)
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
                refm[t["id"]] = cv2.dilate(line_mask(ri, t, reg), np.ones((5, 5), np.uint8))
    stack, okm = [], []
    used = 0
    for c in cands:
        if used >= K:
            break
        ci = _load(proj, gid, c)
        if ci is None:
            continue
        Mc = frame_mask(ci, c, lines, reg, margin=2)
        for t in lines:
            if t["id"] in refm and t["in"] - 2 <= c <= t["out"] + 2:
                Mc |= refm[t["id"]]
        valid_t = ((cv2.dilate(M | Mc, np.ones((9, 9), np.uint8)) == 0)).astype(np.uint8)
        warp, _ = _align(tg, cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY).astype(np.float32), valid_t)
        if warp is None:
            continue
        IM = cv2.WARP_INVERSE_MAP
        wi = cv2.warpAffine(ci, warp, (W, H), flags=cv2.INTER_LINEAR | IM, borderMode=cv2.BORDER_REFLECT)
        inside = cv2.warpAffine(np.full((H, W), 255, np.uint8), warp, (W, H), flags=cv2.INTER_NEAREST | IM)
        wm = cv2.warpAffine(Mc, warp, (W, H), flags=cv2.INTER_NEAREST | IM)
        ring = (cv2.dilate(M, np.ones((25, 25), np.uint8)) > 0) & (M == 0) & (wm == 0) & (inside > 0)
        if ring.sum() < 50:
            continue
        a = img[ring].astype(np.float32); b = wi[ring].astype(np.float32)
        gain = (a.mean(0) + 1) / (b.mean(0) + 1)
        wi = np.clip(wi.astype(np.float32) * gain, 0, 255)
        res = float(np.abs(wi[ring] - a).mean())
        if res > thr_res:
            continue
        stack.append(wi)
        okm.append((wm == 0) & (inside > 0))
        used += 1
    plate = img.astype(np.float32).copy()
    hole = M > 0
    hole_left = np.zeros_like(hole)
    if stack:
        S = np.stack(stack)             # k,H,W,3
        V = np.stack(okm)[..., None]    # k,H,W,1
        Sm = np.where(V, S, np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(Sm[:, hole], axis=0)   # n,3
        good = ~np.isnan(med).any(1)
        idx = np.argwhere(hole)
        plate[idx[good, 0], idx[good, 1]] = med[good]
        hole_left[idx[~good, 0], idx[~good, 1]] = True
    else:
        hole_left = hole.copy()
    plate = plate.astype(np.uint8)
    if hole_left.any():
        plate = synth(plate, hole_left)
    frac = float((hole_left & (M0 > 0)).sum()) / max(1, (M0 > 0).sum())
    if second:
        d = np.abs(img.astype(np.int16) - plate.astype(np.int16)).max(2)
        extra = (env > 0) & (d > 14) & (M0 == 0)
        extra = cv2.morphologyEx(extra.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        M = M0 | (cv2.dilate(extra, np.ones((5, 5), np.uint8)) * 255)
    a = cv2.GaussianBlur((M > 0).astype(np.float32), (0, 0), 1.2)[..., None]
    a = np.maximum(a, (M > 0)[..., None].astype(np.float32))
    out = (img.astype(np.float32) * (1 - a) + plate.astype(np.float32) * a).astype(np.uint8)
    return out, frac, M


def synth(img, hole):
    """Fill pixels never seen in any neighbour. FSR keeps texture; TELEA for tiny specks."""
    known = (~hole).astype(np.uint8) * 255
    if hole.sum() < 150 or not hasattr(cv2, "xphoto"):
        return cv2.inpaint(img, hole.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
    out = np.zeros_like(img)
    try:
        cv2.xphoto.inpaint(img, known, out, cv2.xphoto.INPAINT_FSR_FAST)
        return out
    except cv2.error:
        return cv2.inpaint(img, hole.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)


def run(proj, only=None):
    P = vio.load_json(os.path.join(proj, "project.json"))
    S = vio.load_json(os.path.join(proj, "shots.json"))
    T = vio.load_json(os.path.join(proj, "titles.json"))
    n = P["n_frames"]
    stats = vio.load_json(os.path.join(proj, "clean_stats.json"), {})
    built = {}
    for gid, lines in groups(T).items():
        act = [t for t in lines if t.get("translate", True) and "mask" in t]
        if not act or (only and gid not in only):
            continue
        reg = act[0]["region"]; lo, hi = act[0]["cache"]
        d = os.path.join(proj, "clean", gid)
        shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        _IMG.clear(); _MSK.clear()
        a = min(t["in"] for t in act); b = max(t["out"] for t in act)
        fr = []
        for f in range(a, b + 1):
            out, frac, M = clean_frame(proj, gid, f, lines, reg, S["cuts"], n, lo, hi)
            cv2.imwrite(os.path.join(d, f"{f:06d}.png"), out)
            cv2.imwrite(os.path.join(d, f"m{f:06d}.png"), M)
            fr.append(frac)
        stats[gid] = {"frames": [a, b], "synth_mean": round(float(np.mean(fr)), 3),
                      "synth_max": round(float(np.max(fr)), 3)}
        for t in lines:
            built[t["id"]] = [t["in"], t["out"], bool(t.get("translate", True))]
        print(gid, stats[gid])
    vio.save_json(os.path.join(proj, "clean_stats.json"), stats)
    T2 = vio.load_json(os.path.join(proj, "titles.json"))  # re-read: the editor may have saved meanwhile
    for t in T2["titles"]:
        if t["id"] in built:
            t["built"] = built[t["id"]]
    vio.save_json(os.path.join(proj, "titles.json"), T2)
