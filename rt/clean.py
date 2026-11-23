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
        _, warp = cv2.findTransformECC(ts, ss, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5),
                                       vm, 5)
    except cv2.error:
        return None, float("inf")
    warp[:, 2] /= scale
    return warp, 0.0


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
    # synthesise only around the letters themselves; the wider fade envelope is used only
    # where real background was found
    core = cv2.dilate(M0, np.ones((7, 7), np.uint8)) > 0
    need = hole_left & core
    if need.any():
        plate = synth_cached(gid, f, plate, need, motion)
    unknown = hole_left & ~core
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
    return out, frac, M


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
    _IMG.clear(); _MSK.clear(); SYNTH.clear()
    _CTX.update(proj=proj, gid=gid)
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
    st = {"frames": [a, b], "synth_mean": round(float(np.mean(fr)), 3), "synth_max": round(float(np.max(fr)), 3)}
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


def run(proj, only=None, progress=None):
    T = vio.load_json(os.path.join(proj, "titles.json"))
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
