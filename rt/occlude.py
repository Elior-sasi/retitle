"""Things passing IN FRONT of a title (a character, a branch, snow): found per frame, so the English is erased
only where it shows and the Hebrew is drawn behind them, as in the original.

How it is found, without guessing at the scene: SAM 2 follows the original letters frame by frame
(sam/<gid>/). Where the letters were a moment ago and are now missing, something covers them. The cover is
the moving part of the picture there: the frame against the card's own background (all its frames lined up on
the camera's motion, per-pixel median). Only moving parts that actually hide letters count - a background that
slides by (parallax) never hides the letters, they are on top of it.

Output: occlude/<gid>/<frame>.png, 8-bit soft mask in the card region (255 = in front of the title)."""
import os, shutil
import cv2
import numpy as np
from . import vio

SMALL = 480                                  # alignment and the background model work at this width


def _align(a, b, mask):
    """Affine warp taking frame b onto frame a (grey, small), estimated away from the letters."""
    W = np.eye(2, 3, dtype=np.float32)
    try:
        _, W = cv2.findTransformECC(a, b, W, cv2.MOTION_AFFINE,
                                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-4), mask, 3)
    except cv2.error:
        pass
    return W


def masks(proj, gid, a, b, progress=None):
    """Compute and store the masks for frames a..b of card gid. Returns how many frames have something in front."""
    cache = os.path.join(proj, "cache", gid)
    samd = os.path.join(proj, "sam", gid)
    out = os.path.join(proj, "occlude", gid)
    shutil.rmtree(out, ignore_errors=True)
    if not os.path.isdir(samd):
        return 0
    fs = [f for f in range(a, b + 1) if os.path.exists(os.path.join(cache, f"{f:06d}.png"))]
    if len(fs) < 5:
        return 0
    full = {f: cv2.imread(os.path.join(cache, f"{f:06d}.png")) for f in fs}
    H, W = next(iter(full.values())).shape[:2]
    sc = min(1.0, SMALL / W)
    sw, sh = max(8, int(W * sc)), max(8, int(H * sc))
    lab = {f: cv2.cvtColor(cv2.resize(full[f], (sw, sh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32) for f in fs}
    grey = {f: cv2.resize(cv2.cvtColor(full[f], cv2.COLOR_BGR2GRAY), (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 for f in fs}
    sam = {}
    for f in fs:
        m = cv2.imread(os.path.join(samd, f"{f:06d}.png"), 0)
        sam[f] = (cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) if m is not None else None
    if sum(1 for v in sam.values() if v is not None) < len(fs) // 2:
        return 0
    k = max(3, int(0.012 * sw) | 1)
    keep = {f: (1 - cv2.dilate(sam[f], np.ones((k, k), np.uint8))) if sam[f] is not None else np.ones((sh, sw), np.uint8) for f in fs}
    # camera motion: each frame onto the middle one, chained through neighbours
    mid = fs[len(fs) // 2]
    Wm = {mid: np.eye(3, dtype=np.float32)}
    for seq in (fs[fs.index(mid) + 1:], fs[:fs.index(mid)][::-1]):
        prev = mid
        for f in seq:
            A = _align(grey[prev], grey[f], keep[prev] * keep[f])     # f -> prev
            A3 = np.vstack([A, [0, 0, 1]]).astype(np.float32)
            Wm[f] = Wm[prev] @ A3                                      # f -> mid
            prev = f
    # the card's own background in the middle frame's coordinates: median over frames, letters left out
    stack, valid = [], []
    for f in fs:
        M = Wm[f][:2]
        stack.append(cv2.warpAffine(lab[f], M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        v = cv2.warpAffine(keep[f].astype(np.float32), M, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        valid.append(v > 0.5)
    stack = np.array(stack); valid = np.array(valid)
    st = np.where(valid[..., None], stack, np.nan)
    with np.errstate(all="ignore"):
        bg = np.nanmedian(st, axis=0)
    bg = np.where(np.isnan(bg), np.nanmedian(stack, axis=0), bg).astype(np.float32)
    os.makedirs(out, exist_ok=True)
    n_fg = 0
    for i, f in enumerate(fs):
        if progress and i % 5 == 0:
            progress(i / len(fs))
        Minv = cv2.invertAffineTransform(Wm[f][:2])
        b_f = cv2.warpAffine(bg, Minv, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        d = np.linalg.norm(lab[f] - b_f, axis=2)
        move = (d > 28).astype(np.uint8)
        move = cv2.morphologyEx(move, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        move = cv2.morphologyEx(move, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        # letters that were there a moment ago (frames around f) and are missing now: hidden
        near = np.zeros((sh, sw), np.uint8)
        for g in (f - 2, f - 1, f + 1, f + 2):
            if g in sam and sam[g] is not None:
                near |= sam[g]
        cur = sam[f] if sam[f] is not None else np.zeros((sh, sw), np.uint8)
        hidden = near & (1 - cv2.dilate(cur, np.ones((k, k), np.uint8)))
        fg = np.zeros((sh, sw), np.uint8)
        if hidden.sum() > 0.0005 * sw * sh:
            n, lab_c, stats, _ = cv2.connectedComponentsWithStats(move, 8)
            for c in range(1, n):
                comp = lab_c == c
                if stats[c, cv2.CC_STAT_AREA] < 0.002 * sw * sh:
                    continue
                ov = int((comp & (hidden > 0)).sum())
                if ov > max(0.0004 * sw * sh, 0.15 * hidden.sum()):     # it covers letters that were there
                    fg |= comp.astype(np.uint8)
        # the cover never includes the letters SAM still sees
        fg &= 1 - cur
        if fg.any():
            n_fg += 1
            ff = np.zeros_like(fg)
            cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ff, cs, -1, 1, -1)              # solid object, no holes
            big = cv2.resize(ff.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            big = cv2.GaussianBlur(big, (0, 0), max(1.0, 1.2 / sc))
            fm = cv2.imread(os.path.join(samd, f"{f:06d}.png"), 0)
            if fm is not None and fm.shape == big.shape:     # exact letters at full size stay letters
                big[fm > 127] = 0
            cv2.imwrite(os.path.join(out, f"{f:06d}.png"), np.clip(big * 255, 0, 255).astype(np.uint8))
    if not n_fg:
        shutil.rmtree(out, ignore_errors=True)
    return n_fg


def get(proj, gid, f, shape=None):
    """Soft mask 0..1 (float32) of what is in front of card gid at frame f, or None."""
    p = os.path.join(proj, "occlude", gid, f"{f:06d}.png")
    if not os.path.exists(p):
        return None
    m = cv2.imread(p, 0)
    if m is None or (shape is not None and m.shape != tuple(shape[:2])):
        return None
    return m.astype(np.float32) / 255
