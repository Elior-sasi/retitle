"""Pass 1: shot cuts + sampled full-res frames for OCR."""
import os, cv2, numpy as np
from . import vio


def run(proj, every=4):
    P = vio.load_json(os.path.join(proj, "project.json"))
    info = P["info"]
    w, h = info["width"], info["height"]
    sdir = os.path.join(proj, "samples")
    os.makedirs(sdir, exist_ok=True)
    feats, diffs, prev, prevh = [], [], None, None
    n = 0
    for i, f in enumerate(vio.frames(P["video"], w, h)):
        small = cv2.resize(f, (192, 108), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, None).flatten()
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is None:
            diffs.append((0.0, 0.0))
        else:
            hd = cv2.compareHist(prevh, hist, cv2.HISTCMP_BHATTACHARYYA)
            md = float(np.mean(np.abs(g - prev))) / 255
            diffs.append((float(hd), md))
        feats.append(float(g.mean()))
        prev, prevh = g, hist
        if i % every == 0:
            cv2.imwrite(os.path.join(sdir, f"{i:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
        n = i + 1
    d = np.array(diffs)
    score = d[:, 0] * 0.7 + d[:, 1] * 2.0
    cuts = [0]
    for i in range(1, n):
        lo, hi = max(1, i - 12), min(n, i + 13)
        local = np.median(score[lo:hi])
        if score[i] > max(0.28, local * 3 + 0.08) and score[i] == score[max(1, i - 2):i + 3].max():
            if i - cuts[-1] >= 3:
                cuts.append(i)
    # flash / fade frames: big luma jumps both ways - recorded separately, not cuts
    P["n_frames"] = n
    vio.save_json(os.path.join(proj, "project.json"), P)
    vio.save_json(os.path.join(proj, "shots.json"), {"cuts": cuts, "n_frames": n,
                  "score": [round(float(s), 4) for s in score], "every": every})
    return cuts


def shot_of(cuts, n, f):
    """(start, end_exclusive) of shot containing frame f."""
    import bisect
    k = bisect.bisect_right(cuts, f) - 1
    s = cuts[k]
    e = cuts[k + 1] if k + 1 < len(cuts) else n
    return s, e
