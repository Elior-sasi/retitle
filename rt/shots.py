"""Pass 1: shot cuts + sampled full-res frames for OCR."""
import os, cv2, numpy as np
from . import vio, live


def run(proj, every=None, progress=None):
    """Samples for OCR are taken every `every` frames (default ~1/6 s) but only written when the
    picture changed since the last written sample - long static stretches (interviews, still
    shots) cost one sample, which keeps hour-long episodes practical."""
    import shutil
    P = vio.load_json(os.path.join(proj, "project.json"))
    info = P["info"]
    w, h = info["width"], info["height"]
    fps = info["fps"]
    every = every or max(3, int(round(fps / 6)))
    total = max(1, int(info.get("duration", 0) * fps))
    sdir = os.path.join(proj, "samples")
    shutil.rmtree(sdir, ignore_errors=True)
    os.makedirs(sdir, exist_ok=True)
    feats, diffs, prev, prevh = [], [], None, None
    samples, last_s, last_f = [], None, -10 ** 9
    # camera motion track: frame-to-frame translation on the whole picture (phase correlation),
    # accumulated - later used to align distant frames even when text sits still on screen
    MW, MH = 640, int(round(640 * h / w / 2)) * 2
    TW, TH = 160, max(60, MH // 3)
    win = cv2.createHanningWindow((TW, TH), cv2.CV_32F)
    mprev, pos, conf = None, [(0.0, 0.0)], [1.0]
    from concurrent.futures import ThreadPoolExecutor
    nt = max(2, min(12, os.cpu_count() or 2))

    def feat(f):
        # one big downscale per frame; the small views come from it (OpenCV releases the GIL,
        # so frames are measured on all cores at once)
        m = cv2.resize(f, (MW, MH), interpolation=cv2.INTER_AREA)
        small = cv2.resize(m, (192, 108), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.normalize(cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256]), None).flatten()
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return g, hist, cv2.cvtColor(m, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def shift(pair):
        a, b = pair
        sh = []
        # median over a grid of tiles: titles and subtitles cover only some tiles
        for ty in range(0, MH - TH + 1, TH):
            for tx in range(0, MW - TW + 1, TW):
                (dx, dy), resp = cv2.phaseCorrelate(np.ascontiguousarray(a[ty:ty + TH, tx:tx + TW]),
                                                     np.ascontiguousarray(b[ty:ty + TH, tx:tx + TW]), win)
                if resp > 0.1:
                    sh.append((dx, dy))
        return sh

    def batches(it, k=48):
        buf = []
        for f in it:
            buf.append(f)
            if len(buf) == k:
                yield buf; buf = []
        if buf:
            yield buf

    n = 0
    writes = []
    with ThreadPoolExecutor(nt) as ex:
        for batch in batches(vio.frames(P["video"], w, h)):
            F = list(ex.map(feat, batch))
            mgs = [mprev] + [x[2] for x in F] if mprev is not None else [x[2] for x in F]
            S = list(ex.map(shift, zip(mgs[:-1], mgs[1:])))
            si = 0
            for j, f in enumerate(batch):
                i = n
                g, hist, mg = F[j]
                if prev is None:
                    diffs.append((0.0, 0.0))
                else:
                    hd = cv2.compareHist(prevh, hist, cv2.HISTCMP_BHATTACHARYYA)
                    md = float(np.mean(np.abs(g - prev))) / 255
                    diffs.append((float(hd), md))
                feats.append(float(g.mean()))
                if mprev is not None:
                    sh = S[si]; si += 1
                    dx, dy = (np.median(np.array(sh), 0) if len(sh) >= 3 else (0.0, 0.0))
                    sx = w / MW
                    px, py = pos[-1]
                    pos.append((px + float(dx) * sx, py + float(dy) * sx)); conf.append(len(sh) / 12)
                mprev = mg
                prev, prevh = g, hist
                if i % every == 0:
                    # change test on a 64x36 block grid: any block that changed means new content
                    sm = cv2.resize(g, (64, 36), interpolation=cv2.INTER_AREA)
                    changed = last_s is None or float(np.abs(sm - last_s).max()) > 6 or i - last_f >= 10 * fps
                    if changed:
                        writes.append(ex.submit(cv2.imwrite, os.path.join(sdir, f"{i:06d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92]))
                        live.show(proj, f"מחפש חיתוכים - פריים {i}", f, every=0.7, width=960)
                        samples.append(i); last_s = sm; last_f = i
                n = i + 1
            if progress:
                progress(min(1.0, n / total))
        for x in writes:
            x.result()
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
                  "motion": [[round(a, 2), round(b, 2)] for a, b in pos], "motion_conf": [round(c, 3) for c in conf],
                  "score": [round(float(s), 4) for s in score], "every": every, "samples": samples})
    return cuts


def shot_of(cuts, n, f):
    """(start, end_exclusive) of shot containing frame f."""
    import bisect
    k = bisect.bisect_right(cuts, f) - 1
    s = cuts[k]
    e = cuts[k + 1] if k + 1 < len(cuts) else n
    return s, e
