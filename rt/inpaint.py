"""Exemplar-based inpainting (Criminisi et al. 2004) for background that is never visible in any
other frame. Copies real patches of the surrounding picture, filling along strong edges first,
so structure continues and grain/texture is kept (no soft blur)."""
import cv2, numpy as np


def exemplar(img, hole, patch=9, search=70, max_iter=20000):
    img = img.copy()
    hole = hole.astype(bool).copy()
    if not hole.any():
        return img
    H, W = hole.shape
    r = patch // 2
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    conf = (~hole).astype(np.float32)
    # isophote strength from the known image (filled once with a quick diffusion so the edges
    # running into the hole are seen)
    pre = cv2.inpaint(img, hole.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    g = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 1.0)
    edge = edge / (edge.max() + 1e-6)
    k = np.ones((patch, patch), np.uint8)
    it = 0
    while hole.any() and it < max_iter:
        it += 1
        front = hole & (cv2.dilate((~hole).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        ys, xs = np.nonzero(front)
        if len(ys) == 0:
            break
        csum = cv2.boxFilter(conf, -1, (patch, patch), normalize=True, borderType=cv2.BORDER_REPLICATE)
        pr = csum[ys, xs] * (edge[ys, xs] + 0.05)
        i = int(np.argmax(pr))
        py, px = int(ys[i]), int(xs[i])
        y0, y1 = max(0, py - r), min(H, py + r + 1)
        x0, x1 = max(0, px - r), min(W, px + r + 1)
        tgt = lab[y0:y1, x0:x1]
        tknown = (~hole[y0:y1, x0:x1]).astype(np.float32)
        ph, pw = y1 - y0, x1 - x0
        # search window
        sy0, sy1 = max(0, py - search), min(H, py + search + 1)
        sx0, sx1 = max(0, px - search), min(W, px + search + 1)
        win = lab[sy0:sy1, sx0:sx1]
        if win.shape[0] < ph or win.shape[1] < pw:
            hole[y0:y1, x0:x1] = False
            continue
        m3 = np.repeat(tknown[..., None], 3, axis=2)
        if tknown.sum() < 1:
            ssd = np.zeros((win.shape[0] - ph + 1, win.shape[1] - pw + 1), np.float32)
        else:
            ssd = cv2.matchTemplate(win, tgt, cv2.TM_SQDIFF, mask=m3)
        # source patches must be fully known
        hw = hole[sy0:sy1, sx0:sx1].astype(np.float32)
        hc = cv2.boxFilter(hw, -1, (pw, ph), normalize=False, borderType=cv2.BORDER_CONSTANT)
        hc = hc[ph // 2: ph // 2 + ssd.shape[0], pw // 2: pw // 2 + ssd.shape[1]]
        ssd = np.where(hc > 0.5, np.inf, ssd)
        ssd[~np.isfinite(ssd)] = np.inf
        j = int(np.argmin(ssd))
        if not np.isfinite(ssd.flat[j]):
            # nothing fully known nearby: widen once, else diffuse this patch
            if search < 220:
                search = int(search * 1.6)
                continue
            fill = hole[y0:y1, x0:x1]
            img[y0:y1, x0:x1][fill] = pre[y0:y1, x0:x1][fill]
            lab[y0:y1, x0:x1][fill] = cv2.cvtColor(pre[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)[fill]
            conf[y0:y1, x0:x1][fill] = csum[py, px]
            hole[y0:y1, x0:x1] = False
            continue
        sy, sx = np.unravel_index(j, ssd.shape)
        sy += sy0; sx += sx0
        fill = hole[y0:y1, x0:x1]
        img[y0:y1, x0:x1][fill] = img[sy:sy + ph, sx:sx + pw][fill]
        lab[y0:y1, x0:x1][fill] = lab[sy:sy + ph, sx:sx + pw][fill]
        conf[y0:y1, x0:x1][fill] = csum[py, px]
        hole[y0:y1, x0:x1] = False
    return img
