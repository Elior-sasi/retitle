"""Live preview while a job runs: the frame being erased / built, shown in the progress card."""
import os, time, json
import cv2, numpy as np

_LAST = [0.0]


def show(proj, label, *panels, every=0.5, width=720):
    """Write live.jpg (panels side by side) at most every `every` seconds. Never raises."""
    now = time.time()
    if now - _LAST[0] < every or not proj:
        return
    _LAST[0] = now
    try:
        h = max(p.shape[0] for p in panels)
        ims = []
        for p in panels:
            if p.ndim == 2:
                p = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
            p = cv2.resize(p, (max(1, int(p.shape[1] * h / p.shape[0])), h)) if p.shape[0] != h else p
            ims.append(p)
            ims.append(np.full((h, max(2, h // 60), 3), 40, np.uint8))
        im = np.hstack(ims[:-1])
        sc = min(1.0, width / im.shape[1])
        if sc < 1:
            im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        d = os.path.join(proj, "live")
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, f"tmp{os.getpid()}.jpg")
        cv2.imwrite(tmp, im, [cv2.IMWRITE_JPEG_QUALITY, 80])
        os.replace(tmp, os.path.join(d, "live.jpg"))
        with open(os.path.join(d, f"tmp{os.getpid()}.json"), "w", encoding="utf-8") as f:
            json.dump({"t": now, "label": label}, f, ensure_ascii=False)
        os.replace(os.path.join(d, f"tmp{os.getpid()}.json"), os.path.join(d, "live.json"))
    except Exception:
        pass


def state(proj):
    try:
        with open(os.path.join(proj, "live", "live.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
