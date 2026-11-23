"""Automatic font matching: pick the Hebrew font closest to the on-screen original.

Latin source text: the same text is rendered in every candidate font (most Hebrew fonts also carry Latin)
and compared with the actual letters in the frame - outline shape (chamfer distance), width/height
proportion and stroke weight. Variable fonts are tried at several weights.
Other scripts (Japanese, Arabic...) or stylised lettering: with an Anthropic API key Claude looks at the
title and the top candidates and chooses; without one, the stroke weight decides.
Results: style.font (+ weight) and t["font_top"] (best 5) for one-click alternatives.
"""
import os, re, json, base64, urllib.request
import cv2, numpy as np
from PIL import Image, ImageDraw
from . import vio, masks

H = 48                                 # comparison height
WEIGHTS = (300, 500, 700, 900)          # coarse pass; the best fonts are refined at every 100


def letters(proj, t):
    """Binary mask of the title's fill colour at its reference frame (None if it can't be isolated)."""
    reg = t.get("region"); mp = t.get("mask") or {}
    if not reg or "fill_lab" not in mp:
        return None
    img = cv2.imread(os.path.join(proj, "cache", t["group"], f"{t['ref']:06d}.png"))
    if img is None:
        return None
    x0, y0, x1, y1 = t["ref_box"]
    x0, y0, x1, y1 = x0 - reg[0], y0 - reg[1], x1 - reg[0], y1 - reg[1]
    crop = img[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = masks._wdist(lab, mp["fill_lab"])
    tol = mp.get("tol", 30)
    far = d[d > tol]
    # threshold half way between the fill and what surrounds it (outline / background): unbiased stroke width
    thr = max(tol * 0.6, 0.5 * float(np.median(far))) if far.size > 20 else tol * 0.8
    core = (d < tol * 0.6).astype(np.uint8)
    m = ((d < thr) & (cv2.dilate(core, np.ones((3, 3), np.uint8), iterations=2) > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 6:
            keep[lab_ == i] = 1
    return keep if keep.sum() > 40 else None


def _norm(m):
    ys, xs = np.nonzero(m)
    if len(xs) < 10:
        return None
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    W = max(4, int(round(w * H / h)))
    return (cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)


def _stroke(m):
    """Stroke thickness relative to height (ink area / skeleton length / height)."""
    try:
        sk = cv2.ximgproc.thinning(m * 255) > 0
        L = max(1, sk.sum())
    except Exception:
        L = max(1, (cv2.Canny(m * 255, 50, 150) > 0).sum() / 2)
    return float(m.sum()) / L / m.shape[0]


def descriptor(m):
    n = _norm(m)
    if n is None:
        return None
    edge = cv2.Canny(n * 255, 50, 150) > 0
    dt = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    return {"m": n, "edge": edge, "dt": dt, "aspect": n.shape[1] / H, "stroke": _stroke(n)}


def _score(a, b):
    """Lower is closer."""
    asp = abs(np.log(a["aspect"] / b["aspect"]))
    eb = cv2.resize(b["edge"].astype(np.uint8), (a["m"].shape[1], H), interpolation=cv2.INTER_NEAREST) > 0
    db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)
    ch = (a["dt"][eb].mean() if eb.any() else 9) + (db[a["edge"]].mean() if a["edge"].any() else 9)
    return ch / H * 10 + asp * 2.0 + abs(a["stroke"] - b["stroke"]) * 12


def _render(font, text):
    pad = 20
    try:
        bb = font.getbbox(text)
    except Exception:
        return None
    w, h = bb[2] - bb[0] + 2 * pad, bb[3] - bb[1] + 2 * pad
    if w <= 0 or h <= 0 or w > 8000:
        return None
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((pad - bb[0], pad - bb[1]), text, font=font, fill=255)
    return (np.array(im) > 127).astype(np.uint8)


def _covers(path, text):
    from fontTools.ttLib import TTFont
    try:
        cmap = TTFont(path, lazy=True).getBestCmap() or {}
    except Exception:
        return False
    return all(ord(c) in cmap for c in text if not c.isspace())


def is_latin(s):
    letters_ = [c for c in s if c.isalpha()]
    return len(letters_) >= 3 and sum(c.isascii() for c in letters_) / len(letters_) > 0.9


def candidates(include_remote=True):
    from . import fontlib
    cat = fontlib.catalog()
    out = []
    for n, e in cat.items():
        if not e["installed"] and not include_remote:
            continue
        if re.search(r"italic|oblique", n, re.I):
            continue                                   # titles are rarely italic; keeps the search fast
        out.append(n)
    return out


def _font_obj(name, size, weight):
    from . import render
    return render._font(name, size, weight)


def _is_variable(name):
    from . import render
    try:
        f = render._font(name, 40, None)
        return bool(f.get_variation_axes())
    except Exception:
        return False


def match_latin(desc, text, names):
    """[(score, name, weight)] best first."""
    from . import render
    res = []
    for n in names:
        try:
            path = render.font_path(n)
        except Exception:
            continue
        if not _covers(path, text):
            continue
        ws = WEIGHTS if _is_variable(n) else (None,)
        best = None
        for w in ws:
            try:
                m = _render(_font_obj(n, 120, w), text)
            except Exception:
                m = None
            d = descriptor(m) if m is not None else None
            if d is None:
                continue
            s = _score(desc, d)
            if best is None or s < best[0]:
                best = (s, n, w)
        if best:
            res.append(best)
    res.sort(key=lambda x: x[0])
    # refine the weight of the best variable fonts
    out = []
    for s0, n, w in res[:6]:
        if w is None:
            out.append((s0, n, w)); continue
        best = (s0, n, w)
        for w2 in range(max(100, w - 150), min(900, w + 150) + 1, 50):
            try:
                d = descriptor(_render(_font_obj(n, 120, w2), text))
            except Exception:
                d = None
            if d:
                s2 = _score(desc, d)
                if s2 < best[0]:
                    best = (s2, n, w2)
        out.append(best)
    out.sort(key=lambda x: x[0])
    return out + res[6:]


def match_stroke(stroke, names, sample="שלום עולם"):
    """No usable letters to compare: closest stroke weight in Hebrew."""
    res = []
    for n in names:
        ws = WEIGHTS if _is_variable(n) else (None,)
        for w in ws:
            try:
                m = _render(_font_obj(n, 120, w), sample)
            except Exception:
                continue
            d = descriptor(m) if m is not None else None
            if d:
                res.append((abs(d["stroke"] - stroke) * 25, n, w))
    res.sort(key=lambda x: x[0])
    seen, out = set(), []
    for r in res:
        if r[1] not in seen:
            seen.add(r[1]); out.append(r)
    return out


def ask_claude(api_key, model, img_path, cands, src):
    """Claude looks at the title and picks among candidates (names + Hebrew descriptions)."""
    from . import fontlib
    cat = fontlib.catalog()
    lst = "\n".join(f"- {n}: {cat.get(n, {}).get('he', '')}" for n in cands)
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = {"model": model, "max_tokens": 300, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
        {"type": "text", "text": f"This on-screen title (text: {src!r}) will be replaced with Hebrew. Choose the Hebrew font from this list "
                                 f"whose style is closest (serif/sans, weight, width, roundness, hand-made, texture), and a weight 100-900.\n{lst}\n"
                                 'Answer only JSON: {"font": "<exact name from the list>", "weight": <number>}'}]}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        res = json.loads(r.read().decode())
    text = "".join(b.get("text", "") for b in res.get("content", []))
    m = re.search(r"\{.*\}", text, re.S)
    d = json.loads(m.group(0)) if m else {}
    return (d.get("font"), d.get("weight")) if d.get("font") in cands else (None, None)


def run(proj, api_key="", model="claude-sonnet-5-5", progress=None, log=print):
    """Choose fonts for titles that have no font chosen by hand or by the series."""
    T = vio.load_json(os.path.join(proj, "titles.json"))
    todo = [t for t in T["titles"] if t.get("translate", True) and t.get("mode") != "remove" and t.get("kind") != "sub"
            and "font" not in t.get("style_locked", []) and not t.get("from_series")]
    if not todo:
        return 0
    names = candidates(include_remote=True)
    local = candidates(include_remote=False)
    done = 0
    for i, t in enumerate(todo):
        if progress:
            progress(i / len(todo))
        m = letters(proj, t)
        desc = descriptor(m) if m is not None else None
        src = (t.get("src") or "").strip()
        top = []
        if desc and is_latin(src):
            try:
                top = match_latin(desc, src, names)
            except Exception:
                top = match_latin(desc, src, local)
        stroke = desc["stroke"] if desc else (t.get("style", {}).get("stroke", 6) / max(1, t.get("style", {}).get("letter_h", 40)))
        if True:
            if api_key:
                try:
                    shortlist = [x[1] for x in top[:8]] or [x[1] for x in match_stroke(stroke, local)[:8]]
                    for extra in ("Heebo", "Frank Ruhl Libre", "Secular One", "Suez One", "Karantina", "Amatic SC", "Gveret Levin", "Varela Round", "Rubik", "Bellefair"):
                        if extra in names and extra not in shortlist:
                            shortlist.append(extra)
                    thumb = os.path.join(proj, "thumbs", f"{t['id']}.jpg")
                    if os.path.exists(thumb):
                        f, w = ask_claude(api_key, model, thumb, shortlist, src)
                        if f:
                            keep_w = next((x[2] for x in top if x[1] == f), None)
                            top = [(0, f, keep_w or w)] + [x for x in top if x[1] != f]
                except Exception as e:
                    log(f"התאמת גופן עם Claude נכשלה ({t['id']}): {e}")
            if not top:
                top = match_stroke(stroke, local)[:5]
        if not top:
            continue
        _, f, w = top[0]
        st = t.setdefault("style", {})
        st["font"] = f
        if w:
            st["weight"] = int(w)
        else:
            st.pop("weight", None)
        t["font_top"] = [[x[1], x[2]] for x in top[:5]]
        t["font_auto"] = True
        done += 1
    vio.save_json(os.path.join(proj, "titles.json"), T)
    return done
