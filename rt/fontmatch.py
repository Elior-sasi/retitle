"""Find a similar font: the Hebrew font closest to the typeface of the original on-screen title.

Latin and Hebrew letterforms differ, so letters are never compared one to one. Instead the same "style"
traits are measured on the original letters and on a Hebrew sample in every candidate font, and the
candidates are ranked by a weighted distance:

  weight    stroke thickness (distance transform along the skeleton) / letter height
  contrast  spread of the stroke thickness (monoline vs thick-thin / serif)
  corners   sharpness of corners and stroke ends: share of the outline that turns sharply over ~1/3 of the stroke
            (rounded fonts spread the turn, so almost none; flat-cut and angular fonts many). More stable across
            different texts than an opening-based measure, which matters because the original text is not the sample
  rough     outline roughness (perimeter vs a smoothed copy): brush, grunge, hand-cut
  slant     italic angle (the shear that makes the column profile sharpest)
  width     letter width / height

Candidates: every font in fontlib.catalog(); variable fonts at several weights (300-900) and widths (wdth axis).
Traits per font instance are measured once and cached in fonttraits.json (appdata).
Optionally Claude makes the final pick from a sheet of the top candidates (only with an API key).
"""
import os, re, json, base64, urllib.request
import cv2, numpy as np
from PIL import Image, ImageDraw
from . import vio, masks

SAMPLE = "אבגדהוזחטיכלמנסעפצקרשת"
LH = 64                                     # letter height after normalisation
WEIGHTS = (300, 400, 500, 600, 700, 800, 900)
WIDTHS = (75, 87.5, 100, 112.5, 125)
TRAITS = ("weight", "contrast", "corners", "rough", "slant", "width")
# weight of each trait in the distance, and its typical spread (puts them on one scale)
W = {"weight": 3.0, "contrast": 1.0, "corners": 2.4, "rough": 1.6, "slant": 1.2, "width": 1.2}
SPREAD = {"weight": 0.05, "contrast": 0.15, "corners": 0.015, "rough": 0.06, "slant": 6.0, "width": 0.12}
LATIN_WIDE = 1.12                           # Latin letters are a bit wider than Hebrew ones
SYSTEM = ("You are a type designer helping localize on-screen video titles into Hebrew. Image 1 is the original title. "
          "Image 2 shows numbered candidates: the same Hebrew text in different Hebrew fonts. Pick the candidate whose letterforms "
          "best match the original's typeface design, judged on: stroke weight and evenness (monoline vs contrast), corners and "
          "stroke endings (rounded, sharp, flat-cut, brush), proportions (condensed/normal/wide) and slant, and overall personality "
          "(playful, epic, elegant, technical, hand-made, horror). Ignore colour, outline, shadow, 3D, texture and size. Hebrew and "
          "Latin letterforms differ - compare the design language, not letter shapes. Also name the Latin typeface of the original "
          "if you recognize it (or the closest known one). Reply ONLY with JSON: {\"best\": n, \"ranking\": [n...], \"latin_font\": "
          "\"...\", \"reason_he\": \"one short Hebrew sentence\", \"settings_he\": \"optional Hebrew tip on weight/width\"}")


# ------------------------------------------------------------------ the original letters
def crop_of(proj, t, pad=6):
    reg = t.get("region")
    img = cv2.imread(os.path.join(proj, "cache", t["group"], f"{t['ref']:06d}.png")) if reg else None
    if img is None:
        img = cv2.imread(os.path.join(proj, "samples", f"{t['ref']:06d}.jpg"))
        reg = [0, 0, 0, 0]
    if img is None:
        return None, (0, 0)
    x0, y0, x1, y1 = t["ref_box"]
    ox, oy = reg[0], reg[1]
    a, b = max(0, y0 - oy - pad), max(0, x0 - ox - pad)
    return img[a:y1 - oy + pad, b:x1 - ox + pad], (ox + b, oy + a)


def letters(proj, t):
    """Clean letter mask: fill colour (fill_lab, tol) only - no grow, no shadow - inside the word boxes."""
    mp = t.get("mask") or {}
    if "fill_lab" not in mp:
        return None
    crop, (cx, cy) = crop_of(proj, t)
    if crop is None or crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = masks._wdist(lab, mp["fill_lab"])
    tol = mp.get("tol", 30)
    far = d[d > tol]
    thr = max(tol * 0.6, 0.5 * float(np.median(far))) if far.size > 20 else tol * 0.8   # half way to the surroundings
    core = (d < tol * 0.6).astype(np.uint8)
    m = ((d < thr) & (cv2.dilate(core, np.ones((3, 3), np.uint8), iterations=2) > 0)).astype(np.uint8)
    inside = np.zeros_like(m)
    for w in t.get("words") or [t["ref_box"]]:
        inside[max(0, w[1] - cy):max(0, w[3] - cy), max(0, w[0] - cx):max(0, w[2] - cx)] = 1
    m &= inside
    return m if m.sum() > 40 else None


# ------------------------------------------------------------------ traits
def _components(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    return n, lab, st


def normalise(m):
    """Scale so the letter height (median height of the big components, no dots/punctuation) is LH."""
    m = (m > 0).astype(np.uint8)
    n, lab, st = _components(m)
    if n < 2:
        return None
    hs = st[1:, cv2.CC_STAT_HEIGHT]; areas = st[1:, cv2.CC_STAT_AREA]
    big = (hs >= 0.45 * hs.max()) & (areas >= 8)
    if not big.any():
        return None
    m = np.isin(lab, 1 + np.nonzero(big)[0]).astype(np.uint8)
    h = float(np.median(hs[big]))
    if h < 6:
        return None
    k = LH / h
    ys, xs = np.nonzero(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    m = cv2.resize(m.astype(np.float32), (max(4, int(round(m.shape[1] * k))), max(4, int(round(m.shape[0] * k)))),
                   interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    return np.pad((m > 0.5).astype(np.uint8), 8)


def _skeleton(m):
    try:
        return cv2.ximgproc.thinning(m * 255) > 0
    except Exception:
        sk = np.zeros(m.shape, bool); el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)); img = m.copy()
        while img.any():
            er = cv2.erode(img, el); sk |= (img - cv2.dilate(er, el)) > 0; img = er
        return sk


def _perimeter(m):
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return sum(cv2.arcLength(c, True) for c in cs) or 1.0


def traits(m):
    """Style traits of a binary letter mask (any script). None if there is too little to measure."""
    n = normalise(m)
    if n is None or n.sum() < 200:
        return None
    dt = cv2.distanceTransform(n, cv2.DIST_L2, 5)
    sk = _skeleton(n)
    th = 2 * dt[sk]
    th = th[th > 0.5]
    if th.size < 20:
        return None
    stroke = float(np.median(th))
    weight = stroke / LH
    contrast = float(np.percentile(th, 85) - np.percentile(th, 15)) / max(stroke, 1)
    per = _perimeter(n)
    k = max(2, int(round(stroke * 0.35)))
    cs, _ = cv2.findContours(n, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    tot = sharp = 0
    for c in cs:
        p = c[:, 0, :].astype(np.float32)
        if len(p) < 3 * k:
            continue
        a = np.roll(p, k, 0) - p; b = np.roll(p, -k, 0) - p
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6)
        tot += len(p); sharp += int((np.degrees(np.arccos(np.clip(cos, -1, 1))) < 115).sum())
    corners = sharp / max(1, tot)
    sm = (cv2.GaussianBlur(n.astype(np.float32), (0, 0), 1.6) > 0.5).astype(np.uint8)   # removes only jagged / brush detail
    rough = per / _perimeter(sm) - 1 if sm.any() else 0.0
    best = (0, -1.0)
    for ang in range(-20, 21, 2):
        sh = np.tan(np.radians(ang))
        M = np.float32([[1, sh, -sh * n.shape[0] / 2], [0, 1, 0]])
        w = cv2.warpAffine(n, M, (n.shape[1], n.shape[0]), flags=cv2.INTER_NEAREST)
        col = w.sum(0).astype(np.float64)
        sharp = float((col ** 2).sum()) / max(1.0, col.sum()) ** 2
        if sharp > best[1]:
            best = (ang, sharp)
    _, _, st = _components(n)
    ws = st[1:, cv2.CC_STAT_WIDTH][st[1:, cv2.CC_STAT_HEIGHT] >= 0.45 * LH]
    width = float(np.median(ws)) / LH if ws.size else 0.6
    return {"weight": weight, "contrast": contrast, "corners": corners, "rough": rough, "slant": float(best[0]), "width": width}


# ------------------------------------------------------------------ candidates (cached)
def _cache_path():
    return os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/.config"), "Retitle", "fonttraits.json")


def _axes(path):
    from fontTools.ttLib import TTFont
    try:
        f = TTFont(path, lazy=True)
        return {a.axisTag: (a.minValue, a.defaultValue, a.maxValue) for a in f["fvar"].axes} if "fvar" in f else {}
    except Exception:
        return {}


def render_mask(name, text=SAMPLE, weight=None, wdth=None, size=110):
    from . import render
    f = render._font(name, size, weight, wdth)
    kw = render._kw(direction="rtl", language="he")
    t = render._vis(text)
    bb = f.getbbox(t, **kw)
    w, h = bb[2] - bb[0] + 40, bb[3] - bb[1] + 40
    if w <= 0 or h <= 0 or w > 12000:
        return None
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((20 - bb[0], 20 - bb[1]), t, font=f, fill=255, **kw)
    return (np.array(im) > 127).astype(np.uint8)


def instances(path):
    ax = _axes(path)
    ws = [w for w in WEIGHTS if "wght" in ax and ax["wght"][0] <= w <= ax["wght"][2]] or [None]
    ds = [d for d in WIDTHS if "wdth" in ax and ax["wdth"][0] <= d <= ax["wdth"][2]] or [None]
    return [(w, d) for w in ws for d in ds]


def candidate_traits(names=None, progress=None, include_remote=False):
    """[(name, weight, wdth, traits)] for every installed candidate. Measured once per file/mtime/instance."""
    from . import fontlib
    cat = fontlib.catalog()
    cache = vio.load_json(_cache_path(), {})
    out, dirty = [], False
    items = [(n, e) for n, e in cat.items() if (names is None or n in names) and (e["installed"] or include_remote)
             and not re.search(r"italic|oblique", n, re.I)]
    for i, (n, e) in enumerate(items):
        if progress and i % 3 == 0:
            progress(i / max(1, len(items)))
        if not e["installed"]:
            try:
                fontlib.ensure(n)
            except Exception:
                continue
        p = e["file"]
        try:
            mt = int(os.path.getmtime(p))
        except OSError:
            continue
        for w, d in instances(p):
            k = f"{p}|{mt}|{w}|{d}"
            if k not in cache:
                try:
                    m = render_mask(n, weight=w, wdth=d)
                    cache[k] = traits(m) if m is not None else None
                except Exception:
                    cache[k] = None
                dirty = True
            if cache[k]:
                out.append((n, w, d, cache[k]))
        if dirty and i % 20 == 19:                       # keep partial work if the app closes
            os.makedirs(os.path.dirname(_cache_path()), exist_ok=True)
            vio.save_json(_cache_path(), cache)
    if dirty:
        os.makedirs(os.path.dirname(_cache_path()), exist_ok=True)
        vio.save_json(_cache_path(), cache)
    return out


def download_google(progress=None):
    """Download every Google font with Hebrew that is not on this computer yet (once)."""
    from . import fontlib
    cat = fontlib.catalog()
    todo = [n for n, e in cat.items() if e["source"] == "google" and not e["installed"]]
    ok = 0
    for i, n in enumerate(todo):
        if progress:
            progress(i / max(1, len(todo)))
        try:
            fontlib.ensure(n); ok += 1
        except Exception:
            pass
    return ok, len(todo)


# ------------------------------------------------------------------ ranking
def script_of(s):
    ls = [c for c in s if c.isalpha()]
    if re.search(r"[֐-׿]", s):
        return "hebrew"
    if len(ls) >= 2 and sum(c.isascii() for c in ls) / len(ls) > 0.8:
        return "latin"
    return "other"


def _src_adj(src, script):
    s = dict(src)
    if script == "latin":
        s["width"] = s["width"] / LATIN_WIDE
    return s


def _dist(s, c, script):
    tot = 0.0
    for k in TRAITS:
        wk = W[k] if not (k == "width" and script == "other") else 0.3   # e.g. Japanese: square letters say little
        tot += wk * ((s[k] - c[k]) / SPREAD[k]) ** 2
    return float(np.sqrt(tot))


DIFF = {"weight": ("עבה יותר", "דק יותר"), "contrast": ("עם יותר ניגוד עבה-דק", "אחיד יותר"),
        "corners": ("חד יותר", "עגול יותר"), "rough": ("מחוספס יותר", "חלק יותר"),
        "slant": ("נוטה יותר", "זקוף יותר"), "width": ("רחב יותר", "צר יותר")}


def diff_he(s, c, script):
    parts = []
    for k in sorted(TRAITS, key=lambda k: -abs(c[k] - s[k]) / SPREAD[k] * W[k]):
        if k == "width" and script == "other":
            continue
        z = (abs(c[k]) - abs(s[k])) / SPREAD[k] if k == "slant" else (c[k] - s[k]) / SPREAD[k]
        if abs(z) < 0.8:
            continue
        parts.append(DIFF[k][0 if z > 0 else 1])
        if len(parts) == 2:
            break
    return "ההבדל: " + ", ".join(parts) if parts else "דומה מאוד"


def rank(src, cands, script="latin", top=8):
    """One instance per font, best first: [{font, weight, wdth, score, diff}]."""
    s = _src_adj(src, script)
    best = {}
    for n, w, d, tr in cands:
        sc = _dist(s, tr, script)
        if n not in best or sc < best[n][0]:
            best[n] = (sc, w, d, tr)
    res, seen = [], []
    for n, (sc, w, d, tr) in sorted(best.items(), key=lambda x: x[1][0]):
        if any(tr == x for x in seen):
            continue                                      # the same file listed under two names
        seen.append(tr)
        res.append({"font": n, "weight": w, "wdth": d, "score": round(sc, 3), "diff": diff_he(s, tr, script)})
        if len(res) == top:
            break
    return res


# ------------------------------------------------------------------ Claude (optional)
def sheet(rows, text, path):
    """Numbered sheet: the Hebrew text in each candidate font (what Claude sees as image 2)."""
    from . import render
    kw = render._kw(direction="rtl", language="he")
    t = render._vis(text)
    tiles = []
    for i, r in enumerate(rows, 1):
        im = Image.new("RGB", (900, 110), (255, 255, 255))
        dr = ImageDraw.Draw(im)
        try:
            f = render._font(r["font"], 64, r.get("weight"), r.get("wdth"))
            bb = f.getbbox(t, **kw)
            w = min(1600, bb[2] - bb[0] + 130)
            im = Image.new("RGB", (max(400, w), 110), (255, 255, 255)); dr = ImageDraw.Draw(im)
            dr.text((im.width - 20 - (bb[2] - bb[0]) - bb[0], 22 - bb[1]), t, font=f, fill=(0, 0, 0), **kw)
        except Exception:
            pass
        dr.rectangle([0, 0, 80, 110], fill=(30, 30, 30))
        try:
            dr.text((22, 28), str(i), font=render._font("Heebo", 46, 700), fill=(255, 255, 255))
        except Exception:
            dr.text((30, 40), str(i), fill=(255, 255, 255))
        tiles.append(im)
    out = Image.new("RGB", (max(x.width for x in tiles), 112 * len(tiles)), (190, 190, 190))
    for i, x in enumerate(tiles):
        out.paste(x, (0, i * 112))
    out.save(path, quality=88)
    return path


def ask_claude(api_key, model, crop_jpg, sheet_jpg, n):
    def b64(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    body = {"model": model, "max_tokens": 500, "system": SYSTEM, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Image 1 - the original title:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(crop_jpg)}},
        {"type": "text", "text": f"Image 2 - candidates 1 to {n}:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(sheet_jpg)}}]}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        res = json.loads(r.read().decode())
    text = "".join(b.get("text", "") for b in res.get("content", []))
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


# ------------------------------------------------------------------ one title / all titles
def match(proj, t, top=8, api_key="", model="claude-sonnet-5-5", include_remote=False, progress=None, log=print):
    """Ranking for title t: {"rows": [...], "latin_font", "reason_he", "settings_he", "by_claude", "error"?}."""
    m = letters(proj, t)
    src = traits(m) if m is not None else None
    if src is None:
        return {"rows": [], "error": "לא הצלחתי לבודד את האותיות של הכותרת הזאת (רקע עמוס או אותיות מעוצבות מאוד)"}
    script = script_of(t.get("src") or "")
    rows = rank(src, candidate_traits(progress=progress, include_remote=include_remote), script, top)
    out = {"rows": rows, "traits": src, "latin_font": "", "reason_he": "", "settings_he": "", "by_claude": False}
    if api_key and rows:
        try:
            d = os.path.join(proj, "fontmatch"); os.makedirs(d, exist_ok=True)
            crop, _ = crop_of(proj, t, pad=12)
            cj = os.path.join(d, f"crop_{t['id']}.jpg"); cv2.imwrite(cj, crop)
            sj = sheet(rows, t.get("he") or SAMPLE, os.path.join(d, f"fontmatch_{t['id']}.jpg"))
            r = ask_claude(api_key, model, cj, sj, len(rows))
            order = [int(x) for x in (r.get("ranking") or []) if str(x).isdigit() and 1 <= int(x) <= len(rows)]
            b = int(r.get("best", 0) or 0)
            if 1 <= b <= len(rows):
                order = [b] + [x for x in order if x != b]
            order = list(dict.fromkeys(order)) + [i for i in range(1, len(rows) + 1) if i not in order]
            out["rows"] = [rows[i - 1] for i in order]
            out.update(latin_font=r.get("latin_font", "") or "", reason_he=r.get("reason_he", "") or "",
                       settings_he=r.get("settings_he", "") or "", by_claude=True)
        except Exception as e:
            log(f"Claude לא זמין לבחירת הגופן, נשאר הדירוג המקומי ({e.__class__.__name__})")
    return out


def apply_row(t, row):
    st = t.setdefault("style", {})
    st["font"] = row["font"]
    for k in ("weight", "wdth"):
        if row.get(k):
            st[k] = row[k]
        else:
            st.pop(k, None)


def run(proj, api_key="", model="claude-sonnet-5-5", progress=None, log=print, only=None):
    """Automatic pick at analysis time, for titles whose font nobody chose (not locked, not from the series).
    Local ranking only - the Claude check is per title, from the "find a similar font" button."""
    T = vio.load_json(os.path.join(proj, "titles.json"))
    todo = [t for t in T["titles"] if t.get("translate", True) and t.get("mode") != "remove" and t.get("kind") != "sub"
            and "font" not in t.get("style_locked", []) and not t.get("from_series")
            and (only is None or t["id"] in only)]
    if not todo:
        return 0
    cands = candidate_traits(progress=lambda x: progress(0.8 * x) if progress else None)
    done = 0
    for i, t in enumerate(todo):
        if progress:
            progress(0.8 + 0.2 * i / len(todo))
        m = letters(proj, t)
        src = traits(m) if m is not None else None
        if src is None:
            continue
        rows = rank(src, cands, script_of(t.get("src") or ""), 8)
        if not rows:
            continue
        apply_row(t, rows[0])
        t["font_top"] = [[r["font"], r["weight"], r["wdth"], r["diff"]] for r in rows[:5]]
        t["font_auto"] = True
        done += 1
    vio.save_json(os.path.join(proj, "titles.json"), T)
    return done
