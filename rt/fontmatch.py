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
import os, re, json, base64, glob, shutil, urllib.request
from .house import with_house
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
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
          "\"...\", \"reason_he\": \"one short Hebrew sentence\", \"settings_he\": \"optional Hebrew tip on weight/width\", "
          "\"look\": {\"weight\": 400-900 or null, \"fill\": [\"#top\", \"#bottom\"] (one colour twice if flat), "
          "\"extrude\": {\"dir\": \"down\"|\"down-right\"|\"down-left\"|\"right\"|\"left\", \"depth\": 0.0-0.15 (of letter height), "
          "\"color\": \"#hex\"} or null, \"outline\": {\"width\": 0.0-0.1, \"color\": \"#hex\"} or null, "
          "\"shadow\": {\"dx\": -0.1..0.1, \"dy\": -0.1..0.1, \"blur\": 0-0.1, \"color\": \"#hex\", \"opacity\": 0-1} or null}}. "
          "\"look\" describes the ORIGINAL's colouring and depth exactly as you see it (the 3D side face may be lit in another "
          "colour than the letters, e.g. pale blue-grey); distances are fractions of the letter height.")

# Hebrew display fonts always offered to Claude next to the measured candidates: the measurement alone
# misses whole families (a rounded heavy font can lose to a sans on stroke statistics)
CURATED = [("Fredoka", 700), ("Rubik", 700), ("Secular One", None), ("Suez One", None), ("Karantina", 700),
           ("Heebo", 800), ("Frank Ruhl Libre", 700), ("Bona Nova", 700), ("Playpen Sans Hebrew", 700),
           ("M PLUS Rounded 1c", 800), ("Amatic SC", 700), ("Gveret Levin", None), ("Assistant", 800), ("Bellefair", None)]


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


def letters(proj, t, pad=6):
    """Clean letter mask: fill colour (fill_lab, tol) only - no grow, no shadow - inside the word boxes."""
    mp = t.get("mask") or {}
    if "fill_lab" not in mp:
        return None
    crop, (cx, cy) = crop_of(proj, t, pad)
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


def ask_claude(api_key, model, crop_jpg, sheet_jpg, n, hint=""):
    def b64(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    body = {"model": model, "max_tokens": 500, "system": [{"type": "text", "text": with_house(SYSTEM), "cache_control": {"type": "ephemeral"}}], "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Image 1 - the original title:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(crop_jpg)}},
        {"type": "text", "text": (hint + "\n" if hint else "") + f"Image 2 - candidates 1 to {n}:"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(sheet_jpg)}}]}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    from .translate import call
    res = call(req)                    # API key, or Claude Code on the subscription
    text = "".join(b.get("text", "") for b in res.get("content", []))
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}



# ------------------------------------------------------------------ Lens: the source's Latin font, and every font compared
def latin_fonts():
    """{family: file} of every font on this computer and the user's imported ones (any script), for finding
    the source's Latin font among them. Cached by folder contents."""
    from . import fontlib
    from fontTools.ttLib import TTFont
    cp = os.path.join(os.path.dirname(_cache_path()), "latinfonts.json")
    cache = vio.load_json(cp, {})
    out, dirty = {}, False
    dirs = [d for d, _ in fontlib.system_dirs()] + [fontlib.user_dir()]
    for d in dirs:
        for p in glob.glob(os.path.join(d, "**", "*.*"), recursive=True):
            if not p.lower().endswith((".ttf", ".otf")):
                continue
            try:
                k = f"{p}|{int(os.path.getmtime(p))}"
            except OSError:
                continue
            if k not in cache:
                try:
                    f = TTFont(p, lazy=True)
                    cm = f.getBestCmap() or {}
                    nm = f["name"]
                    fam = (nm.getDebugName(16) or nm.getDebugName(1) or "").strip()
                    sub = (nm.getDebugName(17) or nm.getDebugName(2) or "").strip().lower()
                    cache[k] = [fam, sub, all(ord(c) in cm for c in "AEHRaeo")]
                except Exception:
                    cache[k] = ["", "", False]
                dirty = True
            fam, sub, latin = cache[k]
            if fam and latin and not re.search(r"italic|oblique", sub):
                if fam not in out or sub in ("regular", "bold", "book"):
                    out[fam] = p
    if dirty:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        vio.save_json(cp, cache)
    return out


def _tile(draw_fn, path):
    im = Image.new("RGB", (900, 200), (255, 255, 255))
    if not draw_fn(im):
        return None
    im.save(path)
    return path


def ai_fonts(proj, t, crop_jpg, he_rows, log=print):
    """What Lens adds: the source's Latin font by name (t["lens"]), the closest Latin fonts on this computer,
    and every Hebrew candidate ordered by how much its lettering looks like the source.
    Returns {"lens": [...], "latin_pc": [names], "he_order": {font name: similarity}} or {} without Lens."""
    from . import aimodels, render
    if not aimodels.lens_ready() or not crop_jpg:
        return {}
    d = os.path.join(proj, "fontmatch", "ai_" + t["id"])
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    out = {}
    try:
        out["lens"] = (aimodels.lens([crop_jpg], 5).get(crop_jpg) or [])
    except Exception as e:
        log(f"Lens לא הצליח ({str(e)[:100]})")
        return {}
    src = re.sub(r"\s+", " ", t.get("src") or "").strip()[:24] or "Title"
    imgs, names = [crop_jpg], []
    # Latin: the source text in every font on this computer
    if re.search(r"[A-Za-z]", src):
        for i, (fam, p) in enumerate(sorted(latin_fonts().items())):
            def draw(im, p=p):
                try:
                    f = ImageFont.truetype(p, 120)
                    bb = f.getbbox(src)
                    if bb[2] - bb[0] > 4000 or bb[2] <= bb[0]:
                        return False
                    im2 = Image.new("RGB", (bb[2] - bb[0] + 60, bb[3] - bb[1] + 60), (255, 255, 255))
                    ImageDraw.Draw(im2).text((30 - bb[0], 30 - bb[1]), src, font=f, fill=(0, 0, 0))
                    im.paste(im2.resize((900, max(1, int(im2.height * 900 / im2.width)))).crop((0, 0, 900, 200)))
                    return True
                except Exception:
                    return False
            q = _tile(draw, os.path.join(d, f"l{i:04d}.png"))
            if q:
                imgs.append(q); names.append(("latin", fam))
    # Hebrew: the Hebrew text in every candidate
    he = t.get("he") or SAMPLE
    kw = render._kw(direction="rtl", language="he")
    vt = render._vis(he)
    for i, r in enumerate(he_rows):
        def draw(im, r=r):
            try:
                f = render._font(r["font"], 120, r.get("weight"), r.get("wdth"))
                bb = f.getbbox(vt, **kw)
                im2 = Image.new("RGB", (bb[2] - bb[0] + 60, bb[3] - bb[1] + 60), (255, 255, 255))
                ImageDraw.Draw(im2).text((30 - bb[0], 30 - bb[1]), vt, font=f, fill=(0, 0, 0), **kw)
                im.paste(im2.resize((900, max(1, int(im2.height * 900 / im2.width)))).crop((0, 0, 900, 200)))
                return True
            except Exception:
                return False
        q = _tile(draw, os.path.join(d, f"h{i:04d}.png"))
        if q:
            imgs.append(q); names.append(("he", r["font"]))
    try:
        E = aimodels.embed(imgs)
    except Exception as e:
        log(f"השוואת הגופנים עם Lens לא הצליחה ({str(e)[:100]})")
        E = None
    if E is not None and len(E) == len(imgs):
        sim = E[1:] @ E[0]
        lat = sorted([(float(sim[i]), n) for i, (k, n) in enumerate(names) if k == "latin"], reverse=True)
        out["latin_pc"] = [n for _, n in lat[:5]]
        out["he_order"] = {}
        for i, (k, n) in enumerate(names):
            if k == "he":
                out["he_order"][n] = max(out["he_order"].get(n, -1.0), float(sim[i]))
    shutil.rmtree(d, ignore_errors=True)
    return out


def same_family(hebrew_names, latin_names):
    """Hebrew candidates that are the Hebrew of one of the source's Latin fonts (Fredoka -> Fredoka)."""
    norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower().replace("hebrew", ""))
    want = [norm(n) for n in latin_names if n]
    return [h for h in hebrew_names if any(w and (norm(h) == w or norm(h).startswith(w)) for w in want)]


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
    # Lens: the source's Latin font by name and by comparison with every font on this computer, and every
    # Hebrew font (this computer, Adobe, imported, Google) ordered by how much it looks like the source
    hint, ai = "", {}
    try:
        from . import aimodels
        if aimodels.lens_ready():
            d = os.path.join(proj, "fontmatch"); os.makedirs(d, exist_ok=True)
            crop, _ = crop_of(proj, t, pad=12)
            cj = os.path.join(d, f"crop_{t['id']}.jpg"); cv2.imwrite(cj, crop)
            allc = rank(src, candidate_traits(include_remote=include_remote), script, 400)
            ai = ai_fonts(proj, t, cj, allc, log=log)
            if ai:
                t["lens"] = [[x["name"], x["score"]] for x in ai.get("lens", [])[:5]]
                t["lens_pc"] = ai.get("latin_pc", [])
                ho = ai.get("he_order") or {}
                by_look = sorted(allc, key=lambda r_: -ho.get(r_["font"], -1))
                fam = same_family([r_["font"] for r_ in allc],
                                  [x["name"] for x in ai.get("lens", []) if x["score"] >= 0.2] + ai.get("latin_pc", [])[:2])
                names = {r_["font"] for r_ in rows}
                first = [r_ for r_ in allc if r_["font"] in fam]
                rows[:0] = [r_ for r_ in first if r_["font"] not in names]
                names |= {r_["font"] for r_ in first}
                for r_ in by_look[:8]:
                    if r_["font"] not in names:
                        rows.append(r_); names.add(r_["font"])
                if first and not api_key:            # no Claude: the source's own family in Hebrew wins
                    out["rows"] = rows = first[:1] + [r_ for r_ in rows if r_["font"] != first[0]["font"]]
                lens_txt = ", ".join(f"{x['name']} ({x['score']:.2f})" for x in ai.get("lens", [])[:3])
                hint = (f"A font recognizer (Lens) reads the original's Latin font as: {lens_txt}. "
                        f"Closest Latin fonts on this computer: {', '.join(ai.get('latin_pc', [])[:3])}. ")
                if first:
                    hint += f"The Hebrew version of that family is among the candidates: {', '.join(r_['font'] for r_ in first[:3])}. "
                hint += "Use this as a strong hint, but judge by the image."
                out["lens"] = t["lens"]
    except Exception as e:
        log(f"Lens לא זמין ({str(e)[:100]})")
    if api_key and rows:
        try:
            d = os.path.join(proj, "fontmatch"); os.makedirs(d, exist_ok=True)
            crop, _ = crop_of(proj, t, pad=12)
            cj = os.path.join(d, f"crop_{t['id']}.jpg"); cv2.imwrite(cj, crop)
            names = {r_["font"] for r_ in rows}
            for n_, w_ in CURATED:
                if n_ in names or len(rows) >= 20:
                    continue
                try:
                    from . import render as _r
                    _r._font(n_, 40, w_)
                    rows.append({"font": n_, "weight": w_, "wdth": None, "score": None, "diff": ""})
                    names.add(n_)
                except Exception:
                    pass
            sj = sheet(rows, t.get("he") or SAMPLE, os.path.join(d, f"fontmatch_{t['id']}.jpg"))
            r = ask_claude(api_key, model, cj, sj, len(rows), hint)
            order = [int(x) for x in (r.get("ranking") or []) if str(x).isdigit() and 1 <= int(x) <= len(rows)]
            b = int(r.get("best", 0) or 0)
            if 1 <= b <= len(rows):
                order = [b] + [x for x in order if x != b]
            order = list(dict.fromkeys(order)) + [i for i in range(1, len(rows) + 1) if i not in order]
            out["rows"] = [rows[i - 1] for i in order]
            out.update(latin_font=r.get("latin_font", "") or "", reason_he=r.get("reason_he", "") or "",
                       settings_he=r.get("settings_he", "") or "", by_claude=True, look=r.get("look") or None)
        except Exception as e:
            log(f"Claude לא זמין לבחירת הגופן, נשאר הדירוג המקומי ({e.__class__.__name__})")
    return out


def _bgr(hx):
    hx = str(hx or "").lstrip("#")
    if len(hx) != 6:
        return None
    r, g, b = (int(hx[i:i + 2], 16) for i in (0, 2, 4))
    return [b, g, r]


def apply_look(t, look):
    """Claude's reading of the original's colouring and depth -> the title's style (locked parts stay)."""
    if not isinstance(look, dict):
        return
    st = t.setdefault("style", {}); lock = set(t.get("style_locked", []))
    lh = float(st.get("letter_h") or (t["ref_box"][3] - t["ref_box"][1]) * 0.75)
    fl = [c for c in (_bgr(x) for x in (look.get("fill") or [])) if c]
    if fl and "fill" not in lock:
        st["fill"] = fl[0]
        if len(fl) > 1 and fl[0] != fl[-1]:
            st["gradient"] = [fl[0], fl[-1]]
        else:
            st.pop("gradient", None)
    ex = look.get("extrude")
    if "extrude" not in lock:
        if isinstance(ex, dict) and _bgr(ex.get("color")) and float(ex.get("depth") or 0) > 0:
            dxy = {"down": (0, 1), "down-right": (0.7, 0.7), "down-left": (-0.7, 0.7), "right": (1, 0), "left": (-1, 0)}.get(ex.get("dir"), (0, 1))
            steps = int(max(1, round(float(ex["depth"]) * lh / max(abs(dxy[0]), abs(dxy[1])))))
            st["extrude"] = {"dx": dxy[0], "dy": dxy[1], "steps": min(steps, 60), "color": _bgr(ex["color"])}
        else:
            st.pop("extrude", None)
    ol = look.get("outline")
    if "outline" not in lock:
        st["outline"] = {"width": max(1, round(float(ol.get("width") or 0) * lh)), "color": _bgr(ol.get("color")) or [0, 0, 0]} \
            if isinstance(ol, dict) and float(ol.get("width") or 0) > 0.005 else None
    sh = look.get("shadow")
    if "shadow" not in lock:
        st["shadow"] = {"dx": round(float(sh.get("dx") or 0) * lh, 1), "dy": round(float(sh.get("dy") or 0) * lh, 1),
                        "blur": round(float(sh.get("blur") or 0) * lh, 1), "color": _bgr(sh.get("color")) or [0, 0, 0],
                        "opacity": float(sh.get("opacity") or 0.6)} if isinstance(sh, dict) else None
    w = look.get("weight")
    if w and "font" not in lock:
        try:
            st["weight"] = int(w)
        except (TypeError, ValueError):
            pass


def _fill_holes(m):
    h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    inv = (1 - m).astype(np.uint8).copy()
    cv2.floodFill(inv, ff, (0, 0), 0)                  # what is left of the inverse are the enclosed holes
    return (m | inv).astype(np.uint8)


def body(proj, t, pad=6):
    """(letter body mask, crop). The fill-colour mask is the whole letter for one-colour titles. A two-tone
    letter (orange at the top fading to white, halftone dots - a cartoon's name cards) matches its fill colour
    only in part: there the white inner line encloses the rest of the letter, so the enclosed holes are filled
    in. Without this the checks read such a letter as plain white and dropped its real gradient and outline."""
    m = letters(proj, t, pad)
    c, _ = crop_of(proj, t, pad)
    if m is None or c is None or c.size == 0:
        return None, None
    m = (m > 0).astype(np.uint8)
    ys = np.nonzero(m)[0]
    if ys.size < 60:
        return m, c
    y0, y1 = ys.min(), ys.max(); hh = max(1, y1 - y0)
    top = m[y0:y0 + hh // 4 + 1].mean(); bot = m[y1 - hh // 4:y1 + 1].mean()
    f = _fill_holes(m)
    ratio = f.sum() / max(1, m.sum())
    # sparse at the top and solid at the bottom, or big enclosed parts (more than letter counters):
    # a two-tone letter. Over 3.5x it is rather a frame around background.
    if ((bot > 0 and top / bot < 0.5) or ratio > 1.35) and ratio < 3.5:
        return f, c
    return m, c


def measure_outline(proj, t):
    """The original's outline around the letter body: {'width', 'color'} or None. Bands of growing distance
    from the letters; the outline is the run of bands of one colour that differs from the letters' edge."""
    b, c = body(proj, t, pad=40)
    if b is None or b.sum() < 200:
        return None
    lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    prev = b.copy(); cols = []
    edge = lab[(b > 0) & (cv2.erode(b, k, iterations=2) == 0)]
    if edge.size == 0:
        return None
    e = np.median(edge, 0)
    for w in range(1, 31):
        cur = cv2.dilate(prev, k)
        band = (cur > 0) & (prev == 0)
        prev = cur
        if band.sum() < 30:
            break
        px = lab[band]
        med = np.median(px, 0)
        same = float((np.linalg.norm(px - med, axis=1) < 22).mean())
        cols.append((med, same))
    if len(cols) < 3:
        return None
    ref, rs = cols[1]                                  # band 1 is the anti-aliased edge: judge by band 2
    if rs < 0.55 or np.linalg.norm(ref - e) < 25:
        return None                                    # no clear ring of its own colour around the letters
    width = 1
    for med, same in cols[1:]:
        if np.linalg.norm(med - ref) > 22 or same < 0.45:
            break
        width += 1
    if width < 2:
        return None
    bgr = cv2.cvtColor(np.uint8([[ref]]), cv2.COLOR_LAB2BGR)[0, 0]
    return {"width": int(width), "color": [int(v) for v in bgr]}


def measured_colours(proj, t):
    """The letters' own colour in the original: median of the top, middle and bottom quarter of the letters
    (eroded, so no edge or outline pixel counts). None when the letters cannot be found."""
    m, c = body(proj, t)
    if m is None or c is None or c.size == 0:
        return None
    er = cv2.erode((m > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
    ys = np.nonzero(er)[0]
    if ys.size < 60:
        return None
    y0, y1 = ys.min(), ys.max(); h = max(1, y1 - y0)
    out = []
    for a, b in ((0, 0.25), (0.4, 0.6), (0.75, 1.0)):
        sel = np.zeros_like(er); sel[y0 + int(a * h):y0 + int(b * h) + 1] = 1
        px = c[(er > 0) & (sel > 0)]
        if len(px) < 15:
            return None
        out.append([int(v) for v in np.median(px, 0)])
    return out


def _lab_d(a, b):
    la, lb = [cv2.cvtColor(np.uint8([[x]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0] for x in (a, b)]
    return float(np.linalg.norm(la - lb))


def fix_colours(proj, t):
    """Claude's fill and gradient are a reading of a small picture; the original's pixels decide. A gradient
    is kept only if the original's top and bottom really differ."""
    st = t.get("style") or {}
    if "fill" in set(t.get("style_locked", [])) or t.get("layer", {}).get("type") not in (None, "text"):
        return False
    mc = measured_colours(proj, t)
    if not mc:
        return False
    top, mid, bot = mc
    gr = st.get("gradient")
    want_gr = [top, bot] if _lab_d(top, bot) > 8 else None
    now = gr or [st.get("fill") or mid, st.get("fill") or mid]
    if _lab_d(now[0], top) > 10 or _lab_d(now[-1], bot) > 10 or (gr and not want_gr):
        st["fill"] = mid
        if want_gr:
            st["gradient"] = want_gr
        else:
            st.pop("gradient", None)
        return True
    return False


def _is_logo(t, name):
    """The film's or show's own name (all its words are in the video's name, e.g. ICE AGE in 'Ice Age- Boiling
    Point - Official Trailer'): a logo keeps its own lettering, it is not one of the taglines."""
    if t.get("kind") == "logo" or t.get("logo_full"):
        return True
    w = re.findall(r"[a-z0-9]+", (t.get("src") or "").lower())
    return bool(w) and len("".join(w)) >= 3 and all(x in name for x in w)


def unify_fonts(T, proj=None):
    """One look per video: the taglines of one kind (designed/styled, or plain) get one Hebrew font. Claude
    designs card by card and could pick a round font for one tagline and a block font for the next while the
    original uses one font for all of them. The winner is the used font closest to the originals' letters
    (stroke weight, corners, width...), else the one most cards use. Logos keep their own.
    Returns the ids of designed cards whose picture must be drawn again."""
    def font_of(t):
        d = t.get("design") if isinstance(t.get("design"), dict) else None
        return (d or {}).get("font") or (t.get("style") or {}).get("font")
    name = ""
    if proj:
        pj = vio.load_json(os.path.join(proj, "project.json"), {})
        base = os.path.splitext(os.path.basename(pj.get("video") or proj))[0].lower()
        # only the title part: "Ice Age- Boiling Point - Official Trailer - In Theaters February 5" -> ice age boiling point
        kw = r"\b(official|trailer|teaser|clip|promo|full episode|episode|season|s\d+e\d+|in theaters|\d{4})\b"
        base = re.split(kw, base)[0] or re.sub(kw, " ", base)
        name = " ".join(re.findall(r"[a-z0-9]+", base))
    pool = [t for t in T["titles"] if t.get("translate", True) and t.get("kind") != "sub"
            and t.get("mode") != "remove" and font_of(t) and not _is_logo(t, name)
            and "font" not in set(t.get("style_locked", [])) and not t.get("font_user")]
    groups = {}
    for t in pool:
        groups.setdefault(bool(t.get("styled") or t.get("design")), []).append(t)
    redo = []
    for ls in groups.values():
        votes = {}
        for t in ls:
            votes.setdefault(font_of(t), []).append(t)
        if len(ls) < 2 or len(votes) == 1:
            continue
        win = None
        if proj:
            try:
                srcs = []
                for t in ls:
                    m = letters(proj, t)
                    tr = traits(m) if m is not None else None
                    if tr:
                        srcs.append(_src_adj(tr, "latin"))
                cands = candidate_traits(names=set(votes))
                if srcs and cands:
                    best = {}
                    for n, w, d, tr in cands:
                        sc = float(np.mean([_dist(sv, tr, "latin") for sv in srcs]))
                        if n not in best or sc < best[n]:
                            best[n] = sc
                    if best:
                        win = min(best, key=best.get)
            except Exception:
                win = None
        if win is None:
            win = max(votes, key=lambda f: (len(votes[f]), max(float((t.get("style") or {}).get("letter_h") or 0) for t in votes[f])))
        donor = votes[win][0]
        dd = donor.get("design") if isinstance(donor.get("design"), dict) else {}
        for t in ls:
            if font_of(t) == win:
                continue
            st = t.setdefault("style", {})
            st["font"] = win
            for k in ("weight", "wdth"):
                if (donor.get("style") or {}).get(k) is not None:
                    st[k] = donor["style"][k]
            if isinstance(t.get("design"), dict):
                t["design"]["font"] = win
                for k in ("weight", "wdth"):
                    if dd.get(k) is not None:
                        t["design"][k] = dd[k]
                if (t.get("layer") or {}).get("design"):
                    redo.append(t["id"])
            t["font_unified"] = win
    return redo


def check_looks(proj, T, log=print):
    """Once per title (look_checked): projects whose looks were picked before this check existed. Titles
    with the same picked look are checked together on the biggest one, so a style stays one style."""
    sets = {}
    nc = 0
    for t in T["titles"]:
        if t.get("translate", True) and t.get("kind") != "sub" and not t.get("look_v2"):
            # the first checks read two-tone letters as plain (fill-colour pixels only): check again
            t.pop("colour_checked", None); t.pop("look_checked", None)
            st = t.get("style") or {}
            if not st.get("outline") and "outline" not in set(t.get("style_locked", [])):
                try:
                    o = measure_outline(proj, t)
                    if o:
                        st["outline"] = o; nc += 1
                except Exception:
                    pass
            t["look_v2"] = True
    # a title the font pass never saw (Claude made it a title later, or split/merged it) would fall back to
    # the default font next to cards in the show's chosen font: give it the font of the nearest-looking card
    have = [t for t in T["titles"] if t.get("translate", True) and t.get("kind") != "sub" and (t.get("style") or {}).get("font")]
    for t in T["titles"]:
        st = t.get("style") or {}
        if (not t.get("translate", True) or t.get("kind") == "sub" or st.get("font") or not have
                or "font" in set(t.get("style_locked", [])) or (t.get("layer") or {}).get("type") not in (None, "text")):
            continue
        def dist(o):
            a, b = (t.get("mask") or {}).get("fill_lab"), (o.get("mask") or {}).get("fill_lab")
            d = float(np.linalg.norm(np.array(a) - np.array(b))) if a and b else 60.0
            return d + 0.3 * abs(float(st.get("letter_h") or 0) - float((o.get("style") or {}).get("letter_h") or 0))
        o = min(have, key=dist)
        for k in ("font", "weight", "wdth"):
            if (o.get("style") or {}).get(k):
                st[k] = o["style"][k]
        t["style"] = st; t["font_from"] = o["id"]; nc += 1
    for t in T["titles"]:
        if t.get("translate", True) and t.get("kind") != "sub" and not t.get("colour_checked"):
            try:
                nc += bool(fix_colours(proj, t))
            except Exception:
                pass
            t["colour_checked"] = True
    for t in T["titles"]:
        st = t.get("style") or {}
        if not t.get("translate", True) or t.get("look_checked") or t.get("kind") == "sub":
            continue
        if not (st.get("extrude") or st.get("outline") or st.get("shadow")):
            t["look_checked"] = True
            continue
        key = json.dumps([st.get("extrude"), st.get("outline"), (st.get("shadow") or {}).get("color")], sort_keys=True)
        sets.setdefault(key, []).append(t)
    n = 0
    for ls in sets.values():
        big = max(ls, key=lambda x: (x["ref_box"][3] - x["ref_box"][1]) * (x["ref_box"][2] - x["ref_box"][0]))
        dr = verify_look(proj, big)
        for t in ls:
            lock = set(t.get("style_locked", []))
            for k in dr:
                if k in lock:
                    continue
                if k == "extrude":
                    t["style"].pop("extrude", None)
                else:
                    t["style"][k] = None
            t["look_checked"] = True
        n += len(ls) if dr else 0
    if n:
        log(f"הוסרו צד תלת-ממדי, קו מתאר או צל שאין במקור ({n} כותרות)")
    if nc:
        log(f"הצבעים של {nc} כותרות הותאמו לצבע שנמדד במקור")
    return n + nc


def verify_look(proj, t):
    """Claude may imagine a 3D side, an outline or a shadow the original does not have (a flat orange tagline
    got a light grey extrusion). Each is kept only if the original shows it where it would be: the side's
    colour next to the letters in its direction, the outline's colour around them, a shadow darker than the
    opposite side. Returns what was dropped."""
    st = t.get("style") or {}
    lock = set(t.get("style_locked", []))
    m, crop = body(proj, t, pad=40)                   # room around the letters for outline, side and shadow
    if m is None or crop is None or crop.size == 0:
        return []
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = (m > 0).astype(np.uint8)
    h, w = m.shape
    near = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)

    def shifted(dx, dy):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST)

    def share(color, zone):
        if zone.sum() < 20:
            return None
        c = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        d = np.linalg.norm(lab[zone > 0] - c, axis=1)
        return float((d < 28).mean())
    dropped = []
    ex = st.get("extrude")
    if ex and "extrude" not in lock:
        L = max(2.0, ex.get("steps", 1) * max(abs(ex.get("dx", 0)), abs(ex.get("dy", 0))))
        side = shifted(ex.get("dx", 0) * L, ex.get("dy", 0) * L) & (1 - near)
        sh = share(ex.get("color") or [0, 0, 0], side)
        if sh is not None and sh < 0.15:
            st.pop("extrude", None); dropped.append("extrude")
    ol = st.get("outline")
    if ol and "outline" not in lock:
        ring = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=max(1, int(ol.get("width") or 1) + 1)) & (1 - m)
        sh = share(ol.get("color") or [0, 0, 0], ring)
        if sh is not None and sh < 0.15:
            real = measure_outline(proj, t)
            if real:
                st["outline"] = real                  # there is an outline, in another colour/width
            else:
                st["outline"] = None; dropped.append("outline")
    sd = st.get("shadow")
    if sd and "shadow" not in lock and (abs(sd.get("dx", 0)) + abs(sd.get("dy", 0))) >= 1.5:
        a = shifted(sd["dx"], sd["dy"]) & (1 - near)
        b = shifted(-sd["dx"], -sd["dy"]) & (1 - near)
        if a.sum() > 20 and b.sum() > 20:
            la, lb = float(lab[..., 0][a > 0].mean()), float(lab[..., 0][b > 0].mean())
            if la > lb - 6:                         # not darker on the shadow's side than on the other
                st["shadow"] = None; dropped.append("shadow")
            else:
                # its colour from the original (beyond the outline), not a guess
                ow = int(((st.get("outline") or {}).get("width") or 0)) + 2
                inner = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=ow)
                zone = cv2.warpAffine(inner, np.float32([[1, 0, sd["dx"]], [0, 1, sd["dy"]]]), (w, h),
                                      flags=cv2.INTER_NEAREST) & (1 - inner)
                if zone.sum() > 40:
                    px = crop[zone > 0]
                    med = np.median(px, 0)
                    if float(np.mean(np.linalg.norm(px - med, axis=1) < 40)) > 0.5:
                        sd["color"] = [int(v) for v in med]
    t["look_checked"] = True
    return dropped


def apply_row(t, row):
    st = t.setdefault("style", {})
    st["font"] = row["font"]
    for k in ("weight", "wdth"):
        if row.get(k):
            st[k] = row[k]
        else:
            st.pop(k, None)


def _clusters(todo):
    """Cards in one style (same letter colour, styled or not) - a trailer's taglines - share one font and look."""
    cards = {}
    for t in todo:
        fl = (t.get("mask") or {}).get("fill_lab")
        key = None
        for k, (rep, ls) in cards.items():
            rf = (rep.get("mask") or {}).get("fill_lab")
            if fl and rf and float(np.linalg.norm(np.array(fl) - np.array(rf))) < 22 and \
                    bool(t.get("styled") or t.get("design")) == bool(rep.get("styled") or rep.get("design")):
                key = k
                break
        if key is None:
            cards[t.get("group", t["id"])] = (t, [t])
        else:
            cards[key][1].append(t)
    return {k: v[1] for k, v in cards.items()}


def run(proj, api_key="", model="claude-sonnet-5-5", progress=None, log=print, only=None):
    """Automatic pick at analysis time, for titles whose font nobody chose (not locked, not from the series).
    With an API key Claude picks the font and reads the look (colours, 3D side, outline, shadow) per card;
    without one, the local ranking."""
    T = vio.load_json(os.path.join(proj, "titles.json"))
    todo = [t for t in T["titles"] if t.get("translate", True) and t.get("mode") != "remove" and t.get("kind") != "sub"
            and "font" not in t.get("style_locked", []) and not t.get("from_series")
            and (only is None or t["id"] in only)]
    if not todo:
        return 0
    cands = candidate_traits(progress=lambda x: progress(0.8 * x) if progress else None)
    done = 0
    if api_key:
        # Claude as the typographer: one look per card (the lines of a card share a typeface and look)
        # cards in one style (same letter colour, close shapes) - a trailer's taglines - share one look:
        # one Claude call per style, not per card
        cards = _clusters(todo)
        for i, (gid, ls) in enumerate(cards.items()):
            if progress:
                progress(0.8 + 0.2 * i / max(1, len(cards)))
            big = max(ls, key=lambda x: (x["ref_box"][3] - x["ref_box"][1]))
            try:
                r = match(proj, big, top=8, api_key=api_key, model=model, log=log)
            except Exception as e:
                log(f"בחירת הגופן עם Claude לא הצליחה ({e.__class__.__name__})")
                r = {}
            if not r.get("rows") or not r.get("by_claude"):
                continue
            for t in ls:
                apply_row(t, r["rows"][0])
                apply_look(t, r.get("look"))
                t["font_top"] = [[x["font"], x["weight"], x["wdth"], x.get("diff", "")] for x in r["rows"][:5]]
                t["font_auto"] = True; t["font_by_claude"] = True
                done += 1
            for t in ls:                      # the colours as measured in the original, per line
                try:
                    fix_colours(proj, t)
                except Exception:
                    pass
                t["colour_checked"] = True
            # only what the original really shows: checked on the style's biggest line, dropped for all
            try:
                dr = verify_look(proj, big)
                for t in ls:
                    for k in dr:
                        if k in set(t.get("style_locked", [])):
                            continue
                        if k == "extrude":
                            t["style"].pop("extrude", None)
                        else:
                            t["style"][k] = None
                    t["look_checked"] = True
                if dr:
                    log(f"הוסר מהעיצוב מה שאין במקור: {', '.join({'extrude': 'צד תלת-ממדי', 'outline': 'קו מתאר', 'shadow': 'צל'}[k] for k in dr)}")
            except Exception as e:
                log(f"בדיקת העיצוב מול המקור לא הצליחה ({e.__class__.__name__})")
        todo = [t for t in todo if not t.get("font_by_claude")]
    # without Claude: one font per style (the biggest line of the style decides), never a different font per line
    cl = list(_clusters(todo).values())
    for i, ls in enumerate(cl):
        if progress:
            progress(0.8 + 0.2 * i / len(cl))
        rows = []
        for t in sorted(ls, key=lambda x: -(x["ref_box"][3] - x["ref_box"][1])):
            m = letters(proj, t)
            src = traits(m) if m is not None else None
            if src is not None:
                rows = rank(src, cands, script_of(t.get("src") or ""), 8)
                if rows:
                    break
        if not rows:
            continue
        for t in ls:
            apply_row(t, rows[0])
            t["font_top"] = [[r["font"], r["weight"], r["wdth"], r["diff"]] for r in rows[:5]]
            t["font_auto"] = True
            done += 1
    vio.save_json(os.path.join(proj, "titles.json"), T)
    return done
