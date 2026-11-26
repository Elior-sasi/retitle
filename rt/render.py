"""Pass 5: measure the source text animation per frame (position / scale / opacity), and render the
Hebrew layer with the source style, driven by that exact per-frame track."""
import os, json
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, features

RAQM = features.check("raqm")


def _vis(text):
    """Visual order for the BASIC layout engine (Windows without FriBiDi). Hebrew needs no shaping."""
    if RAQM:
        return text
    from bidi.algorithm import get_display
    return get_display(text, base_dir="R")


def _kw(**k):
    return k if RAQM else {}
from . import vio
from .extract import groups

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.environ.get("RETITLE_FONTS", os.path.join(os.path.dirname(HERE), "fonts"))


# ---------------------------------------------------------------- tracking
def _moments(D):
    m = D.sum()
    if m <= 1e-6:
        return None
    ys, xs = np.mgrid[0:D.shape[0], 0:D.shape[1]]
    cx = (D * xs).sum() / m; cy = (D * ys).sum() / m
    sx = np.sqrt((D * (xs - cx) ** 2).sum() / m); sy = np.sqrt((D * (ys - cy) ** 2).sum() / m)
    return m, cx, cy, sx, sy


def _match(Dt, Dr, sc0):
    """Find (scale, dx, dy, score) placing reference text map Dr inside Dt."""
    best = None
    for sc in sc0 * np.linspace(0.9, 1.1, 11):
        tw, th = int(Dr.shape[1] * sc), int(Dr.shape[0] * sc)
        if tw < 8 or th < 8 or tw >= Dt.shape[1] or th >= Dt.shape[0]:
            continue
        tp = cv2.resize(Dr, (tw, th), interpolation=cv2.INTER_AREA)
        # zero-mean correlation: plain correlation of all-positive maps favours smaller and smaller templates
        r = cv2.matchTemplate(Dt, tp, cv2.TM_CCOEFF_NORMED)
        r = np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)      # a flat template has no zero-mean match
        _, v, _, loc = cv2.minMaxLoc(r)
        if best is None or v > best[3]:
            best = (sc, loc[0] + tw / 2, loc[1] + th / 2, v)
    return best


def _sharp(D):
    """Edge energy per unit of ink of a text map (falls as the text blurs)."""
    m = D > 0.05
    if m.sum() < 30:
        return 0.0
    lap = np.abs(cv2.Laplacian(D.astype(np.float32), cv2.CV_32F))
    return float(lap[m].mean() / max(1e-3, D[m].mean()))


def _blur_of(Sn, Rn, sc, sharp_ref):
    """Gaussian sigma (frame px) that makes the reference text as soft as it is in this frame."""
    s_now = _sharp(Sn)
    if not sharp_ref or not s_now or s_now >= 0.8 * sharp_ref:
        return 0.0
    R = cv2.resize(Rn, (max(8, int(Rn.shape[1] * sc)), max(8, int(Rn.shape[0] * sc))), interpolation=cv2.INTER_AREA) if abs(sc - 1) > 0.02 else Rn
    best = 0.0
    for sg in (1, 1.5, 2, 3, 4, 6, 8, 11, 15, 20):
        if _sharp(cv2.GaussianBlur(R, (0, 0), sg)) <= s_now:
            best = sg
            break
    return float(best or 20)


def _fix_jumps(fr, h):
    """A frame whose position or size jumps far from both neighbours (the match locked on to something
    else, often in the blur of a fast zoom) takes the motion its neighbours describe."""
    ks = sorted(fr)
    if len(ks) < 3:
        return
    big = lambda a, b: abs(a[0] - b[0]) > max(25, 0.6 * h) or abs(a[1] - b[1]) > max(25, 0.6 * h) or abs(a[2] - b[2]) > 0.25
    # inside: both neighbours agree with each other but not with this frame
    for i in range(1, len(ks) - 1):
        a, b, c = fr[ks[i - 1]], fr[ks[i]], fr[ks[i + 1]]
        if big(a, b) and big(b, c) and not big(a, c):
            fr[ks[i]] = [round((a[j] + c[j]) / 2, 4) for j in range(3)] + [b[3]]
    # the ends: carry the trend of the next two frames
    for i, n1, n2 in ((0, 1, 2), (len(ks) - 1, len(ks) - 2, len(ks) - 3)):
        a, b, c = fr[ks[i]], fr[ks[n1]], fr[ks[n2]]
        if big(a, b) and not big(b, c):
            fr[ks[i]] = [round(2 * b[j] - c[j], 4) for j in range(3)] + [a[3]]


def _steady_ref(proj, gid, lines, t):
    """The reference frame must show the title at its steady size. A logo detected on a frame of its fly-in or
    zoom-through (bigger, cut off, blurred) made every other frame look 'smaller' and the Hebrew was drawn huge.
    With SAM 2's exact masks of the card: if the reference frame's mask is far from the card's usual size, move
    the reference to a steady frame and carry the boxes over (same scale and shift as the mask's box)."""
    d = os.path.join(proj, "sam", gid)
    if not os.path.isdir(d) or t.get("ref_fixed"):
        return
    box = {}
    for f in range(t["in"], t["out"] + 1):
        m = cv2.imread(os.path.join(d, f"{f:06d}.png"), 0)
        if m is None:
            continue
        ys, xs = np.where(m > 127)
        if len(xs) > 50:
            ox, oy = t["region"][0], t["region"][1]          # the masks are crops of the card's region
            box[f] = (float(xs.min() + ox), float(ys.min() + oy), float(xs.max() + ox), float(ys.max() + oy), float(len(xs)))
    if len(box) < 8 or t["ref"] not in box:
        return
    areas = np.array([v[4] for v in box.values()])
    med = float(np.median(areas))
    if abs(box[t["ref"]][4] / med - 1) < 0.35:
        return
    steady = [f for f, v in box.items() if abs(v[4] / med - 1) < 0.15]
    if not steady:
        return
    mid = (t["in"] + t["out"]) / 2
    new = min(steady, key=lambda f: abs(f - mid))
    a, b = box[t["ref"]], box[new]
    k = float(np.sqrt(((b[2] - b[0]) * (b[3] - b[1])) / max(1.0, (a[2] - a[0]) * (a[3] - a[1]))))
    move = lambda r: [int(round(b[0] + (r[0] - a[0]) * k)), int(round(b[1] + (r[1] - a[1]) * k)),
                      int(round(b[0] + (r[2] - a[0]) * k)), int(round(b[1] + (r[3] - a[1]) * k))]
    for x in lines:                                    # the whole card shares one reference
        if x.get("ref") != t["ref"] or x.get("ref_fixed"):
            continue
        x["ref_box"] = move(x["ref_box"])
        lay = x.get("layer") or {}
        if lay.get("box"):
            lay["box"] = move(lay["box"])
        x["ref"] = new
        x["ref_fixed"] = True
        x.pop("anim", None) if not (x.get("anim") or {}).get("locked") else None
    print(f"reference frame of {gid} moved to {new} (steady size)")


def track(proj, only=None):
    """Per frame [dx, dy, scale, alpha] of the source text relative to its reference frame.
    Text map = |source - cleaned| clipped (shape, independent of background brightness);
    scale+position by multi-scale template match, alpha by p90 contrast ratio."""
    T = vio.load_json(os.path.join(proj, "titles.json"))
    for gid, lines in groups(T).items():
        if only is not None and gid not in only:
            continue
        for t in lines:
            if not t.get("translate", True) or "mask" not in t:
                continue
            if t.get("anim", {}).get("locked"):
                continue
            _steady_ref(proj, gid, lines, t)
            reg = t["region"]; bx = t["ref_box"]; h = bx[3] - bx[1]
            pad = int(h * 0.4)
            rx0, ry0 = max(0, bx[0] - reg[0] - pad), max(0, bx[1] - reg[1] - pad)
            rx1, ry1 = bx[2] - reg[0] + pad, bx[3] - reg[1] + pad
            s = t.get("search", int(h * 1.5))
            sx0, sy0 = max(0, rx0 - s), max(0, ry0 - s)
            sx1, sy1 = rx1 + s, ry1 + s

            def tmap(f):
                o = cv2.imread(os.path.join(proj, "cache", gid, f"{f:06d}.png"))
                c = cv2.imread(os.path.join(proj, "clean", gid, f"{f:06d}.png"))
                if o is None or c is None:
                    return None
                D = np.abs(o.astype(np.int16) - c.astype(np.int16)).max(2).astype(np.float32)
                D[D < 10] = 0
                for o2 in lines:  # hide sibling lines of the same card
                    if o2 is t or not o2.get("translate", True):
                        continue
                    b2 = o2["ref_box"]; p2 = int((b2[3] - b2[1]) * 0.3)
                    X0, Y0, X1, Y1 = b2[0] - reg[0] - p2, b2[1] - reg[1] - p2, b2[2] - reg[0] + p2, b2[3] - reg[1] + p2
                    if not (X1 < rx0 + pad or X0 > rx1 - pad or Y1 < ry0 + pad or Y0 > ry1 - pad):
                        continue
                    D[max(0, Y0):max(0, Y1), max(0, X0):max(0, X1)] = 0
                return D
            Dref = tmap(t["ref"])
            if Dref is None:
                continue
            R = Dref[ry0:ry1, rx0:rx1]
            p_ref = float(np.percentile(R[R > 0], 90)) if (R > 0).any() else 1.0
            Rn = np.minimum(R, p_ref * 0.5) / (p_ref * 0.5)
            rcx = (rx0 + rx1) / 2; rcy = (ry0 + ry1) / 2
            fr = {}
            blur = {}
            Rraw = R / max(1e-3, p_ref)                 # not clipped: clipping would hide the blur
            sharp_ref = _sharp(Rraw)
            prev_sc = 1.0
            for f in range(t["in"], t["out"] + 1):
                D = tmap(f)
                if D is None:
                    continue
                S = D[sy0:sy1, sx0:sx1]
                if not (S > 0).any():
                    fr[f] = [0, 0, 1, 0]; continue
                p = float(np.percentile(S[S > 0], 90))
                al = float(np.clip(p / p_ref, 0, 1))
                Sn = np.minimum(S, p * 0.5) / max(1e-3, p * 0.5)
                # coarse scale from mass, refined by matching
                m_ref = (Rn > 0.5).sum(); m = (Sn > 0.5).sum()
                sc0 = float(np.clip(np.sqrt(m / max(1, m_ref)), 0.1, 5)) if al > 0.3 else prev_sc
                # the mass guess counts everything in the search window (a bigger line next to this one
                # too), so the steady guesses - the last frame's scale and no change - also compete
                if f == t["ref"]:
                    fr[f] = [0, 0, 1.0, 1.0]                  # the reference frame is the reference, by definition
                    prev_sc = 1.0
                    continue
                # the steady guesses first; the mass guess only when it is clearly better (a bigger line in the
                # window, e.g. the rest of a logo, inflates it - it had won even on the reference frame itself,
                # and the whole logo was then drawn zoomed and shifted)
                bm = None
                for c in (_match(Sn, Rn, prev_sc), _match(Sn, Rn, 1.0)):
                    if c and (bm is None or c[3] > bm[3]):
                        bm = c
                c = _match(Sn, Rn, sc0)
                if c and (bm is None or c[3] > bm[3] + 0.08):
                    bm = c
                if bm is None:
                    fr[f] = [0, 0, 1, round(al, 3)]; continue
                sc, cx, cy, v = bm
                dx = cx + sx0 - rcx; dy = cy + sy0 - rcy
                if abs(dx) < 1.5 and abs(dy) < 1.5 and abs(sc - 1) < 0.025:
                    dx = dy = 0; sc = 1.0
                if al > 0.85:
                    al = 1.0
                prev_sc = sc
                fr[f] = [round(dx, 2), round(dy, 2), round(float(sc), 4), round(al, 3)]
                # motion / focus blur of the original (fast zoom, pop-in): the Hebrew gets the same blur
                if al > 0.3:
                    sg = _blur_of(S / max(1e-3, p), Rraw, sc, sharp_ref)
                    if sg > 1.0:
                        blur[f] = round(sg, 1)
            _fix_jumps(fr, h)
            vals = np.array(list(fr.values()))
            static = (np.abs(vals[:, :2]).max() < 1.5 and np.abs(vals[:, 2] - 1).max() < 0.025
                      and vals[:, 3].min() > 0.9)
            t["anim"] = {"mode": "static", "frames": {str(k): v for k, v in fr.items()}} if static else \
                {"mode": "match", "frames": {str(k): v for k, v in fr.items()}}
            # rotation / perspective, reveal matte, light, layered logos
            from . import motion
            try:
                rects = [rc for _, rc in motion.parts_layout(proj, t)] if (t.get("layer") or {}).get("type") == "parts" else None
                motion.refine(proj, t, gid, lines, rects)
            except Exception as e:
                print("motion refine failed", t["id"], e)
            if blur and t["anim"].get("mode") == "match":
                t["anim"]["blur"] = {str(k): v for k, v in blur.items()}
            if t["anim"].get("mode") == "static":
                t["anim"].pop("frames", None)
        # a designed card's picture moves like the card's biggest line (the logo itself), not like the
        # line it happens to be stored on (often a small "THE")
        import copy
        for t in lines:
            if not (t.get("layer") or {}).get("design") or t.get("anim", {}).get("locked"):
                continue
            mem = [x for x in lines if (x is t or x.get("design_member") == t["id"]) and x.get("anim")]
            big = max(mem, key=lambda x: (x["ref_box"][2] - x["ref_box"][0]) * (x["ref_box"][3] - x["ref_box"][1]), default=None)
            if big is not None and big is not t:
                a = copy.deepcopy(big["anim"])
                for k in ("matte", "gain", "parts", "mref"):
                    a.pop(k, None)
                a["from"] = big["id"]
                # the big line's H takes ITS reference frame to each frame; the picture is laid out at THIS
                # card's reference frame - rebase, or the logo is drawn with the wrong zoom and offset
                # (a reference frame near the end of a zoom made the logo huge and cut off)
                from .motion import _nearest
                if a.get("H") and t.get("ref") != big.get("ref"):
                    b = _nearest(a["H"], t["ref"])
                    if b is not None:
                        Bi = np.linalg.inv(np.array(b, np.float64).reshape(3, 3))
                        a["H"] = {k: [round(float(x), 6) for x in (np.array(v, np.float64).reshape(3, 3) @ Bi).ravel()]
                                  for k, v in a["H"].items()}
                t["anim"] = a
        # how blurred the original is in each frame (a logo that flies in out of focus): measured on the
        # original itself, the letters' edge sharpness against the reference frame
        for t in lines:
            a = t.get("anim") or {}
            if a.get("H") and not a.get("blur") and not a.get("locked"):
                try:
                    bl = _measure_blur(proj, gid, t)
                    if bl:
                        a["blur"] = bl
                except Exception as e:
                    print("blur measure failed", t["id"], e)
    vio.save_json(os.path.join(proj, "titles.json"), T)


def _measure_blur(proj, gid, t):
    """{frame: sigma px} where the original's letters are softer than in the reference frame. Edge energy
    (99th percentile gradient / contrast) inside the title's box as the homography moves it; a ratio r of
    the reference's sharpness means a blur of about 4.5 * (1/r - 1) px at the frame's scale."""
    H = t["anim"]["H"]
    reg = t["region"]; x0, y0, x1, y1 = t.get("ref_box") or t["box"]

    def sharp(f):
        img = cv2.imread(os.path.join(proj, "cache", gid, f"{int(f):06d}.png"))
        M = np.array(H.get(str(f)) or [], np.float64)
        if img is None or M.size != 9:
            return None, None
        M = M.reshape(3, 3)
        pts = cv2.perspectiveTransform(np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]]), M).reshape(-1, 2)
        pts -= (reg[0], reg[1])
        h, w = img.shape[:2]
        a, b = np.clip(pts.min(0), 0, [w - 1, h - 1]).astype(int)
        c, d = np.clip(pts.max(0), 0, [w - 1, h - 1]).astype(int)
        if c - a < 20 or d - b < 10:
            return None, None
        g = cv2.cvtColor(img[b:d, a:c], cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        return float(np.percentile(m, 99)) / max(1.0, float(g.std())), float(np.sqrt(abs(np.linalg.det(M[:2, :2]))))
    r0, _ = sharp(t.get("ref", t["in"]))
    if not r0:
        return {}
    out = {}
    for f in sorted(H, key=int):
        r, s = sharp(f)
        if r is None:
            continue
        sg = 4.5 * (r0 / max(1e-3, r) - 1) * s
        if sg > 0.8:
            out[f] = round(min(12.0, sg), 1)
    return out


# ---------------------------------------------------------------- fonts
def font_path(name):
    if os.path.isabs(name) and os.path.exists(name):
        return name
    cat = vio.load_json(os.path.join(FONT_DIR, "catalog.json"), {})
    if name in cat:
        return os.path.join(FONT_DIR, cat[name]["file"])
    from . import fontlib                     # Google (downloaded on first use), Windows, Adobe Fonts, imported
    try:
        p = fontlib.ensure(name)
    except Exception:
        p = None
    if p:
        return p
    p = os.path.join(FONT_DIR, name)
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"font {name} - run: python -m rt fonts")


def _font(name, size, weight=None, wdth=None):
    f = ImageFont.truetype(font_path(name), int(round(size)), layout_engine=ImageFont.Layout.RAQM if RAQM else ImageFont.Layout.BASIC)
    try:
        axes = f.get_variation_axes()
    except Exception:
        axes = []
    if axes:
        vals = []
        for ax in axes:
            nm = ax.get("name"); nm = nm.decode() if isinstance(nm, bytes) else str(nm)
            if nm.lower().startswith("weight"):
                vals.append(min(ax.get("maximum", 1000), max(ax.get("minimum", 1), weight or ax.get("default", 400))))
            elif nm.lower().startswith("width") and wdth:
                vals.append(min(ax.get("maximum", 200), max(ax.get("minimum", 25), wdth)))
            else:
                vals.append(ax.get("default", ax.get("minimum")))
        try:
            f.set_variation_by_axes(vals)
        except Exception:
            pass
    return f


def auto_size(st):
    """Font px so that the Hebrew letter body matches the source letter height * size_ratio."""
    f = _font(st.get("font", "Heebo"), 200, st.get("weight"), st.get("wdth"))
    b = f.getbbox(_vis("המש"), **_kw(direction="rtl", language="he"))
    body = (b[3] - b[1]) / 200
    return st.get("letter_h", 60) * st.get("size_ratio", 0.9) / max(body, 0.1)


def _bgr2rgb(c):
    return (int(c[2]), int(c[1]), int(c[0]))


def text_layer(he, st, ss=2):
    """Render Hebrew text to an RGBA PIL image at ss x supersample. Returns (img, anchor_xy)
    where anchor is the centre of the letter body in the image."""
    size = st.get("size_px") or auto_size(st)
    font = _font(st.get("font", "Heebo"), size * ss, st.get("weight"), st.get("wdth"))
    lines = [_vis(l) for l in he.split("\n")]
    spacing = st.get("line_gap", 0.25) * size * ss
    ow = int(round((st.get("outline") or {}).get("width", 0) * ss))
    kw = _kw(direction="rtl", language="he")
    boxes = [font.getbbox(l, **kw, stroke_width=ow) for l in lines]
    lw = [b[2] - b[0] for b in boxes]
    asc, desc = font.getmetrics()
    lh = asc + desc
    pad = int(size * ss * 0.6) + ow * 2
    sh = st.get("shadow") or {}
    ex = st.get("extrude") or {}
    pad += int(max(abs(sh.get("dx", 0)), abs(sh.get("dy", 0)), abs(ex.get("dx", 0) * ex.get("steps", 0)),
                   abs(ex.get("dy", 0) * ex.get("steps", 0))) * ss) + int(sh.get("blur", 0) * ss * 3)
    W = int(max(lw) + 2 * pad); Hh = int(len(lines) * lh + (len(lines) - 1) * spacing + 2 * pad)
    align = st.get("text_align", "center")

    def draw_into(img, color, dx=0, dy=0, stroke=0, stroke_fill=None, fill_override=None):
        d = ImageDraw.Draw(img)
        for i, l in enumerate(lines):
            if align == "center":
                x = (W - lw[i]) / 2 - boxes[i][0]
            elif align == "right":
                x = W - pad - lw[i] - boxes[i][0]
            else:
                x = pad - boxes[i][0]
            y = pad + i * (lh + spacing)
            d.text((x + dx, y + dy), l, font=font, fill=fill_override or color, stroke_width=stroke,
                   stroke_fill=stroke_fill, **kw)

    base = Image.new("RGBA", (W, Hh), (0, 0, 0, 0))
    fill = st.get("fill", [255, 255, 255])
    # glyph alpha (with stroke) used for shadow/extrude
    galpha = Image.new("L", (W, Hh), 0)
    draw_into(galpha, 255, stroke=ow, stroke_fill=255)
    if sh:
        so = Image.new("RGBA", (W, Hh), _bgr2rgb(sh.get("color", [0, 0, 0])) + (0,))
        a = Image.new("L", (W, Hh), 0)
        a.paste(galpha, (int(sh.get("dx", 0) * ss), int(sh.get("dy", 0) * ss)))
        if sh.get("blur"):
            a = a.filter(ImageFilter.GaussianBlur(sh["blur"] * ss))
        a = a.point(lambda v: int(v * sh.get("opacity", 0.7)))
        so.putalpha(a)
        base = Image.alpha_composite(base, so)
    if ex:
        col = _bgr2rgb(ex.get("color", [0, 0, 0]))
        for k in range(int(ex.get("steps", 0)), 0, -1):
            e = Image.new("RGBA", (W, Hh), col + (0,))
            a = Image.new("L", (W, Hh), 0)
            a.paste(galpha, (int(round(ex.get("dx", 0) * k * ss)), int(round(ex.get("dy", 0) * k * ss))))
            e.putalpha(a)
            base = Image.alpha_composite(base, e)
    if ow:
        ol = Image.new("RGBA", (W, Hh), (0, 0, 0, 0))
        oc = _bgr2rgb(st["outline"]["color"])
        draw_into(ol, oc + (255,), stroke=ow, stroke_fill=oc + (255,))
        base = Image.alpha_composite(base, ol)
    fa = Image.new("L", (W, Hh), 0)
    draw_into(fa, 255)
    grad = st.get("gradient")  # [top_bgr, bottom_bgr]
    if grad:
        g = np.zeros((Hh, W, 3), np.float32)
        top, bot = np.array(_bgr2rgb(grad[0]), np.float32), np.array(_bgr2rgb(grad[1]), np.float32)
        for i in range(len(lines)):
            y0 = pad + i * (lh + spacing); y1 = y0 + lh
            ys = np.clip((np.arange(Hh) - y0) / max(1, y1 - y0), 0, 1)[:, None]
            sel = (np.arange(Hh) >= y0 - lh * 0.3) & (np.arange(Hh) <= y1 + lh * 0.3)
            g[sel] = (top * (1 - ys) + bot * ys)[sel][:, None, :].repeat(W, 1)
        fl = Image.fromarray(g.astype(np.uint8), "RGB").convert("RGBA")
    else:
        fl = Image.new("RGBA", (W, Hh), _bgr2rgb(fill) + (255,))
    fl.putalpha(fa)
    base = Image.alpha_composite(base, fl)
    # anchor: centre of the ink of the fill
    arr = np.array(fa)
    ys, xs = np.nonzero(arr > 64)
    if len(xs):
        anchor = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
        ink = (xs.min(), ys.min(), xs.max(), ys.max())
    else:
        anchor = (W / 2, Hh / 2); ink = (0, 0, W, Hh)
    return base, anchor, ink


def build_layers(proj, only=None):
    """Render each line's Hebrew layer once (supersampled) -> layers/<id>.png + meta."""
    T = vio.load_json(os.path.join(proj, "titles.json"))
    try:                                   # looks picked before the check against the original existed
        from . import fontmatch
        redo = fontmatch.unify_fonts(T, proj)
        if redo:
            vio.save_json(os.path.join(proj, "titles.json"), T)
            from . import design
            for tid in redo:              # designed cards: draw their picture again in the shared font
                try:
                    design.rerender(proj, tid)
                except Exception as e:
                    print("rerender", tid, e)
            only = None
        if fontmatch.check_looks(proj, T):
            vio.save_json(os.path.join(proj, "titles.json"), T)
            only = None                    # the changed titles need new layers too
        elif any(t.get("look_checked") for t in T["titles"]):
            vio.save_json(os.path.join(proj, "titles.json"), T)
    except Exception as e:
        print("check_looks", e)
    P = vio.load_json(os.path.join(proj, "project.json"))
    W = P["info"]["width"]
    d = os.path.join(proj, "layers"); os.makedirs(d, exist_ok=True)
    meta = vio.load_json(os.path.join(d, "meta.json"), {}) if only else {}
    for t in T["titles"]:
        if only is not None and t["id"] not in only:
            continue
        lay = t.get("layer") or {"type": "text"}
        if (not t.get("translate", True) or t.get("mode") == "remove"
                or (lay["type"] == "text" and not t.get("he"))):
            meta.pop(t["id"], None)
            continue
        ss = 2
        if lay["type"] == "text":
            st = t.get("style", {})
            img, anc, ink = text_layer(t["he"], st, ss)
            # keep inside safe width
            maxw = st.get("max_width", W * 0.92) * ss
            if ink[2] - ink[0] > maxw:
                st2 = dict(st); st2["size_px"] = (st.get("size_px") or auto_size(st)) * maxw / (ink[2] - ink[0])
                img, anc, ink = text_layer(t["he"], st2, ss)
                t["warn_shrunk"] = round(st2["size_px"], 1)
            else:
                t.pop("warn_shrunk", None)
        elif lay["type"] == "parts":
            from . import motion
            parts = []
            for i, (im, rc) in enumerate(motion.parts_layout(proj, t)):
                pp = os.path.join(d, f"{t['id']}#{i}.png"); im.save(pp)
                ssp = im.width / max(1e-6, rc[2] - rc[0])
                parts.append({"file": pp, "ss": ssp, "anchor": [im.width / 2, im.height / 2],
                              "target": [(rc[0] + rc[2]) / 2, (rc[1] + rc[3]) / 2]})
            meta[t["id"]] = {"parts": parts}
            continue
        elif lay["type"] == "image":
            # a designed logo (transparent PNG): fitted to the width of the source text block
            img = Image.open(os.path.join(proj, lay["path"]))
            ink = _ink(img)
            img = img.convert("RGBA")
            bb = img.getbbox() or (0, 0, img.width, img.height)
            img = img.crop(bb)
            ink = (ink[0] - bb[0], ink[1] - bb[1], ink[2] - bb[0], ink[3] - bb[1]) if ink else (0, 0, img.width, img.height)
            iw0, ih0 = max(1, ink[2] - ink[0]), max(1, ink[3] - ink[1])
            rb = lay.get("box") or t.get("ref_box", t["box"])      # a designed card covers the whole card
            tw = max(8.0, (rb[2] - rb[0]) * t.get("style", {}).get("size_ratio", 1.0))
            ss = iw0 / tw
            if lay.get("box"):
                # the Hebrew letters cover the original letters (what is left of them must hide behind
                # the Hebrew), within reason; underline, shadow and sparkles come on top of that
                th = max(8.0, (rb[3] - rb[1]) * t.get("style", {}).get("size_ratio", 1.0))
                sw, sh = tw / iw0, th / ih0
                ss = 1 / min(max(sw, sh), 1.2 * min(sw, sh))
            anc = ((ink[0] + ink[2]) / 2, (ink[1] + ink[3]) / 2)
        else:
            continue
        p = os.path.join(d, f"{t['id']}.png")
        img.save(p)
        # target anchor in frame coords: source box centre (or aligned edge)
        bx = t["box"]; st = t.get("style", {})
        rb = lay.get("box") or t.get("ref_box", bx)
        cy = (rb[1] + rb[3]) / 2 + st.get("nudge_y", 0)
        al = st.get("align", "center")
        iw = (ink[2] - ink[0]) / ss
        if al == "right":
            cx = rb[2] - iw / 2
        elif al == "left":
            cx = rb[0] + iw / 2
        else:
            cx = (rb[0] + rb[2]) / 2
        cx += st.get("nudge_x", 0)
        meta[t["id"]] = {"file": p, "ss": ss, "anchor": anc, "target": [cx, cy]}
    _no_overlap(T, meta, W, P["info"]["height"])
    vio.save_json(os.path.join(proj, "layers", "meta.json"), meta)
    vio.save_json(os.path.join(proj, "titles.json"), T)
    return meta


def _no_overlap(T, meta, W=None, H=None):
    """Lines of one card must not run into each other: Hebrew letters are often taller than the source's
    (final letters reach down, ל up), so lines set at the source's size can touch. Both lines are made
    a little smaller, around their own centres, until there is a small gap."""
    from .extract import groups
    for gid, lines in groups(T).items():
        boxes = []
        for t in lines:
            m = meta.get(t["id"])
            if not m or "file" not in m or not os.path.exists(m["file"]):
                continue
            try:
                im = Image.open(m["file"]); bb = im.getbbox()
            except Exception:
                continue
            if not bb:
                continue
            ss = m["ss"]; ax, ay = m["anchor"]; tx, ty = m["target"]
            # not much wider than the source line, and never past the frame's edges
            if not (t.get("layer") or {}).get("type") == "image":
                rb = t.get("ref_box", t["box"])
                iw = (bb[2] - bb[0]) / ss
                lim = 1.15 * (rb[2] - rb[0])
                if W:
                    room = 2 * min(tx - W * 0.03, W * 0.97 - tx)
                    lim = min(lim, room) if room > 0 else lim
                lim = max(lim, (rb[2] - rb[0]) * 0.8)
                if iw > lim:
                    ss = ss * iw / lim; m["ss"] = ss
            boxes.append([t["id"], tx + (bb[0] - ax) / ss, ty + (bb[1] - ay) / ss, tx + (bb[2] - ax) / ss, ty + (bb[3] - ay) / ss])
        boxes.sort(key=lambda b: b[2])
        for a, b in zip(boxes, boxes[1:]):
            ha, hb = a[4] - a[2], b[4] - b[2]
            gap = 0.2 * min(ha, hb)
            ov = a[4] - b[2] + gap
            if a[1] >= b[3] or b[1] >= a[3] or ov <= 0:     # side by side, or already apart
                continue
            k = max(0.7, 1 - ov / max(1.0, (ha + hb) / 2))
            for bb_ in (a, b):
                meta[bb_[0]]["ss"] = meta[bb_[0]]["ss"] / k
                cy = (bb_[2] + bb_[4]) / 2; cx = (bb_[1] + bb_[3]) / 2
                bb_[1] = cx - (cx - bb_[1]) * k; bb_[3] = cx + (bb_[3] - cx) * k
                bb_[2] = cy - (cy - bb_[2]) * k; bb_[4] = cy + (bb_[4] - cy) * k


def _ink(img):
    """The letters' box inside a designed logo PNG (written by design.save), or None."""
    try:
        v = [int(x) for x in str(img.info.get("ink", "")).split(",")]
        return tuple(v) if len(v) == 4 and v[2] > v[0] and v[3] > v[1] else None
    except ValueError:
        return None


def _similar(Hm, target):
    """The zoom and position of a homography without its stretch: uniform scale sqrt(det), the title's
    centre mapped where the homography maps it."""
    A = Hm[:2, :2]
    s = float(np.sqrt(abs(np.linalg.det(A))))
    tx, ty = target
    p = Hm @ np.array([tx, ty, 1.0])
    px, py = p[0] / p[2], p[1] / p[2]
    return np.array([[s, 0, px - s * tx], [0, s, py - s * ty], [0, 0, 1]], np.float64)


def place(layer_rgba, meta, anim, f, W, H, proj=None, tid=None, part=None, shift=(0, 0)):
    """Return premultiplied float RGBA full-frame for frame f, or None.
    Motion: full homography of the original when measured (rotation / perspective), else [dx, dy, scale];
    then the original's reveal matte and light/colour gain (motion.py)."""
    from . import motion
    dx = dy = 0.0; sc = 1.0; al = 1.0
    mode = anim.get("mode", "static")
    if mode == "match":
        fr = anim.get("frames") or {}
        v = fr.get(str(f))
        if v is None and fr:  # frames outside the measured range (timing extended by hand): hold the edge
            keys = sorted(int(k) for k in fr)
            k = keys[0] if f < keys[0] else keys[-1] if f > keys[-1] else min(keys, key=lambda x: abs(x - f))
            v = fr[str(k)]
            if f < keys[0] or f > keys[-1]:
                v = [v[0], v[1], v[2], 1.0]
        if v is not None:
            dx, dy, sc, al = v
    elif mode == "fade":
        a, b = anim["in"], anim["out"]
        fi, fo = anim.get("fade_in", 0), anim.get("fade_out", 0)
        if fi and f < a + fi:
            al = (f - a + 1) / (fi + 1)
        if fo and f > b - fo:
            al = min(al, (b - f + 1) / (fo + 1))
    ss = meta["ss"]; ax, ay = meta["anchor"]; tx, ty = meta["target"]
    Hm = motion.homography(anim, f, part) if mode == "match" else None
    if Hm is not None and part is None:
        # the full fit is trusted unless it stretches the title much more one way than the other
        # (in fast zoom blur it can wander off) - then the plain move and size are used
        sv = np.linalg.svd(np.asarray(Hm, np.float64)[:2, :2], compute_uv=False)
        if sv[1] <= 1e-6:
            Hm = None
        elif sv[0] / sv[1] > 1.35:
            # still keep its size and place: the same zoom and position without the stretch (a logo that
            # enters huge and blurred has a wandering shape but a sound scale - the plain match loses both)
            Hm = _similar(np.asarray(Hm, np.float64), meta["target"])
    if Hm is not None:
        A0 = np.array([[1 / ss, 0, tx - ax / ss], [0, 1 / ss, ty - ay / ss], [0, 0, 1]], np.float64)
        M = Hm @ A0
        k = np.sqrt(abs(np.linalg.det(M[:2, :2])))
        out = cv2.warpPerspective(layer_rgba, M, (W, H), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    else:
        k = sc / ss
        M = np.float32([[k, 0, tx + dx - ax * k], [0, k, ty + dy - ay * k]])
        out = cv2.warpAffine(layer_rgba, M, (W, H), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    bl = (anim.get("blur") or {}).get(str(f))
    if bl:
        a = out[..., 3]
        ys, xs = np.nonzero(a > 0.002)
        if len(xs):
            p = int(bl * 3) + 2
            y0, y1, x0, x1 = max(0, ys.min() - p), min(H, ys.max() + p + 1), max(0, xs.min() - p), min(W, xs.max() + p + 1)
            out[y0:y1, x0:x1] = cv2.GaussianBlur(out[y0:y1, x0:x1], (0, 0), float(bl))
    out, matted = motion.apply_maps(out, proj, tid, anim, f, W, H, shift) if (proj and tid) else (out, False)
    if not matted:
        if al <= 0.001:
            return None
        out *= al
    return out


def title_layers(proj, t, meta, cache, f, W, H):
    """Every premultiplied full-frame layer of title t at frame f (text, image, layered logo, sequence)."""
    if not t.get("translate", True) or t.get("mode") == "remove" or t["id"] not in meta:
        return []
    lay = t.get("layer") or {"type": "text"}
    if lay["type"] == "sequence":
        p = os.path.join(proj, lay["dir"], f"{f:06d}.png")
        return [load_layer(p)] if os.path.exists(p) else []
    if not (t["in"] <= f <= t["out"]):
        return []
    anim = dict(t.get("anim", {"mode": "static"}))
    anim.setdefault("in", t["in"]); anim.setdefault("out", t["out"])
    m = meta[t["id"]]
    st = t.get("style", {})
    shift = (st.get("nudge_x", 0), st.get("nudge_y", 0))
    out = []
    if "parts" in m:
        for i, pm in enumerate(m["parts"]):
            key = f"{t['id']}#{i}"
            if key not in cache:
                cache[key] = load_layer(pm["file"])
            L = place(cache[key], pm, anim, f, W, H, proj, t["id"], part=i, shift=shift)
            if L is not None:
                out.append(L)
        return out
    if t["id"] not in cache:
        cache[t["id"]] = load_layer(m["file"])
    L = place(cache[t["id"]], m, anim, f, W, H, proj, t["id"], shift=shift)
    return [L] if L is not None else []


def load_layer(path):
    im = np.array(Image.open(path).convert("RGBA")).astype(np.float32) / 255.0
    im = im[..., [2, 1, 0, 3]]  # to BGRA
    im[..., :3] *= im[..., 3:4]  # premultiply
    return im
