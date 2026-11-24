"""Claude as the graphic designer of a title card or logo.

For a styled card (a film logo, a painted or 3D title) a font alone is not enough. Claude looks at the
original, picks a Hebrew font from a sheet of candidates, and writes a design spec: per-letter layout,
fill (colours sampled from the original, gradient, texture), rough edges, outlines, extrusion, shadow,
glow, decorations (stripes, underline) and elements carried over from the original (a gem, a star).
The spec is rendered here, put over the clean background next to the original, and Claude reviews it
like an art director and corrects it, up to MAX_ROUNDS times.

Result: logos/<id>_design.png on the first line of the card (layer "image" with the card's box);
the other lines of the card are erased only. The spec is kept on the title, so it can be edited and
rendered again without asking Claude.
"""
import os, re, json, math, base64, urllib.request
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from . import vio
from .translate import call, small_jpeg

MAX_ROUNDS = 2
MODEL = "claude-opus-5-5"          # design is judgement: the strongest model, only for styled cards

SYSTEM = """You are a senior graphic designer who localizes film and TV title cards and logos into Hebrew
for theatrical release. The Hebrew must look as if the original designer made it: same character, weight,
colour, texture, depth, decorations and energy - adapted to Hebrew letterforms (not a copy of Latin shapes).
Work only with what the spec below can express. Use the measured palette for colours.
Spec (JSON; sizes are relative to the letter height "em" unless marked px):
{
 "font": "<exact name from the candidate sheet>", "weight": <number or null>, "wdth": <number or null>,
 "lines": [{"text": "<Hebrew>", "scale": 1.0}],        # top to bottom; you may split or join lines for a better layout
 "line_gap": 0.15, "align": "center"|"right"|"left",
 "stretch_x": 1.0,                                       # horizontal scale of the letters
 "letters": {"spacing": 0.02, "rot": 0, "scale": 0, "dy": 0, "seed": 1},   # random per-letter tilt (deg), size and baseline jitter for hand-made looks; 0 = straight
 "edges": {"rough": 0, "soften": 0},                     # rough: edge wobble in px at 200 px letters (painted/brush 3-8)
 "fill": {"type": "solid"|"vertical"|"horizontal"|"radial", "colors": ["#rrggbb", ...], "texture": 0, "texture_scale": 0.15},
 "highlight": {"color": "#rrggbb", "width": 0.05, "side": "top"|"bottom"|"left"|"right"} or null,
 "outlines": [{"color": "#rrggbb", "width": 0.04}],      # inner first
 "extrude": {"dx": 0.0, "dy": 0.06, "steps": 8, "color": "#rrggbb", "shade": 0.3} or null,
 "shadow": {"dx": 0.03, "dy": 0.05, "blur": 0.06, "color": "#000000", "opacity": 0.6} or null,
 "glow": {"color": "#rrggbb", "size": 0.15, "opacity": 0.5} or null,
 "marks": {"color": "#rrggbb", "per_letter": [1, 3], "length": 0.35, "width": 0.08, "chance": 0.7} or null,   # stripes from the letter edges (tiger, claws)
 "underline": {"color": "#rrggbb", "outline": "#rrggbb" or null, "outline_width": 0.12, "thickness": 0.08, "y": 0.18, "from": 0.0, "to": 1.0,
               "taper": 1.0, "curve": 0.0, "rim": "#rrggbb" or null, "core": "#rrggbb" or null, "texture": 0, "glint": "#rrggbb" or null} or null,
   # a stick / swoosh under the letters, starting at the right: taper = tip thickness (0.2 = pointed brush stroke), curve = sag,
   # rim = colour of its lit top face, core = a light streak through it, glint = sparkle at the tip. Match the original's stick.
 "transplants": [{"box": [x0, y0, x1, y1], "line": 0, "letter": 2, "mode": "in"|"replace"|"part", "rx": 0.5, "ry": 0.5, "scale": 0.4}]   # copy an element from the ORIGINAL image (pixel box in image 1) onto a Hebrew letter (index in the line, right to left from 0), centred at rx, ry of that letter, scale = height relative to the letter. "in": inside the letter - ONLY into a closed hole of that letter (ם ס ט), never on top of strokes. "replace": the element takes the letter's place (the letter is removed), for an element that stands for a letter in the original (a jewel as the O): pick the Hebrew letter with the same sound or shape (O -> ו, I -> י), so the word still reads. "part": the element takes the place of the letter's detached part (the short left leg of ה, the leg of ק). With no fitting letter, leave the element out
}
Legibility comes first: every Hebrew letter must read at a glance at video size. Keep the look-alike
pairs apart - ה/ח (the gap of ה stays open), ד/ר, ו/ז/ן, נ/ג/כ, ב/כ, ס/ם, ט/מ. Condensed fonts, rough edges,
extrusion, outlines and letter tilt must never close a gap or thin a letter into another; when a font's
letters are too narrow for this, pick a fuller one (the shape and mood of the original still lead).
Reply ONLY with JSON: {"spec": {...}, "notes_he": "one short Hebrew sentence about your choices"}"""

REVIEW = """Image 1 is the original. Image 2 is the Hebrew version on the same background, at the same size.
Review it as an art director. First read the Hebrew letter by letter: if any letter could be misread
(ה like ח, נ like ג or ו, ד like ר...), it is not approved - fix the font or the effects. Then: would this pass
for the original designer's work? Compare character, weight,
colour, texture, depth, decorations, proportions and layout. If it is good, reply {"approved": true}.
Otherwise return the corrected full spec: {"approved": false, "spec": {...}, "notes_he": "what you changed, short Hebrew"}.
Reply ONLY with JSON."""


# ------------------------------------------------------------------ the card in the original
def card(proj, gid):
    T = vio.load_json(os.path.join(proj, "titles.json"))
    lines = [t for t in T["titles"] if t["group"] == gid and t.get("translate", True)]
    lines.sort(key=lambda t: t["ref_box"][1])
    return T, lines


def card_box(lines):
    xs0 = min(t["ref_box"][0] for t in lines); ys0 = min(t["ref_box"][1] for t in lines)
    xs1 = max(t["ref_box"][2] for t in lines); ys1 = max(t["ref_box"][3] for t in lines)
    return [xs0, ys0, xs1, ys1]


def _frame(proj, kind, t):
    """Full frame at the card's reference frame: 'src' from the cache/samples, 'clean' from the clean pass."""
    f = t["ref"]
    reg = t.get("region")
    if kind == "clean":
        p = os.path.join(proj, "clean", t["group"], f"{f:06d}.png")
    else:
        p = os.path.join(proj, "cache", t["group"], f"{f:06d}.png")
    img = cv2.imread(p) if reg else None
    if img is not None:
        return img, (reg[0], reg[1])
    if kind == "clean":
        return None, (0, 0)
    img = cv2.imread(os.path.join(proj, "samples", f"{f:06d}.jpg"))
    return img, (0, 0)


def crops(proj, lines, margin=0.35):
    """Source crop of the card, a clean background crop (clean pass, else a quick inpaint), and the
    letter mask (source minus background)."""
    t0 = lines[0]
    box = card_box(lines)
    src, (ox, oy) = _frame(proj, "src", t0)
    if src is None:
        return None
    h = box[3] - box[1]; w = box[2] - box[0]
    m = int(max(h, w * 0.25) * margin)
    x0, y0 = max(0, box[0] - ox - m), max(0, box[1] - oy - m)
    x1, y1 = min(src.shape[1], box[2] - ox + m), min(src.shape[0], box[3] - oy + m)
    s = src[y0:y1, x0:x1].copy()
    cl, (cx, cy) = _frame(proj, "clean", t0)
    if cl is not None and cl.shape == src.shape:
        c = cl[y0:y1, x0:x1].copy()
    else:
        # preview background: erase the letters with a quick inpaint (the real one comes from the build)
        mk = np.zeros(s.shape[:2], np.uint8)
        for t in lines:
            bx = t["ref_box"]
            mk[max(0, bx[1] - oy - y0 - 4):bx[3] - oy - y0 + 4, max(0, bx[0] - ox - x0 - 4):bx[2] - ox - x0 + 4] = 255
        c = cv2.inpaint(s, mk, 9, cv2.INPAINT_TELEA)
    diff = cv2.GaussianBlur(np.abs(s.astype(np.int16) - c.astype(np.int16)).max(2).astype(np.uint8), (3, 3), 0)
    letters = (diff > 40).astype(np.uint8)
    inner = [box[0] - ox - x0, box[1] - oy - y0, box[2] - ox - x0, box[3] - oy - y0]
    return {"src": s, "clean": c, "letters": letters, "box": inner, "origin": (ox + x0, oy + y0)}


def palette(img, mask, k=5):
    px = img[mask > 0].reshape(-1, 3).astype(np.float32)
    if len(px) < 50:
        px = img.reshape(-1, 3).astype(np.float32)
    k = min(k, max(1, len(px) // 20))
    _, lab, cen = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    cnt = np.bincount(lab.ravel(), minlength=k)
    out = []
    for i in np.argsort(-cnt):
        b, g, r = (int(v) for v in cen[i])
        out.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "share": round(float(cnt[i] / cnt.sum()), 2)})
    return out


# ------------------------------------------------------------------ rendering a spec
def _hex(c, default=(255, 255, 255)):
    try:
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return default


def _noise(w, h, cell, seed):
    """Smooth noise 0..1 with features about `cell` px apart."""
    r = np.random.default_rng(seed)
    sw, sh = max(2, int(w / max(2.0, cell))), max(2, int(h / max(2.0, cell)))
    n = cv2.resize(r.random((sh, sw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), max(1.0, w / sw / 3))
    return (n - n.min()) / max(1e-6, n.max() - n.min())


def _glyph(ch, font, size_px):
    from . import render
    kw = render._kw(direction="rtl", language="he")
    bb = font.getbbox(ch, **kw)
    w, h = bb[2] - bb[0] + 8, bb[3] - bb[1] + 8
    im = Image.new("L", (max(1, w), max(1, h)), 0)
    ImageDraw.Draw(im).text((4 - bb[0], 4 - bb[1]), ch, font=font, fill=255, **kw)
    return im, bb


def letter_mask(spec, em=200):
    """Alpha mask of all lines (float 0..1) at letter height `em`, plus each letter's box."""
    from . import render
    lt = spec.get("letters") or {}
    rng = np.random.default_rng(int(lt.get("seed", 1)))
    sx = float(spec.get("stretch_x", 1.0) or 1.0)
    jit = any(float(lt.get(k, 0) or 0) for k in ("rot", "scale", "dy"))
    rows = []
    for ln in spec.get("lines") or []:
        sc = float(ln.get("scale", 1.0) or 1.0)
        size = int(em * sc * 1.35)
        font = render._font(spec.get("font", "Heebo"), size, spec.get("weight"), spec.get("wdth"))
        text = ln.get("text", "")
        pieces = []
        if not jit:
            g, _ = _glyph(render._vis(text), font, size)
            if sx != 1:
                g = g.resize((max(1, int(g.width * sx)), g.height), Image.LANCZOS)
            pieces.append((g, 0, [(0, g.width)]))        # whole line, one box
        else:
            # letters right to left as read; each drawn alone, then placed right to left
            for ch in text:
                if ch == " ":
                    pieces.append((None, int(size * 0.28 * sx), None)); continue
                g, _ = _glyph(ch, font, size)
                s2 = 1 + rng.uniform(-1, 1) * float(lt.get("scale", 0) or 0)
                g = g.resize((max(1, int(g.width * sx * s2)), max(1, int(g.height * s2))), Image.LANCZOS)
                rot = rng.uniform(-1, 1) * float(lt.get("rot", 0) or 0)
                if rot:
                    g = g.rotate(rot, resample=Image.BICUBIC, expand=True)
                dy = int(rng.uniform(-1, 1) * float(lt.get("dy", 0) or 0) * em)
                pieces.append((g, dy, None))
        rows.append((pieces, size))
    gap = float(spec.get("line_gap", 0.15) or 0) * em
    sp = float(lt.get("spacing", 0.02) or 0) * em
    # measure rows
    row_imgs = []
    for pieces, size in rows:
        if len(pieces) == 1 and pieces[0][2] is not None:
            g = pieces[0][0]
            a = np.array(g).astype(np.float32) / 255
            ys, xs = np.where(a > 0.3)
            if len(xs) == 0:
                continue
            a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            row_imgs.append((a, [(0, 0, a.shape[1], a.shape[0])]))
            continue
        hs = [g.height + abs(dy) for g, dy, _ in pieces if g is not None]
        H = max(hs) if hs else 1
        W = sum((g.width if g is not None else dy) for g, dy, _ in pieces) + sp * max(0, len(pieces) - 1)
        canvas = np.zeros((int(H + em * 0.4), int(W) + 2), np.float32)
        x = canvas.shape[1]
        boxes = []
        base = canvas.shape[0] - int(em * 0.2)
        for g, dy, _ in pieces:
            if g is None:
                x -= dy; continue
            a = np.array(g).astype(np.float32) / 255
            y = base - a.shape[0] + dy
            x -= a.shape[1]
            y = max(0, min(canvas.shape[0] - a.shape[0], y)); xx = max(0, x)
            region = canvas[y:y + a.shape[0], xx:xx + a.shape[1]]
            region[:] = np.maximum(region, a[:region.shape[0], :region.shape[1]])
            boxes.append((xx, y, xx + a.shape[1], y + a.shape[0]))
            x -= int(sp)
        ys, xs = np.where(canvas > 0.3)
        if len(xs) == 0:
            continue
        x0, y0 = xs.min(), ys.min()
        canvas = canvas[y0:ys.max() + 1, x0:xs.max() + 1]
        row_imgs.append((canvas, [(b[0] - x0, b[1] - y0, b[2] - x0, b[3] - y0) for b in boxes]))
    if not row_imgs:
        return np.zeros((10, 10), np.float32), [], 0
    W = max(a.shape[1] for a, _ in row_imgs)
    H = int(sum(a.shape[0] for a, _ in row_imgs) + gap * (len(row_imgs) - 1))
    pad = int(em * 0.9)
    M = np.zeros((H + 2 * pad, W + 2 * pad), np.float32)
    al = spec.get("align", "center")
    y = pad
    boxes = []
    for i, (a, bx) in enumerate(row_imgs):
        x = pad + (W - a.shape[1]) if al == "right" else pad if al == "left" else pad + (W - a.shape[1]) // 2
        M[y:y + a.shape[0], x:x + a.shape[1]] = np.maximum(M[y:y + a.shape[0], x:x + a.shape[1]], a)
        boxes.append([(b[0] + x, b[1] + y, b[2] + x, b[3] + y) for b in bx])
        y += a.shape[0] + int(gap)
    return M, boxes, pad


def _shift(a, dx, dy):
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    dx, dy = int(round(dx)), int(round(dy))
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


def render_spec(spec, source=None, em=200):
    """spec -> RGBA PIL image (tight). source: {"src": BGR crop} for transplants."""
    M, boxes, pad = letter_mask(spec, em)
    h, w = M.shape
    trs = [dict(tr) for tr in spec.get("transplants") or []]
    for tr in trs:
        if tr.get("mode") == "part":
            # the element takes the place of a detached part of the letter (the short leg of ה,
            # the leg of ק, the dot...): the smallest piece of that letter goes, the element sits there
            try:
                row = boxes[int(tr.get("line", 0))]
                lb = row[min(len(row) - 1, int(tr.get("letter", 0)))]
                n_, lab_, st_, cen_ = cv2.connectedComponentsWithStats((M > 0.3).astype(np.uint8))
                ks = [k for k in range(1, n_) if lb[0] <= cen_[k][0] <= lb[2] and lb[1] <= cen_[k][1] <= lb[3]]
                if len(ks) >= 2:
                    k = min(ks, key=lambda k: st_[k, 4])
                    x, y, ww, hh = st_[k, :4]
                    tr["_part"] = [int(x), int(y), int(x + ww), int(y + hh)]
                    M[cv2.dilate((lab_ == k).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
            except Exception:
                pass
    for tr in trs:
        if tr.get("mode") == "replace":
            # the element takes the letter's place (as the jewel is the O of LOST): the letter goes
            try:
                row = boxes[int(tr.get("line", 0))]
                lb = row[min(len(row) - 1, int(tr.get("letter", 0)))]
                n_, lab_, st_, cen_ = cv2.connectedComponentsWithStats((M > 0.3).astype(np.uint8))
                for k in range(1, n_):
                    cx_, cy_ = cen_[k]
                    if lb[0] <= cx_ <= lb[2] and lb[1] <= cy_ <= lb[3]:
                        M[cv2.dilate((lab_ == k).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 0
            except Exception:
                pass
    seed = int((spec.get("letters") or {}).get("seed", 1))
    # rough edges
    ed = spec.get("edges") or {}
    rough = float(ed.get("rough", 0) or 0) * em / 200
    if rough > 0:
        # a slow wobble (hand-cut shapes) plus a little grain (paint on the edge)
        nx = ((_noise(w, h, em * 0.18, seed + 1) - 0.5) * 2 * rough + (_noise(w, h, em * 0.04, seed + 3) - 0.5) * rough * 0.6)
        ny = ((_noise(w, h, em * 0.18, seed + 2) - 0.5) * 2 * rough + (_noise(w, h, em * 0.04, seed + 4) - 0.5) * rough * 0.6)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        M = cv2.remap(M, gx + nx.astype(np.float32), gy + ny.astype(np.float32), cv2.INTER_LINEAR)
    if float(ed.get("soften", 0) or 0) > 0:
        M = cv2.GaussianBlur(M, (0, 0), float(ed["soften"]) * em / 200)
    # fill
    fl = spec.get("fill") or {}
    cols = [np.array(_hex(c), np.float32) for c in (fl.get("colors") or ["#ffffff"])]
    ys, xs = np.where(M > 0.3)
    y0, y1 = (ys.min(), ys.max()) if len(ys) else (0, h)
    x0, x1 = (xs.min(), xs.max()) if len(xs) else (0, w)
    t = None
    kind = fl.get("type", "solid")
    if kind == "vertical" and len(cols) > 1:
        t = np.clip((np.arange(h, dtype=np.float32)[:, None] - y0) / max(1, y1 - y0), 0, 1) * np.ones((1, w), np.float32)
    elif kind == "horizontal" and len(cols) > 1:
        t = np.ones((h, 1), np.float32) * np.clip((np.arange(w, dtype=np.float32)[None] - x0) / max(1, x1 - x0), 0, 1)
    elif kind == "radial" and len(cols) > 1:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = np.clip(np.hypot((xx - (x0 + x1) / 2) / max(1, (x1 - x0) / 2), (yy - (y0 + y1) / 2) / max(1, (y1 - y0) / 2)), 0, 1)
    if t is None:
        fill = np.ones((h, w, 1), np.float32) * cols[0]
    else:
        seg = t * (len(cols) - 1)
        i = np.clip(seg.astype(int), 0, len(cols) - 2)
        f = (seg - i)[..., None]
        C = np.stack(cols)
        fill = C[i] * (1 - f) + C[i + 1] * f
    tex = float(fl.get("texture", 0) or 0)
    if tex > 0:
        cs = max(0.02, float(fl.get("texture_scale", 0.15) or 0.15)) * em
        n = 0.65 * _noise(w, h, cs, seed + 5) + 0.35 * _noise(w, h, cs * 0.25, seed + 6)
        fill = fill * (1 + (n[..., None] - 0.5) * 0.7 * tex)
    hl = spec.get("highlight")
    if hl:
        wd = max(1.0, float(hl.get("width", 0.05)) * em)
        side = hl.get("side", "top")
        d = {"top": (0, wd), "bottom": (0, -wd), "left": (wd, 0), "right": (-wd, 0)}.get(side, (0, wd))
        rim = np.clip(M - _shift(M, d[0], d[1]), 0, 1)
        rim = cv2.GaussianBlur(rim, (0, 0), max(0.6, wd / 4))
        fill = fill * (1 - rim[..., None]) + np.array(_hex(hl.get("color"), (255, 255, 255)), np.float32) * rim[..., None]
    mk = spec.get("marks")
    if mk:
        S = _marks(M, boxes, mk, em, seed)
        fill = fill * (1 - S[..., None]) + np.array(_hex(mk.get("color"), (60, 30, 10)), np.float32) * S[..., None]
    out = np.zeros((h, w, 4), np.float32)

    def over(rgb, a):
        a = np.clip(a, 0, 1)
        out[..., :3] = out[..., :3] * (1 - a[..., None]) + np.broadcast_to(rgb, out[..., :3].shape) * a[..., None]
        out[..., 3] = out[..., 3] * (1 - a) + a

    # stack from the back: glow, shadow, extrusion, outlines (outer first), fill
    body = M.copy()
    outl = spec.get("outlines") or []
    widths = [max(0.0, float(o.get("width", 0.03)) * em) for o in outl]
    total = sum(widths)
    grown = cv2.dilate(M, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * total) + 1, int(2 * total) + 1))) if total >= 1 else M
    ex = spec.get("extrude")
    E = np.zeros_like(M)
    if ex:
        steps = int(ex.get("steps", 8) or 8)
        dx, dy = float(ex.get("dx", 0)) * em, float(ex.get("dy", 0.06)) * em
        for k in range(1, steps + 1):
            E = np.maximum(E, _shift(grown, dx * k / steps, dy * k / steps))
    sh = spec.get("shadow")
    if sh:
        base = np.maximum(grown, E)
        S = _shift(base, float(sh.get("dx", 0.03)) * em, float(sh.get("dy", 0.05)) * em)
        b = float(sh.get("blur", 0.05)) * em
        if b > 0.5:
            S = cv2.GaussianBlur(S, (0, 0), b)
        over(np.array(_hex(sh.get("color"), (0, 0, 0)), np.float32), S * float(sh.get("opacity", 0.6)))
    gl = spec.get("glow")
    if gl:
        G = cv2.GaussianBlur(np.maximum(grown, E), (0, 0), max(1.0, float(gl.get("size", 0.15)) * em))
        G = np.clip(G * 1.6, 0, 1)
        over(np.array(_hex(gl.get("color"), (255, 255, 255)), np.float32), G * float(gl.get("opacity", 0.5)))
    if ex:
        c = np.array(_hex(ex.get("color"), (40, 40, 40)), np.float32)
        shade = float(ex.get("shade", 0.3) or 0)
        # darker towards the far end of the extrusion
        far = cv2.GaussianBlur(np.clip(E - grown, 0, 1), (0, 0), max(1.0, em * 0.02))
        col = np.ones((h, w, 1), np.float32) * c * (1 - shade * far[..., None])
        out_a = np.clip(E - grown, 0, 1)
        out[..., :3] = out[..., :3] * (1 - out_a[..., None]) + col * out_a[..., None]
        out[..., 3] = out[..., 3] * (1 - out_a) + out_a
    acc = total
    for o, wd in reversed(list(zip(outl, widths))):     # outermost first
        r = int(round(acc))
        if r >= 1:
            A = cv2.dilate(M, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            A = cv2.GaussianBlur(A, (0, 0), 0.6)
            over(np.array(_hex(o.get("color"), (0, 0, 0)), np.float32), A)
        acc -= wd
    over(fill, body)
    img = Image.fromarray(np.dstack([out[..., :3].clip(0, 255), (out[..., 3] * 255).clip(0, 255)]).astype(np.uint8), "RGBA")
    ul = spec.get("underline")
    if ul:
        img = _underline(img, M, ul, em)
    for tr in trs:
        img = _transplant(img, boxes, tr, source, (spec.get("fill") or {}).get("colors"))
    ys, xs = np.where(M > 0.3)
    ink = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else [0, 0, img.width, img.height]
    for tr in trs:
        if tr.get("mode") in ("replace", "part"):       # the element stands for a letter: it is part of the lettering
            try:
                row = boxes[int(tr.get("line", 0))]
                lb = row[min(len(row) - 1, int(tr.get("letter", 0)))]
                ink = [min(ink[0], int(lb[0])), min(ink[1], int(lb[1])), max(ink[2], int(lb[2])), max(ink[3], int(lb[3]))]
            except Exception:
                pass
    bb = img.getbbox()
    if bb:
        img = img.crop(bb)
        ink = [ink[0] - bb[0], ink[1] - bb[1], ink[2] - bb[0], ink[3] - bb[1]]
    img.info["ink"] = ",".join(str(v) for v in ink)
    return img


def save(img, path):
    """PNG with the letters' box kept in it (the build fits the letters, not the decorations)."""
    from PIL import PngImagePlugin
    pi = PngImagePlugin.PngInfo()
    if img.info.get("ink"):
        pi.add_text("ink", img.info["ink"])
    img.save(path, pnginfo=pi)


def _marks(M, boxes, mk, em, seed):
    r = np.random.default_rng(seed + 9)
    h, w = M.shape
    L = Image.new("L", (w, h), 0); d = ImageDraw.Draw(L)
    B = M > 0.5
    lo, hi = (mk.get("per_letter") or [1, 3])[:2]
    for row in boxes:
        for (a, b, c, e) in row:
            if r.random() > float(mk.get("chance", 0.7)):
                continue
            side = r.choice([-1, 1])
            for _ in range(int(r.integers(int(lo), int(hi) + 1))):
                y = int(r.uniform(b + (e - b) * 0.15, b + (e - b) * 0.85))
                if y >= h:
                    continue
                row_px = np.where(B[y, a:c])[0]
                if len(row_px) == 0:
                    continue
                x = a + (row_px.min() if side == 1 else row_px.max())
                ln = float(mk.get("length", 0.35)) * em * r.uniform(0.7, 1.2)
                t = float(mk.get("width", 0.08)) * em / 2 * r.uniform(0.8, 1.2)
                ang = np.deg2rad(r.uniform(-25, 10))
                dx, dy = side * np.cos(ang), np.sin(ang); px, py = -dy, dx
                d.polygon([(x - px * t, y - py * t), (x + px * t, y + py * t), (x + dx * ln, y + dy * ln)], fill=255)
    return np.array(L.filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32) / 255 * M


def _underline(img, M, ul, em):
    """A painted stick under the letters: thick rounded end at the start of the reading direction
    (the right in Hebrew), tapering to a point, with a top bevel (rim), a light streak through its
    core, paint texture, an optional outline and a sparkle at the tip."""
    ys, xs = np.where(M > 0.5)
    if not len(xs):
        return img
    x0, x1, yb = xs.min(), xs.max(), ys.max()
    th = max(3.0, float(ul.get("thickness", 0.08)) * em)
    y = yb + float(ul.get("y", 0.18)) * em
    # right to left: "from" 0 is the right end
    xa = int(x1 - (x1 - x0) * float(ul.get("to", 1.0)))
    xb = int(x1 - (x1 - x0) * float(ul.get("from", 0.0)))
    need = int(y + th * 2 + em * 0.4)
    if need > img.height:
        new = Image.new("RGBA", (img.width, need), (0, 0, 0, 0)); new.paste(img, (0, 0)); img = new
    H, W = img.height, img.width
    taper = float(ul.get("taper", 1.0))                 # tip thickness / start thickness
    curve = float(ul.get("curve", 0.0)) * em            # sag of the middle (px, + = down)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    L = max(1, xb - xa)
    tt = np.clip((xb - xx) / L, 0, 1)                   # 0 at the thick start (right), 1 at the tip
    half = th / 2 * (1 - (1 - taper) * tt ** 2.5)      # stays thick, narrows towards the end
    yc = y + th / 2 + curve * np.sin(np.pi * tt)
    inside = (xx >= xa) & (xx <= xb)
    dist = np.abs(yy - yc) - half                       # <0 inside the stick
    m = np.where(inside, dist, 1e3)
    # round caps
    for cx, r in ((xb, th / 2), (xa, max(1.0, th / 2 * taper))):
        cy = y + th / 2
        m = np.minimum(m, np.hypot(xx - cx, yy - cy) - r)
    A = np.clip(0.5 - m, 0, 1)                          # 1 px antialias
    if not A.any():
        return img
    rgb = np.array(_hex(ul.get("color"), (255, 160, 30)), np.float32)
    col = np.ones((H, W, 3), np.float32) * rgb
    tex = float(ul.get("texture", 0) or 0)
    if tex > 0:
        n = _noise(W, H, max(4.0, em * 0.05), 11)
        col = col * (1 + (n[..., None] - 0.5) * 0.6 * tex)
    rim = ul.get("rim")
    if rim:
        # the top face of the stick
        rel = (yy - (yc - half)) / np.maximum(1e-3, 2 * half)   # 0 top edge .. 1 bottom edge
        rb = 0.8 * np.clip((0.3 - rel) / 0.08, 0, 1) * (rel > -0.2)
        rc = np.array(_hex(rim), np.float32)
        if tex > 0:
            rc = rc * (1 + (n[..., None] - 0.5) * 0.8 * tex)
        col = col * (1 - rb[..., None]) + rc * rb[..., None]
    core = ul.get("core")
    if core:
        # a light streak along the middle, fading towards the tip
        cyc = yc + half * 0.15
        g = np.exp(-((yy - cyc) / np.maximum(0.6, half * 0.22)) ** 2) * (1 - 0.7 * tt) * inside
        col = col * (1 - g[..., None]) + np.array(_hex(core), np.float32) * g[..., None]
    out = np.array(img).astype(np.float32)
    oc = ul.get("outline")
    if oc:
        ow = max(1.0, th * float(ul.get("outline_width", 0.12)))
        O = np.clip(0.5 - (m - ow), 0, 1)
        oa = O * (1 - A)
        o3 = np.array(_hex(oc), np.float32)
        out[..., :3] = out[..., :3] * (1 - oa[..., None]) + o3 * oa[..., None]
        out[..., 3] = np.maximum(out[..., 3], oa * 255)
    out[..., :3] = out[..., :3] * (1 - A[..., None]) + col * A[..., None]
    out[..., 3] = np.maximum(out[..., 3], A * 255)
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGBA")
    if ul.get("glint"):
        g = Image.new("RGBA", img.size, (0, 0, 0, 0)); gd = ImageDraw.Draw(g)
        c = _hex(ul["glint"], (200, 255, 255))
        t2 = max(2, int(th * max(taper, 0.35)))
        r = int(t2 * 2.2); cy = int(y + th / 2)
        gd.ellipse((xa - r, cy - r, xa + r, cy + r), fill=c + (200,))
        g = g.filter(ImageFilter.GaussianBlur(t2)); gd = ImageDraw.Draw(g)
        gd.line([(xa - r * 1.3, cy), (xa + r * 1.3, cy)], fill=c + (255,), width=max(1, t2 // 4))
        gd.line([(xa, cy - r), (xa, cy + r)], fill=c + (255,), width=max(1, t2 // 4))
        img.alpha_composite(g)
    return img


def _transplant(img, boxes, tr, source, fills=None):
    """Copy an element (box in the source crop) onto a Hebrew letter, keyed out of its background."""
    if not source or "src" not in source:
        return img
    try:
        x0, y0, x1, y1 = (int(v) for v in tr["box"])
        s = source["src"][max(0, y0):y1, max(0, x0):x1]
        c = source["clean"][max(0, y0):y1, max(0, x0):x1]
        if s.size == 0:
            return img
        a = np.abs(s.astype(np.int16) - c.astype(np.int16)).max(2).astype(np.float32)
        a = np.clip((a - 25) / 40, 0, 1)
        # not the letter it sat on: drop pixels close to the letters' main colour
        # (the letter colours: the measured fill, else the colours the spec paints the letters in)
        cols = [source["fill_hex"]] if source.get("fill_hex") not in (None, "#000000") else list(fills or [])
        for hx in cols:
            fc = np.array(_hex(hx), np.float32)[::-1]
            dist = np.linalg.norm(s.astype(np.float32) - fc, axis=2)
            a *= np.clip((dist - 45) / 45, 0, 1)
        # the hole around the element (shadow inside a letter) also differs from the clean plate:
        # split by lightness and keep the part that stands out most from the plate
        L = cv2.cvtColor(s, cv2.COLOR_BGR2LAB)[..., 0]
        Lc = float(cv2.cvtColor(c, cv2.COLOR_BGR2LAB)[..., 0].mean())
        reg = a > 0.4
        if reg.sum() > 50:
            th, _ = cv2.threshold(L[reg].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            hi_, lo_ = reg & (L > th), reg & (L <= th)
            if hi_.any() and lo_.any():
                keep = hi_ if abs(L[hi_].mean() - Lc) >= abs(L[lo_].mean() - Lc) else lo_
                a = a * cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.0)
        # the element is one piece: keep the biggest blob (convex hull), soft edge
        m = (a > 0.4).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:
            k = 1 + int(np.argmax(st[1:, 4]))
            cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = np.zeros_like(m); cv2.fillPoly(hull, [cv2.convexHull(cnts[0])], 1)
            a = cv2.GaussianBlur(hull.astype(np.float32), (5, 5), 0)
            ys, xs = np.where(hull > 0)
            s = s[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        el = Image.fromarray(np.dstack([cv2.cvtColor(s, cv2.COLOR_BGR2RGB), (a * 255).astype(np.uint8)]), "RGBA")
        row = boxes[int(tr.get("line", 0))]
        lb = tr.get("_part") or row[min(len(row) - 1, int(tr.get("letter", 0)))]
        lh = lb[3] - lb[1]
        sc = float(tr.get("scale", 1.0 if tr.get("_part") else 0.9 if tr.get("mode") == "replace" else 0.4)) * lh / max(1, el.height)
        el = el.resize((max(1, int(el.width * sc)), max(1, int(el.height * sc))), Image.LANCZOS)
        cx = lb[0] + (lb[2] - lb[0]) * float(tr.get("rx", 0.5)); cy = lb[1] + lh * float(tr.get("ry", 0.5))
        img.alpha_composite(el, (int(cx - el.width / 2), int(cy - el.height / 2)))
    except Exception:
        pass
    return img


def composite(design, crops_, scale_to="box"):
    """The design over the clean background, fitted to the original card box (same width)."""
    bg = cv2.cvtColor(crops_["clean"], cv2.COLOR_BGR2RGB)
    x0, y0, x1, y1 = crops_["box"]
    tw, th = x1 - x0, y1 - y0
    from .render import _ink
    ink = _ink(design) or (0, 0, design.width, design.height)
    iw, ih = max(1, ink[2] - ink[0]), max(1, ink[3] - ink[1])
    sw, sh = tw / iw, th / ih
    s = min(max(sw, sh), 1.2 * min(sw, sh))          # same rule as the build
    d = design.resize((max(1, int(design.width * s)), max(1, int(design.height * s))), Image.LANCZOS)
    im = Image.fromarray(bg).convert("RGBA")
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax, ay = (ink[0] + ink[2]) / 2 * s, (ink[1] + ink[3]) / 2 * s
    im.alpha_composite(d, (int(cx - ax), int(cy - ay)))
    return im.convert("RGB")


# ------------------------------------------------------------------ Claude
def _ask(api_key, content, system, model=None):
    body = {"model": model or MODEL, "max_tokens": 3000,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    r = call(req)
    text = "".join(b.get("text", "") for b in r.get("content", []))
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def _img(arr_or_pil, w):
    a = arr_or_pil if isinstance(arr_or_pil, np.ndarray) else cv2.cvtColor(np.array(arr_or_pil), cv2.COLOR_RGB2BGR)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": small_jpeg(a, w, 82)}}


def candidates(proj, lines, he_text, n=14):
    """Candidate fonts: the closest by measured style, plus display fonts (painted, 3D, playful...)."""
    from . import fontmatch
    rows = []
    try:
        m = fontmatch.letters(proj, lines[0])
        src = fontmatch.traits(m) if m is not None else None
        if src is not None:
            rows = fontmatch.rank(src, fontmatch.candidate_traits(), fontmatch.script_of(lines[0].get("src", "")), n)
    except Exception:
        rows = []
    extra = ["Karantina", "Fredoka", "Rubik Dirt", "Rubik Wet Paint", "Secular One", "Suez One", "Gveret Levin",
             "Amatic SC", "Rubik", "Bona Nova", "Frank Ruhl Libre", "Rubik Mono One", "Varela Round", "Bellefair"]
    have = {r["font"] for r in rows}
    from . import render
    for f in extra:
        if len(rows) >= n + 6:
            break
        if f in have:
            continue
        try:
            render._font(f, 40)
            rows.append({"font": f, "weight": 700 if f in ("Rubik", "Fredoka") else None, "wdth": None})
        except Exception:
            pass
    return rows


def design_card(proj, gid, api_key, log=print, progress=None):
    """Design the card `gid`; returns (spec, png path) or raises."""
    from . import fontmatch
    T, lines = card(proj, gid)
    if not lines:
        raise RuntimeError("אין שורות לעיצוב")
    he = [t.get("he", "").strip() for t in lines if t.get("mode") != "remove" or len(lines) == 1]
    he = [x for x in he if x]
    if not he:
        raise RuntimeError("קודם צריך עברית לכותרת")
    cr = crops(proj, lines)
    if cr is None:
        raise RuntimeError("לא מצאתי את הפריים של הכותרת")
    pal = palette(cr["src"], cr["letters"])
    cr["fill_hex"] = pal[0]["hex"] if pal else "#000000"
    rows = candidates(proj, lines, " ".join(he))
    d = os.path.join(proj, "design"); os.makedirs(d, exist_ok=True)
    sj = fontmatch.sheet(rows, " ".join(he), os.path.join(d, f"fonts_{gid}.jpg"))
    names = "\n".join(f"{i + 1}. {r['font']}" + (f" (weight {r['weight']})" if r.get("weight") else "") for i, r in enumerate(rows))
    sh, sw = cr["src"].shape[:2]
    if progress:
        progress(0.1)
    content = [{"type": "text", "text": f"Image 1 - the original card ({sw}x{sh} px; transplant boxes use these pixels):"},
               _img(cr["src"], 900),
               {"type": "text", "text": f"Image 2 - Hebrew candidate fonts:\n{names}"},
               {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": small_jpeg(sj, 900, 80)}},
               {"type": "text", "text": "Original text, top to bottom: " + json.dumps([t["src"] for t in lines], ensure_ascii=False)
                + "\nHebrew to set: " + json.dumps(he, ensure_ascii=False)
                + "\nMeasured palette of the original letters (most used first): " + json.dumps(pal)
                + f"\nThe original lettering block is {cr['box'][2] - cr['box'][0]}x{cr['box'][3] - cr['box'][1]} px: lay the Hebrew out to fill a block of about the same shape (it is scaled to cover it, so what is left of the original hides behind it)."
                + "\nDesign the Hebrew version."}]
    r = _ask(api_key, content, SYSTEM)
    spec = r.get("spec") or {}
    if not spec.get("lines"):
        spec["lines"] = [{"text": x} for x in he]
    notes = [r.get("notes_he", "")]
    for k in range(MAX_ROUNDS):
        if progress:
            progress(0.3 + 0.6 * k / MAX_ROUNDS)
        img = render_spec(spec, cr)
        prev = composite(img, cr)
        r2 = _ask(api_key, [{"type": "text", "text": "Image 1 - original:"}, _img(cr["src"], 900),
                            {"type": "text", "text": "Image 2 - Hebrew version:"}, _img(prev, 900),
                            {"type": "text", "text": "Current spec: " + json.dumps(spec, ensure_ascii=False)}], SYSTEM + "\n\n" + REVIEW)
        if r2.get("approved") or not r2.get("spec"):
            break
        spec = r2["spec"]; notes.append(r2.get("notes_he", ""))
    img = render_spec(spec, cr)
    os.makedirs(os.path.join(proj, "logos"), exist_ok=True)
    rel = f"logos/{lines[0]['id']}_design.png"
    save(img, os.path.join(proj, rel))
    composite(img, cr).save(os.path.join(d, f"preview_{gid}.jpg"), quality=88)
    log("Claude עיצב: " + " ".join(n for n in notes if n))
    return spec, rel, lines, card_box(lines)


def apply(proj, lines, spec, rel, box):
    """The design goes on the first line (whole card box); the other lines of the card are erased only."""
    T = vio.load_json(os.path.join(proj, "titles.json"))
    ids = [t["id"] for t in lines]
    for t in T["titles"]:
        if t["id"] == ids[0]:
            t["layer"] = {"type": "image", "path": rel, "box": box, "design": True}
            t["design"] = spec
            t.pop("mode", None); t["translate"] = True
            # the picture stays as long as any line of the card is on screen
            t["in"] = min(x["in"] for x in lines)
            t["out"] = max(x["out"] for x in lines)
            S = vio.load_json(os.path.join(proj, "shots.json"), {})
            if S.get("cuts"):                       # ...but never past the shot the card is in
                from .shots import shot_of
                n = vio.load_json(os.path.join(proj, "project.json")).get("n_frames") or 10 ** 9
                _, eb = shot_of(S["cuts"], n, max(x["sample_out"] for x in lines))
                t["out"] = min(t["out"], eb - 1)
        elif t["id"] in ids:
            t["mode"] = "remove"; t["design_member"] = ids[0]
    vio.save_json(os.path.join(proj, "titles.json"), T)


EDIT = """The user (a Hebrew localizer) asks for a change to this Hebrew logo design. Image 1 is the original card,
image 2 the current Hebrew version. Apply exactly what is asked - nothing else changes - and keep the letters
legible. Reply ONLY with JSON: {"spec": {...full spec...}, "notes_he": "what you changed, one short Hebrew sentence"}"""


def edit_card(proj, tid, api_key, request, log=print):
    """A change in the user's own words ("put the jewel in the leg of ה"): Claude edits the stored spec,
    the logo is rendered again. The previous spec is kept for one-step undo."""
    T = vio.load_json(os.path.join(proj, "titles.json"))
    t = next(x for x in T["titles"] if x["id"] == tid)
    if not t.get("design"):
        raise RuntimeError("הכרטיס הזה עוד לא עוצב")
    _, lines = card(proj, t["group"])
    cr = crops(proj, [x for x in lines if x["id"] == tid or x.get("design_member") == tid] or lines)
    prev = composite(render_spec(t["design"], cr), cr)
    sh, sw = cr["src"].shape[:2]
    r = _ask(api_key, [{"type": "text", "text": f"Image 1 - original ({sw}x{sh} px; transplant boxes use these pixels):"}, _img(cr["src"], 900),
                       {"type": "text", "text": "Image 2 - current Hebrew version:"}, _img(prev, 900),
                       {"type": "text", "text": "Current spec: " + json.dumps(t["design"], ensure_ascii=False)
                        + "\nThe user's request (Hebrew): " + request}], SYSTEM + "\n\n" + EDIT)
    spec = r.get("spec")
    if not spec or not spec.get("lines"):
        raise RuntimeError("Claude לא החזיר עיצוב מתוקן - נסה לנסח אחרת")
    t["design_prev"] = t["design"]; t["design"] = spec
    vio.save_json(os.path.join(proj, "titles.json"), T)
    rel = rerender(proj, tid)
    if r.get("notes_he"):
        log("Claude: " + r["notes_he"])
    return rel


def back_card(proj, tid):
    """Undo the last change in words (one step)."""
    T = vio.load_json(os.path.join(proj, "titles.json"))
    t = next(x for x in T["titles"] if x["id"] == tid)
    if not t.get("design_prev"):
        return False
    t["design"] = t.pop("design_prev")
    vio.save_json(os.path.join(proj, "titles.json"), T)
    rerender(proj, tid)
    return True


def rerender(proj, tid):
    """Render a (hand-edited) stored spec again, without Claude."""
    T = vio.load_json(os.path.join(proj, "titles.json"))
    t = next(x for x in T["titles"] if x["id"] == tid)
    _, lines = card(proj, t["group"])
    cr = crops(proj, [x for x in lines if x["id"] == tid or x.get("design_member") == tid] or lines)
    img = render_spec(t["design"], cr)
    save(img, os.path.join(proj, t["layer"]["path"]))
    return t["layer"]["path"]
