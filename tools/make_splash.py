"""Draws assets/splash.png - the picture shown while Retitle starts (PyInstaller splash)."""
import os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
W, H, R = 860, 480, 22
S = 2                                          # draw at 2x, scale down: smooth edges
F = lambda size, wght=400: _f(size * S, wght)
def _f(size, wght):
    f = ImageFont.truetype(os.path.join(ROOT, "fonts", "Heebo[wght].ttf"), size)
    f.set_variation_by_axes([wght]); return f
w, h = W * S, H * S
# background: near-black with a violet glow on the art side and a warm one under the playhead
y, x = np.mgrid[0:h, 0:w].astype(np.float32)
base = np.array([20, 19, 27], np.float32)
g1 = np.exp(-(((x - 0.25 * w) / (0.45 * w)) ** 2 + ((y - 0.45 * h) / (0.55 * h)) ** 2))[..., None]
g2 = np.exp(-(((x - 0.38 * w) / (0.18 * w)) ** 2 + ((y - 0.95 * h) / (0.25 * h)) ** 2))[..., None]
img = base + g1 * np.array([95, 70, 190]) * 0.55 + g2 * np.array([255, 200, 90]) * 0.18
noise = np.random.default_rng(7).normal(0, 2.2, (h, w, 1))
im = Image.fromarray(np.clip(img + noise, 0, 255).astype(np.uint8)).convert("RGBA")
d = ImageDraw.Draw(im)

# art: three frames of a film, the middle one with a Hebrew title replacing the source one
def frame(cx, cy, fw, fh, ang, tone, he):
    L = Image.new("RGBA", (fw + 40, fh + 40), (0, 0, 0, 0)); q = ImageDraw.Draw(L)
    q.rounded_rectangle((20, 20, 20 + fw, 20 + fh), 14 * S, fill=tone + (255,))
    # a soft scene: hills
    for k, (col, amp, off) in enumerate([((255, 255, 255, 28), 0.10, 0.62), ((0, 0, 0, 60), 0.07, 0.78)]):
        pts = [(20 + i, 20 + fh * off + math.sin(i / fw * 6.3 + k) * fh * amp) for i in range(0, fw + 1, 8)]
        q.polygon(pts + [(20 + fw, 20 + fh), (20, 20 + fh)], fill=col)
    if he:
        t = "כותרת בעברית"; f = F(30, 800)
        bb = q.textbbox((0, 0), t, font=f, direction="rtl", language="he")
        q.text((20 + (fw - (bb[2] - bb[0])) / 2 - bb[0], 20 + fh * 0.30), t, font=f, fill=(255, 255, 255, 255), direction="rtl", language="he")
        q.rounded_rectangle((20 + fw * 0.2, 20 + fh * 0.80, 20 + fw * 0.8, 20 + fh * 0.80 + 12 * S), 6 * S, fill=(214, 205, 255, 235))
    else:
        q.rounded_rectangle((20 + fw * 0.25, 20 + fh * 0.36, 20 + fw * 0.75, 20 + fh * 0.36 + 18 * S), 6 * S, fill=(255, 255, 255, 70))
    sh = L.filter(ImageFilter.GaussianBlur(14)); shp = Image.new("RGBA", L.size, (0, 0, 0, 0))
    shp.putalpha(sh.getchannel("A").point(lambda v: int(v * 0.6)))
    L2 = Image.alpha_composite(shp, L).rotate(ang, resample=Image.BICUBIC, expand=True)
    im.alpha_composite(L2, (int(cx - L2.width / 2), int(cy - L2.height / 2)))
frame(0.15 * w, 0.40 * h, 250 * S, 150 * S, 8, (74, 58, 140), False)
frame(0.44 * w, 0.42 * h, 250 * S, 150 * S, -6, (70, 54, 130), False)
frame(0.29 * w, 0.52 * h, 330 * S, 200 * S, 0, (98, 80, 200), True)
d = ImageDraw.Draw(im)
# playhead across the art
px = 0.36 * w
d.polygon([(px - 16 * S, 0.12 * h), (px + 16 * S, 0.12 * h), (px, 0.12 * h + 22 * S)], fill=(255, 209, 102, 255))
d.rounded_rectangle((px - 3 * S, 0.12 * h + 14 * S, px + 3 * S, 0.88 * h), 3 * S, fill=(255, 209, 102, 255))

# right: logo, name, tagline
icon = Image.open(os.path.join(ROOT, "assets", "icon.png")).convert("RGBA").resize((96 * S, 96 * S), Image.LANCZOS)
rx = w - 56 * S
im.alpha_composite(icon, (rx - icon.width, 70 * S))
def rtext(txt, yy, f, col, ltr=False):
    kw = {} if ltr else {"direction": "rtl", "language": "he"}
    bb = d.textbbox((0, 0), txt, font=f, **kw)
    d.text((rx - bb[2], yy), txt, font=f, fill=col, **kw)
rtext("Retitle", 180 * S, F(72, 800), (255, 255, 255, 255), ltr=True)
rtext("לוקליזציה לעברית לכל וידאו", 278 * S, F(25, 500), (214, 205, 255, 255))
rtext("כותרות על המסך · כתוביות · תמלול", 316 * S, F(18, 400), (170, 164, 196, 255))
# bottom: accent line and credit
for i in range(int(w - 112 * S)):
    t = i / (w - 112 * S)
    c = tuple(int(a + (b - a) * t) for a, b in zip((124, 92, 240), (255, 209, 102)))
    d.line((56 * S + i, h - 64 * S, 56 * S + i, h - 61 * S), fill=c + (255,))
rtext("נוצר ע\"י אליאור מצליח סאסי", h - 46 * S, F(17, 500), (190, 186, 210, 255))
d.text((56 * S, h - 46 * S), "טוען", font=F(17, 500), fill=(150, 146, 172, 255), direction="rtl", language="he")
# rounded window shape
out = im.resize((W, H), Image.LANCZOS)
m = Image.new("L", (W * 4, H * 4), 0); ImageDraw.Draw(m).rounded_rectangle((0, 0, W * 4 - 1, H * 4 - 1), R * 4, fill=255)
out.putalpha(m.resize((W, H), Image.LANCZOS).point(lambda v: 255 if v > 127 else 0))
out.save(os.path.join(ROOT, "assets", "splash.png"))
print("assets/splash.png", out.size)
