"""Builds assets/icon.png and assets/icon.ico (option 3A + 3F: R, playhead, subtitle bar)."""
import base64, io, sys
from PIL import Image, ImageDraw, ImageFont

FONT = sys.argv[1] if len(sys.argv) > 1 else "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"
S = 1024
VIOLET, YELLOW, BAR, WHITE = (123, 92, 240), (255, 209, 102), (214, 203, 255), (255, 255, 255)


def draw(size=S, small=False):
    k = size / 100
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=22 * k, fill=VIOLET)
    f = ImageFont.truetype(FONT, int((60 if small else 54) * k))
    cx, base = (40 if small else 40) * k, (60 if small else 60) * k
    d.text((cx, base), "R", font=f, fill=WHITE, anchor="ms")
    # playhead
    px, w = 70 * k, (6 if small else 4.5) * k
    d.rounded_rectangle([px - w / 2, 24 * k, px + w / 2, 64 * k], radius=w / 2, fill=YELLOW)
    d.polygon([(px - 9 * k, 17 * k), (px + 9 * k, 17 * k), (px, 27 * k)], fill=YELLOW)
    # subtitle bar
    d.rounded_rectangle([18 * k, 72 * k, 82 * k, (86 if small else 83) * k], radius=5 * k, fill=BAR)
    return im


big = draw()
big.save("assets/icon.png")
sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
frames = [(draw(S, small=s <= 32)).resize((s, s), Image.LANCZOS) for s in sizes]
frames[-1].save("assets/icon.ico", sizes=[(s, s) for s in sizes], append_images=frames[:-1])
buf = io.BytesIO()
big.resize((64, 64), Image.LANCZOS).save(buf, "PNG", optimize=True)
open("assets/icon64.b64", "w").write(base64.b64encode(buf.getvalue()).decode())
prev = Image.new("RGBA", (560, 300), (32, 32, 40, 255))
x = 20
for s in [256, 128, 64, 48, 32, 24, 16]:
    fr = next(f for f, z in zip(frames, sizes) if z == s) if s in sizes else big.resize((s, s), Image.LANCZOS)
    prev.alpha_composite(fr, (x, 150 - s // 2)); x += s + 12
prev.save("assets/icon_preview.png")
print("ok")
