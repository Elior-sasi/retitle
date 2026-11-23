"""Test clip: a title that is revealed by a wipe, rotates, and gets a light flash (for motion.py)."""
import sys, os, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "logo.mp4"
FONT = os.path.join(HERE, "..", "fonts", "Heebo[wght].ttf")
SC = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
W, H, N, FPS = int(1280 * SC) // 2 * 2, int(720 * SC) // 2 * 2, 110, 24
rng = np.random.default_rng(3)
bg = cv2.GaussianBlur(rng.integers(0, 255, (90, 160, 3)).astype(np.uint8), (0, 0), 2)
bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
bg = (bg * 0.55 + 30).astype(np.uint8)
F = ImageFont.truetype(FONT, int(150 * SC)); F.set_variation_by_axes([850])
lay = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
b = d.textbbox((0, 0), "TIGER", font=F)
x, y = (W - (b[2] - b[0])) // 2 - b[0], (H - (b[3] - b[1])) // 2 - b[1]
d.text((x + int(6 * SC), y + int(6 * SC)), "TIGER", font=F, fill=(10, 10, 10, 255))
d.text((x, y), "TIGER", font=F, fill=(255, 140, 0, 255))
L = np.array(lay).astype(np.float32) / 255  # RGBA
cx0, cx1 = x + b[0] - 10, x + b[2] + 16
p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                      "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", OUT], stdin=subprocess.PIPE)
for f in range(N):
    fr = bg.astype(np.float32)
    if 8 <= f < 100:
        ang = 0 if f < 40 else min(18, (f - 40) * 0.6)           # rotation after the reveal
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        Lf = cv2.warpAffine(L, M, (W, H))
        a = Lf[..., 3:4]
        if f < 30:                                               # wipe from left to right
            edge = cx0 + (cx1 - cx0) * (f - 8) / 22
            xs = np.arange(W)[None, :, None]
            a = a * np.clip((edge - xs) / (20 * SC), 0, 1)
        rgb = Lf[..., :3][..., ::-1] * 255
        if 60 <= f < 68:                                         # light flash on the letters
            rgb = np.minimum(255, rgb * 1.0 + 110 * (1 - abs(f - 64) / 4))
        fr = fr * (1 - a) + rgb * a
    p.stdin.write(np.clip(fr, 0, 255).astype(np.uint8).tobytes())
p.stdin.close(); p.wait()
print(OUT)
