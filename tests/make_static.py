"""20 s static interview-like shot with a fading two-line lower third (doc/reality style)."""
import numpy as np, cv2, subprocess, os, sys
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, '..', 'fonts', 'Heebo[wght].ttf')
OUT, BG = sys.argv[1], sys.argv[2]
W, H, FPS = 1280, 720, 25
r = np.random.default_rng(3)
base = np.zeros((H, W, 3), np.float32)
for k in range(120):
    c = r.uniform(30, 200, 3)
    cv2.circle(base, (int(r.integers(0, W)), int(r.integers(0, H))), int(r.integers(20, 160)), c.tolist(), -1)
base = cv2.GaussianBlur(base, (0, 0), 6)
f1 = ImageFont.truetype(FONT, 44); f1.set_variation_by_axes([700])
f2 = ImageFont.truetype(FONT, 30); f2.set_variation_by_axes([400])
def card(alpha):
    L = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    d.rectangle((60, 540, 560, 650), fill=(10, 30, 80, 200))
    d.text((80, 548), "JOHN SMITH", font=f1, fill=(255, 255, 255, 255))
    d.text((80, 604), "Director of the museum", font=f2, fill=(255, 210, 90, 255))
    a = np.array(L).astype(np.float32) / 255; a[..., 3] *= alpha
    return a
ff = os.environ.get('FFMPEG', 'ffmpeg')
mk = lambda o: subprocess.Popen([ff, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                                  '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', o], stdin=subprocess.PIPE)
p, q = mk(OUT), mk(BG)
A = card(1.0)
for f in range(20 * FPS):
    noise = r.normal(0, 2.0, base.shape).astype(np.float32)   # sensor noise only: a locked-off camera
    bg = np.clip(base + noise, 0, 255)
    img = bg.copy()
    if 125 <= f <= 312:   # 5.0 s - 12.48 s, 6-frame fade in/out
        al = min(1, (f - 125 + 1) / 6, (312 - f + 1) / 6)
        rgb = img[..., ::-1] / 255
        o = A[..., :3] * (A[..., 3:] * al) + rgb * (1 - A[..., 3:] * al)
        img = o[..., ::-1] * 255
    p.stdin.write(img.astype(np.uint8).tobytes()); q.stdin.write(bg.astype(np.uint8).tobytes())
for x in (p, q):
    x.stdin.close(); x.wait()
