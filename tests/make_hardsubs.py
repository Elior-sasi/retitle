"""12 s anime-like clip with English hardsubs at the bottom (6 lines, 2 s each) over a slow pan."""
import numpy as np, cv2, subprocess, os, sys
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, '..', 'fonts', 'Heebo[wght].ttf')
OUT = sys.argv[1]
W, H, FPS = 1280, 720, 24
r = np.random.default_rng(5)
big = np.zeros((900, 2200, 3), np.float32)
for k in range(300):
    cv2.circle(big, (int(r.integers(0, 2200)), int(r.integers(0, 900))), int(r.integers(15, 120)), r.uniform(20, 230, 3).tolist(), -1)
big = cv2.GaussianBlur(big, (0, 0), 4).astype(np.uint8)
f = ImageFont.truetype(FONT, 40); f.set_variation_by_axes([600])
LINES = ["Where are you going?", "I have to find her.", "Wait for me!", "We don't have much time.", "Then let's run.", "Together, always."]
ff = os.environ.get('FFMPEG', 'ffmpeg')
p = subprocess.Popen([ff, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                      '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', OUT], stdin=subprocess.PIPE)
for i in range(12 * FPS):
    img = big[90:90 + H, 100 + i * 2:100 + i * 2 + W].copy()
    k = i // 48
    if i % 48 < 44:
        L = Image.fromarray(img[..., ::-1]); d = ImageDraw.Draw(L)
        b = d.textbbox((0, 0), LINES[k], font=f, stroke_width=3)
        d.text(((W - (b[2] - b[0])) / 2 - b[0], 640 - b[3]), LINES[k], font=f, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        img = np.array(L)[..., ::-1]
    p.stdin.write(np.ascontiguousarray(img).tobytes())
p.stdin.close(); p.wait()
