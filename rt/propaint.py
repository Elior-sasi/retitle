"""ProPainter (github.com/sczhou/ProPainter, S-Lab License 1.0: free for non-commercial use) as a strong erase
for hard cards: letters that move over a moving background, thick decorated text. It fills each title's hole
from the whole stretch of frames at once (optical flow + a video transformer), so the background does not
flicker. Retitle's own masks say what to fill; only the masked pixels are taken from ProPainter.

It runs in its own Python with PyTorch, set up once on demand in %LOCALAPPDATA%/Retitle/propainter
(about 1 GB without an NVIDIA card, about 3 GB with one): uv (a Python installer) is downloaded from GitHub,
then Python, PyTorch, ProPainter's code (pinned commit) and its weights (GitHub release v0.1.0)."""
import os, sys, json, shutil, subprocess, threading, zipfile, glob, urllib.request
import numpy as np
import cv2
from . import vio

COMMIT = "326782529a63d997cab80ab21439b92ed422b3d3"
CODE_ZIP = f"https://github.com/sczhou/ProPainter/archive/{COMMIT}.zip"
WEIGHTS = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"
WEIGHT_FILES = ("ProPainter.pth", "recurrent_flow_completion.pth", "raft-things.pth")
UV_ZIP = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
PKGS = ["einops", "imageio", "imageio-ffmpeg", "scipy", "av", "opencv-python-headless", "numpy<2.3", "Pillow"]
STATE = {"state": "idle", "progress": None, "what": "", "error": ""}
_T = [None]


def home():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    return os.path.join(base, "Retitle", "propainter")


def _py():
    h = home()
    p = os.path.join(h, "venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(h, "venv", "bin", "python")
    return p


def code_dir():
    return os.path.join(home(), "ProPainter-" + COMMIT)


def ready():
    w = os.path.join(code_dir(), "weights")
    return os.path.exists(_py()) and os.path.exists(os.path.join(code_dir(), "inference_propainter.py")) and \
        all(os.path.exists(os.path.join(w, f)) for f in WEIGHT_FILES) and os.path.exists(os.path.join(home(), "ok"))


def vram_gb():
    """Memory of the NVIDIA card in GB (nvidia-smi comes with the driver); 6 when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        return max(float(x) for x in out.split()) / 1024
    except Exception:
        return 6.0


def gpu():
    from .asr import has_nvidia
    return has_nvidia()


def _get(url, dst, part=None):
    tmp = dst + ".part"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Retitle"}), timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0); done = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b); done += len(b)
            if part and total:
                STATE["progress"] = part[0] + (part[1] - part[0]) * done / total
    os.replace(tmp, dst)


def _run(cmd, part=None, what=""):
    STATE["what"] = what
    p = subprocess.run(cmd, capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "").strip()[-400:])
    if part:
        STATE["progress"] = part[1]


def install():
    """Set the engine up (blocking). Safe to run again: finished parts are kept."""
    h = home(); os.makedirs(h, exist_ok=True)
    STATE.update(state="installing", progress=0.0, error="")
    try:
        uv = shutil.which("uv") or os.path.join(h, "uv.exe" if os.name == "nt" else "uv")
        if not os.path.exists(uv):
            STATE["what"] = "מוריד את uv"
            z = os.path.join(h, "uv.zip"); _get(UV_ZIP, z, (0.0, 0.05))
            with zipfile.ZipFile(z) as zf:
                zf.extractall(h)
            os.remove(z)
        if not os.path.exists(_py()):
            _run([uv, "venv", "--python", "3.11", os.path.join(h, "venv")], (0.05, 0.12), "מתקין Python")
        idx = "https://download.pytorch.org/whl/cu121" if gpu() else "https://download.pytorch.org/whl/cpu"
        _run([uv, "pip", "install", "--python", _py(), "torch", "torchvision", "--index-url", idx], (0.12, 0.7),
             "מוריד את PyTorch" + (" ל-NVIDIA" if gpu() else ""))
        _run([uv, "pip", "install", "--python", _py()] + PKGS, (0.7, 0.75), "מתקין ספריות")
        if not os.path.exists(os.path.join(code_dir(), "inference_propainter.py")):
            STATE["what"] = "מוריד את ProPainter"
            z = os.path.join(h, "code.zip"); _get(CODE_ZIP, z, (0.75, 0.78))
            with zipfile.ZipFile(z) as zf:
                zf.extractall(h)
            os.remove(z)
        wd = os.path.join(code_dir(), "weights"); os.makedirs(wd, exist_ok=True)
        for i, f in enumerate(WEIGHT_FILES):
            if not os.path.exists(os.path.join(wd, f)):
                STATE["what"] = f"מוריד את המודל ({i + 1}/3)"
                _get(WEIGHTS + f, os.path.join(wd, f), (0.78 + 0.07 * i, 0.85 + 0.07 * i))
        open(os.path.join(h, "ok"), "w").close()
        STATE.update(state="ready", progress=1.0, what="")
    except Exception as e:
        STATE.update(state="error", error=str(e)[:300], what="")
        raise


def install_async():
    if _T[0] and _T[0].is_alive():
        return
    _T[0] = threading.Thread(target=lambda: _safe(install), daemon=True)
    _T[0].start()


def _safe(fn):
    try:
        fn()
    except Exception:
        pass


def status():
    if ready() and STATE["state"] != "installing":
        STATE.update(state="ready")
    return dict(STATE, ready=ready(), gpu=gpu())


def card(proj, gid, log=print, progress=None):
    """Refill the hole of card gid with ProPainter, over the cleaned frames. The masks are the ones the normal
    erase wrote (clean/<gid>/m<frame>.png); outside them the cleaned frames stay as they were."""
    if not ready():
        raise RuntimeError("המנוע ProPainter עוד לא מותקן - מתקינים אותו בהעדפות")
    d = os.path.join(proj, "clean", gid)
    fs = sorted(int(os.path.basename(p)[1:7]) for p in glob.glob(os.path.join(d, "m*.png")))
    if not fs:
        raise RuntimeError("קודם צריך לבנות את הכרטיס")
    work = os.path.join(proj, "propainter", gid)
    shutil.rmtree(work, ignore_errors=True)
    src, msk = os.path.join(work, "in"), os.path.join(work, "mask")
    os.makedirs(src); os.makedirs(msk)
    shape = None
    for i, f in enumerate(fs):
        im = cv2.imread(os.path.join(proj, "cache", gid, f"{f:06d}.png"))
        m = cv2.imread(os.path.join(d, f"m{f:06d}.png"), 0)
        if im is None or m is None:
            continue
        shape = im.shape
        cv2.imwrite(os.path.join(src, f"{i:05d}.png"), im)
        cv2.imwrite(os.path.join(msk, f"{i:05d}.png"), m)
    h, w = shape[:2]
    on_gpu = gpu()
    # working size by the graphics card's memory (a 6 GB laptop card takes about 0.45 megapixel per frame
    # in half precision); the CPU only a small copy (slow, but it works). Out of memory: again, smaller.
    vram = vram_gb() if on_gpu else 0
    budget = (0.95e6 if vram >= 12 else 0.65e6 if vram >= 8 else 0.45e6) if on_gpu else 0.18e6
    sub = "40" if vram >= 8 else "24" if on_gpu else "14"
    r = min(1.0, (budget / float(h * w)) ** 0.5)
    log(f"ProPainter על {len(fs)} פריימים" + (f" (כרטיס מסך, {vram:.0f}GB)" if on_gpu else " (מעבד - איטי)"))
    for attempt in range(3):
        cmd = [_py(), "inference_propainter.py", "-i", src, "-m", msk, "-o", os.path.join(work, "out"), "--save_frames",
               "--resize_ratio", f"{r:.3f}", "--subvideo_length", sub, "--neighbor_length", "10" if on_gpu else "6"]
        if on_gpu:
            cmd.append("--fp16")
        p = subprocess.Popen(cmd, cwd=code_dir(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        tail = []
        for line in p.stdout:
            tail = (tail + [line])[-20:]
            if progress and "%|" in line:
                try:
                    progress(min(0.95, float(line.split("%|")[0].split()[-1]) / 100))
                except ValueError:
                    pass
        if p.wait() == 0:
            break
        msg = "".join(tail)
        if "out of memory" in msg.lower() and attempt < 2:
            r *= 0.75; sub = str(max(8, int(sub) * 2 // 3))
            log("ProPainter: חסר זיכרון בכרטיס המסך - מנסה שוב בגודל קטן יותר")
            continue
        raise RuntimeError("ProPainter נכשל: " + msg[-300:])
    outs = sorted(glob.glob(os.path.join(work, "out", "in", "frames", "*.png")))
    if len(outs) < len(fs):
        raise RuntimeError("ProPainter לא החזיר את כל הפריימים")
    for i, f in enumerate(fs):
        o = cv2.resize(cv2.imread(outs[i]), (w, h), interpolation=cv2.INTER_CUBIC)
        cur = cv2.imread(os.path.join(d, f"{f:06d}.png"))
        m = cv2.imread(os.path.join(d, f"m{f:06d}.png"), 0)
        a = cv2.GaussianBlur((cv2.dilate(m, np.ones((5, 5), np.uint8)) > 0).astype(np.float32), (0, 0), 1.5)[..., None]
        res = (cur.astype(np.float32) * (1 - a) + o.astype(np.float32) * a).astype(np.uint8)
        cv2.imwrite(os.path.join(d, f"{f:06d}.png"), res)
    shutil.rmtree(work, ignore_errors=True)
    st = vio.load_json(os.path.join(proj, "clean_stats.json"), {})
    st.setdefault(gid, {})["engine"] = "propainter"
    vio.save_json(os.path.join(proj, "clean_stats.json"), st)
    return len(fs)
