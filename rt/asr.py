"""Transcription engine setup: the Whisper model is fetched once, in the background, right after the app starts,
so it is ready before you need it. Kept in %LOCALAPPDATA%/Retitle/whisper/<model> (not in the installer: it is
1.5-3 GB, and every update would download it again)."""
import os, threading, time

SIZES = {"large-v3": 3.09e9, "large-v3-turbo": 1.62e9, "turbo": 1.62e9, "medium": 1.53e9, "small": 0.48e9}
STATE = {"model": "", "state": "idle", "progress": None, "error": ""}
_LOCK = threading.Lock()
_THREAD = [None]


def model_dir(name):
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    return os.path.join(base, "Retitle", "whisper", name)


def ready(name):
    d = model_dir(name)
    return os.path.isfile(os.path.join(d, "model.bin")) and os.path.isfile(os.path.join(d, "config.json")) \
        and not os.path.exists(os.path.join(d, ".incomplete"))


def _size(d):
    n = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                n += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n


def _prepare_all(name):
    if not ready(name):
        _download(name)
    if STATE["state"] != "error" and has_nvidia() and not gpu_ready():
        try:
            _get_gpu_libs()
            STATE.update(state="ready", progress=1.0, what="")
        except Exception as e:                       # the CPU still works; try again next start
            STATE.update(state="ready", progress=1.0, what="", gpu_error=str(e))


def _download(name):
    from faster_whisper.utils import download_model
    d = model_dir(name)
    os.makedirs(d, exist_ok=True)
    flag = os.path.join(d, ".incomplete")
    open(flag, "w").close()
    STATE.update(model=name, state="downloading", progress=0.0, error="")
    stop = threading.Event()

    def watch():
        total = SIZES.get(name, 2e9)
        cache = os.path.join(os.path.dirname(d), ".cache")
        while not stop.wait(1.0):
            STATE["progress"] = min(0.99, (_size(d) + _size(cache)) / total)
    threading.Thread(target=watch, daemon=True).start()
    try:
        download_model(name, output_dir=d, cache_dir=os.path.join(os.path.dirname(d), ".cache"))
        stop.set()
        os.remove(flag)
        STATE.update(state="ready", progress=1.0)
        _cleanup(name)
    except Exception as e:
        STATE.update(state="error", error=str(e), progress=None)
    finally:
        stop.set()


def _cleanup(keep):
    """Free the space of other models downloaded before (only the one in use is kept)."""
    import shutil
    base = os.path.dirname(model_dir(keep))
    for d in os.listdir(base):
        if d not in (keep, ".cache") and os.path.isfile(os.path.join(base, d, "model.bin")):
            shutil.rmtree(os.path.join(base, d), ignore_errors=True)
    shutil.rmtree(os.path.join(base, ".cache"), ignore_errors=True)


def prepare(name):
    """Start the background download if the model is missing. Returns immediately."""
    with _LOCK:
        if ready(name) and (gpu_ready() or not has_nvidia()):
            STATE.update(model=name, state="ready", progress=1.0, error="")
            try:
                _cleanup(name)
            except Exception:
                pass
            return
        if _THREAD[0] and _THREAD[0].is_alive():
            return
        t = threading.Thread(target=_prepare_all, args=(name,), daemon=True)
        _THREAD[0] = t
        t.start()


def wait(name, progress=None):
    """Block until the model is on disk (used by the transcribe job). Returns its folder."""
    prepare(name)
    while not ready(name) or (_THREAD[0] and _THREAD[0].is_alive()):    # also the GPU libraries, if they are coming
        if STATE["state"] == "error":
            raise RuntimeError("הורדת מנוע התמלול נכשלה: " + STATE["error"] + " - בדוק את החיבור לאינטרנט ונסה שוב")
        if progress and STATE["progress"] is not None:
            progress(STATE["progress"])
        time.sleep(1)
        if STATE["state"] == "idle":
            prepare(name)
    return model_dir(name)


# ------------------------------------------------------------------ NVIDIA GPU
# ctranslate2 ships only the cuDNN loader; cuBLAS and the cuDNN sub-libraries come from NVIDIA's wheels on PyPI
# (~1.1 GB, once), matched to the CUDA 12 / cuDNN version ctranslate2 was built with.
GPU_PKGS = {"nvidia-cublas-cu12": "12.4.", "nvidia-cuda-runtime-cu12": "12.4.", "nvidia-cudnn-cu12": None}


def gpu_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    return os.path.join(base, "Retitle", "cuda")


def has_nvidia():
    if os.name != "nt":
        return False
    try:
        import ctypes
        ctypes.WinDLL("nvcuda.dll")                       # comes with the NVIDIA driver
        return True
    except OSError:
        return False


def _dll_version(path):
    import ctypes
    from ctypes import wintypes
    v = ctypes.windll.version
    n = v.GetFileVersionInfoSizeW(path, None)
    if not n:
        return None
    buf = ctypes.create_string_buffer(n)
    v.GetFileVersionInfoW(path, 0, n, buf)
    p = ctypes.c_void_p(); ln = wintypes.UINT()
    v.VerQueryValueW(buf, "\\", ctypes.byref(p), ctypes.byref(ln))
    ms, ls = ctypes.cast(p, ctypes.POINTER(ctypes.c_uint32 * 13)).contents[2:4]
    return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"


def _cudnn_wanted():
    try:
        import importlib.util
        import sys
        cands = [os.path.join(getattr(sys, "_MEIPASS", ""), "ctranslate2", "cudnn64_9.dll")]
        spec = importlib.util.find_spec("ctranslate2")
        if spec and spec.origin:
            cands.append(os.path.join(os.path.dirname(spec.origin), "cudnn64_9.dll"))
        for c in cands:
            if os.path.isfile(c):
                return _dll_version(c)
        return None
    except Exception:
        return None


def gpu_ready():
    d = gpu_dir()
    return os.path.isdir(d) and any(f.startswith("cublas64_12") for f in os.listdir(d)) \
        and any(f.startswith("cudnn_ops64_9") for f in os.listdir(d)) and not os.path.exists(os.path.join(d, ".incomplete"))


def _wheel_url(pkg, prefix):
    import json, urllib.request
    with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/json", timeout=30) as r:
        d = json.load(r)
    def key(v):
        return [int(x) if x.isdigit() else 0 for x in v.split(".")]
    for v in sorted(d["releases"], key=key, reverse=True):
        if prefix and not v.startswith(prefix):
            continue
        for f in d["releases"][v]:
            if f["filename"].endswith("win_amd64.whl"):
                return f["url"], f["size"]
    raise RuntimeError(f"no Windows build of {pkg} {prefix or ''}")


def _get_gpu_libs():
    import zipfile, urllib.request, shutil
    d = gpu_dir(); os.makedirs(d, exist_ok=True)
    flag = os.path.join(d, ".incomplete"); open(flag, "w").close()
    want = _cudnn_wanted()
    pk = dict(GPU_PKGS); pk["nvidia-cudnn-cu12"] = want if want else "9."
    urls = [(p, *_wheel_url(p, v)) for p, v in pk.items()]
    total = sum(x[2] for x in urls) or 1
    done = 0
    STATE.update(state="downloading", progress=0.0, error="", what="gpu")
    for p, url, size in urls:
        tmp = os.path.join(d, p + ".whl.part")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b); done += len(b)
                STATE["progress"] = min(0.99, done / total)
        with zipfile.ZipFile(tmp) as z:
            for n in z.namelist():
                if n.lower().endswith(".dll") and "/bin/" in n:
                    with z.open(n) as src, open(os.path.join(d, os.path.basename(n)), "wb") as dst:
                        shutil.copyfileobj(src, dst)
        os.remove(tmp)
    os.remove(flag)


def enable_gpu_libs():
    """Make NVIDIA's DLLs findable; preload our full cuDNN before ctranslate2 loads its own loader stub."""
    if os.name != "nt" or not gpu_ready():
        return False
    import ctypes
    d = gpu_dir()
    try:
        os.add_dll_directory(d)
    except Exception:
        pass
    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    for f in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"):
        try:
            ctypes.WinDLL(os.path.join(d, f))
        except OSError:
            pass
    return True


def load(name):
    """WhisperModel on the GPU when it works, else the CPU."""
    gpu = enable_gpu_libs()
    from faster_whisper import WhisperModel
    path = model_dir(name) if ready(name) else name
    if gpu:
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                STATE["device"] = "gpu"
                return WhisperModel(path, device="cuda", compute_type="float16")
        except Exception:
            pass
    STATE["device"] = "cpu"
    return WhisperModel(path, device="cpu", compute_type="int8")
