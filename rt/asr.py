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
        if ready(name):
            STATE.update(model=name, state="ready", progress=1.0, error="")
            try:
                _cleanup(name)
            except Exception:
                pass
            return
        if _THREAD[0] and _THREAD[0].is_alive():
            return
        t = threading.Thread(target=_download, args=(name,), daemon=True)
        _THREAD[0] = t
        t.start()


def wait(name, progress=None):
    """Block until the model is on disk (used by the transcribe job). Returns its folder."""
    prepare(name)
    while not ready(name):
        if STATE["state"] == "error":
            raise RuntimeError("הורדת מנוע התמלול נכשלה: " + STATE["error"] + " - בדוק את החיבור לאינטרנט ונסה שוב")
        if progress and STATE["progress"] is not None:
            progress(STATE["progress"])
        time.sleep(1)
        if STATE["state"] == "idle":
            prepare(name)
    return model_dir(name)


def load(name):
    """WhisperModel on the GPU when it works, else the CPU."""
    from faster_whisper import WhisperModel
    path = model_dir(name) if ready(name) else name
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return WhisperModel(path, device="cuda", compute_type="float16")
    except Exception:
        pass
    return WhisperModel(path, device="cpu", compute_type="int8")
