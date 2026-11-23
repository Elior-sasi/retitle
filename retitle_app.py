"""Entry point for the Windows app (PyInstaller)."""
import os, sys

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
if sys.stdout is None or sys.stderr is None:  # windowed exe: no console - log to a file
    d = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "Retitle")
    os.makedirs(d, exist_ok=True)
    f = open(os.path.join(d, "retitle.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stdout or f
    sys.stderr = sys.stderr or f
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

if getattr(sys, "frozen", False):
    base = sys._MEIPASS
    os.environ.setdefault("FFMPEG", os.path.join(base, "bin", "ffmpeg.exe"))
    os.environ.setdefault("FFPROBE", os.path.join(base, "bin", "ffprobe.exe"))
    os.environ.setdefault("RETITLE_FONTS", os.path.join(base, "fonts"))
    os.environ["PATH"] = os.path.join(base, "bin") + os.pathsep + os.environ.get("PATH", "")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    from rt.app import main
    main()
