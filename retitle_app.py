"""Entry point for the Windows app (PyInstaller)."""
import os, sys

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
    print("Retitle is running. Keep this window open; close it to quit.")
    main()
