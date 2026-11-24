"""rea-video (github.com/JuneYaooo/rea-video, Apache 2.0) inside Retitle - the same tool, not an imitation.

It is installed into Retitle's private Python (next to ProPainter, SAM 2 and Lens) together with its Claude Code
skill. In full mode Retitle runs `rea-video decompose` on the video first (shots, every overlay with its timing
and entrance, element crops, first/last frames per shot, stems), and the Claude Code session gets the
rea-video skill and the `rea-video` command - exactly the setup of the chat where the trailer came out right."""
import os, shutil, subprocess, zipfile
from . import vio, propaint

COMMIT = "fc69e194bdb4da91f72420c58debbc38eee647f5"
ZIP = f"https://github.com/JuneYaooo/rea-video/archive/{COMMIT}.zip"


def code_dir():
    return os.path.join(propaint.home(), "rea-video-" + COMMIT)


def exe():
    d = os.path.dirname(propaint._py())
    p = os.path.join(d, "rea-video.exe" if os.name == "nt" else "rea-video")
    return p if os.path.exists(p) else ""


def skill_dir():
    return os.path.join(code_dir(), "skills", "rea-video")


def ready():
    return propaint.ready() and bool(exe()) and os.path.exists(os.path.join(skill_dir(), "SKILL.md")) and \
        os.path.exists(os.path.join(propaint.home(), "ok_rea"))


def install(part=(0.0, 1.0)):
    """rea-video with its OCR into ProPainter's Python, and its source (for the Claude Code skill)."""
    h = propaint.home()
    S = propaint.STATE

    def p(a):
        return part[0] + (part[1] - part[0]) * a
    if not os.path.isdir(code_dir()):
        S["what"] = "מוריד את rea-video"
        z = os.path.join(h, "rea.zip")
        propaint._get(ZIP, z, (p(0), p(0.3)))
        with zipfile.ZipFile(z) as zf:
            zf.extractall(h)
        os.remove(z)
    uv = shutil.which("uv") or os.path.join(h, "uv.exe" if os.name == "nt" else "uv")
    propaint._run([uv, "pip", "install", "--python", propaint._py(), code_dir() + "[ocr]"], (p(0.3), p(0.9)), "מתקין את rea-video")
    # rapidocr-onnxruntime pulls the GUI OpenCV over the headless one; rea-video's own installer puts headless back
    try:
        propaint._run([uv, "pip", "uninstall", "--python", propaint._py(), "opencv-python"], None, "")
    except Exception:
        pass
    propaint._run([uv, "pip", "install", "--python", propaint._py(), "--reinstall", "opencv-python-headless"],
                  (p(0.9), p(1.0)), "מסיים את rea-video")
    open(os.path.join(h, "ok_rea"), "w").close()


def env():
    """rea-video needs ffmpeg on PATH: the one Retitle ships."""
    e = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    ff = vio.FFMPEG
    if os.path.isabs(ff):
        e["PATH"] = os.path.dirname(ff) + os.pathsep + e.get("PATH", "")
    e["PATH"] = os.path.dirname(exe() or propaint._py()) + os.pathsep + e["PATH"]
    return e


def decompose(proj, log=print, cancel=lambda: False, lang="en"):
    """rea-video decompose of the project's video into <proj>/rea (once; kept for later runs)."""
    out = os.path.join(proj, "rea")
    if os.path.exists(os.path.join(out, "VIDEO.md")):
        return out
    if not ready():
        raise RuntimeError("rea-video עוד לא מותקן")
    video = vio.load_json(os.path.join(proj, "project.json"))["video"]
    s = vio.load_json(os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/.config"), "Retitle", "settings.json"), {})
    ocr = {"eng": "en", "jpn": "ja", "kor": "ko"}.get(s.get("src_lang") or "eng", "en")
    cmd = [exe(), "decompose", video, "-o", out, "--lang", lang, "--ocr-lang", ocr]
    log("rea-video מפרק את הסרטון (שוטים, טקסטים על המסך, אנימציות, סאונד)")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env=env(), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    tail = []
    for line in p.stdout:
        tail = (tail + [line])[-20:]
        if cancel():
            p.kill()
            break
    if p.wait() != 0 or not os.path.exists(os.path.join(out, "VIDEO.md")):
        raise RuntimeError("rea-video נכשל: " + "".join(tail)[-300:])
    return out


def give_skill(proj):
    """The rea-video skill for the Claude Code session that works in <proj> (project skills folder)."""
    if not os.path.isdir(skill_dir()):
        return False
    d = os.path.join(proj, ".claude", "skills", "rea-video")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(skill_dir(), d)
    return True
