"""Self-update from GitHub releases (Windows exe). Private repo: needs a read-only token in settings."""
import os, sys, json, zipfile, tempfile, subprocess, urllib.request
from . import vio

REPO = "Elior-sasi/retitle"
try:
    from ._version import BUILD
except Exception:
    BUILD = 0


def _get(url, token, accept="application/vnd.github+json"):
    h = {"Accept": accept, "User-Agent": "Retitle"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60)


def check(token=""):
    with _get(f"https://api.github.com/repos/{REPO}/releases/latest", token) as r:
        rel = json.loads(r.read().decode())
    tag = rel.get("tag_name", "")
    n = int(tag.split("-")[-1]) if tag.split("-")[-1].isdigit() else 0
    asset = next((a for a in rel.get("assets", []) if a["name"].endswith(".zip")), None)
    return {"current": BUILD, "latest": n, "newer": n > BUILD, "tag": tag,
            "asset_api": asset["url"] if asset else None, "size": asset["size"] if asset else 0,
            "notes": rel.get("name", "")}


def apply(info, token="", progress=None):
    """Download the zip, unpack next to the app, and hand over to a small script that waits for
    this process to exit, swaps the files and starts the new version. Settings live in %APPDATA%."""
    if not getattr(sys, "frozen", False):
        raise RuntimeError("עדכון עצמי עובד רק בגרסת ה-exe")
    app_dir = os.path.dirname(sys.executable)
    tmp = tempfile.mkdtemp(prefix="retitle_upd_")
    zp = os.path.join(tmp, "Retitle.zip")
    with _get(info["asset_api"], token, "application/octet-stream") as r, open(zp, "wb") as f:
        total = info.get("size") or 0; done = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b); done += len(b)
            if progress and total:
                progress(done / total)
    with zipfile.ZipFile(zp) as z:
        z.extractall(tmp)
    new_dir = os.path.join(tmp, "Retitle")
    if not os.path.isdir(new_dir):
        new_dir = tmp
    exe = os.path.join(app_dir, "Retitle.exe")
    cmd = os.path.join(tmp, "update.cmd")
    with open(cmd, "w", encoding="utf-8") as f:
        f.write(f"""@echo off
chcp 65001 >nul
:wait
tasklist /FI "PID eq {os.getpid()}" | find "{os.getpid()}" >nul && (timeout /t 1 /nobreak >nul & goto wait)
robocopy "{new_dir}" "{app_dir}" /E /IS /IT /R:3 /W:1 /NFL /NDL /NJH /NJS >nul
start "" "{exe}"
""")
    subprocess.Popen(["cmd", "/c", cmd], creationflags=0x08000000 | 0x00000008)  # no window, detached
    return True
