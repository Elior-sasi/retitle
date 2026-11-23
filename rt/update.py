"""Self-update from GitHub releases (Windows exe). Private repo: needs a read-only token in settings."""
import os, sys, json, zipfile, tempfile, subprocess, urllib.request
from . import vio

REPO = "Elior-sasi/retitle"                 # private: needs a token
PUBLIC = "Elior-sasi/retitle-releases"      # public mirror of the releases: no token needed
try:
    from ._version import BUILD
except Exception:
    BUILD = 0


def _get(url, token, accept="application/vnd.github+json"):
    h = {"Accept": accept, "User-Agent": "Retitle"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60)


def _latest(token):
    errs = []
    for repo, tok in ((PUBLIC, ""), (REPO, token)):
        if repo == REPO and not tok and errs:
            pass
        try:
            with _get(f"https://api.github.com/repos/{repo}/releases/latest", tok) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            errs.append(f"{repo}: {e}")
    raise RuntimeError("; ".join(errs))


def check(token=""):
    rel = _latest(token)
    tag = rel.get("tag_name", "")
    n = int(tag.split("-")[-1]) if tag.split("-")[-1].isdigit() else 0
    assets = rel.get("assets", [])
    asset = next((a for a in assets if a["name"].lower().endswith("setup.exe")), None) or \
        next((a for a in assets if a["name"].endswith(".zip")), None)
    return {"current": BUILD, "latest": n, "newer": n > BUILD, "tag": tag,
            "asset_api": asset["url"] if asset else None, "size": asset["size"] if asset else 0,
            "asset_name": asset["name"] if asset else "",
            "notes": rel.get("name", "")}


def apply(info, token="", progress=None):
    """Download the new installer and run it silently; it closes this app, installs over it
    and starts the new version. Settings live in %APPDATA% and are kept."""
    if not getattr(sys, "frozen", False):
        raise RuntimeError("עדכון עצמי עובד רק בגרסה המותקנת")
    if not info.get("asset_api"):
        raise RuntimeError("לא נמצא קובץ התקנה בגרסה החדשה")
    tmp = tempfile.mkdtemp(prefix="retitle_upd_")
    dst = os.path.join(tmp, info.get("asset_name") or "Retitle-Setup.exe")
    with _get(info["asset_api"], token, "application/octet-stream") as r, open(dst, "wb") as f:
        total = info.get("size") or 0; done = 0
        while True:
            b = r.read(min(1 << 20, total - done) if total else 1 << 20)
            if not b:
                break
            f.write(b); done += len(b)
            if progress and total:
                progress(min(0.999, done / total))
            if total and done >= total:          # all of it: do not wait on a connection that stays open
                break
    if total and os.path.getsize(dst) < total:
        raise RuntimeError("ההורדה נקטעה - נסה שוב")
    if dst.lower().endswith(".exe"):
        subprocess.Popen([dst, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", "/SP-"],
                         creationflags=0x00000008)  # detached
        return True
    # portable zip fallback
    app_dir = os.path.dirname(sys.executable)
    with zipfile.ZipFile(dst) as z:
        z.extractall(tmp)
    new_dir = os.path.join(tmp, "Retitle") if os.path.isdir(os.path.join(tmp, "Retitle")) else tmp
    exe = os.path.join(app_dir, "Retitle.exe")
    cmd = os.path.join(tmp, "update.cmd")
    with open(cmd, "w", encoding="utf-8") as f:
        f.write(f"""@echo off
:wait
tasklist /FI "PID eq {os.getpid()}" | find "{os.getpid()}" >nul && (timeout /t 1 /nobreak >nul & goto wait)
robocopy "{new_dir}" "{app_dir}" /E /IS /IT /R:3 /W:1 /NFL /NDL /NJH /NJS >nul
start "" "{exe}"
""")
    subprocess.Popen(["cmd", "/c", cmd], creationflags=0x08000000 | 0x00000008)
    return True
