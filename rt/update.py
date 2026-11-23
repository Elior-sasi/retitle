"""Self-update from GitHub releases (Windows exe). Private repo: needs a read-only token in settings."""
import os, sys, json, zipfile, tempfile, subprocess, urllib.request, shutil
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
    z = next((a for a in assets if a["name"].endswith(".zip")), None)
    return {"current": BUILD, "latest": n, "newer": n > BUILD, "tag": tag,
            "zip_url": z.get("browser_download_url") if z else None, "zip_size": z["size"] if z else 0,
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
    try:                                          # only the files that changed (usually a few MB)
        if apply_delta(info, progress):
            return True
    except Exception as e:
        print("delta update failed, full installer:", e)
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
        subprocess.Popen([dst, "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS", "/SP-"],   # a small progress window, no questions
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


# ------------------------------------------------------------------ delta update
class HttpFile:
    """Read-only, seekable view of a remote file through HTTP Range requests (enough for zipfile)."""

    def __init__(self, url, size):
        self.url, self.size, self.pos, self.got = url, size, 0, 0
        self._cache = (0, b"")

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        c0, cb = self._cache
        if c0 <= self.pos and self.pos + n <= c0 + len(cb):
            b = cb[self.pos - c0:self.pos - c0 + n]
        else:
            want = max(n, 1 << 16)                 # small reads (headers) come in one request
            end = min(self.size, self.pos + want) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "Retitle"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise RuntimeError("server does not send parts of the file")
                data = r.read()
            self.got += len(data)
            self._cache = (self.pos, data)
            b = data[:n]
        self.pos += len(b)
        return b


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def apply_delta(info, progress=None):
    """Fetch only the changed files out of the release zip, then swap them in after this app exits
    and start it again. False when not possible (the caller uses the full installer)."""
    if not getattr(sys, "frozen", False) or not info.get("zip_url") or not info.get("zip_size"):
        return False
    app_dir = os.path.dirname(sys.executable)
    local = vio.load_json(os.path.join(app_dir, "manifest.json"))
    if not local:
        return False
    z = zipfile.ZipFile(HttpFile(info["zip_url"], info["zip_size"]))
    names = {n.replace("\\", "/"): n for n in z.namelist()}
    pre = next((n[:-len("manifest.json")] for n in names if n.endswith("manifest.json") and n.count("/") <= 1), None)
    if pre is None:
        return False
    remote = json.loads(z.read(names[pre + "manifest.json"]).decode())
    lf, rf = local["files"], remote["files"]
    # what is on disk now counts, not what the old manifest says
    need = [k for k, (sz, sh) in rf.items()
            if lf.get(k) != [sz, sh] or not os.path.exists(os.path.join(app_dir, k))]
    gone = [k for k in lf if k not in rf]
    total = sum(rf[k][0] for k in need)
    if total > 0.6 * sum(v[0] for v in rf.values()):
        return False                                # most of it changed: the installer is as quick
    tmp = tempfile.mkdtemp(prefix="retitle_delta_")
    stage = os.path.join(tmp, "new")
    done = 0
    for k in need:
        src = names.get(pre + k)
        if src is None:
            return False
        dst = os.path.join(stage, *k.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with z.open(src) as r, open(dst, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b); done += len(b)
                if progress and total:
                    progress(min(0.98, done / total))
        if _sha(dst) != rf[k][1]:
            raise RuntimeError("checksum " + k)
    with open(os.path.join(stage, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(remote, f)
    swap(stage, app_dir, need + ["manifest.json"], gone)
    # start the new version; it waits until this process is gone before it opens its window
    subprocess.Popen([os.path.join(app_dir, "Retitle.exe"), "--wait-pid", str(os.getpid())],
                     creationflags=0x00000008, close_fds=True)
    print(f"delta update: {len(need)} files, {round(total / 1e6, 1)} MB, {len(gone)} removed - swapped in place")
    shutil.rmtree(tmp, ignore_errors=True)
    return True


OLD = ".retitle-old"


def swap(stage, app_dir, files, gone):
    """Put the new files in place while this app is still running. Windows lets a running exe and
    loaded DLLs be renamed (not overwritten), so each old file is renamed aside first and the new
    one moved in. All or nothing: on any failure everything is put back."""
    done = []                                       # (target, aside or None)
    try:
        for k in files:
            src = os.path.join(stage, *k.split("/")); dst = os.path.join(app_dir, *k.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            aside = None
            if os.path.exists(dst):
                aside = dst + OLD
                if os.path.exists(aside):
                    os.remove(aside)
                os.replace(dst, aside)
            done.append((dst, aside))
            shutil.copy2(src, dst)
        for k in gone:
            dst = os.path.join(app_dir, *k.split("/"))
            if os.path.exists(dst):
                os.replace(dst, dst + OLD)
                done.append((None, dst + OLD))
    except Exception:
        for dst, aside in reversed(done):
            try:
                if dst and os.path.exists(dst):
                    os.remove(dst)
                if aside:
                    os.replace(aside, dst or aside[:-len(OLD)])
            except Exception:
                pass
        raise


def cleanup_old(app_dir=None):
    """At start: remove what the last update set aside (no longer in use now)."""
    app_dir = app_dir or os.path.dirname(sys.executable)
    for d, _, fs in os.walk(app_dir):
        for f in fs:
            if f.endswith(OLD):
                try:
                    os.remove(os.path.join(d, f))
                except OSError:
                    pass


def wait_pid(pid, timeout=60):
    """Block until process `pid` has exited (Windows), at most `timeout` seconds."""
    if os.name != "nt":
        return
    import ctypes
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x00100000, False, int(pid))      # SYNCHRONIZE
    if h:
        k.WaitForSingleObject(h, int(timeout * 1000))
        k.CloseHandle(h)
