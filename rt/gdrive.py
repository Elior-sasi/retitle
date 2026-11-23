"""Backup to Google Drive, signing in with Google from inside Retitle (like Unbox).

What is backed up: series (with their logos), the name glossary and the shared preferences, and the fonts you
imported. It goes to the app's hidden folder in your Drive (appDataFolder: only Retitle can see it), as one
zip that Drive keeps revisions of. Your videos and projects are not uploaded.

Sign-in: OAuth for installed apps - the browser opens Google's page, Google sends the answer back to a
one-time local address (127.0.0.1) with PKCE. The client id comes from the build (rt/_google.py, written by CI
from the repo secrets GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET; for a desktop client the "secret" is not secret).
The refresh token is kept only in this computer's settings.
"""
import os, io, json, time, base64, hashlib, secrets, zipfile, threading, webbrowser, urllib.request, urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from . import vio

SCOPES = "https://www.googleapis.com/auth/drive.appdata openid email"
NAME = "retitle-backup.zip"
STATE = {"busy": False, "last": "", "error": ""}


def client():
    try:
        from ._google import CLIENT_ID, CLIENT_SECRET
        return CLIENT_ID, CLIENT_SECRET
    except Exception:
        return os.environ.get("RETITLE_GOOGLE_ID", ""), os.environ.get("RETITLE_GOOGLE_SECRET", "")


def configured():
    return bool(client()[0])


def _cfg_dir():
    return os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/.config"), "Retitle")


def _settings():
    return vio.load_json(os.path.join(_cfg_dir(), "settings.json"), {})


def _save_settings(upd):
    s = _settings(); s.update(upd)
    os.makedirs(_cfg_dir(), exist_ok=True)
    vio.save_json(os.path.join(_cfg_dir(), "settings.json"), s)


def _post(url, data):
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


# ------------------------------------------------------------------ sign in
def sign_in(timeout=300):
    cid, csec = client()
    if not cid:
        raise RuntimeError("הגיבוי לגוגל עוד לא הוגדר בגרסה הזאת")
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)
    got = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            if q.get("state", [""])[0] == state:
                got.update({k: v[0] for k, v in q.items()})
            ok = "code" in got
            body = ("<html dir=rtl lang=he><body style='font:20px sans-serif;background:#111;color:#eee;padding:40px'>"
                    + ("<h1>התחברת ל-Google</h1><p>אפשר לסגור את החלון ולחזור ל-Retitle.</p>" if ok else
                       "<h1>ההתחברות לא הושלמה</h1><p>חזור ל-Retitle ונסה שוב.</p>") + "</body></html>").encode()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    srv = HTTPServer(("127.0.0.1", 0), H)
    redirect = f"http://127.0.0.1:{srv.server_address[1]}"
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
        "client_id": cid, "redirect_uri": redirect, "response_type": "code", "scope": SCOPES,
        "code_challenge": challenge, "code_challenge_method": "S256", "state": state,
        "access_type": "offline", "prompt": "consent"})
    webbrowser.open(url)
    srv.timeout = 1
    t0 = time.time()
    while "code" not in got and "error" not in got and time.time() - t0 < timeout:
        srv.handle_request()
    srv.server_close()
    if "code" not in got:
        raise RuntimeError("ההתחברות בוטלה או לא הושלמה")
    tok = _post("https://oauth2.googleapis.com/token", {
        "code": got["code"], "client_id": cid, "client_secret": csec, "redirect_uri": redirect,
        "grant_type": "authorization_code", "code_verifier": verifier})
    email = ""
    if tok.get("id_token"):
        try:
            p = tok["id_token"].split(".")[1]; p += "=" * (-len(p) % 4)
            email = json.loads(base64.urlsafe_b64decode(p)).get("email", "")
        except Exception:
            pass
    if not tok.get("refresh_token"):
        raise RuntimeError("Google לא החזיר הרשאה קבועה - נסה שוב")
    _save_settings({"g_refresh": tok["refresh_token"], "g_email": email})
    return email


def sign_out():
    s = _settings()
    rt = s.get("g_refresh")
    if rt:
        try:
            _post("https://oauth2.googleapis.com/revoke", {"token": rt})
        except Exception:
            pass
    _save_settings({"g_refresh": "", "g_email": "", "g_last_push": "", "g_last_pull": "", "g_hash": ""})


def _access():
    cid, csec = client()
    rt = _settings().get("g_refresh")
    if not rt:
        raise RuntimeError("לא מחובר ל-Google")
    return _post("https://oauth2.googleapis.com/token", {
        "client_id": cid, "client_secret": csec, "refresh_token": rt, "grant_type": "refresh_token"})["access_token"]


def _api(method, url, token, data=None, ctype=None):
    h = {"Authorization": "Bearer " + token}
    if ctype:
        h["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    with urllib.request.urlopen(req, timeout=120) as r:
        b = r.read()
        return json.loads(b) if b and r.headers.get("Content-Type", "").startswith("application/json") else b


def _remote(token):
    q = urllib.parse.urlencode({"spaces": "appDataFolder", "q": f"name='{NAME}'", "fields": "files(id,modifiedTime,size)"})
    fs = _api("GET", "https://www.googleapis.com/drive/v3/files?" + q, token).get("files", [])
    return fs[0] if fs else None


# ------------------------------------------------------------------ what is backed up
def _sources():
    """(path in zip, path on disk) for everything that is backed up."""
    from . import series, fontlib
    out = []
    for base, prefix in ((series.root(), "series"), (os.path.join(fontlib.user_dir(), "mine"), "fonts")):
        if os.path.isdir(base):
            for root, _, files in os.walk(base):
                for f in files:
                    if f.endswith((".tmp", ".part")):
                        continue
                    p = os.path.join(root, f)
                    out.append((prefix + "/" + os.path.relpath(p, base).replace("\\", "/"), p))
    return out


def _shared():
    from .app import SHARED, settings
    s = settings()
    return {k: s[k] for k in SHARED if k in s}


def snapshot_hash():
    h = hashlib.sha1(json.dumps(_shared(), sort_keys=True, ensure_ascii=False).encode())
    for name, p in sorted(_sources()):
        st = os.stat(p)
        h.update(f"{name}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()


def push(force=False):
    """Upload the backup if something changed since the last one."""
    hsh = snapshot_hash()
    if not force and hsh == _settings().get("g_hash"):
        return False
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("settings.json", json.dumps(_shared(), ensure_ascii=False, indent=1))
        for name, p in _sources():
            z.write(p, name)
    data = buf.getvalue()
    token = _access()
    cur = _remote(token)
    bnd = "retitle" + secrets.token_hex(8)
    meta = {"name": NAME} if cur else {"name": NAME, "parents": ["appDataFolder"]}
    body = (f"--{bnd}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{json.dumps(meta)}\r\n"
            f"--{bnd}\r\nContent-Type: application/zip\r\n\r\n").encode() + data + f"\r\n--{bnd}--".encode()
    if cur:
        r = _api("PATCH", f"https://www.googleapis.com/upload/drive/v3/files/{cur['id']}?uploadType=multipart&fields=id,modifiedTime",
                 token, body, f"multipart/related; boundary={bnd}")
    else:
        r = _api("POST", "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,modifiedTime",
                 token, body, f"multipart/related; boundary={bnd}")
    _save_settings({"g_hash": hsh, "g_last_push": r.get("modifiedTime", ""), "g_last_pull": r.get("modifiedTime", "")})
    STATE["last"] = time.strftime("%H:%M")
    return True


def pull(force=False):
    """Bring the backup into this computer: newer files win; settings from the backup when it is newer."""
    from . import series, fontlib
    token = _access()
    cur = _remote(token)
    if not cur:
        return False
    s = _settings()
    if not force and cur["modifiedTime"] <= (s.get("g_last_pull") or ""):
        return False
    data = _api("GET", f"https://www.googleapis.com/drive/v3/files/{cur['id']}?alt=media", token)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            n = info.filename
            if n == "settings.json":
                sh = json.loads(z.read(n).decode("utf-8"))
                from .app import SHARED, sync_root
                _save_settings({k: v for k, v in sh.items() if k in SHARED})
                r = sync_root()
                if r:
                    vio.save_json(os.path.join(r, "settings.json"), {k: v for k, v in sh.items() if k in SHARED})
                continue
            if ".." in n.split("/"):
                continue
            base = series.root() if n.startswith("series/") else os.path.join(fontlib.user_dir(), "mine") if n.startswith("fonts/") else None
            if not base:
                continue
            dst = os.path.join(base, *n.split("/")[1:])
            ts = time.mktime(info.date_time + (0, 0, -1))
            if os.path.exists(dst) and os.path.getmtime(dst) >= ts - 2:
                continue                                  # the copy here is the same or newer
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(z.read(n))
            os.utime(dst, (ts, ts))
    _save_settings({"g_last_pull": cur["modifiedTime"], "g_hash": snapshot_hash()})
    return True


def status():
    s = _settings()
    return {"configured": configured(), "email": s.get("g_email", ""), "signed_in": bool(s.get("g_refresh")),
            "last_push": s.get("g_last_push", ""), "busy": STATE["busy"], "error": STATE["error"]}


def auto_loop(stop=None):
    """Background: get the backup at start (another computer may have added to it), then back up every
    5 minutes when something changed."""
    first = True
    while True:
        try:
            if _settings().get("g_refresh") and configured():
                STATE["busy"] = True
                if first:
                    pull()
                push()
                STATE["error"] = ""
        except Exception as e:
            STATE["error"] = str(e)
        finally:
            STATE["busy"] = False
        first = False
        for _ in range(300):
            if stop and stop.is_set():
                return
            time.sleep(1)
