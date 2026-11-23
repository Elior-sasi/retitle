"""Retitle desktop app (Hebrew video localization): editor UI (native window via pywebview, browser fallback) over the rt engine."""
import os, sys, json, threading, traceback, webbrowser, socket, subprocess, glob, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import cv2, numpy as np
from . import vio

UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
CFG_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/.config"), "Retitle")
CFG = os.path.join(CFG_DIR, "settings.json")
STATE = {"proj": None, "job": None, "log": [], "running": False, "error": None, "progress": None}
WINDOW = None
LOCK = threading.Lock()          # titles.json / dialog.json writes
FRAMES = {}                      # decoded source frames (small LRU)
LAYERS = {}                      # id -> (mtime, premultiplied layer)


def settings():
    s = {"api_key": "", "model": "claude-sonnet-5-5", "glossary": "", "src_lang": "eng", "content": "trailer",
         "whisper_model": "large-v3", "whisper_lang": ""}
    s.update(vio.load_json(CFG, {}))
    return s


def log(msg):
    STATE["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
    STATE["log"] = STATE["log"][-300:]


def P(*a):
    return os.path.join(STATE["proj"], *a)


def prog(x):
    STATE["progress"] = None if x is None else round(float(x), 3)


# ------------------------------------------------------------------ helpers
def thumbs():
    T = vio.load_json(P("titles.json"))
    os.makedirs(P("thumbs"), exist_ok=True)
    for t in T["titles"]:
        img = None
        if "region" in t and "cache" in t:
            img = cv2.imread(P("cache", t["group"], f"{t['ref']:06d}.png"))
            reg = t["region"]
            if img is not None:
                bx = [t["ref_box"][0] - reg[0], t["ref_box"][1] - reg[1], t["ref_box"][2] - reg[0], t["ref_box"][3] - reg[1]]
        if img is None:
            img = cv2.imread(P("samples", f"{t['ref']:06d}.jpg"))
            bx = t["ref_box"]
        if img is None:
            continue
        h = bx[3] - bx[1]
        c = img[max(0, bx[1] - h // 2):bx[3] + h // 2, max(0, bx[0] - h // 2):bx[2] + h // 2]
        cv2.imwrite(P("thumbs", f"{t['id']}.jpg"), c)


def dirty_groups(T):
    """Cards whose background must be rebuilt (timing or translate flag changed since the last clean)."""
    from .extract import groups
    out = set()
    for gid, lines in groups(T).items():
        for t in lines:
            want = [t["in"], t["out"], bool(t.get("translate", True))]
            if t.get("translate", True) and t.get("built") != want:
                out.add(gid)
            if not t.get("translate", True) and (t.get("built") or [0, 0, False])[2]:
                out.add(gid)
    return out


def needs_extract(T):
    for t in T["titles"]:
        if not t.get("translate", True):
            continue
        if "mask" not in t or "cache" not in t:
            return True
        if t["in"] < t["cache"][0] or t["out"] > t["cache"][1]:
            return True
    return False


# ------------------------------------------------------------------ jobs
def job_analyze():
    from . import shots, detect, extract
    s = settings()
    log("מחפש חיתוכי שוטים...")
    c = shots.run(STATE["proj"], progress=prog)
    prog(None)
    log(f"{len(c)} שוטים")
    log("מזהה טקסט על המסך...")
    ts = detect.run(STATE["proj"], s.get("src_lang") or "eng", progress=prog)
    prog(None)
    log(f"נמצאו {len(ts)} שורות טקסט")
    log("מודד צבע, צל ותזמון מדויק...")
    extract.run(STATE["proj"])
    thumbs()
    log("הניתוח הסתיים. בחר כותרת ברשימה והקלד עברית.")


def job_build(proxy=True):
    from . import clean, render, extract, compose
    proj = STATE["proj"]
    T = vio.load_json(P("titles.json"))
    if needs_extract(T):
        log("התזמון השתנה מעבר לטווח שנשמר - שומר פריימים מחדש...")
        extract.run(proj)
        thumbs()
        T = vio.load_json(P("titles.json"))
    dirty = dirty_groups(T)
    if dirty:
        log(f"מוחק את המקור ומשחזר רקע ({len(dirty)} כרטיסים)...")
        clean.run(proj, sorted(dirty))
        log("מודד את אנימציית המקור...")
        render.track(proj, only=dirty)
    log("מרנדר עברית...")
    render.build_layers(proj)
    if proxy:
        log("מכין תצוגת ניגון...")
        compose.run(proj, out_name="../proxy.mp4", crf=30, preset="ultrafast", scale=960, progress=prog)
        prog(None)
    log("מוכן. אפשר לנגן במצב תוצאה.")


def job_export():
    from . import compose, check, srt
    T = vio.load_json(P("titles.json"))
    if dirty_groups(T) or needs_extract(T) or not os.path.exists(P("layers", "meta.json")):
        job_build(proxy=False)
    log("מקודד את הסרטון הסופי...")
    out = compose.run(STATE["proj"], progress=prog)
    prog(None)
    log("בודק כל כותרת לפני ואחרי...")
    check.run(STATE["proj"])
    D = vio.load_json(P("dialog.json"), {"cues": []})
    if any(c.get("he") for c in D["cues"]):
        p, w = srt.build(STATE["proj"])
        for x in w:
            log("אזהרת כתוביות: " + x)
        log(f"SRT: {p}")
    log(f"הסרטון מוכן: {out}")


def job_transcribe():
    from . import srt
    s = settings()
    PJ = vio.load_json(P("project.json"))
    log("מתמלל עם Whisper (בפעם הראשונה יורד מודל, זה לוקח זמן)...")
    srt.transcribe(PJ["video"], P("whisper.json"), s.get("whisper_lang") or None, s.get("whisper_model", "large-v3"))
    srt.prep(STATE["proj"], P("whisper.json"))
    log("התמלול מוכן. בחר כתובית והקלד עברית.")


def job_srt():
    from . import srt
    p, w = srt.build(STATE["proj"])
    for x in w:
        log("אזהרה: " + x)
    log(f"SRT מוכן: {p}")


def job_translate(kind):
    from . import translate
    s = settings()
    if not s.get("api_key"):
        raise RuntimeError("אין מפתח API בהגדרות")
    if kind == "titles":
        T = vio.load_json(P("titles.json"))
        items = [{"id": t["id"], "src": t["src"], "group": t["group"]} for t in T["titles"]
                 if t.get("translate", True) and not t.get("he")]
        log(f"מתרגם {len(items)} כותרות...")
        imgs = {it["id"]: P("thumbs", f"{it['id']}.jpg") for it in items if os.path.exists(P("thumbs", f"{it['id']}.jpg"))}
        res = translate.translate(items, s["api_key"], s.get("glossary", ""),
                                  translate.context(s.get("content"), "titles"), s.get("model"), images=imgs)
        with LOCK:
            T = vio.load_json(P("titles.json"))
            for t in T["titles"]:
                if t["id"] in res and not t.get("he"):
                    t["he"] = res[t["id"]]
            vio.save_json(P("titles.json"), T)
        if os.path.exists(P("layers", "meta.json")):
            from . import render
            render.build_layers(STATE["proj"], only=set(res))
    else:
        D = vio.load_json(P("dialog.json"))
        items = [{"id": str(c["n"]), "src": c["src"]} for c in D["cues"] if not c.get("he")]
        log(f"מתרגם {len(items)} כתוביות...")
        res = {}
        for i in range(0, len(items), 60):
            res.update(translate.translate(items[i:i + 60], s["api_key"], s.get("glossary", ""),
                                           translate.context(s.get("content"), "dialog"), s.get("model")))
        with LOCK:
            D = vio.load_json(P("dialog.json"))
            for c in D["cues"]:
                if str(c["n"]) in res and not c.get("he"):
                    c["he"] = res[str(c["n"])]
            vio.save_json(P("dialog.json"), D)
    log("התרגום הושלם. עבור ותקן.")


JOBS = {"analyze": job_analyze, "build": job_build, "export": job_export,
        "transcribe": job_transcribe, "srt": job_srt,
        "translate_titles": lambda: job_translate("titles"), "translate_dialog": lambda: job_translate("dialog")}


def start(name):
    if STATE["running"]:
        return False
    STATE.update(running=True, job=name, error=None, progress=None)

    def run():
        try:
            JOBS[name]()
        except Exception as e:
            STATE["error"] = str(e)
            log("שגיאה: " + str(e))
            traceback.print_exc()
        finally:
            STATE["running"] = False
            STATE["progress"] = None
    threading.Thread(target=run, daemon=True).start()
    return True


def pick_file():
    if WINDOW is not None:
        import webview
        r = WINDOW.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False,
                                      file_types=("Video (*.mp4;*.mov;*.mkv;*.m4v;*.avi;*.webm)", "All files (*.*)"))
        return r[0] if r else ""
    import tkinter as tk
    from tkinter import filedialog
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilename(title="בחר סרטון", filetypes=[("Video", "*.mp4 *.mov *.mkv *.m4v *.avi *.webm"), ("All", "*.*")])
    r.destroy()
    return p


def open_project(video):
    proj = os.path.splitext(video)[0] + "_retitle"
    os.makedirs(proj, exist_ok=True)
    STATE["proj"] = proj
    FRAMES.clear(); LAYERS.clear()
    if not os.path.exists(P("project.json")):
        vio.save_json(P("project.json"), {"video": os.path.abspath(video), "info": vio.probe(video)})
    STATE["log"] = []
    log(f"פרויקט: {proj}")
    return proj


# ------------------------------------------------------------------ live edits
STYLE_KEYS = {"font", "weight", "size_ratio", "fill", "align", "nudge_x", "nudge_y"}


def apply_titles(edits):
    """Apply editor state for titles. Returns ids whose Hebrew layer must be re-rendered."""
    with LOCK:
        T = vio.load_json(P("titles.json"))
        byid = {t["id"]: t for t in T["titles"]}
        relayer = set()
        for e in edits:
            t = byid.get(e.get("id"))
            if not t:
                continue
            if "he" in e and e["he"] != t.get("he", ""):
                t["he"] = e["he"]; relayer.add(t["id"])
            if "translate" in e and bool(e["translate"]) != t.get("translate", True):
                t["translate"] = bool(e["translate"]); relayer.add(t["id"])
            st = e.get("style") or {}
            cur = t.setdefault("style", {})
            ch = {k: v for k, v in st.items() if k in STYLE_KEYS and cur.get(k) != v}
            if ch:
                cur.update(ch); relayer.add(t["id"])
                t["style_locked"] = sorted(set(t.get("style_locked", [])) | set(ch))
            for k in ("in", "out"):
                if k in e and e[k] is not None and int(e[k]) != t.get(k):
                    t[k] = int(e[k]); t[k + "_override"] = int(e[k])
        vio.save_json(P("titles.json"), T)
    if relayer and os.path.exists(P("layers", "meta.json")):
        from . import render
        render.build_layers(STATE["proj"], only=relayer)
    return relayer


def apply_dialog(cues):
    with LOCK:
        D = vio.load_json(P("dialog.json"), {"cues": []})
        m = {str(c["n"]): c for c in D["cues"]}
        for e in cues:
            c = m.get(str(e["n"]))
            if not c:
                continue
            if "he" in e:
                c["he"] = e["he"]
            for k in ("start", "end"):
                if k in e and e[k] is not None and abs(float(e[k]) - c[k]) > 1e-4:
                    c[k] = round(float(e[k]), 4); c["manual"] = True
        vio.save_json(P("dialog.json"), D)


def render_frame(f, mode, width):
    from . import compose, render
    PJ = vio.load_json(P("project.json"))
    info = PJ["info"]; W, H = info["width"], info["height"]
    if f not in FRAMES:
        fr = vio.grab(PJ["video"], f, info)
        if fr is None:
            return None
        if len(FRAMES) > 40:
            FRAMES.pop(next(iter(FRAMES)))
        FRAMES[f] = fr
    fr = FRAMES[f]
    if mode == "result" and os.path.exists(P("titles.json")):
        T = vio.load_json(P("titles.json"))
        meta = vio.load_json(P("layers", "meta.json"), {})
        cache = {}
        for tid, m in meta.items():
            if not m.get("file") or not os.path.exists(m["file"]):
                continue
            mt = os.path.getmtime(m["file"])
            if LAYERS.get(tid, (None,))[0] != mt:
                LAYERS[tid] = (mt, render.load_layer(m["file"]))
            cache[tid] = LAYERS[tid][1]
        fr = compose.frame_result(STATE["proj"], f, fr, T, meta, cache, W, H)
    if width and width < W:
        fr = cv2.resize(fr, (width, int(H * width / W)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpg.tobytes()


# ------------------------------------------------------------------ http
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _video(self, path):
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        a, b = 0, size - 1
        if rng and rng.startswith("bytes="):
            s, _, e = rng[6:].partition("-")
            if s:
                a = int(s); b = int(e) if e else size - 1
            else:
                a = size - int(e)
        b = min(b, size - 1, a + 8 * 1024 * 1024 - 1)
        self.send_response(206 if rng else 200)
        ext = os.path.splitext(path)[1].lower()
        self.send_header("Content-Type", {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
                                          ".webm": "video/webm", ".mkv": "video/x-matroska"}.get(ext, "video/mp4"))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(b - a + 1))
        if rng:
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(a)
            left = b - a + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return
                left -= len(chunk)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(UI, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if u.path == "/api/status":
            return self._send(200, {"proj": STATE["proj"], "running": STATE["running"], "job": STATE["job"],
                                    "error": STATE["error"], "log": STATE["log"], "progress": STATE["progress"]})
        if u.path == "/api/settings":
            s = settings(); s["api_key_set"] = bool(s.get("api_key")); s["api_key"] = ""
            return self._send(200, s)
        if u.path == "/api/fonts":
            from . import render
            return self._send(200, vio.load_json(os.path.join(render.FONT_DIR, "catalog.json"), {}))
        if not STATE["proj"]:
            return self._send(400, {"error": "no project"})
        if u.path == "/api/project":
            PJ = vio.load_json(P("project.json"))
            T = vio.load_json(P("titles.json"), {"titles": []})
            S = vio.load_json(P("shots.json"), {"cuts": [], "n_frames": 0})
            px = P("proxy.mp4")
            return self._send(200, {
                "project": PJ, "titles": T, "dialog": vio.load_json(P("dialog.json"), {"cues": []}),
                "cuts": S.get("cuts", []),
                "n_frames": PJ.get("n_frames") or S.get("n_frames") or int(round(PJ["info"]["duration"] * PJ["info"]["fps"])),
                "dirty": sorted(dirty_groups(T)) if T["titles"] and any("built" in t for t in T["titles"]) else [],
                "built": os.path.exists(P("layers", "meta.json")),
                "proxy": os.path.exists(px), "proxy_mtime": os.path.getmtime(px) if os.path.exists(px) else 0,
                "outputs": sorted(glob.glob(P("out", "*"))),
                "checks": sorted(os.path.basename(x) for x in glob.glob(P("check", "*.jpg"))),
                "stats": vio.load_json(P("clean_stats.json"), {})})
        if u.path == "/api/video":
            which = q.get("which", ["src"])[0]
            path = P("proxy.mp4") if which == "proxy" else vio.load_json(P("project.json"))["video"]
            if not os.path.exists(path):
                return self._send(404, {"error": "no video"})
            return self._video(path)
        if u.path == "/api/frame":
            f = int(q.get("f", ["0"])[0]); mode = q.get("mode", ["result"])[0]; w = int(q.get("w", ["0"])[0])
            try:
                jpg = render_frame(f, mode, w)
            except Exception as e:
                traceback.print_exc()
                return self._send(500, {"error": str(e)})
            return self._send(200, jpg, "image/jpeg") if jpg else self._send(404, {"error": "frame"})
        if u.path == "/api/file":
            rel = unquote(q.get("p", [""])[0])
            full = os.path.normpath(os.path.join(STATE["proj"], rel))
            if not full.startswith(os.path.normpath(STATE["proj"])) or not os.path.isfile(full):
                return self._send(404, {"error": "not found"})
            ct = "image/jpeg" if full.endswith(".jpg") else "image/png" if full.endswith(".png") else "application/octet-stream"
            with open(full, "rb") as f:
                return self._send(200, f.read(), ct)
        self._send(404, {"error": "?"})

    def do_POST(self):
        u = urlparse(self.path)
        b = self._body()
        if u.path == "/api/pick":
            p = pick_file()
            if p:
                open_project(p)
            return self._send(200, {"proj": STATE["proj"]})
        if u.path == "/api/open":
            return self._send(200, {"proj": open_project(b["video"])})
        if u.path == "/api/settings":
            s = settings()
            for k, v in b.items():
                if k == "api_key" and not v:
                    continue
                s[k] = v
            os.makedirs(CFG_DIR, exist_ok=True)
            vio.save_json(CFG, s)
            return self._send(200, {"ok": True})
        if u.path == "/api/run":
            return self._send(200, {"started": start(b["job"])})
        if not STATE["proj"]:
            return self._send(400, {"error": "no project"})
        if u.path == "/api/titles":
            try:
                r = apply_titles(b.get("titles", []))
            except Exception as e:
                traceback.print_exc()
                return self._send(500, {"error": str(e)})
            return self._send(200, {"ok": True, "relayered": sorted(r)})
        if u.path == "/api/dialog":
            apply_dialog(b.get("cues", []))
            return self._send(200, {"ok": True})
        if u.path == "/api/reveal":
            target = P("out") if os.path.isdir(P("out")) else STATE["proj"]
            if sys.platform.startswith("win"):
                os.startfile(target)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                subprocess.Popen(["xdg-open", target])
            return self._send(200, {"ok": True})
        self._send(404, {"error": "?"})


def free_port(start=8765):
    for p in range(start, start + 50):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p)); return p
            except OSError:
                continue
    return 0


def main():
    global WINDOW
    port = free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print("Retitle:", url)
    if "--server" in sys.argv or "--no-browser" in sys.argv:
        srv.serve_forever(); return
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    if "--browser" not in sys.argv:
        try:
            import webview
            WINDOW = webview.create_window("Retitle", url, width=1600, height=980, min_size=(1100, 700),
                                           background_color="#101114")
            print("window: pywebview", flush=True)
            webview.start()
            return
        except Exception:
            traceback.print_exc()
            print("window: fallback to browser", flush=True)
            WINDOW = None
    webbrowser.open(url)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
