"""Retitle desktop app: local web UI (opens in the browser) over the rt engine."""
import os, sys, json, threading, traceback, webbrowser, socket, subprocess, glob, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import cv2, numpy as np
from . import vio

UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
CFG_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/.config"), "Retitle")
CFG = os.path.join(CFG_DIR, "settings.json")
STATE = {"proj": None, "job": None, "log": [], "running": False, "error": None}


def settings():
    return vio.load_json(CFG, {"api_key": "", "model": "claude-sonnet-5-5", "glossary": "",
                               "src_lang": "eng", "whisper_model": "large-v3", "whisper_lang": ""})


def log(msg):
    STATE["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
    STATE["log"] = STATE["log"][-300:]


def P(*a):
    return os.path.join(STATE["proj"], *a)


# ------------------------------------------------------------------ jobs
def thumbs():
    T = vio.load_json(P("titles.json"))
    os.makedirs(P("thumbs"), exist_ok=True)
    for t in T["titles"]:
        img = None
        if "region" in t:
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


def job_analyze():
    from . import shots, detect, extract
    s = settings()
    log("מחפש חיתוכי שוטים...")
    c = shots.run(STATE["proj"])
    log(f"{len(c)} שוטים")
    log("מזהה טקסט על המסך...")
    ts = detect.run(STATE["proj"], s.get("src_lang") or "eng")
    log(f"נמצאו {len(ts)} שורות טקסט")
    log("מודד צבע, צל ותזמון מדויק...")
    extract.run(STATE["proj"])
    thumbs()
    log("הניתוח הסתיים. סמן מה לתרגם והקלד עברית.")


def job_build():
    from . import clean, render, extract
    from .__main__ import preview
    log("מוחק את הטקסט המקורי ומשחזר רקע...")
    clean.run(STATE["proj"])
    log("מודד את אנימציית המקור...")
    render.track(STATE["proj"])
    log("מרנדר עברית...")
    render.build_layers(STATE["proj"])
    T = vio.load_json(P("titles.json"))
    for gid, lines in extract.groups(T).items():
        if any(t.get("translate", True) and "region" in t for t in lines):
            preview(STATE["proj"], gid, [])
    log("תצוגה מקדימה מוכנה.")


def job_layers():
    from . import render, extract
    from .__main__ import preview
    render.build_layers(STATE["proj"])
    T = vio.load_json(P("titles.json"))
    for gid, lines in extract.groups(T).items():
        if any(t.get("translate", True) and "region" in t for t in lines) and os.path.isdir(P("clean", gid)):
            preview(STATE["proj"], gid, [])
    log("תצוגה מקדימה עודכנה.")


def job_export():
    from . import compose, check
    log("מקודד את הסרטון...")
    out = compose.run(STATE["proj"])
    log("בודק כל כותרת לפני ואחרי...")
    check.run(STATE["proj"])
    log(f"מוכן: {out}")


def job_transcribe():
    from . import srt
    s = settings()
    PJ = vio.load_json(P("project.json"))
    log("מתמלל עם Whisper (בפעם הראשונה יורד מודל, זה לוקח זמן)...")
    srt.transcribe(PJ["video"], P("whisper.json"), s.get("whisper_lang") or None, s.get("whisper_model", "large-v3"))
    srt.prep(STATE["proj"], P("whisper.json"))
    log("התמלול מוכן. הקלד עברית לכל כתובית.")


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
        res = translate.translate(items, s["api_key"], s.get("glossary", ""), "on-screen trailer titles (taglines, title cards)",
                                  s.get("model"), images=imgs)
        for t in T["titles"]:
            if t["id"] in res and not t.get("he"):
                t["he"] = res[t["id"]]
        vio.save_json(P("titles.json"), T)
    else:
        D = vio.load_json(P("dialog.json"))
        items = [{"id": str(c["n"]), "src": c["src"]} for c in D["cues"] if not c.get("he")]
        log(f"מתרגם {len(items)} כתוביות...")
        for i in range(0, len(items), 60):
            res = translate.translate(items[i:i + 60], s["api_key"], s.get("glossary", ""),
                                      "trailer dialogue subtitles, max 2 lines of 42 chars", s.get("model"))
            for c in D["cues"]:
                if str(c["n"]) in res and not c.get("he"):
                    c["he"] = res[str(c["n"])]
        vio.save_json(P("dialog.json"), D)
    log("התרגום הושלם. עבור ותקן.")


JOBS = {"analyze": job_analyze, "build": job_build, "layers": job_layers, "export": job_export,
        "transcribe": job_transcribe, "srt": job_srt,
        "translate_titles": lambda: job_translate("titles"), "translate_dialog": lambda: job_translate("dialog")}


def start(name):
    if STATE["running"]:
        return False
    STATE.update(running=True, job=name, error=None)

    def run():
        try:
            JOBS[name]()
        except Exception as e:
            STATE["error"] = str(e)
            log("שגיאה: " + str(e))
            traceback.print_exc()
        finally:
            STATE["running"] = False
    threading.Thread(target=run, daemon=True).start()
    return True


def pick_file():
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
    if not os.path.exists(P("project.json")):
        vio.save_json(P("project.json"), {"video": os.path.abspath(video), "info": vio.probe(video)})
    STATE["log"] = []
    log(f"פרויקט: {proj}")
    return proj


# ------------------------------------------------------------------ http
class H(BaseHTTPRequestHandler):
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

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(UI, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/api/status":
            return self._send(200, {"proj": STATE["proj"], "running": STATE["running"], "job": STATE["job"],
                                    "error": STATE["error"], "log": STATE["log"]})
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
            outs = sorted(glob.glob(P("out", "*")))
            prev = sorted(os.path.basename(x) for x in glob.glob(P("preview_*.jpg")))
            checks = sorted(os.path.basename(x) for x in glob.glob(P("check", "*.jpg")))
            return self._send(200, {"project": PJ, "titles": vio.load_json(P("titles.json"), {"titles": []}),
                                    "dialog": vio.load_json(P("dialog.json"), {"cues": []}),
                                    "outputs": outs, "previews": prev, "checks": checks})
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
            T = vio.load_json(P("titles.json"))
            byid = {t["id"]: t for t in T["titles"]}
            for e in b.get("titles", []):
                t = byid.get(e["id"])
                if not t:
                    continue
                for k in ("he", "translate"):
                    if k in e:
                        t[k] = e[k]
                if "style" in e:
                    t.setdefault("style", {}).update(e["style"])
                    t["style_locked"] = sorted(set(t.get("style_locked", [])) | set(e["style"]))
                for k in ("in_override", "out_override"):
                    if e.get(k) not in (None, ""):
                        t[k] = int(e[k])
            vio.save_json(P("titles.json"), T)
            return self._send(200, {"ok": True})
        if u.path == "/api/dialog":
            D = vio.load_json(P("dialog.json"), {"cues": []})
            m = {str(c["n"]): c for c in D["cues"]}
            for e in b.get("cues", []):
                if str(e["n"]) in m:
                    m[str(e["n"])]["he"] = e["he"]
            vio.save_json(P("dialog.json"), D)
            return self._send(200, {"ok": True})
        if u.path == "/api/reveal":
            target = P("out")
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
    port = free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = f"http://127.0.0.1:{port}/"
    print("Retitle:", url)
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
