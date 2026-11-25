"""Retitle desktop app (Hebrew video localization): editor UI (native window via pywebview, browser fallback) over the rt engine."""
import os, sys, json, threading, traceback, webbrowser, socket, subprocess, glob, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import cv2, numpy as np
from . import vio

UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
CFG_DIR = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/.config"), "Retitle")
CFG = os.path.join(CFG_DIR, "settings.json")
STATE = {"proj": None, "job": None, "log": [], "running": False, "error": None, "progress": None,
         "stage": "", "step": 0, "steps": 0, "t0": 0, "cancel": False, "cancelled": False}
RANGE = [0.0, 1.0]               # current stage's share of the whole job
PHASE = {"range": [0.0, 1.0], "name": ""}   # one-click / queue: the current part of the run
WINDOW = None
FULL = False
LOCK = threading.Lock()          # titles.json / dialog.json writes
FRAMES = {}                      # decoded source frames (small LRU)
LAYERS = {}                      # id -> (mtime, premultiplied layer)
INK = {}                         # layer file -> drawn area


# settings that travel between computers through the sync folder (secrets and recent files stay local)
SHARED = ("notify", "open_maximized", "glossary", "content", "src_lang", "model", "srt_rle", "sep_vocals", "whisper_model", "whisper_lang", "ui_fs")


def sync_root():
    d = vio.load_json(CFG, {}).get("sync_dir", "")
    return os.path.join(d, "Retitle") if d and os.path.isdir(d) else ""


def detect_sync_dirs():
    """Folders that a cloud client already syncs on this computer (free OneDrive / Google Drive / Dropbox)."""
    home = os.path.expanduser("~")
    c = [os.environ.get("OneDrive"), os.environ.get("OneDriveConsumer"), os.path.join(home, "OneDrive"),
         os.path.join(home, "Google Drive"), "G:\\My Drive", "G:\\האחסון שלי", os.path.join(home, "Dropbox")]
    out = []
    for x in c:
        if x and os.path.isdir(x) and os.path.normcase(x) not in [os.path.normcase(y) for y in out]:
            out.append(x)
    return out


def enable_sync(d):
    """Use folder d for series + shared settings. Local series are copied in (newer file wins)."""
    import shutil
    from . import series
    s = vio.load_json(CFG, {})
    old_series = series.root()
    s["sync_dir"] = d
    os.makedirs(CFG_DIR, exist_ok=True)
    vio.save_json(CFG, s)
    if not d:
        return
    root = os.path.join(d, "Retitle")
    os.makedirs(root, exist_ok=True)
    new_series = series.root()
    if os.path.isdir(old_series) and os.path.normcase(old_series) != os.path.normcase(new_series):
        for dirpath, _, files in os.walk(old_series):
            rel = os.path.relpath(dirpath, old_series)
            os.makedirs(os.path.join(new_series, rel), exist_ok=True)
            for f in files:
                a, b = os.path.join(dirpath, f), os.path.join(new_series, rel, f)
                if not os.path.exists(b) or os.path.getmtime(a) > os.path.getmtime(b):
                    shutil.copy2(a, b)
    sh = os.path.join(root, "settings.json")
    if not os.path.exists(sh):                     # first computer: publish its settings
        vio.save_json(sh, {k: s[k] for k in SHARED if k in s})


def settings():
    s = {"api_key": "", "model": "claude-sonnet-5-5", "glossary": "", "src_lang": "eng", "content": "trailer",
         "srt_rle": True, "sep_vocals": True, "gh_token": "", "open_maximized": True, "notify": True,
         "whisper_model": "large-v3", "whisper_lang": ""}
    s.update(vio.load_json(CFG, {}))
    if not s.get("asr_v3"):                    # one-time: back to large-v3, the most accurate (undoes the turbo switch of build ~20)
        if s.get("whisper_model") in ("large-v3-turbo", "turbo"):
            s["whisper_model"] = "large-v3"
        s["asr_v3"] = True
        try:
            raw = vio.load_json(CFG, {}); raw.update(whisper_model=s["whisper_model"], asr_v3=True)
            os.makedirs(CFG_DIR, exist_ok=True); vio.save_json(CFG, raw)
        except Exception:
            pass
    r = sync_root()
    if r:
        sh = vio.load_json(os.path.join(r, "settings.json"), {})
        s.update({k: v for k, v in sh.items() if k in SHARED})
    return claude_route(s)


def claude_route(s):
    """Which way Claude is reached: "auto" (the API key when there is one, else Claude Code on the
    subscription when it is installed), "api", "code" or "off" (local engine only). Everything that calls
    Claude gets s["api_key"]: the real key, the Claude Code marker, or "" (= no Claude)."""
    from .translate import CODE, claude_cli
    via = s.get("claude_via") or "auto"
    raw = s.get("api_key") or ""
    s["api_key_saved"] = bool(raw)
    if via == "off":
        key = ""
    elif via == "api":
        key = raw
    elif via == "code":
        key = CODE if claude_cli() else ""
    else:
        key = raw or (CODE if claude_cli() else "")
    s["api_key"] = key
    s["claude_code_found"] = bool(claude_cli())
    return s


def log(msg):
    STATE["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
    STATE["log"] = STATE["log"][-300:]


def P(*a):
    return os.path.join(STATE["proj"], *a)


class Cancelled(Exception):
    pass


def _live():
    """The live preview of the running job (only frames written since it started)."""
    if not STATE["running"] or not STATE["proj"]:
        return None
    from . import live
    st = live.state(STATE["proj"])
    since = PHASE.get("turn_t0", 0) if PHASE.get("claude_turn") else (STATE["t0"] or 0)
    return st if st and st.get("t", 0) >= since else None      # no frozen frame from an earlier stage


def prog(x):
    """Stage progress 0..1, mapped onto the whole job. Also the cancel point for long loops."""
    if STATE["cancel"]:
        raise Cancelled()
    if x is None:
        return
    a, b = RANGE
    v = a + (b - a) * min(1.0, max(0.0, float(x)))
    pa, pb = PHASE["range"]                    # inside "all in one click": this job's share of the whole
    STATE["progress"] = round(pa + (pb - pa) * v, 3)


def run_steps(steps):
    """steps: (name, weight, fn). Shows 'step i of n', one overall percentage, and checks cancel between steps."""
    steps = [x for x in steps if x]
    tot = sum(w for _, w, _ in steps) or 1
    acc = 0.0
    for i, (name, w, fn) in enumerate(steps):
        STATE.update(stage=name, step=i + 1, steps=len(steps))
        RANGE[:] = [acc / tot, (acc + w) / tot]
        prog(0)
        log(name + "...")
        fn()
        acc += w
    RANGE[:] = [1.0, 1.0]
    STATE["progress"] = 1.0


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
            if t.get("translate", True) and t.get("built_region") and t.get("region") and list(t["built_region"]) != list(t["region"]):
                out.add(gid)                          # the card's area changed (e.g. "whole logo")
            if not t.get("translate", True) and (t.get("built") or [0, 0, False])[2]:
                out.add(gid)
    return out


def needs_extract(T):
    for t in T["titles"]:
        if not t.get("translate", True):
            continue
        if "mask" not in t or "cache" not in t or t.get("reextract"):
            return True
        if t["in"] < t["cache"][0] or t["out"] > t["cache"][1]:
            return True
    return False


# ------------------------------------------------------------------ jobs
def job_analyze():
    from . import shots, detect, extract
    s = settings()
    proj = STATE["proj"]

    rng = vio.load_json(P("project.json")).get("range")
    before = {t["id"] for t in vio.load_json(P("titles.json"), {"titles": []})["titles"]} if rng else set()
    if rng:
        log(f"עובד רק על הטווח המסומן: {tcode(rng[0])} עד {tcode(rng[1])}")

    def a():
        c = shots.run(proj, progress=prog, rng=rng)
        log(f"{len(c)} שוטים")

    def b():
        ts = detect.run(proj, s.get("src_lang") or "eng", progress=prog, rng=rng)
        log(f"נמצאו {len([t for t in ts if t['id'] not in before])} שורות טקסט" + (" בטווח" if rng else ""))

    def b2():
        from . import review
        only = None
        if rng:
            only = {t["id"] for t in vio.load_json(P("titles.json"))["titles"] if t["id"] not in before}
        try:
            with LOCK:
                cnt = review.classify(proj, s["api_key"], s.get("model"), only=only, progress=prog, log=log)
        except Exception as e:
            log(f"הבדיקה של Claude לא הצליחה ({e}) - נשארים עם הסימון האוטומטי")
            return
        names = {"title": "כותרות לתרגום", "logo": "לוגואים", "credits": "קרדיטים", "scene": "טקסט בתוך התמונה",
                 "dialogue": "כתוביות צרובות", "noise": "רעש"}
        if cnt:
            log("Claude מיין: " + ", ".join(f"{names[k]} {v}" for k, v in cnt.items()))

    def c():
        extract.run(proj, progress=prog)
        prog(0.9)
        thumbs()
    def d():
        if rng:
            fonts_auto(only={t["id"] for t in vio.load_json(P("titles.json"))["titles"] if t["id"] not in before})
        else:
            fonts_auto()
    run_steps([("מחפש חיתוכי שוטים", 0.15, a), ("מזהה טקסט על המסך", 0.45, b),
               ("Claude בודק מה כל טקסט - כותרת, לוגו, קרדיט או רעש", 0.06, b2) if s.get("api_key") and not PHASE.get("agent") else None,
               ("מודד צבע, צל ותזמון מדויק", 0.32, c),
               ("מתאים גופן עברי לכל כותרת", 0.08, d)])
    n = series_apply()
    if n:
        log(f"{n} כותרות מולאו מהסדרה \"{series_name()}\" - מסומנות \"מהסדרה\" ברשימה")
    log("הניתוח הסתיים. בחר כותרת ברשימה והקלד עברית.")


def build_steps(proxy=True):
    from . import clean, render, extract, compose
    proj = STATE["proj"]
    T = vio.load_json(P("titles.json"))
    ext = needs_extract(T)
    dirty = set() if ext else dirty_groups(T)
    st = {}

    def do_extract():
        before = {t["id"]: (t.get("region"), t.get("reextract")) for t in vio.load_json(P("titles.json"))["titles"]}
        extract.run(proj, progress=prog)
        thumbs()
        T2 = vio.load_json(P("titles.json"))
        changed = {t["group"] for t in T2["titles"] if t["id"] in before and
                   (before[t["id"]][0] != t.get("region") or before[t["id"]][1])}
        st["dirty"] = dirty_groups(T2) | changed

    def do_clean():
        d = st.get("dirty", dirty)
        if d:
            log(f"{len(d)} כרטיסים")
            from . import propaint, aimodels
            from .extract import groups
            auto = settings().get("auto_propainter", True) and propaint.gpu() and aimodels.sam_ready()

            def sam(gs, why):
                for g in gs:
                    try:
                        STATE.update(stage=f"SAM 2 - מסכה מדויקת ל{why} ({g})")
                        aimodels.sam_card(proj, g, log=log)
                    except Exception as e:
                        log(f"SAM 2 על {g} לא הצליח ({str(e)[:120]}) - נשארות המסכות הרגילות")
            if auto:
                # logos, designed titles and cards asked to erase harder: exact masks before the erase
                G = groups(vio.load_json(P("titles.json")))
                sam([g for g in sorted(d) if g in G and not any(t.get("sam") for t in G[g]) and
                     any(t.get("styled") or t.get("strong") or t.get("kind") == "logo" for t in G[g])], "לוגו")
            clean.run(proj, sorted(d), progress=prog)
            if auto:
                # the self check still found the old letters: follow them with SAM 2 and erase those cards again
                cs = vio.load_json(P("clean_stats.json"), {})
                G = groups(vio.load_json(P("titles.json")))
                again = [g for g in sorted(d) if (cs.get(g) or {}).get("left_frames", 0) > 2 and g in G and
                         not any(t.get("sam") for t in G[g])]
                if again:
                    sam(again, "שאריות")
                    again = [g for g in again if any(t.get("sam") for t in groups(vio.load_json(P("titles.json")))[g])]
                    if again:
                        STATE.update(stage="מוחק שוב עם המסכות של SAM 2")
                        clean.run(proj, again)
            # hard cards (the self check still found the old letters, or most of the hole had to be invented)
            # go through ProPainter by themselves when it runs on the graphics card
            if settings().get("auto_propainter", True) and propaint.gpu() and propaint.ready():
                cs = vio.load_json(P("clean_stats.json"), {})
                hard = [g for g in sorted(d) if (cs.get(g) or {}).get("left_frames", 0) > 0 or (cs.get(g) or {}).get("synth_mean", 0) > 0.5]
                for g in hard:
                    try:
                        STATE.update(stage=f"מחיקה חזקה לכרטיס קשה ({g})")
                        propaint.card(proj, g, log=log)
                    except Exception as e:
                        log(f"ProPainter על {g} לא הצליח ({e}) - נשארת המחיקה הרגילה")

    def do_track():
        d = st.get("dirty", dirty)
        if d:
            render.track(proj, only=d)
    steps = [("התזמון השתנה - שומר פריימים מחדש", 0.15, do_extract) if ext else None,
             ("מוחק את המקור ומשחזר רקע", 0.5, do_clean) if (ext or dirty) else None,
             ("מודד את אנימציית המקור", 0.08, do_track) if (ext or dirty) else None,
             ("מרנדר עברית", 0.05, lambda: render.build_layers(proj))]
    if proxy:
        steps.append(("מכין תצוגת ניגון", 0.22, lambda: compose.run(proj, out_name="../proxy.mp4", crf=30,
                                                                   preset="ultrafast", scale=960, progress=prog)))
    return steps


def job_build():
    run_steps(build_steps(True))
    log("מוכן. אפשר לנגן במצב תוצאה.")


def job_export():
    from . import compose, check, srt, qa
    T = vio.load_json(P("titles.json"))
    pre = build_steps(False) if (dirty_groups(T) or needs_extract(T) or not os.path.exists(P("layers", "meta.json"))) else []
    res = {}

    def enc():
        res["out"] = compose.run(STATE["proj"], progress=prog)

    def chk():
        check.run(STATE["proj"])
        g = vio.load_json(P("check", "guard.json"), {}).get("changed") or []
        if g:
            log(f"אזהרה: ב-{len(g)} פריימים השתנה משהו מחוץ לאזורי הכותרות (הראשון ב-{g[0]['tc']}) - "
                f"התמונות בתיקיית check, guard_*.jpg")
        else:
            log("נבדק: מחוץ לאזורי הכותרות שום דבר לא השתנה.")

    def vqa():
        from . import visualqa, clean, render
        proj = STATE["proj"]
        try:
            iss = visualqa.run(proj, s["api_key"], s.get("model"), progress=prog)
        except Exception as e:
            log(f"הבדיקה של Claude לא הצליחה ({e})")
            return
        # what Claude saw left of the English, or Hebrew cut: those cards are done again, harder, once
        bad = sorted({i["group"] for i in iss if i.get("sev") == "err" and i.get("kind") in ("left", "cut") and i.get("group")})
        if bad:
            log(f"Claude מצא שאריות או עברית חתוכה ב-{len(bad)} כרטיסים - מתקן אותם אוטומטית")
            with LOCK:
                T = vio.load_json(P("titles.json"))
                for t in T["titles"]:
                    if t.get("group") in bad and t.get("translate", True):
                        t["strong"] = True; t["built"] = None
                vio.save_json(P("titles.json"), T)
            STATE.update(stage="מתקן את הכרטיסים ש-Claude סימן")
            clean.run(proj, bad)
            from . import propaint
            if propaint.ready():
                for g in bad:
                    try:
                        propaint.card(proj, g, log=log)
                    except Exception as e:
                        log(f"ProPainter על {g} לא הצליח ({e})")
            render.track(proj, only=set(bad))
            render.build_layers(proj)
            res["out"] = compose.run(proj)
            check.run(proj)
            try:
                iss2 = visualqa.run(proj, s["api_key"], s.get("model"), only=set(bad))
                left = [i for i in iss2 if i.get("group") in bad and i.get("sev") == "err"]
                log("אחרי התיקון: " + (f"נשארו {len(left)} הערות - פתח בדיקת QA" if left else "הכרטיסים נקיים"))
            except Exception as e:
                log(f"הבדיקה החוזרת של Claude לא הצליחה ({e})")
            iss = vio.load_json(P("visual_qa.json"), {"issues": []})["issues"]
        log(f"Claude עבר על התוצאה: {len(iss)} הערות" + (" - פתח בדיקת QA" if iss else " - נראה טוב"))

    def subs():
        D = vio.load_json(P("dialog.json"), {"cues": []})
        if any(c.get("he") for c in D["cues"]):
            p, w = srt.build(STATE["proj"], rle=settings().get("srt_rle", True))
            for x in w:
                log("אזהרת כתוביות: " + x)
            log(f"SRT: {p}")
        issues = qa.run(STATE["proj"])
        errs = sum(1 for i in issues if i["sev"] == "err")
        if issues:
            log(f"בדיקת QA: {errs} שגיאות, {len(issues) - errs} אזהרות (תפריט וידאו > בדיקת QA)")
    s = settings()
    run_steps(pre + [("מקודד את הווידאו הסופי", 0.5, enc), ("בודק כל כותרת לפני ואחרי", 0.05, chk),
                     ("Claude בודק את התוצאה כמו צופה", 0.05, vqa) if s.get("api_key") and not PHASE.get("agent") else None,
                     ("כתוביות ובדיקת QA", 0.02, subs)])
    log(f"הווידאו מוכן: {res['out']}")


def job_overlay():
    """Only the Hebrew titles on a transparent background, to lay over another cut of the same video."""
    from . import compose
    T = vio.load_json(P("titles.json"))
    if not any(t.get("translate", True) and t.get("mode") != "remove" and (t.get("he") or (t.get("layer") or {}).get("type") == "image")
               for t in T.get("titles", [])):
        raise RuntimeError("אין כותרות בעברית לייצוא - הקלד עברית לפחות לכותרת אחת")
    pre = build_steps(False) if (dirty_groups(T) or needs_extract(T) or not os.path.exists(P("layers", "meta.json"))) else []
    res = {}
    run_steps(pre + [("מקודד כותרות על רקע שקוף (ProRes 4444)", 0.4, lambda: res.update(out=compose.run_overlay(STATE["proj"], progress=prog)))])
    log(f"כותרות בלבד, רקע שקוף: {res['out']}")
    log("להדבקה על וידאו אחר באותו אורך וקצב פריימים (למשל גרסה בלי טקסט): שים את הקובץ ברצועה מעל הווידאו, מתחילת הציר.")


def fonts_auto(only=None):
    from . import fontmatch
    s = settings()
    with LOCK:
        n = fontmatch.run(STATE["proj"], api_key=s.get("api_key", ""), model=s.get("model") or "claude-sonnet-5-5",
                          progress=prog, log=log, only=only)
    if n:
        log(f"נבחר גופן אוטומטית ל-{n} כותרות" + (" (בעזרת Claude)" if s.get("api_key") else ""))
    return n


def job_fontmatch():
    """Find a similar font for one title (button in the inspector). Result: fontmatch/<id>.json."""
    from . import fontmatch
    a = STATE.get("args") or {}
    tid = a.get("id")
    T = vio.load_json(P("titles.json"))
    t = next((x for x in T["titles"] if x["id"] == tid), None)
    if not t:
        raise RuntimeError("הכותרת לא נמצאה")
    s = settings()
    res = {}

    def measure():
        res.update(fontmatch.match(STATE["proj"], t, top=8, api_key="", progress=prog, log=log))

    def claude():
        r = fontmatch.match(STATE["proj"], t, top=8, api_key=s["api_key"], model=s.get("model") or "claude-sonnet-5-5", log=log)
        if r.get("rows"):
            res.update(r)
    steps = []
    if a.get("google"):
        steps.append(("מוריד את כל גופני Google עם עברית (פעם אחת)", 0.4, lambda: fontmatch.download_google(progress=prog)))
    steps.append(("מודד את סגנון האותיות ומשווה לכל הגופנים", 0.4, measure))
    if s.get("api_key"):
        steps.append(("Claude משווה את המועמדים למקור", 0.2, claude))
    run_steps(steps)
    os.makedirs(P("fontmatch"), exist_ok=True)
    res["id"] = tid
    res.pop("traits", None)
    vio.save_json(P("fontmatch", f"{tid}.json"), res)
    if res.get("error"):
        log(res["error"])
    else:
        log(f"נמצאו {len(res['rows'])} גופנים דומים" + (" - Claude בחר את הראשון" if res.get("by_claude") else ""))


def job_fonts():
    run_steps([("מתאים גופן עברי לכל כותרת", 1.0, fonts_auto)])
    if os.path.exists(P("layers", "meta.json")):
        from . import render
        render.build_layers(STATE["proj"])


def job_transcribe():
    from . import srt
    s = settings()
    PJ = vio.load_json(P("project.json"))
    from . import asr
    name = s.get("whisper_model", "large-v3")
    if not asr.ready(name):
        STATE.update(stage="מוריד את מנוע התמלול (פעם אחת)", step=1, steps=2)
        log("מוריד את מנוע התמלול (פעם אחת)...")
        asr.wait(name, progress=lambda x: STATE.update(progress=round(0.5 * x, 3)) if not STATE["cancel"] else prog(0))
    audio = PJ["video"]
    rng = PJ.get("range")
    fps = PJ["info"]["fps"] or 24
    start, dur = (rng[0] / fps, (rng[1] + 1 - rng[0]) / fps) if rng else (None, None)
    if rng:
        log(f"מתמלל רק את הטווח המסומן: {tcode(rng[0])} עד {tcode(rng[1])}")
        audio = P("range.wav")
        subprocess.run([vio.FFMPEG, "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", PJ["video"],
                        "-map", "0:a:0", "-ac", "1", "-ar", "16000", audio], check=True, **vio.NOWIN)
    if s.get("sep_vocals", True) and PJ.get("info", {}).get("has_audio", True):
        from . import vocals
        RANGE[:] = [0.5 if STATE["steps"] == 2 else 0.0, 0.75]
        STATE.update(stage="מפריד את הדיבור מהמוזיקה", step=STATE.get("step") or 1, steps=STATE["steps"] or 1)
        try:
            if not vocals.ready():
                log("מוריד את מודל הפרדת הדיבור (פעם אחת)...")
                vocals.fetch(progress=lambda x: prog(0.2 * x))
            log("מפריד את הדיבור מהמוזיקה...")
            audio = vocals.speech_wav(PJ["video"], P("speech.wav"), progress=lambda x: prog(0.2 + 0.8 * x), start=start, dur=dur)
        except Cancelled:
            raise
        except Exception as e:
            traceback.print_exc()
            log(f"הפרדת הדיבור לא הצליחה ({e}) - מתמלל מהפסקול המלא")
            audio = P("range.wav") if rng else PJ["video"]
        RANGE[:] = [0.75, 1.0]
    STATE.update(stage="מתמלל", step=2 if STATE["steps"] == 2 else 1, steps=STATE["steps"] or 1)
    log("מתמלל עם Whisper" + (" על כרטיס המסך..." if asr.has_nvidia() and asr.gpu_ready() else " על המעבד..."))
    rs = (start, start + dur) if rng else None
    srt.transcribe(audio, P("whisper.json"), s.get("whisper_lang") or None, name, offset=start or 0.0, rng_s=rs)
    srt.prep(STATE["proj"], P("whisper.json"), rng_s=rs)
    log("התמלול מוכן. בחר כתובית והקלד עברית.")


def job_youtube():
    """Dialogue from the video's own captions on YouTube (youtube-transcript-api, MIT) instead of Whisper: the
    studio's subtitles are exact text (names, terms) even where music covers the voice. Generated (automatic)
    captions only when asked - they are speech recognition too."""
    import re as _re
    from youtube_transcript_api import YouTubeTranscriptApi
    a = STATE.get("args") or {}
    url = (a.get("url") or "").strip()
    m = _re.search(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([\w-]{11})", url) or _re.fullmatch(r"([\w-]{11})", url)
    if not m:
        raise RuntimeError("זה לא נראה כמו קישור לסרטון ביוטיוב")
    vid = m.group(1)
    want = {"eng": "en", "jpn": "ja", "kor": "ko"}.get(settings().get("src_lang") or "eng", "en")
    api = YouTubeTranscriptApi()
    tl = list(api.list(vid))
    man = [t for t in tl if not t.is_generated]
    gen = [t for t in tl if t.is_generated]
    pick = next((t for t in man if t.language_code.split("-")[0] == want), None) or (man[0] if man else None)
    if pick is None and a.get("auto"):
        pick = next((t for t in gen if t.language_code.split("-")[0] == want), None) or (gen[0] if gen else None)
    if pick is None:
        raise RuntimeError("לסרטון הזה אין ביוטיוב כתוביות רשמיות" + (" (יש רק אוטומטיות - אפשר לסמן שמותר להשתמש בהן)" if gen else ""))
    log(f"מוריד כתוביות {'רשמיות' if not pick.is_generated else 'אוטומטיות'} מיוטיוב ({pick.language})")
    sn = [x for x in pick.fetch() if x.text.strip()]
    off = float(a.get("offset") or 0)
    cues = []
    for i, x in enumerate(sn):
        txt = _re.sub(r"\s+", " ", x.text.replace("\n", " ")).strip()
        if _re.fullmatch(r"[\[(♪].*[\])♪]", txt):                 # [Music], (laughs), ♪...♪
            continue
        st = x.start + off
        en = x.start + x.duration + off
        if i + 1 < len(sn):
            en = min(en, sn[i + 1].start + off)
        cues.append({"n": len(cues) + 1, "start": round(max(0, st), 3), "end": round(max(st + 0.3, en), 3), "src": txt, "he": ""})
    with LOCK:
        vio.save_json(P("dialog.json"), {"cues": cues})
        PJ = vio.load_json(P("project.json")); PJ["youtube"] = url; vio.save_json(P("project.json"), PJ)
    log(f"{len(cues)} כתוביות נכנסו מיוטיוב. התזמון הוא של הסרטון ביוטיוב - אם הקובץ שלך חתוך אחרת, הזז בשדה ההזזה.")


def job_srt():
    from . import srt
    p, w = srt.build(STATE["proj"], rle=settings().get("srt_rle", True))
    for x in w:
        log("אזהרה: " + x)
    log(f"SRT מוכן: {p}")


def job_translate(kind):
    from . import translate
    s = settings()
    use_dicta = s.get("translate_engine") == "dicta"
    if not s.get("api_key") and not use_dicta:
        raise RuntimeError("אין מפתח API בהגדרות")
    engine = "DictaLM" if use_dicta else "Claude"

    def tr(batch, ctx, images=None):
        if use_dicta:                       # the Hebrew model on this computer (free, slow)
            from . import dicta
            return dicta.translate(batch, s.get("glossary", ""), ctx, log=log)
        return translate.translate(batch, s["api_key"], s.get("glossary", ""), ctx, s.get("model"), images=images)
    workers = 1 if use_dicta else 4
    if kind == "titles":
        from .detect import junk_reason
        with LOCK:                              # OCR noise is set to stay as it is, not sent to translation
            T = vio.load_json(P("titles.json"))
            cnt = {}
            for t in T["titles"]:
                cnt[t["group"]] = cnt.get(t["group"], 0) + 1
            junk = 0
            for t in T["titles"]:
                if t.get("translate", True) and not t.get("he") and not t.get("auto_reason"):
                    why = junk_reason(t["src"], s.get("src_lang") or "eng", cnt[t["group"]])
                    if why:
                        t["translate"] = False; t["auto_reason"] = why; junk += 1
            if junk:
                vio.save_json(P("titles.json"), T)
                log(f"{junk} שורות של רעש זיהוי (כמו אותיות סיניות או \"DD\") סומנו \"נשאר\"")
        items = [{"id": t["id"], "src": t["src"], "group": t["group"]} for t in T["titles"]
                 if t.get("translate", True) and not t.get("he")]
        if not items:
            log("אין כותרות שמחכות לתרגום (כולן כבר בעברית או מסומנות \"נשאר\").")
            return
        STATE.update(stage=f"מתרגם {len(items)} כותרות עם {engine}", step=1, steps=1)
        log(f"מתרגם {len(items)} כותרות...")
        # the same lone line again (a repeated caption or name card): translated once, reused - no tokens
        from .detect import norm
        size = {}
        for t in T["titles"]:
            size[t["group"]] = size.get(t["group"], 0) + 1
        known = {norm(t["src"]): t["he"] for t in T["titles"] if t.get("he") and size[t["group"]] == 1 and norm(t["src"])}
        res, copies, send = {}, {}, []
        for it in items:
            k = norm(it["src"])
            if size[it["group"]] == 1 and k and k in known:
                res[it["id"]] = known[k]
            elif size[it["group"]] == 1 and k and any(norm(x["src"]) == k and size[x["group"]] == 1 for x in send):
                first = next(x for x in send if norm(x["src"]) == k and size[x["group"]] == 1)
                copies.setdefault(first["id"], []).append(it["id"])
            else:
                send.append(it)
        if len(send) < len(items):
            log(f"{len(items) - len(send)} כותרות חוזרות - משתמש בתרגום הקיים במקום לשלוח שוב")
        items = send
        imgs = {it["id"]: P("thumbs", f"{it['id']}.jpg") for it in items if os.path.exists(P("thumbs", f"{it['id']}.jpg"))}
        # whole cards stay together in one request (their lines are translated as one sentence)
        batches, cur = [], []
        for it in items:
            if cur and len(cur) >= 40 and cur[-1]["group"] != it["group"]:   # a trailer's cards together: one sentence can run across them
                batches.append(cur); cur = []
            cur.append(it)
        if cur:
            batches.append(cur)
        batches = [b for b in batches if b]
        prog(0.02)
        from concurrent.futures import ThreadPoolExecutor
        tctx = translate.context(s.get("content"), "titles")
        if s.get("tmdb", True):                        # the official Hebrew title from TMDB (own key, or via Unwatch)
            from . import tmdb
            h = tmdb.hint(STATE["proj"], s.get("tmdb_key", ""), log=log)
            if h:
                tctx = (tctx + "\n" if tctx else "") + h
        with ThreadPoolExecutor(workers) as ex:        # requests side by side (and one Message Batch in batch mode)
            futs = [ex.submit(tr, bt, tctx,
                              {i: imgs[i] for i in imgs if any(x["id"] == i for x in bt)}) for bt in batches]
            for k, fu in enumerate(futs):
                res.update(fu.result())
                prog((k + 1) / len(batches))
        for a, others in copies.items():
            for o in others:
                if a in res:
                    res[o] = res[a]
        with LOCK:
            T = vio.load_json(P("titles.json"))
            for t in T["titles"]:
                if t["id"] in res and not t.get("he"):
                    t["he"] = res[t["id"]]
            # a card Claude left wholly empty is not text to translate (OCR noise on a face, a wheel, leaves):
            # it stays untouched instead of being erased. A card with some Hebrew keeps its empty lines erased
            # (their words went into the card's other lines).
            grp = {}
            for t in T["titles"]:
                if t.get("translate", True) and t.get("mode") != "remove":
                    grp.setdefault(t["group"], []).append(t)
            noise = 0
            for g, ls in grp.items():
                if all(t["id"] in res and not (t.get("he") or "").strip() for t in ls):
                    for t in ls:
                        t["translate"] = False; t["auto_reason"] = "Claude: אין כאן טקסט לתרגום"; noise += 1
            vio.save_json(P("titles.json"), T)
            if noise:
                log(f"{noise} שורות ש-Claude השאיר ריקות (לא טקסט אמיתי) סומנו \"נשאר\" ולא יימחקו")
        if os.path.exists(P("layers", "meta.json")):
            from . import render
            render.build_layers(STATE["proj"], only=set(res))
    else:
        D = vio.load_json(P("dialog.json"))
        items = [{"id": str(c["n"]), "src": c["src"]} for c in D["cues"] if not c.get("he")]
        if not items:
            log("אין כתוביות שמחכות לתרגום.")
            return
        STATE.update(stage=f"מתרגם {len(items)} כתוביות עם {engine}", step=1, steps=1)
        log(f"מתרגם {len(items)} כתוביות...")
        res = {}
        prog(0.02)
        from concurrent.futures import ThreadPoolExecutor
        chunks = [items[i:i + 60] for i in range(0, len(items), 60)]
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(tr, ch, translate.context(s.get("content"), "dialog")) for ch in chunks]
            for k, fu in enumerate(futs):
                res.update(fu.result())
                prog((k + 1) / len(chunks))
        with LOCK:
            D = vio.load_json(P("dialog.json"))
            for c in D["cues"]:
                if str(c["n"]) in res and not c.get("he"):
                    c["he"] = res[str(c["n"])]
            vio.save_json(P("dialog.json"), D)
    log("התרגום הושלם. עבור ותקן.")



def job_translate_compare():
    """The same titles translated by Claude and by DictaLM, side by side (compare.json); the user picks per line."""
    from . import translate, dicta
    s = settings()
    if not s.get("api_key"):
        raise RuntimeError("להשוואה צריך גם את Claude (העדפות > Claude)")
    T = vio.load_json(P("titles.json"))
    items = [{"id": t["id"], "src": t["src"], "group": t["group"]} for t in T["titles"]
             if t.get("translate", True) and t.get("kind") != "sub" and t.get("mode") != "remove"]
    if not items:
        raise RuntimeError("אין כותרות לתרגום")
    ctx = translate.context(s.get("content"), "titles")
    STATE.update(stage="Claude מתרגם")
    log(f"השוואת תרגום: {len(items)} כותרות")
    a = translate.translate(items, s["api_key"], s.get("glossary", ""), ctx, s.get("model"))
    prog(0.2)
    STATE.update(stage="DictaLM מתרגם (איטי)")
    b = dicta.translate(items, s.get("glossary", ""), ctx, log=log)
    rows = [{"id": it["id"], "group": it["group"], "src": it["src"], "claude": a.get(it["id"], ""), "dicta": b.get(it["id"], ""),
             "he": next((t.get("he", "") for t in T["titles"] if t["id"] == it["id"]), "")} for it in items]
    vio.save_json(P("compare.json"), {"rows": rows})
    log("ההשוואה מוכנה: וידאו > השוואת תרגום")


UPDATE = {}


def job_update():
    from . import update
    STATE.update(stage="מוריד את הגרסה החדשה", step=1, steps=2)
    log("מוריד את הגרסה החדשה...")
    update.apply(UPDATE, settings().get("gh_token", ""),
                 progress=lambda x: (STATE.update(stage="מתקין ופותח מחדש", step=2) if x >= 0.999 else None, prog(x)))
    STATE.update(stage="מתקין ופותח מחדש - התוכנה תיסגר ותיפתח לבד", step=2)
    log("הגרסה החדשה תיפתח עכשיו.")
    # leave right away: the installer is waiting for this program to close
    def bye():
        try:
            if WINDOW is not None:
                WINDOW.destroy()
        except Exception:
            pass
        os._exit(0)
    threading.Timer(0.6, bye).start()


PLAYABLE = {"h264", "vp8", "vp9"}


def needs_playable(proj=None):
    """WebView2 cannot play some codecs (HEVC without the extension, AV1, ProRes, MPEG-2...).
    Those get a playback copy; the original stays the source for every frame Retitle edits."""
    proj = proj or STATE["proj"]
    if not proj:
        return False
    info = vio.load_json(os.path.join(proj, "project.json"), {}).get("info", {})
    return bool(info.get("codec")) and info["codec"] not in PLAYABLE


def play_path():
    p = P("play.mp4")
    return p if os.path.exists(p) else None


def job_propaint():
    """Strong erase of one card with ProPainter (installed on demand from the settings)."""
    from . import propaint, render
    gid = (STATE.get("args") or {}).get("gid")
    if not gid:
        raise RuntimeError("לא נבחר כרטיס")
    STATE.update(stage="מחיקה חזקה עם ProPainter")
    RANGE[:] = [0.0, 0.85]
    with LOCK:
        n = propaint.card(STATE["proj"], gid, log=log, progress=prog)
    RANGE[:] = [0.85, 1.0]
    render.track(STATE["proj"], only={gid})
    render.build_layers(STATE["proj"])
    RANGE[:] = [0.0, 1.0]
    log(f"המחיקה החזקה הסתיימה ({n} פריימים). אפשר לנגן במצב תוצאה.")


def job_playable():
    import re as _re, subprocess
    src = vio.load_json(P("project.json"))["video"]
    n = vio.load_json(P("project.json")).get("n_frames") or 1
    out = P("play.tmp.mp4")
    STATE.update(stage="מכין עותק לניגון (הווידאו בקידוד שהנגן לא מכיר)", step=1, steps=1)
    cmd = [vio.FFMPEG, "-v", "error", "-y", "-i", src, "-map", "0:v:0", "-map", "0:a:0?", "-fps_mode", "passthrough",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart", "-progress", "pipe:1", "-nostats", out]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, **vio.NOWIN)
    try:
        for line in p.stdout:
            m = _re.match(r"frame=(\d+)", line)
            if m:
                prog(int(m.group(1)) / n)
    except Cancelled:
        p.kill(); p.wait()
        try:
            os.remove(out)
        except OSError:
            pass
        raise
    if p.wait() != 0:
        raise RuntimeError("לא הצלחתי להכין עותק לניגון")
    os.replace(out, P("play.mp4"))
    log("הווידאו מוכן לניגון.")


def job_review():
    """Menu: Claude sorts the detected texts again (titles not yet translated or edited)."""
    from . import review
    s = settings()
    if not s.get("api_key"):
        raise RuntimeError("אין מפתח API בהגדרות")
    STATE.update(stage="Claude בודק מה כל טקסט - כותרת, לוגו, קרדיט או רעש", step=1, steps=1)
    only = {t["id"] for t in vio.load_json(P("titles.json"))["titles"] if not t.get("he") and not t.get("user_set")}
    with LOCK:
        cnt = review.classify(STATE["proj"], s["api_key"], s.get("model"), only=only, progress=prog, log=log)
    names = {"title": "כותרות לתרגום", "logo": "לוגואים", "credits": "קרדיטים", "scene": "טקסט בתוך התמונה",
             "dialogue": "כתוביות צרובות", "noise": "רעש"}
    log("Claude מיין: " + (", ".join(f"{names[k]} {v}" for k, v in cnt.items()) or "אין מה לבדוק"))


def job_design(gids=None):
    """Claude designs styled cards like a graphic designer (font, layout, colour, texture, depth,
    decorations). gids: the cards to design; None = every styled card that has Hebrew and no design yet."""
    from . import design, render
    s = settings()
    if not s.get("api_key"):
        raise RuntimeError("אין מפתח API בהגדרות")
    args = STATE.get("args") or {}
    if gids is None and args.get("gid"):
        gids = [args["gid"]]
    T = vio.load_json(P("titles.json"))
    if gids is None:
        gids = []
        for t in T["titles"]:
            if t.get("styled") and t.get("translate", True) and t.get("he") and not t.get("design") \
                    and not t.get("design_member") and t["group"] not in gids:
                gids.append(t["group"])
    if not gids:
        log("אין כותרות מעוצבות שמחכות לעיצוב.")
        return
    done = set()
    for k, g in enumerate(gids):
        STATE.update(stage=f"Claude מעצב כמו מעצב גרפי ({k + 1} מתוך {len(gids)})")
        RANGE[:] = [k / len(gids), (k + 1) / len(gids)]
        spec, rel, lines, box = design.design_card(STATE["proj"], g, s["api_key"], log=log, progress=prog)
        with LOCK:
            design.apply(STATE["proj"], lines, spec, rel, box)
        done.update(t["id"] for t in lines)
    RANGE[:] = [0.0, 1.0]
    if os.path.exists(P("layers", "meta.json")):
        render.build_layers(STATE["proj"], only=done)
    log(f"עוצבו {len(gids)} כרטיסים. אפשר לראות בתצוגה \"תוצאה\" ולבנות.")


def job_design_edit():
    """A change to a designed logo in the user's own words; Claude edits the design."""
    from . import design, render
    s = settings()
    if not s.get("api_key"):
        raise RuntimeError("אין מפתח API בהגדרות")
    a = STATE.get("args") or {}
    tid, text = a.get("id"), (a.get("text") or "").strip()
    if not tid or not text:
        raise RuntimeError("כתוב מה לשנות")
    STATE.update(stage="Claude משנה את העיצוב")
    prog(0.1)
    with LOCK:
        design.edit_card(STATE["proj"], tid, s["api_key"], text, log=log)
    prog(0.8)
    if os.path.exists(P("layers", "meta.json")):
        render.build_layers(STATE["proj"], only={tid})
    log("השינוי בוצע. אפשר לבטל אותו בכפתור \"בטל שינוי אחרון\".")


def undo_design(tid):
    T = vio.load_json(P("titles.json"))
    for t in T["titles"]:
        if t["id"] == tid:
            t.pop("design", None); t["layer"] = {"type": "text"}
        elif t.get("design_member") == tid:
            t.pop("design_member", None); t.pop("mode", None)
    vio.save_json(P("titles.json"), T)


def auto_parts(opts=None):
    """What "all in one click" will do for the open project (the dialog shows the same list)."""
    s = settings(); opts = opts or {}
    T = vio.load_json(P("titles.json"), {"titles": []})["titles"]
    D = vio.load_json(P("dialog.json"), {"cues": []})["cues"]
    key = bool(s.get("api_key"))
    tkey = key or s.get("translate_engine") == "dicta"
    want_t, want_d = opts.get("titles", True), opts.get("dialog", True)
    parts = []
    if want_t:
        if not T:
            parts.append(("analyze", "ניתוח הכותרות", 0.35))
        elif key:
            # analysed before Claude was connected: Claude sorts the texts and picks the fonts now
            if any("ai_kind" not in t and not t.get("from_series") for t in T if t.get("translate", True)):
                parts.append(("classify", "Claude בודק מה כל טקסט - כותרת, לוגו, קרדיט או רעש", 0.04))
            if any(t.get("translate", True) and t.get("font_auto") and not t.get("font_by_claude") for t in T):
                parts.append(("fonts", "Claude בוחר גופן לכל כותרת", 0.05))
        if tkey:
            parts.append(("translate_titles", "תרגום הכותרות", 0.05))
        if key:
            parts.append(("design", "עיצוב כותרות מעוצבות ולוגואים (Claude)", 0.06))
    if want_d and vio.load_json(P("project.json")).get("info", {}).get("has_audio", True):
        if not D:
            parts.append(("transcribe", "תמלול הדיאלוג", 0.25))
        if tkey:
            parts.append(("translate_dialog", "תרגום הכתוביות", 0.05))
        parts.append(("srt", "קובץ כתוביות SRT", 0.01))
    if want_t:
        parts.append(("export", "בנייה וייצוא + בדיקה", 0.3))
    return parts


def job_auto(opts=None):
    """Everything in a row: analyse, translate, transcribe, build, export, check. Steps that are
    already done are skipped; a missing Claude key just skips the translation steps."""
    opts = opts if opts is not None else (STATE.get("args") or {})
    parts = auto_parts(opts)
    tot = sum(w for *_, w in parts) or 1
    acc = 0.0
    base = PHASE["range"][:]
    for i, (job, name, w) in enumerate(parts):
        a = base[0] + (base[1] - base[0]) * acc / tot
        b = base[0] + (base[1] - base[0]) * (acc + w) / tot
        PHASE.update(range=[a, b], name=(PHASE.get("prefix", "") + f"חלק {i + 1} מתוך {len(parts)}: {name}"))
        RANGE[:] = [0.0, 1.0]
        STATE.update(stage=name, step=0, steps=0)
        prog(0)
        claude_part = job in ("translate_titles", "translate_dialog", "design", "classify", "fonts")
        if claude_part and PHASE.get("no_claude"):
            acc += w
            continue
        try:
            if job == "translate_titles":
                job_translate("titles")
            elif job == "classify":
                from . import review
                with LOCK:
                    review.classify(STATE["proj"], settings()["api_key"], settings().get("model"), progress=prog, log=log)
            elif job == "fonts":
                fonts_auto(only={t["id"] for t in vio.load_json(P("titles.json"))["titles"]
                                 if t.get("translate", True) and t.get("font_auto") and not t.get("font_by_claude")})
            elif job == "translate_dialog":
                job_translate("dialog")
            else:
                JOBS[job]()
        except Cancelled:
            raise
        except Exception as e:
            if not claude_part:
                raise
            # no credit / bad key / network: the rest (build, export, SRT) still runs without Claude
            PHASE["no_claude"] = True
            log(f"דילגתי על השלבים של Claude: {e}")
        acc += w
    if not settings().get("api_key"):
        log("בלי מפתח Claude דילגתי על התרגום האוטומטי - אפשר לכתוב עברית ולייצא שוב")
    log("הכול הסתיים.")


AGENT = {"token": "", "port": 0}
_AGENT_RUN = threading.Lock()


def agent_run(argv):
    """One engine command for Claude Code in full mode, inside this process (the same engine and libraries
    the app uses - no Python needed on the PC): `rt.cmd <command> ...` as in `python -m rt`, or
    `rt.cmd --py script.py args` for a script Claude wrote (numpy, cv2 and the rt modules importable)."""
    import io, contextlib, runpy
    buf = io.StringIO()
    code = 0
    with _AGENT_RUN, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            if os.path.isabs(vio.FFMPEG) and os.path.dirname(vio.FFMPEG) not in os.environ.get("PATH", ""):
                os.environ["PATH"] = os.path.dirname(vio.FFMPEG) + os.pathsep + os.environ.get("PATH", "")
            if argv and argv[0] == "--py":
                old = sys.argv[:]
                sys.argv = argv[1:]
                try:
                    runpy.run_path(argv[1], run_name="__main__")
                finally:
                    sys.argv = old
            else:
                from .__main__ import main as cli
                cli(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except BaseException:
            traceback.print_exc()
            code = 1
    if argv:
        log("Claude הריץ: " + " ".join(os.path.basename(a) if os.sep in a or "/" in a else a for a in argv)[:160])
    return buf.getvalue() + f"\n[exit {code}]\n"


def agent_tool(proj):
    """The rt command Claude Code calls (a .cmd on Windows, a shell script elsewhere). Returns its path."""
    import secrets
    AGENT["token"] = AGENT["token"] or secrets.token_hex(16)
    d = os.path.join(proj, "agent")
    os.makedirs(d, exist_ok=True)
    url = f"http://127.0.0.1:{AGENT['port']}/api/agent_run"
    if os.name == "nt":
        p = os.path.join(d, "rt.cmd")
        body = ("@echo off\r\nchcp 65001 >nul\r\nsetlocal EnableDelayedExpansion\r\nset \"D=\"\r\n"
                ":next\r\nif \"%~1\"==\"\" goto go\r\nset D=!D! --data-urlencode \"a=%~1\"\r\nshift\r\ngoto next\r\n"
                f":go\r\ncurl.exe -s --max-time 14400 --data-urlencode \"token={AGENT['token']}\" !D! {url}\r\n")
    else:
        p = os.path.join(d, "rt")
        body = (f"#!{sys.executable}\nimport sys, urllib.request, urllib.parse\n"
                f"d = urllib.parse.urlencode([('token', '{AGENT['token']}')] + [('a', x) for x in sys.argv[1:]]).encode()\n"
                f"print(urllib.request.urlopen('{url}', d, timeout=14400).read().decode())\n")
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(body)
    if os.name != "nt":
        os.chmod(p, 0o755)
    return p


def job_agent():
    """Full mode: Claude Code works the whole video like an editor in chat - looks at every sheet, edits the
    titles itself, asks the engine for fixes and looks again, until every card is right."""
    from . import agent, render, propaint, design, fontmatch
    from .extract import groups
    s = settings()
    if not s.get("claude_code_found"):
        raise RuntimeError("המצב המלא עובד דרך Claude Code - התקן אותו והתחבר (/login), ואז נסה שוב")
    opts = STATE.get("args") or {}
    proj = STATE["proj"]
    rounds = max(1, min(10, int(opts.get("rounds") or 10)))   # until Claude approves every card; 10 is only a safety cap
    ses = agent.Session(proj, s.get("agent_model") or "opus", log, cancel=lambda: STATE["cancel"], tool=agent_tool(proj))
    PHASE["agent"] = True
    base = PHASE["range"][:]

    skip = [0.0]                                  # analysis already done: the bar starts at what really runs

    def part(a, b, name):
        a, b = [(x - skip[0]) / (1 - skip[0]) for x in (a, b)]
        PHASE.update(range=[base[0] + (base[1] - base[0]) * a, base[0] + (base[1] - base[0]) * b], name=name)
        RANGE[:] = [0.0, 1.0]
        STATE.update(stage="", step=0, steps=0)          # the phase name is shown already - no repeat
        prog(0)

    def turn(prompt, stage):
        STATE.update(stage=stage)
        before = agent.snapshot(proj)
        n = [0]

        def seen(kind, text, path):
            # what Claude does right now; the bar creeps inside this part (the length of a turn is unknown)
            n[0] += 1
            STATE.update(stage="Claude: " + text)
            prog(min(0.95, 1 - 0.965 ** n[0]))
            if kind == "look" and path and os.path.exists(path):
                from . import live
                im = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
                if im is not None:
                    live.show(proj, text, im, every=0)
        PHASE["claude_turn"] = True
        PHASE["turn_t0"] = time.time()
        try:
            r = ses.ask(prompt, on_event=seen)
        finally:
            PHASE["claude_turn"] = False
        with LOCK:
            changed, err = agent.settle(proj, before, log)
        if err:
            log(err)
        if r.get("notes_he"):
            log("Claude: " + r["notes_he"])
        return r, changed, err

    try:
        T = vio.load_json(P("titles.json"), {"titles": []})["titles"]
        if not T:
            part(0.0, 0.25, "ניתוח הכותרות")
            job_analyze()
        else:
            skip[0] = 0.25
        from . import reavideo
        if reavideo.ready():
            part(0.25, 0.28, "rea-video מפרק את הסרטון")
            try:
                reavideo.decompose(proj, log=log, cancel=lambda: STATE["cancel"])
                reavideo.give_skill(proj)
            except Exception as e:
                log(f"rea-video לא הצליח ({str(e)[:160]}) - ממשיך בלעדיו")
        else:
            log("rea-video עוד לא מותקן (העדפות > מחיקה חזקה וזיהוי גופנים) - ממשיך בלעדיו")
        import shutil                          # skills that ship with Retitle (qc-video, MIT) for Claude Code
        sk = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
        for name in (os.listdir(sk) if os.path.isdir(sk) else []):
            dst = os.path.join(proj, ".claude", "skills", name)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(os.path.join(sk, name), dst)
        part(0.28, 0.4, "Claude עובר על כל הטקסטים שזוהו")
        gl = (s.get("glossary") or "").strip()
        _, _, err = turn(f"""Project folder: {proj}
Step 1 - what was detected. Read detect_sheet.jpg (each detected line at its frame, with its box) and titles.json.
For every line decide translate true/false (credits, studio logos, English names and text inside the scene stay),
fix lines that the OCR split or misread (src), and write the Hebrew (he) for every line that is translated -
the whole video's titles together, so a sentence across cards reads as one.
{("Name glossary (always wins):" + chr(10) + gl) if gl else ""}
Lines:
{agent.card_list(proj)}""", "Claude עובר על כל הטקסטים שזוהו ומתרגם")
        fl = {t["id"] for t in vio.load_json(P("titles.json"))["titles"]
              if t.get("translate", True) and t.get("kind") != "sub" and not t.get("font_by_claude")
              and "font" not in t.get("style_locked", [])}
        if fl:
            part(0.4, 0.42, "Claude בוחר גופן לכל סגנון")
            try:
                fonts_auto(only=fl)
            except Exception as e:
                log(f"בחירת הגופנים לא הצליחה ({e})")
        if opts.get("dialog", True) and vio.load_json(P("project.json")).get("info", {}).get("has_audio", True):
            D = vio.load_json(P("dialog.json"), {"cues": []})["cues"]
            part(0.42, 0.47, "תמלול ותרגום הדיאלוג")
            if not D:
                job_transcribe()
            if s.get("api_key"):
                try:
                    job_translate("dialog")
                except Exception as e:
                    log(f"תרגום הדיאלוג לא הצליח ({e})")
            job_srt()
        part(0.47, 0.5, "עיצוב כותרות מעוצבות ולוגואים")
        try:
            job_design()
        except Exception as e:
            log(f"העיצוב לא הצליח ({e})")
        last = []
        for k in range(rounds):
            # most videos need 2-4 rounds: each round takes half of what is left of the bar
            a, w = 0.5 + 0.5 * (1 - 0.5 ** k), 0.5 * 0.5 ** (k + 1)
            if k == rounds - 1:
                w = 1.0 - a
            part(a, a + 0.5 * w, f"סבב {k + 1}: בנייה, קידוד ובדיקה")
            job_export()
            sheets = sorted(f for f in os.listdir(P("check")) if f.endswith(".jpg")) if os.path.isdir(P("check")) else []
            if not sheets:
                log("אין כרטיסים לבדיקה.")
                break
            part(a + 0.5 * w, a + 0.9 * w, f"סבב {k + 1}: Claude בודק כל כרטיס")
            r, changed, _ = turn(f"""Round {k + 1} of {rounds}. The video was built and encoded. Look at EVERY sheet below
(Read each one): per card, the source on the left and the Hebrew result on the right, from the frame before the title
comes in to the frame after it goes out. {"Last round you asked for: " + json.dumps(last, ensure_ascii=False) + "." if last else ""}
Check: source letters or ghosts left, a smeared background, Hebrew cut, overlapping, too small or big, wrong place,
colour, light or font, animation that does not follow the source, mistakes in the Hebrew.
Fix it: edit titles.json, run the engine yourself (RT ...), or write code for what the engine cannot do - then look
again at what you changed (RT preview). Put in actions only what you want the app to do for you. After your turn the
app builds, encodes and makes new sheets. Set done only when everything is right.
Sheets: {", ".join("check/" + f for f in sheets)}""", f"סבב {k + 1}: Claude בודק כל כרטיס")
            acts = [x for x in r.get("actions") or [] if isinstance(x, dict)]
            last = acts
            if r.get("done") and not acts and not changed:
                log(f"Claude אישר את כל הכרטיסים אחרי {k + 1} סבבים.")
                break
            if k == rounds - 1:
                log("הגעתי לתקרה של 10 סבבים בלי ש-Claude אישר הכול. הייצוא האחרון הוא הגרסה הסופית - ההערות שלו ביומן.")
                break
            part(a + 0.9 * w, a + w, f"סבב {k + 1}: מתקן מה ש-Claude ביקש")
            with LOCK:
                T = vio.load_json(P("titles.json"))
                G = groups(T)
                for x in acts:
                    if x.get("do") == "clean" and x.get("gid") in G:
                        for t in G[x["gid"]]:
                            t["strong"] = True; t["built"] = None
                vio.save_json(P("titles.json"), T)
            sm = [x["gid"] for x in acts if x.get("do") == "sam" and x.get("gid") in G]
            if sm:
                from . import aimodels
                for g in sm:
                    try:
                        STATE.update(stage=f"SAM 2 ({g})")
                        aimodels.sam_card(proj, g, log=log)
                    except Exception as e:
                        log(f"SAM 2 על {g} לא הצליח ({str(e)[:120]})")
            pp = [x["gid"] for x in acts if x.get("do") == "propaint" and x.get("gid") in G]
            rt = {x["gid"] for x in acts if x.get("do") == "retrack" and x.get("gid") in G}
            if any(x.get("do") in ("clean", "sam") for x in acts) or pp:
                run_steps(build_steps(False))
            for g in pp:
                if not propaint.ready():
                    log("ProPainter עוד לא מותקן - דילגתי על המחיקה החזקה")
                    break
                try:
                    STATE.update(stage=f"מחיקה חזקה ({g})")
                    with LOCK:
                        propaint.card(proj, g, log=log)
                    rt.add(g)
                except Exception as e:
                    log(f"ProPainter על {g} לא הצליח ({e})")
            if rt:
                render.track(proj, only=rt)
            for x in acts:
                try:
                    if x.get("do") == "fontmatch" and x.get("id"):
                        with LOCK:
                            T = vio.load_json(P("titles.json"))
                            for t in T["titles"]:
                                if t["id"] == x["id"]:
                                    t["style_locked"] = [k for k in t.get("style_locked", []) if k != "font"]; t.pop("from_series", None)
                            vio.save_json(P("titles.json"), T)
                        fontmatch.run(proj, s.get("api_key"), s.get("model") or "claude-sonnet-5-5", log=log, only={x["id"]})
                    elif x.get("do") == "design" and x.get("gid") in G:
                        spec, rel, lines, box = design.design_card(proj, x["gid"], s["api_key"], log=log)
                        with LOCK:
                            design.apply(proj, lines, spec, rel, box)
                    elif x.get("do") == "redesign" and x.get("id") and x.get("request"):
                        with LOCK:
                            design.edit_card(proj, x["id"], s["api_key"], x["request"], log=log)
                except Exception as e:
                    log(f"{x.get('do')} לא הצליח ({e})")
            render.build_layers(proj)
    finally:
        PHASE.pop("agent", None)
    log("המצב המלא הסתיים. התוצאה בתיקיית out.")


QUEUE = []                       # [{"path", "status", "msg"}] - shown in the queue dialog


def job_queue():
    """Runs "all in one click" on each video/project in the queue, one after the other."""
    todo = [q for q in QUEUE if q["status"] in ("waiting", "error")]
    opts = STATE.get("args") or {}
    from . import translate as _tr
    _tr.MODE.update(batch=bool(opts.get("batch", True)), cancel=lambda: STATE["cancel"])
    if _tr.MODE["batch"]:
        log("תור: הבקשות ל-Claude יוצאות כעבודה מושהית - חצי מחיר, אותה איכות, לוקח יותר זמן")
    for k, q in enumerate(todo):
        q.update(status="running", msg="")
        PHASE.update(range=[k / len(todo), (k + 1) / len(todo)], prefix=f"פרק {k + 1} מתוך {len(todo)} · ")
        try:
            open_file(q["path"])
            job_auto(opts)
            try:
                from . import project
                if STATE.get("file"):
                    project.write(STATE["file"], STATE["proj"], vio.load_json(P("project.json"))["video"])
            except Exception:
                traceback.print_exc()
            q.update(status="done", msg="")
        except Cancelled:
            q.update(status="waiting", msg="בוטל")
            raise
        except Exception as e:
            traceback.print_exc()
            q.update(status="error", msg=str(e))
            log(f"{os.path.basename(q['path'])}: שגיאה - {e}. ממשיך לבא בתור.")
    PHASE.pop("prefix", None)
    _tr.MODE["batch"] = False
    log("התור הסתיים.")


JOBS = {"youtube": job_youtube, "translate_compare": job_translate_compare, "agent": job_agent, "design": job_design, "design_edit": job_design_edit, "propaint": job_propaint, "auto": job_auto, "queue": job_queue, "review": job_review, "playable": job_playable, "analyze": job_analyze, "update": job_update, "build": job_build, "export": job_export,
        "transcribe": job_transcribe, "overlay": job_overlay, "fonts": job_fonts, "fontmatch": job_fontmatch, "srt": job_srt,
        "translate_titles": lambda: job_translate("titles"), "translate_dialog": lambda: job_translate("dialog")}


JOB_HE = {"youtube": "כתוביות מיוטיוב", "translate_compare": "השוואת תרגום", "agent": "מצב מלא עם Claude Code", "propaint": "מחיקה חזקה",
          "design_edit": "שינוי עיצוב עם Claude", "design": "עיצוב עם Claude", "auto": "הכול אוטומטית", "queue": "תור עבודה",
          "review": "מיון עם Claude", "playable": "הכנת וידאו לניגון", "fontmatch": "חיפוש גופן דומה", "fonts": "התאמת גופנים",
          "overlay": "ייצוא כותרות בלבד", "update": "עדכון", "analyze": "ניתוח כותרות", "build": "בניית תצוגה", "export": "ייצוא",
          "transcribe": "תמלול", "srt": "יצירת SRT", "translate_titles": "תרגום כותרות", "translate_dialog": "תרגום כתוביות"}


def start(name):
    if STATE["running"]:
        return False
    STATE.update(running=True, job=name, error=None, progress=None, stage="", step=0, steps=0,
                 t0=time.time(), cancel=False, cancelled=False)
    RANGE[:] = [0.0, 1.0]
    PHASE.update(range=[0.0, 1.0], name="")
    from . import translate as _tr
    _tr.USAGE.clear()

    def run():
        try:
            JOBS[name]()
        except Cancelled:
            STATE["cancelled"] = True
            log("בוטל.")
        except Exception as e:
            STATE["error"] = str(e)
            log("שגיאה: " + str(e))
            traceback.print_exc()
        finally:
            try:
                line = _tr.usage_line()
                if line:
                    log(line)
            except Exception:
                pass
            STATE["t1"] = time.time()
            PHASE.update(range=[0.0, 1.0], name="")
            PHASE.pop("no_claude", None)
            PHASE["claude_turn"] = False
            _tr.MODE["batch"] = False
            STATE["running"] = False
            STATE["progress"] = None
            if name != "update":
                try:
                    from . import notify
                    last = STATE["log"][-1][10:].strip() if STATE["log"] else ""
                    notify.job_ended(JOB_HE.get(name, name), STATE["t1"] - STATE["t0"], STATE["error"],
                                     STATE["cancelled"], settings().get("notify", True), extra=last)
                except Exception:
                    pass
    threading.Thread(target=run, daemon=True).start()
    return True


PICK = {"video": ("Video (*.mp4;*.mov;*.mkv;*.m4v;*.avi;*.webm)", "Video", "*.mp4 *.mov *.mkv *.m4v *.avi *.webm"),
        "srt": ("Subtitles (*.srt)", "SRT", "*.srt"),
        "png": ("PNG image (*.png)", "PNG", "*.png"),
        "retitle": ("Retitle project (*.retitle)", "Retitle", "*.retitle"),
        "font": ("Fonts (*.ttf;*.otf;*.ttc)", "Fonts", "*.ttf *.otf *.ttc"),
        "layers": ("Photoshop or PNG layers (*.psd;*.png)", "Layers", "*.psd *.png")}


def pick_files(kind):
    wv, tkname, tkpat = PICK[kind]
    if WINDOW is not None:
        import webview
        r = WINDOW.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=True, file_types=(wv, "All files (*.*)"))
        return list(r) if r else []
    import tkinter as tk
    from tkinter import filedialog
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilenames(filetypes=[(tkname, tkpat), ("All", "*.*")])
    r.destroy()
    return list(p)


def pick_folder():
    if WINDOW is not None:
        import webview
        r = WINDOW.create_file_dialog(webview.FOLDER_DIALOG)
        return (r[0] if isinstance(r, (list, tuple)) else r) if r else ""
    import tkinter as tk
    from tkinter import filedialog
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askdirectory()
    r.destroy()
    return p


def save_file(name, folder):
    if WINDOW is not None:
        import webview
        r = WINDOW.create_file_dialog(webview.SAVE_DIALOG, directory=folder, save_filename=name,
                                      file_types=("Retitle project (*.retitle)",))
        return (r[0] if isinstance(r, (list, tuple)) else r) if r else ""
    import tkinter as tk
    from tkinter import filedialog
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.asksaveasfilename(initialdir=folder, initialfile=name, defaultextension=".retitle",
                                     filetypes=[("Retitle", "*.retitle")])
    r.destroy()
    return p


def pick_file(kind="video"):
    wv, tkname, tkpat = PICK[kind]
    if WINDOW is not None:
        import webview
        r = WINDOW.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=(wv, "All files (*.*)"))
        return (r[0] if isinstance(r, (list, tuple)) else r) if r else ""
    import tkinter as tk
    from tkinter import filedialog
    r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
    p = filedialog.askopenfilename(filetypes=[(tkname, tkpat), ("All", "*.*")])
    r.destroy()
    return p


def _activate(rfile, data):
    from . import project
    if STATE["running"] and STATE["job"] != "update":
        raise RuntimeError("פעולה אחרת עדיין רצה - חכה שתסתיים או בטל אותה")
    STATE["proj"] = data
    STATE["file"] = rfile
    STATE["missing_video"] = not os.path.isfile(vio.load_json(os.path.join(data, "project.json"), {}).get("video", ""))
    FRAMES.clear(); LAYERS.clear()
    STATE["log"] = []
    log(f"פרויקט: {rfile}")
    s = vio.load_json(CFG, {})
    s["recent"] = [rfile] + [x for x in s.get("recent", []) if os.path.normcase(x) != os.path.normcase(rfile)][:11]
    os.makedirs(CFG_DIR, exist_ok=True)
    vio.save_json(CFG, s)
    set_title()
    try:
        n = series_apply()
        if n:
            log(f"{n} כותרות עודכנו מהסדרה")
    except Exception:
        traceback.print_exc()
    if not STATE["missing_video"]:
        threading.Thread(target=build_media, args=(data,), daemon=True).start()
        if needs_playable(data) and not os.path.exists(os.path.join(data, "play.mp4")) and not STATE["running"]:
            threading.Timer(0.3, lambda: start("playable")).start()
    return data


def bring_front():
    """Show the window in front. Un-minimize it only if it is minimized - restore() would also
    take a maximized window out of maximized (it shrank when a project was opened)."""
    if WINDOW is None:
        return
    try:
        form = getattr(WINDOW, "native", None)
        st = str(getattr(form, "WindowState", "")) if form is not None else ""
        if "Minimized" in st:
            WINDOW.restore()
            if settings().get("open_maximized", True):
                WINDOW.maximize()
        WINDOW.show()
    except Exception:
        pass


def set_title():
    if WINDOW is None:
        return
    try:
        name = os.path.splitext(os.path.basename(STATE["file"]))[0] if STATE.get("file") else ""
        WINDOW.set_title(f"{name} - Retitle" if name else "Retitle")
    except Exception:
        pass


def open_project(video):
    """New project from a video (or the existing one of that video)."""
    from . import project
    rfile = project.for_video(video)
    data, _ = project.read(rfile)
    return _activate(rfile, data)


def open_file(path):
    """Open a .retitle file, or a video (its project)."""
    from . import project
    if not path.lower().endswith(project.EXT):
        return open_project(path)
    data, video = project.read(path)
    return _activate(path, data)


def recent():
    out = []
    for f in vio.load_json(CFG, {}).get("recent", []):
        d = vio.load_json(f, {}) if os.path.isfile(f) else {}
        out.append({"path": f, "name": os.path.splitext(os.path.basename(f))[0], "folder": os.path.dirname(f),
                    "saved": d.get("saved", ""), "exists": bool(d)})
    return out


# ------------------------------------------------------------------ timeline media (video thumbnails + audio waveform)
MEDIA_BUSY = set()


def build_media(proj):
    """media/thumbs/NNNNN.jpg every `step` s and media/meta.json with audio peaks (100 per second)."""
    if proj in MEDIA_BUSY or os.path.exists(os.path.join(proj, "media", "meta.json")):
        return
    MEDIA_BUSY.add(proj)
    try:
        PJ = vio.load_json(os.path.join(proj, "project.json"))
        src, dur = PJ["video"], float(PJ["info"]["duration"])
        d = os.path.join(proj, "media", "thumbs")
        os.makedirs(d, exist_ok=True)
        step = max(0.5, dur / 400)
        subprocess.run([vio.FFMPEG, "-v", "error", "-y", "-i", src, "-map", "0:v:0",
                        "-vf", f"fps=1/{step:.4f}:round=near,scale=-2:90", "-q:v", "6",
                        os.path.join(d, "%05d.jpg")], **vio.NOWIN)
        count = len(glob.glob(os.path.join(d, "*.jpg")))
        PPS, SR = 100, 8000
        r = subprocess.run([vio.FFMPEG, "-v", "error", "-i", src, "-map", "0:a:0?", "-ac", "1", "-ar", str(SR),
                            "-f", "s16le", "-"], capture_output=True, **vio.NOWIN)
        a = np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0
        peaks = []
        if a.size:
            per = SR // PPS
            a = a[:a.size // per * per].reshape(-1, per)
            pk = np.sqrt(np.abs(a).max(1))          # sqrt: quiet passages stay visible, like Premiere's log view
            peaks = np.clip(pk * 255, 0, 255).astype(int).tolist()
        vio.save_json(os.path.join(proj, "media", "meta.json"),
                      {"step": step, "count": count, "pps": PPS, "audio": bool(peaks), "peaks": peaks})
    except Exception:
        traceback.print_exc()
    finally:
        MEDIA_BUSY.discard(proj)


# ------------------------------------------------------------------ live edits
STYLE_KEYS = {"font", "weight", "wdth", "size_ratio", "fill", "align", "nudge_x", "nudge_y"}


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
            was = (t.get("he", ""), t.get("translate", True), t.get("mode", "replace"), t.get("layer"), dict(t.get("style") or {}))
            if "he" in e and e["he"] != t.get("he", ""):
                t["he"] = e["he"]; relayer.add(t["id"])
            if "translate" in e and bool(e["translate"]) != t.get("translate", True):
                t["translate"] = bool(e["translate"]); relayer.add(t["id"])
            if "mode" in e and e["mode"] in ("replace", "remove") and e["mode"] != t.get("mode", "replace"):
                t["mode"] = e["mode"]; relayer.add(t["id"])
            if "logo_full" in e and bool(e["logo_full"]) != bool(t.get("logo_full")):
                t["logo_full"] = bool(e["logo_full"]); t["reextract"] = True; t.pop("logo_bg_done", None)
                if not t["logo_full"]:                  # back to the text only
                    for k in ("logo_bg",):
                        t.pop(k, None)
                    if "text_box" in t:
                        t["box"] = t.pop("text_box")
                    if "text_ref_box" in t:
                        t["ref_box"] = t.pop("text_ref_box")
                for o in T["titles"]:                   # the card's region is recomputed
                    if o.get("group") == t.get("group"):
                        o.pop("region", None)
                relayer.add(t["id"])
            if e.get("reveal_dir") in ("auto", "mirror", "same") and e["reveal_dir"] != t.get("reveal_dir", "auto"):
                from . import motion
                t["reveal_dir"] = e["reveal_dir"]
                if t.get("anim"):
                    t["anim"]["mirror"] = motion.mirror_default(t) if e["reveal_dir"] == "auto" else e["reveal_dir"] == "mirror"
            if "layer" in e and e["layer"] != t.get("layer"):
                if e["layer"]:
                    t["layer"] = e["layer"]
                else:
                    t.pop("layer", None)
                relayer.add(t["id"])
            st = e.get("style") or {}
            cur = t.setdefault("style", {})
            for k in [k for k, v in st.items() if k in STYLE_KEYS and v is None and k in cur]:   # undo to "never set"
                cur.pop(k); relayer.add(t["id"])
                t["style_locked"] = [x for x in t.get("style_locked", []) if x != k]
            ch = {k: v for k, v in st.items() if k in STYLE_KEYS and v is not None and cur.get(k) != v}
            if ch:
                cur.update(ch); relayer.add(t["id"])
                t["style_locked"] = sorted(set(t.get("style_locked", [])) | set(ch))
            for k in ("in", "out"):
                if k in e and e[k] is not None and int(e[k]) != t.get(k):
                    t[k] = int(e[k]); t[k + "_override"] = int(e[k])
            if t.get("from_series") and was != (t.get("he", ""), t.get("translate", True), t.get("mode", "replace"), t.get("layer"), dict(t.get("style") or {})):
                t["series_override"] = True       # changed here: stays local until pushed to the whole series
        vio.save_json(P("titles.json"), T)
        name = series_name()
        if name and relayer:
            from . import series
            try:
                series.learn(name, STATE["proj"], T["titles"])
            except Exception:
                traceback.print_exc()
    if relayer and os.path.exists(P("layers", "meta.json")):
        from . import render
        render.build_layers(STATE["proj"], only=relayer)
    return relayer


def series_name():
    return vio.load_json(P("project.json"), {}).get("series", "") if STATE["proj"] else ""


def series_apply(only=None):
    """Fill titles from the project's series; re-render what changed."""
    from . import series
    name = series_name()
    if not name or not os.path.exists(P("titles.json")):
        return 0
    with LOCK:
        T = vio.load_json(P("titles.json"))
        ch = series.apply(name, STATE["proj"], T, only=only)
        if ch:
            vio.save_json(P("titles.json"), T)
    if ch and os.path.exists(P("layers", "meta.json")):
        from . import render
        render.build_layers(STATE["proj"], only=ch)
    return len(ch)


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


def _checker(H, W, s=16):
    y, x = np.mgrid[0:H, 0:W]
    c = (((x // s) + (y // s)) % 2).astype(np.uint8)
    return np.repeat((40 + c * 22)[..., None], 3, 2)


def render_frame(f, mode, width, layers=True, bg=True):
    """layers=False: the cleaned picture without the Hebrew. bg=False: the Hebrew alone, on a checkerboard."""
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
        if not bg:
            L = compose.overlay_frame(STATE["proj"], f, T, meta, cache, W, H) if layers else None
            base = _checker(H, W).astype(np.float32) / 255
            fr = np.clip(((L[..., :3] + base * (1 - L[..., 3:4])) if L is not None else base) * 255 + 0.5, 0, 255).astype(np.uint8)
        else:
            fr = compose.frame_result(STATE["proj"], f, fr, T, meta if layers else {}, cache, W, H)
    elif not bg:
        fr = _checker(H, W)
    if width and width < W:
        fr = cv2.resize(fr, (width, int(H * width / W)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jpg.tobytes()


def title_box(tid, f):
    """Where the Hebrew layer of a title sits at frame f: corner polygons (frame px) of every layer + the
    position it would have with no manual offset (for snapping)."""
    from . import motion
    from PIL import Image
    T = vio.load_json(P("titles.json")); meta = vio.load_json(P("layers", "meta.json"), {})
    t = next((x for x in T["titles"] if x["id"] == tid), None)
    if not t or tid not in meta:
        return None
    m = meta[tid]
    anim = t.get("anim", {"mode": "static"})
    v = (anim.get("frames") or {}).get(str(f)) if anim.get("mode") == "match" else None
    dx, dy, sc = (v[0], v[1], v[2]) if v else (0, 0, 1)
    polys = []
    for i, pm in enumerate(m.get("parts") or [m]):
        if not pm.get("file") or not os.path.exists(pm["file"]):
            continue
        k_ = (pm["file"], os.path.getmtime(pm["file"]))
        if k_ not in INK:
            im_ = Image.open(pm["file"])
            INK[k_] = (im_.getchannel("A").getbbox() if im_.mode == "RGBA" else None) or (0, 0, im_.width, im_.height)
        bx0, by0, bx1, by1 = INK[k_]                 # the drawn letters, not the padding around them
        ss = pm["ss"]; ax, ay = pm["anchor"]; tx, ty = pm["target"]
        Hm = motion.homography(anim, f, i if "parts" in m else None) if anim.get("mode") == "match" else None
        pts = []
        for x, y in ((bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)):
            if Hm is not None:
                X = x / ss + tx - ax / ss; Y = y / ss + ty - ay / ss
                q = Hm @ np.array([X, Y, 1.0]); pts.append([q[0] / q[2], q[1] / q[2]])
            else:
                k = sc / ss
                pts.append([k * x + tx + dx - ax * k, k * y + ty + dy - ay * k])
        polys.append(pts)
    if not polys:
        return None
    st = t.get("style", {})
    return {"polys": polys, "nudge": [st.get("nudge_x", 0), st.get("nudge_y", 0)], "size_ratio": st.get("size_ratio", 0.9 if (t.get("layer") or {}).get("type", "text") == "text" else 1.0),
            "in": t["in"], "out": t["out"]}


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
        raw = self.rfile.read(n) or b"{}"
        if "x-www-form-urlencoded" in (self.headers.get("Content-Type") or ""):
            return parse_qs(raw.decode("utf-8"), keep_blank_values=True)
        return json.loads(raw)

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
        if u.path == "/cuelint.web.js":             # CueLint (MIT), subtitle QC, runs in the page
            with open(os.path.join(UI, "cuelint.web.js"), "rb") as f:
                return self._send(200, f.read(), "text/javascript; charset=utf-8")
        if u.path == "/splash.png":                  # the start-up picture, drawn by the page itself (same size as the splash)
            with open(os.path.join(UI, "splash.png"), "rb") as f:
                return self._send(200, f.read(), "image/png")
        if u.path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if u.path == "/api/google":
            from . import gdrive
            return self._send(200, gdrive.status())
        if u.path == "/api/ping":
            return self._send(200, {"app": "retitle"})
        if u.path == "/api/recent":
            return self._send(200, {"items": recent()})
        if u.path == "/api/series":
            from . import series
            return self._send(200, {"current": series_name(), "list": series.list_all()})
        if u.path == "/api/status":
            from . import asr
            if "f" in q:
                from . import notify
                notify.report_focus(q["f"][0] == "1")
            return self._send(200, {"asr": asr.STATE, "proj": STATE["proj"], "file": STATE.get("file"), "missing_video": STATE.get("missing_video", False), "running": STATE["running"], "job": STATE["job"],
                                    "error": STATE["error"], "log": STATE["log"], "progress": STATE["progress"],
                                    "stage": STATE["stage"], "step": STATE["step"], "steps": STATE["steps"], "phase": PHASE["name"],
                                    "queue": QUEUE,
                                    "elapsed": round((time.time() if STATE["running"] else STATE.get("t1", time.time())) - STATE["t0"], 1) if STATE["t0"] else 0,
                                    "cancelled": STATE["cancelled"], "cancelling": STATE["cancel"] and STATE["running"], "claude_turn": bool(PHASE.get("claude_turn")),
                                    "live": _live()})
        if u.path == "/api/live":
            f = os.path.join(STATE["proj"] or "", "live", "live.jpg")
            if not STATE["proj"] or not os.path.exists(f):
                return self._send(404, {"error": "none"})
            with open(f, "rb") as fh:
                data = fh.read()
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data); return
        if u.path == "/api/dicta":
            from . import dicta
            return self._send(200, dicta.status())
        if u.path == "/api/compare":
            return self._send(200, vio.load_json(P("compare.json"), {"rows": []}) if STATE.get("proj") else {"rows": []})
        if u.path == "/api/propainter":
            from . import propaint
            return self._send(200, propaint.status())
        if u.path == "/api/settings":
            s = settings(); s["api_key_set"] = s.get("api_key_saved", False); s["api_key"] = ""
            s["gh_token_set"] = bool(s.get("gh_token")); s["gh_token"] = ""
            s["tmdb_key_set"] = bool(s.get("tmdb_key")); s["tmdb_key"] = ""
            s["sync_dir"] = vio.load_json(CFG, {}).get("sync_dir", ""); s["sync_found"] = detect_sync_dirs()
            return self._send(200, s)
        if u.path == "/api/fonts":
            from . import fontlib
            cat = fontlib.catalog(refresh=q.get("refresh", ["0"])[0] == "1")
            return self._send(200, {n: {"he": v.get("he", ""), "source": v["source"], "installed": v["installed"],
                                        "url": v.get("url", "") if not v["installed"] else ""} for n, v in cat.items()})
        if u.path == "/api/fontfile":
            from . import fontlib
            try:
                p = fontlib.ensure(q.get("name", [""])[0])
            except Exception:
                p = None
            if not p:
                return self._send(404, {"error": "font"})
            with open(p, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "font/otf" if p.lower().endswith(".otf") else "font/ttf")
            self.send_header("Cache-Control", "max-age=86400")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if not STATE["proj"]:
            return self._send(400, {"error": "no project"})
        if u.path == "/api/project":
            PJ = vio.load_json(P("project.json"))
            T = vio.load_json(P("titles.json"), {"titles": []})
            S = vio.load_json(P("shots.json"), {"cuts": [], "n_frames": 0})
            px = P("proxy.mp4")
            return self._send(200, {
                "project": PJ, "file": STATE.get("file"), "missing_video": not os.path.isfile(PJ.get("video", "")), "titles": T, "dialog": vio.load_json(P("dialog.json"), {"cues": []}),
                "cuts": S.get("cuts", []),
                "n_frames": PJ.get("n_frames") or S.get("n_frames") or int(round(PJ["info"]["duration"] * PJ["info"]["fps"])),
                "dirty": sorted(dirty_groups(T)) if T["titles"] and any("built" in t for t in T["titles"]) else [],
                "built": os.path.exists(P("layers", "meta.json")),
                "proxy": os.path.exists(px), "proxy_mtime": os.path.getmtime(px) if os.path.exists(px) else 0,
                "outputs": sorted(glob.glob(P("out", "*"))),
                "checks": sorted(os.path.basename(x) for x in glob.glob(P("check", "*.jpg"))),
                "stats": vio.load_json(P("clean_stats.json"), {})})
        if u.path == "/api/media":
            m = P("media", "meta.json")
            if os.path.exists(m):
                with open(m, "rb") as f:
                    return self._send(200, f.read())
            if STATE["proj"] not in MEDIA_BUSY:
                threading.Thread(target=build_media, args=(STATE["proj"],), daemon=True).start()
            return self._send(200, {"ready": False})
        if u.path == "/api/titlebox":
            try:
                return self._send(200, title_box(q.get("id", [""])[0], int(q.get("f", ["0"])[0])) or {})
            except Exception as e:
                traceback.print_exc()
                return self._send(200, {"error": str(e)})
        if u.path == "/api/fontmatch":
            f = P("fontmatch", q.get("id", [""])[0] + ".json")
            return self._send(200, vio.load_json(f, {}) if os.path.exists(f) else {})
        if u.path == "/api/video":
            which = q.get("which", ["src"])[0]
            path = P("proxy.mp4") if which == "proxy" else (play_path() or vio.load_json(P("project.json"))["video"])
            if not os.path.exists(path):
                return self._send(404, {"error": "no video"})
            return self._video(path)
        if u.path == "/api/frame":
            f = int(q.get("f", ["0"])[0]); mode = q.get("mode", ["result"])[0]; w = int(q.get("w", ["0"])[0])
            try:
                jpg = render_frame(f, mode, w, q.get("lay", ["1"])[0] != "0", q.get("bg", ["1"])[0] != "0")
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
        if u.path == "/api/agent_run":                 # full mode: Claude Code runs the engine (rt.cmd)
            if not AGENT["token"] or (b.get("token") or [""])[0] != AGENT["token"]:
                return self._send(403, {"error": "token"})
            out = agent_run(b.get("a") or [])
            return self._send(200, out.encode("utf-8"), "text/plain; charset=utf-8")
        if u.path == "/api/pick":
            p = pick_file()
            if p:
                try:
                    open_project(p)
                except Exception as e:
                    return self._send(200, {"proj": None, "error": str(e)})
            return self._send(200, {"proj": STATE["proj"] if p else None, "file": STATE.get("file")})
        if u.path == "/api/open":
            return self._send(200, {"proj": open_project(b["video"])})
        if u.path == "/api/queue":
            act = b.get("act")
            if act == "add":
                for pth in (b.get("paths") or ([pick_file("video")] if not b.get("path") else [b["path"]])):
                    if pth and not any(q["path"] == pth for q in QUEUE):
                        QUEUE.append({"path": pth, "name": os.path.basename(pth), "status": "waiting", "msg": ""})
            elif act == "remove":
                QUEUE[:] = [q for q in QUEUE if q["path"] != b.get("path") or q["status"] == "running"]
            elif act == "clear_done":
                QUEUE[:] = [q for q in QUEUE if q["status"] != "done"]
            return self._send(200, {"queue": QUEUE})
        if u.path == "/api/auto_plan":
            if not STATE["proj"]:
                return self._send(200, {"parts": []})
            return self._send(200, {"parts": [n for _, n, _ in auto_parts(b.get("opts") or {})],
                                    "key": bool(settings().get("api_key"))})
        if u.path == "/api/design_back":
            from . import design, render
            with LOCK:
                ok = design.back_card(STATE["proj"], b["id"])
            if ok and os.path.exists(P("layers", "meta.json")):
                render.build_layers(STATE["proj"], only={b["id"]})
            return self._send(200, {"ok": ok})
        if u.path == "/api/undo_design":
            with LOCK:
                undo_design(b["id"])
            from . import render
            if os.path.exists(P("layers", "meta.json")):
                render.build_layers(STATE["proj"])
            return self._send(200, {"ok": True})
        if u.path == "/api/range":
            if not STATE["proj"]:
                return self._send(400, {"error": "no project"})
            PJ = vio.load_json(P("project.json"))
            r = b.get("range")
            if r:
                n = PJ.get("n_frames") or int(round(PJ["info"]["duration"] * PJ["info"]["fps"]))
                a, z = sorted(int(x) for x in r[:2])
                PJ["range"] = [max(0, a), min(n - 1, z)]
            else:
                PJ.pop("range", None)
            with LOCK:
                vio.save_json(P("project.json"), PJ)
            return self._send(200, {"range": PJ.get("range")})
        if u.path == "/api/settings":
            s = vio.load_json(CFG, {})          # the stored file (settings() holds the resolved Claude route)
            for k, v in b.items():
                if k in ("api_key", "gh_token", "tmdb_key") and not v:
                    continue
                s[k] = v
            os.makedirs(CFG_DIR, exist_ok=True)
            vio.save_json(CFG, s)
            r = sync_root()
            if r:
                vio.save_json(os.path.join(r, "settings.json"), {k: s[k] for k in SHARED if k in s})
            if "whisper_model" in b:
                from . import asr
                asr.prepare(s.get("whisper_model", "large-v3"))
            return self._send(200, {"ok": True})
        if u.path == "/api/notify_test":
            from . import notify
            notify.toast("Retitle", "כך תיראה התראה כשעבודה ארוכה מסתיימת")
            return self._send(200, {"ok": True})
        if u.path == "/api/fonts/import":
            from . import fontlib
            paths = pick_files("font")
            if not paths:
                return self._send(200, {"ok": False})
            added, bad = fontlib.import_files(paths)
            return self._send(200, {"ok": True, "added": added, "bad": bad})
        if u.path.startswith("/api/google/"):
            from . import gdrive
            act = u.path.rsplit("/", 1)[1]
            try:
                if act == "signin":
                    email = gdrive.sign_in()
                    gdrive.pull(force=True); gdrive.push(force=True)
                    return self._send(200, {"ok": True, "email": email})
                if act == "backup":
                    gdrive.push(force=True); return self._send(200, {"ok": True})
                if act == "restore":
                    gdrive.pull(force=True); return self._send(200, {"ok": True})
                if act == "signout":
                    gdrive.sign_out(); return self._send(200, {"ok": True})
            except Exception as e:
                return self._send(200, {"ok": False, "error": str(e)})
            return self._send(404, {"error": "?"})
        if u.path == "/api/sync":
            d = b.get("dir")
            if d == "pick":
                d = pick_folder()
                if not d:
                    return self._send(200, {"ok": False})
            if d and not os.path.isdir(d):
                return self._send(200, {"ok": False, "error": "התיקייה לא קיימת"})
            try:
                enable_sync(d or "")
            except Exception as e:
                return self._send(200, {"ok": False, "error": str(e)})
            return self._send(200, {"ok": True, "dir": d or ""})
        if u.path == "/api/fullscreen":
            global FULL
            on = bool(b.get("on"))
            if WINDOW is not None and on != FULL:
                try:
                    WINDOW.toggle_fullscreen(); FULL = on
                except Exception:
                    traceback.print_exc()
            return self._send(200, {"ok": True})
        if u.path == "/api/cancel":
            if STATE["running"]:
                STATE["cancel"] = True
                log("מבטל...")
            return self._send(200, {"ok": True})
        if u.path == "/api/run":
            if STATE["running"]:
                return self._send(200, {"started": False})
            STATE["args"] = b.get("args") or {}
            if (b["job"] in ("review", "design", "design_edit", "translate_compare") or
                    (b["job"] in ("translate_titles", "translate_dialog") and settings().get("translate_engine") != "dicta")) \
                    and not settings().get("api_key"):
                return self._send(200, {"started": False, "need": "api_key"})
            return self._send(200, {"started": start(b["job"])})
        if u.path == "/api/dicta/install":
            from . import dicta
            dicta.install_async()
            return self._send(200, {"ok": True})
        if u.path == "/api/compare/pick":
            with LOCK:
                T = vio.load_json(P("titles.json"))
                ch = {str(k): v for k, v in (b.get("he") or {}).items()}
                for t in T["titles"]:
                    if t["id"] in ch:
                        t["he"] = ch[t["id"]]
                vio.save_json(P("titles.json"), T)
            from . import render
            if os.path.exists(P("layers", "meta.json")):
                render.build_layers(STATE["proj"], only=set(ch))
            return self._send(200, {"ok": True, "n": len(ch)})
        if u.path == "/api/propainter/install":
            from . import propaint
            propaint.install_async()
            return self._send(200, {"ok": True})
        if u.path == "/api/update/check":
            from . import update
            try:
                r = update.check(settings().get("gh_token", ""))
            except Exception as e:
                return self._send(200, {"error": str(e), "current": update.BUILD})
            UPDATE.clear(); UPDATE.update(r)
            return self._send(200, r)
        if u.path in ("/api/open_file", "/api/open_project"):
            path = b.get("path") or pick_file("retitle")
            if not path:
                return self._send(200, {"proj": None})
            try:
                open_file(path)
                bring_front()
            except Exception as e:
                return self._send(200, {"proj": None, "error": str(e)})
            return self._send(200, {"proj": STATE["proj"], "file": STATE["file"], "missing_video": STATE["missing_video"]})
        if not STATE["proj"]:
            return self._send(400, {"error": "no project"})
        if u.path == "/api/series":
            from . import series
            name = (b.get("name") or "").strip()
            pj = vio.load_json(P("project.json")); pj["series"] = name; vio.save_json(P("project.json"), pj)
            learned = applied = 0
            if name and os.path.exists(P("titles.json")):
                applied = series_apply()
                learned = series.learn(name, STATE["proj"], vio.load_json(P("titles.json"))["titles"])
            return self._send(200, {"ok": True, "applied": applied, "learned": learned})
        if u.path in ("/api/series/push", "/api/series/revert"):
            from . import series
            name = series_name(); tid = b.get("id")
            if not name:
                return self._send(200, {"ok": False})
            with LOCK:
                T = vio.load_json(P("titles.json"))
                t = next((x for x in T["titles"] if x["id"] == tid), None)
                if t:
                    t.pop("series_override", None)
                    if u.path.endswith("push"):
                        t["from_series"] = True
                    vio.save_json(P("titles.json"), T)
            if t and u.path.endswith("push"):
                series.learn(name, STATE["proj"], [t], force_ids={tid})
            series_apply(only={tid}) if t else None
            return self._send(200, {"ok": bool(t)})
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
        if u.path == "/api/import_srt":
            from . import srt
            p = pick_file("srt")
            if not p:
                return self._send(200, {"ok": False})
            with LOCK:
                cues = srt.import_srt(STATE["proj"], p, bool(b.get("hebrew")))
            log(f"יובאו {len(cues)} כתוביות מ-{os.path.basename(p)}")
            return self._send(200, {"ok": True, "n": len(cues)})
        if u.path == "/api/hardsubs":
            from . import srt
            with LOCK:
                cues = srt.from_hardsubs(STATE["proj"])
            log(f"{len(cues)} כתוביות צרובות הועברו לרשימת הכתוביות")
            return self._send(200, {"ok": True, "n": len(cues)})
        if u.path == "/api/logo":
            import shutil
            tid = b["id"]
            if b.get("layers"):
                from . import motion, render
                paths = pick_files("layers")
                if not paths:
                    return self._send(200, {"ok": False})
                try:
                    files = motion.import_layers(STATE["proj"], tid, paths)
                except Exception as e:
                    return self._send(200, {"ok": False, "error": str(e)})
                apply_titles([{"id": tid, "layer": {"type": "parts", "files": files, "t": time.time()}}])
                T = vio.load_json(P("titles.json"))
                t = next(x for x in T["titles"] if x["id"] == tid)
                if os.path.isdir(P("clean", t["group"])):          # each layer gets the motion of the part under it
                    render.track(STATE["proj"], only={t["group"]})
                if os.path.exists(P("layers", "meta.json")):
                    render.build_layers(STATE["proj"], only={tid})
                return self._send(200, {"ok": True, "n": len(files)})
            if b.get("clear"):
                apply_titles([{"id": tid, "layer": None}])
                return self._send(200, {"ok": True})
            p = pick_file("png")
            if not p:
                return self._send(200, {"ok": False})
            os.makedirs(P("logos"), exist_ok=True)
            shutil.copyfile(p, P("logos", f"{tid}.png"))
            apply_titles([{"id": tid, "layer": {"type": "image", "path": f"logos/{tid}.png", "t": time.time()}}])
            return self._send(200, {"ok": True})
        if u.path == "/api/qa":
            from . import qa
            vis = vio.load_json(P("visual_qa.json"), {"issues": []})["issues"]
            return self._send(200, {"issues": qa.run(STATE["proj"]) + vis})
        if u.path == "/api/save":
            from . import project
            if not STATE.get("file"):
                return self._send(200, {"ok": False})
            pj = vio.load_json(P("project.json"))
            project.write(STATE["file"], STATE["proj"], pj["video"])
            return self._send(200, {"ok": True, "file": STATE["file"]})
        if u.path == "/api/save_as":
            from . import project
            if not STATE.get("file"):
                return self._send(200, {"ok": False})
            if STATE["running"]:
                return self._send(200, {"ok": False, "error": "פעולה אחרת עדיין רצה"})
            cur = STATE["file"]
            path = save_file(os.path.splitext(os.path.basename(cur))[0] + " - עותק.retitle", os.path.dirname(cur))
            if not path:
                return self._send(200, {"ok": False})
            if os.path.normcase(os.path.abspath(path if path.lower().endswith(".retitle") else path + ".retitle")) == os.path.normcase(os.path.abspath(cur)):
                project.write(cur, STATE["proj"], vio.load_json(P("project.json"))["video"])
                return self._send(200, {"ok": True, "file": cur})
            try:
                nf, nd = project.save_as(cur, STATE["proj"], path)
                _activate(nf, nd)
            except Exception as e:
                return self._send(200, {"ok": False, "error": str(e)})
            return self._send(200, {"ok": True, "file": nf, "proj": nd})
        if u.path == "/api/relink":
            from . import project
            path = pick_file("video")
            if not path:
                return self._send(200, {"ok": False})
            try:
                project.relink(STATE["file"], STATE["proj"], path)
                _activate(STATE["file"], STATE["proj"])
            except Exception as e:
                return self._send(200, {"ok": False, "error": str(e)})
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


def _handoff(path):
    """If Retitle is already open, ask it to open `path` and return True."""
    import urllib.request
    for p in range(8765, 8775):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{p}/api/ping", timeout=0.4) as r:
                if json.load(r).get("app") != "retitle":
                    continue
            req = urllib.request.Request(f"http://127.0.0.1:{p}/api/open_file", data=json.dumps({"path": path}).encode(), method="POST")
            urllib.request.urlopen(req, timeout=30).read()
            return True
        except Exception:
            continue
    return False


def tcode(f):
    fps = vio.load_json(P("project.json"))["info"]["fps"] or 24
    t = f / fps
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def close_splash():
    """The start-up picture (PyInstaller splash) goes away once the window is ready."""
    try:
        import pyi_splash
        pyi_splash.close()
    except Exception:
        pass


def _after_update():
    """Started by an update: wait for the old copy to close, then tidy up what it set aside."""
    if "--wait-pid" in sys.argv:
        from . import update
        i = sys.argv.index("--wait-pid")
        try:
            update.wait_pid(int(sys.argv[i + 1]))
        except Exception:
            traceback.print_exc()
        del sys.argv[i:i + 2]
    if getattr(sys, "frozen", False):
        try:
            from . import update
            update.cleanup_old()
        except Exception:
            traceback.print_exc()


PREP_HTML = """<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8"><title>Retitle - מכין רכיבים</title>
<style>body{font:18px/1.6 "Segoe UI",Arial,sans-serif;background:#1b1b1f;color:#f2f2f2;margin:0;padding:28px}
h1{font-size:24px;margin:0 0 6px}p{margin:0 0 18px;color:#cfcfcf}.it{margin:16px 0}.nm{font-weight:600}
progress{width:100%;height:22px;accent-color:#ffb84d}.st{font-size:16px;color:#d8d8d8;min-height:1.4em}
button{font:inherit;font-size:18px;padding:10px 22px;border-radius:8px;border:2px solid #ffb84d;background:#2a2a30;color:#fff;margin-top:18px}
button:focus{outline:3px solid #8cc8ff;outline-offset:2px}@media (prefers-reduced-motion:reduce){*{transition:none!important}}</style></head>
<body><main><h1>מכין את הרכיבים של Retitle</h1><p>הורדה חד-פעמית. אפשר לסגור את החלון - מה שלא הסתיים יורד כשהתוכנה נפתחת.</p>
<div id="list"></div><div role="status" aria-live="polite" id="done"></div><button id="close">סגור</button></main>
<script>
const L=document.getElementById('list');document.getElementById('close').onclick=()=>window.pywebview.api.close();
async function tick(){const r=await window.pywebview.api.status();L.innerHTML='';
 for(const it of r.items){const d=document.createElement('div');d.className='it';
  d.innerHTML='<div class="nm" id="n'+it.k+'">'+it.name+'</div>'+(it.p!=null?'<progress max="100" value="'+Math.round(it.p*100)+'" aria-labelledby="n'+it.k+'"></progress>':'')+'<div class="st">'+it.st+'</div>';L.appendChild(d)}
 document.getElementById('done').textContent=r.done?'הכול מוכן.':'';if(!r.done)setTimeout(tick,1000)}
window.addEventListener('pywebviewready',tick);
</script></body></html>"""


def prepare_ui():
    """Retitle.exe --prepare (run by the installer): downloads the transcription model, the speech separation
    model and, with an NVIDIA card, the ProPainter engine, with a small accessible progress window."""
    import webview
    from . import asr, vocals, propaint
    name = settings().get("whisper_model", "large-v3")
    asr.prepare(name)
    threading.Thread(target=lambda: vocals.ready() or vocals.fetch(), daemon=True).start()
    want_pp = propaint.gpu()
    if want_pp and not propaint.all_ready():
        propaint.install_async()

    class Api:
        def status(self):
            items = []
            a = asr.STATE
            ok = asr.ready(name) and a.get("state") != "downloading"
            items.append({"k": 1, "name": "מנוע התמלול (Whisper)", "p": 1.0 if ok else a.get("progress"),
                          "st": "מוכן" if ok else ("שגיאה: " + a.get("error", "")) if a.get("state") == "error" else "מוריד..."})
            v = vocals.ready()
            items.append({"k": 2, "name": "הפרדת דיבור ממוזיקה", "p": 1.0 if v else (getattr(vocals, "STATE", {}) or {}).get("progress"),
                          "st": "מוכן" if v else "מוריד..."})
            done = ok and v
            if want_pp:
                ps = propaint.status()
                items.append({"k": 3, "name": "מחיקה חזקה (ProPainter, לכרטיס NVIDIA)", "p": 1.0 if ps["ready"] else ps.get("progress"),
                              "st": "מוכן" if ps["ready"] else ("שגיאה: " + ps.get("error", "")) if ps.get("state") == "error" else (ps.get("what") or "מכין...")})
                done = done and (ps["ready"] or ps.get("state") == "error")
            return {"items": items, "done": done}

        def close(self):
            for w in list(webview.windows):
                w.destroy()

    webview.create_window("Retitle - מכין רכיבים", html=PREP_HTML, js_api=Api(), width=620, height=520)
    webview.start()


def main():
    global WINDOW
    try:
        from . import notify
        notify.set_app_id()
    except Exception:
        pass
    _after_update()
    if "--prepare" in sys.argv:
        close_splash()
        prepare_ui()
        return
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args and os.path.isfile(args[0]):
        if _handoff(os.path.abspath(args[0])):
            close_splash()
            return
        try:
            open_file(os.path.abspath(args[0]))
        except Exception:
            traceback.print_exc()
    def _warm():
        try:
            from . import fontlib
            fontlib.catalog()
        except Exception:
            traceback.print_exc()
        try:                                   # Google backup: restore what other computers added, then keep it current
            from . import gdrive
            if "--server" not in sys.argv:
                threading.Thread(target=gdrive.auto_loop, daemon=True).start()
        except Exception:
            traceback.print_exc()
        try:                                   # transcription engine: fetched once in the background
            from . import asr
            if "--server" not in sys.argv:
                asr.prepare(settings().get("whisper_model", "large-v3"))
        except Exception:
            traceback.print_exc()
        try:                                   # speech/music separation model: fetched once too
            from . import vocals
            if "--server" not in sys.argv and settings().get("sep_vocals", True) and not vocals.ready():
                vocals.fetch()
        except Exception:
            traceback.print_exc()
        try:                                   # every Google font with Hebrew, once: font matching compares them all
            from . import fontlib, fontmatch
            if "--server" not in sys.argv and not settings().get("google_fonts_done"):
                ok, n = fontmatch.download_google()
                if ok == n:
                    cfg = vio.load_json(CFG, {}); cfg["google_fonts_done"] = True; vio.save_json(CFG, cfg)
        except Exception:
            traceback.print_exc()
        try:                                   # the strong erase: set up by itself where it runs well (NVIDIA)
            from . import propaint
            if "--server" not in sys.argv and settings().get("auto_propainter", True) and propaint.gpu() and not propaint.all_ready():
                propaint.install_async()
        except Exception:
            traceback.print_exc()
    threading.Thread(target=_warm, daemon=True).start()
    port = free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    AGENT["port"] = port
    srv.daemon_threads = True
    url = f"http://127.0.0.1:{port}/"
    print("Retitle:", url)
    if "--server" in sys.argv or "--no-browser" in sys.argv:
        close_splash()
        srv.serve_forever(); return
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    if "--browser" not in sys.argv:
        try:
            import webview
            kw = dict(width=1600, height=980, min_size=(1100, 700), background_color="#101114")
            try:
                WINDOW = webview.create_window("Retitle", url, maximized=settings().get("open_maximized", True), **kw)
            except TypeError:                 # older pywebview without `maximized`
                WINDOW = webview.create_window("Retitle", url, **kw)
            print("window: pywebview", flush=True)
            # the start-up picture goes before the browser engine starts: when it starts, Windows switches the
            # process to sharp (DPI-aware) drawing and the picture would shrink on screen. The page shows the same
            # picture at the same size until the app is ready.
            close_splash()
            webview.start()
            return
        except Exception:
            traceback.print_exc()
            print("window: fallback to browser", flush=True)
            WINDOW = None
    close_splash()
    webbrowser.open(url)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
