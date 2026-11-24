"""Full mode: Claude Code works the video like an editor in chat.

The engine does the heavy lifting; Claude Code (the user's subscription) looks at every sheet the engine
makes, edits titles.json itself, asks for fixes (stronger erase, ProPainter, a design change, a new font)
and looks again, round after round, until every card is right. One Claude Code session runs through the
whole video, so what it learnt on the first look carries to the last one (and the repeated context is
cached, which keeps the subscription use down)."""
import json, os, re, shutil, subprocess, time

from . import vio
from .house import with_house

SYSTEM_T = """You are the editor of a Hebrew localization of a video's burned-in titles. A local engine
(Retitle) erases the source text, tracks its animation and renders the Hebrew; you are the eyes and the judgement,
exactly like a human editor working with it in a chat.

You work in the project folder. The file that matters is titles.json: {"titles":[...]}, one entry per detected text line:
  id, group (the card: lines shown together), src (source text read by OCR), he (the Hebrew), translate (false = leave
  the source as it is: credits, studio logos, English names, text inside the scene), in/out (first/last frame),
  region [x0,y0,x1,y1] (the area that is erased and rebuilt), styled (a designed logo/title), strong (erase harder),
  style {font, size_px, size_ratio, weight, fill [b,g,r], gradient, outline, shadow, extrude, align, text_align,
  max_width, line_gap, nudge_x, nudge_y} (how the Hebrew is drawn), design (Claude's spec of a designed logo; do not
  hand-edit it - ask for "redesign" instead), anim, mask, cache, built (engine bookkeeping - never touch these).
Edit titles.json with the Edit tool (small exact edits; keep it valid JSON).

You have the engine itself, like an editor working with it in a chat. Run it with the command given in the first
message (call it RT; in PowerShell put & before the quoted path). RT <command> ... is the same as `python -m rt`:
{CLI}
RT --py <script.py> [args] runs a Python script you wrote (Write it under the project's agent folder) inside the
engine, with numpy, cv2 and the engine's modules (from rt import vio, clean, render, design, motion, masks ...)
importable; from rt import aimodels gives SAM 2 (aimodels.sam_card(proj, gid) writes sam/<gid>/<frame>.png, used by
the next clean) and Lens (aimodels.lens([image]) names a Latin font, aimodels.embed([images]) compares lettering;
titles carry "lens" = the source's Latin font). When the engine has no tool for a case (a background it smears, a light it does not match, a shape it
cannot copy), write the code for it, the way a developer in the chat would: e.g. fix the cleaned plates of one card,
paint a mask by hand, measure the source's motion yourself, draw an element. Look at what you made before moving on
(preview <proj> <gid> writes crops you can Read; masks writes mask sheets). Never delete the source video or the
project's files you did not make; keep your own files in the agent folder.

Hebrew rules: natural Hebrew a viewer reads in a glance, the official Hebrew title when there is one, no niqqud,
plain " quotes, a regular hyphen - never an em dash, the user's name glossary always wins. A sentence split over
several cards is translated as one sentence and split at the same places.

At the end of every turn answer with ONE JSON object and nothing after it:
{"actions":[...], "done":true|false, "notes_he":"one or two short Hebrew sentences for the user"}
actions (only what is needed; titles.json edits need no action):
  {"do":"clean","gid":G}            erase card G again, harder (letters or a shadow of the source still show)
  {"do":"propaint","gid":G}         erase card G with ProPainter (strong video inpainting; for a background that the
                                    normal erase smears, or text over moving people/water/fire)
  {"do":"sam","gid":G}              follow card G's letters with SAM 2 (exact masks through all its frames, with the
                                    3D side, bevel, glow and shadow) and erase it again - for ghosts and edges left
  {"do":"retrack","gid":G}          measure the source animation of G again (Hebrew drifts, jumps or has the wrong size)
  {"do":"fontmatch","id":ID}        pick the font and look of a title again from the source
  {"do":"design","gid":G}           design a styled card from scratch
  {"do":"redesign","id":ID,"request":"..."}  change a designed logo in words (Hebrew or English), e.g. the light,
                                    the stick, a jewel, the size
"done": true only when every card you looked at is right: no source letters or ghosts left, background clean, the
Hebrew readable, the same size, place, colour, light and animation as the source, and nothing cut or overlapping."""


def system():
    from . import __main__ as cli
    return with_house(SYSTEM_T.replace("{CLI}", cli.__doc__.split("\n", 1)[1]))


def _exe():
    from .translate import claude_cli
    exe = claude_cli()
    if not exe:
        raise RuntimeError("המצב המלא עובד דרך Claude Code - הוא לא נמצא במחשב")
    return exe


class Session:
    """One Claude Code conversation over the whole video (resumed turn after turn)."""

    def __init__(self, proj, model="opus", log=print, cancel=lambda: False, tool=None):
        self.proj, self.model, self.log, self.cancel, self.tool = proj, model, log, cancel, tool
        self.sid = None
        self.turns = 0

    def ask(self, prompt, max_turns=200, timeout=4 * 3600):
        from .translate import USAGE
        exe = _exe()
        cmd = [exe, "-p", "--output-format", "json", "--model", self.model,
               "--allowedTools", "Read", "Edit", "Write", "Glob", "Grep", "Bash", "PowerShell",
               "--add-dir", self.proj, "--max-turns", str(max_turns)]
        if self.sid:
            cmd += ["--resume", self.sid]
        else:
            cmd += ["--append-system-prompt", system()]
            if self.tool:
                prompt = f'RT = "{self.tool}"\n\n' + prompt
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8", cwd=self.proj,
                             env=dict(os.environ, BASH_DEFAULT_TIMEOUT_MS="1800000", BASH_MAX_TIMEOUT_MS="7200000"),
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        import threading
        box = {}
        th = threading.Thread(target=lambda: box.update(zip(("out", "err"), p.communicate(prompt))), daemon=True)
        th.start()
        t0 = time.time()
        while th.is_alive():
            th.join(1.0)
            if self.cancel() or time.time() - t0 > timeout:
                p.kill(); th.join(5)
                if self.cancel():
                    from .app import Cancelled
                    raise Cancelled()
                raise RuntimeError("Claude Code לא ענה בזמן")
        try:
            out = json.loads(box.get("out") or "{}")
        except ValueError:
            out = {}
        if p.returncode != 0 and not out.get("result") or out.get("is_error") and out.get("subtype") != "error_max_turns":
            msg = (out.get("result") or box.get("err") or "").strip()[:300]
            low = msg.lower()
            if "login" in low or "auth" in low:
                raise RuntimeError("Claude Code לא מחובר - פתח אותו פעם אחת וכתוב /login")
            if "limit" in low:
                raise RuntimeError("הגעת למגבלת השימוש של המנוי - נסה שוב מאוחר יותר")
            raise RuntimeError(f"Claude Code: {msg or 'שגיאה'}")
        self.sid = out.get("session_id") or self.sid
        self.turns += 1
        u = out.get("usage") or {}
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            USAGE[k] = USAGE.get(k, 0) + int(u.get(k) or 0)
        USAGE["calls"] = USAGE.get("calls", 0) + 1
        USAGE["via_code"] = True
        return _answer(out.get("result") or "")


def _answer(text):
    """The JSON object at the end of Claude's answer (empty actions when there is none)."""
    for m in reversed(list(re.finditer(r"\{", text))):
        try:
            r = json.loads(text[m.start():text.rindex("}") + 1])
            if isinstance(r, dict) and ("actions" in r or "done" in r):
                r.setdefault("actions", [])
                return r
        except ValueError:
            continue
    return {"actions": [], "done": False, "notes_he": ""}


def snapshot(proj):
    p = os.path.join(proj, "titles.json")
    shutil.copyfile(p, p + ".agent_bak")
    return vio.load_json(p)


def settle(proj, before, log=print):
    """After Claude edited titles.json: restore it when the JSON broke or engine bookkeeping was touched,
    and say which cards changed. Returns (changed group ids, error text or "")."""
    p = os.path.join(proj, "titles.json")
    try:
        T = vio.load_json(p)
        assert isinstance(T.get("titles"), list)
    except Exception:
        shutil.copyfile(p + ".agent_bak", p)
        return set(), "titles.json לא היה JSON תקין אחרי העריכה - שוחזר"
    old = {t["id"]: t for t in before["titles"]}
    keep = ("mask", "cache", "built", "built_region", "anim", "design", "design_prev")
    changed, fixed = set(), False
    for t in T["titles"]:
        o = old.get(t.get("id"))
        if not o:
            changed.add(t.get("group"))
            continue
        if any(t.get(k) != o.get(k) for k in set(t) | set(o) if k not in keep):
            changed.add(t["group"])
            if t.get("he") != o.get("he") or t.get("style") != o.get("style"):
                if not t.get("user_set"):
                    t["user_set"] = True; fixed = True
    if fixed:
        vio.save_json(p, T)
    return changed, ""


def card_list(proj):
    """A short text table of the cards (the model reads the full JSON itself when it needs to)."""
    T = vio.load_json(os.path.join(proj, "titles.json"))
    fps = vio.load_json(os.path.join(proj, "project.json"))["info"]["fps"]
    rows = []
    for t in T["titles"]:
        rows.append(f'{t["id"]} card {t["group"]} {vio.tc(t.get("in", 0), fps)} translate={t.get("translate", True)}'
                    f'{" styled" if t.get("styled") else ""} src="{t.get("src", "")}" he="{t.get("he", "")}"')
    return "\n".join(rows)
