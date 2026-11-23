"""Hebrew QA for on-screen titles and subtitles (Elior's delivery conventions)."""
import os, re
from . import vio

NIQQUD = re.compile("[֑-ׇֽֿׁׂׅׄ]")
LATIN = re.compile(r"[A-Za-z]")
CTRL = re.compile("[‎‏‪-‮⁦-⁩]")
CURLY = re.compile("[“”„״‘’׳]")
DASHES = re.compile("[–—‒―]")
# Arabic-origin slang kept out of Hebrew subtitles
SLANG = ["יאללה", "וואלה", "יעני", "חלאס", "סחתיין", "אהלן", "דחילק", "כפרה", "מבסוט", "אחלה", "סבבה", "בסאטה", "מסטול", "פדיחה"]


def text_issues(he, subtitle):
    out = []
    if NIQQUD.search(he):
        out.append(("err", "יש ניקוד"))
    if CURLY.search(he):
        out.append(("err", "מירכאות או גרש מסולסלים / גרשיים - להחליף ב-\" או '"))
    if DASHES.search(he):
        out.append(("err", "קו ארוך (– / —)"))
    lines = he.split("\n")
    for k, l in enumerate(lines):
        body = l.strip()
        # dash used as punctuation inside the text (a speaker dash at the start of line 2 is fine)
        inner = body[1:] if (subtitle and k > 0 and body.startswith("-")) else body
        if re.search(r"\s-\s|\s-$|^-\s", inner) or (k == 0 and subtitle and body.startswith("-") and len(lines) > 1):
            out.append(("err", f"מקף כסימן פיסוק בשורה {k + 1}" if not (k == 0 and body.startswith("-")) else "מקף דובר בשורה הראשונה - בשורה הראשונה אין מקף"))
        if subtitle and k > 0 and body.startswith("- "):
            out.append(("err", f"מקף דובר עם רווח בשורה {k + 1} - צריך \"-טקסט\""))
    if LATIN.search(he):
        out.append(("warn", "יש אותיות באנגלית"))
    if CTRL.search(he):
        out.append(("warn", "תווי כיווניות בתוך הטקסט"))
    if "  " in he or he != he.strip() or any(l != l.strip() for l in lines):
        out.append(("warn", "רווחים מיותרים"))
    for w in SLANG:
        if re.search(r"(^|[^א-ת])" + w + r"($|[^א-ת])", he):
            out.append(("warn", f"סלנג ממקור ערבי: {w}"))
    if "..." in he and "…" in he:
        out.append(("warn", "שלוש נקודות בשני סגנונות"))
    return out


def run(proj):
    P = vio.load_json(os.path.join(proj, "project.json"))
    T = vio.load_json(os.path.join(proj, "titles.json"), {"titles": []})
    D = vio.load_json(os.path.join(proj, "dialog.json"), {"cues": []})
    fps = P["info"]["fps"]
    res = []
    for t in T["titles"]:
        mode = t.get("mode", "replace")
        if not t.get("translate", True) or mode == "remove":
            continue
        if (t.get("layer") or {}).get("type") == "image":
            continue
        he = t.get("he", "")
        if not he.strip():
            res.append({"k": "t", "id": t["id"], "sev": "err", "msg": "חסר תרגום"})
            continue
        for sev, m in text_issues(he, False):
            res.append({"k": "t", "id": t["id"], "sev": sev, "msg": m})
        if t.get("warn_shrunk"):
            res.append({"k": "t", "id": t["id"], "sev": "warn", "msg": "רחב מדי למסך, הוקטן אוטומטית"})
    cues = D["cues"]
    for i, c in enumerate(cues):
        he = c.get("he", "")
        if not he.strip():
            res.append({"k": "c", "id": c["n"], "sev": "err", "msg": "חסר תרגום"})
            continue
        for sev, m in text_issues(he, True):
            res.append({"k": "c", "id": c["n"], "sev": sev, "msg": m})
        lines = he.split("\n")
        if len(lines) > 2:
            res.append({"k": "c", "id": c["n"], "sev": "err", "msg": "יותר משתי שורות"})
        ml = max(len(l) for l in lines)
        if ml > 42:
            res.append({"k": "c", "id": c["n"], "sev": "warn", "msg": f"שורה ארוכה ({ml} תווים)"})
        dur = c["end"] - c["start"]
        ch = len(he.replace("\n", ""))
        if dur > 0 and ch / dur > 17:
            res.append({"k": "c", "id": c["n"], "sev": "warn", "msg": f"מהירות קריאה {ch / dur:.1f} תווים לשנייה"})
        if dur < 1.0 - 1e-6:
            res.append({"k": "c", "id": c["n"], "sev": "warn", "msg": f"קצרה משנייה ({dur:.2f} שנ')"})
        if i + 1 < len(cues):
            gap = (cues[i + 1]["start"] - c["end"]) * fps
            if gap < 0:
                res.append({"k": "c", "id": c["n"], "sev": "err", "msg": "חופפת לכתובית הבאה"})
            elif gap < 2 - 1e-6 and c.get("manual"):
                res.append({"k": "c", "id": c["n"], "sev": "warn", "msg": "פחות מ-2 פריימים לכתובית הבאה"})
    return res
