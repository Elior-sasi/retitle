"""Claude looks at every text the OCR found and says what it is: a title to translate, a logo,
credits, text that belongs to the scene (signs, plates, screens), burned-in dialogue, or OCR noise.
It also corrects the reading ("FROMTHE" -> "FROM THE"). Runs at the end of the analysis when a
Claude key is set; the rules in detect.py stay as the fallback without a key."""
import os, json, base64, re, urllib.request
import cv2
from . import vio
from .translate import call, small_jpeg

KINDS = {"title": None, "logo": "לוגו של חברה", "credits": "קרדיטים", "scene": "טקסט בתוך התמונה",
         "dialogue": None, "noise": "רעש של זיהוי"}

SYSTEM = """You check on-screen text found by OCR in a video that will be localized into Hebrew.
For each item you get the line as the OCR read it, a close crop of it, and the whole frame with the
line boxed in yellow. Decide what it is:
- "title": text the editor put on screen for the viewer and that a Hebrew version would translate:
  trailer cards ("COMING SOON", "FROM THE STUDIO THAT BROUGHT YOU"), the film/series title logo,
  chapter cards, lower thirds with names or places, captions, on-screen jokes or labels made for the viewer.
- "logo": a company, studio, channel or distributor logo (Universal, Sky Kids, Netflix, a production company).
- "credits": cast/crew credits, billing blocks, copyright lines, "A ... PRODUCTION", "PRESENTS".
- "scene": text that is part of the filmed/drawn world: signs, licence plates, shop names, screens,
  clothing, packaging, graffiti - unless the shot is clearly about reading it.
- "dialogue": burned-in subtitles of the spoken dialogue.
- "noise": not real text (patterns, drawings or textures read as letters).
Also give "text": the correct reading of the line (fix OCR mistakes and missing spaces), same language,
and for titles "styled": true when the lettering is a designed logo or display art (painted, 3D, textured,
decorated, custom letters) that a plain font would not match; false for ordinary typeset text.
Return ONLY JSON: {"items": [{"id": "...", "kind": "...", "text": "...", "styled": false, "why": "few words"}]}"""


def _jpg(img, w):
    return small_jpeg(img, w, 70)

# sorting is an easy look: the small, cheap model does it well; the strong one is kept for writing
MODEL = "claude-haiku-5-5"


def _content(proj, items):
    out = []
    for t in items:
        img = cv2.imread(os.path.join(proj, "samples", f"{t['ref']:06d}.jpg"))
        if img is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in t["ref_box"])
        h = max(8, y1 - y0)
        crop = img[max(0, y0 - h // 2):y1 + h // 2, max(0, x0 - h):x1 + h]
        full = img.copy()
        cv2.rectangle(full, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 220, 255), max(2, img.shape[1] // 320))
        out.append({"type": "text", "text": f"Item {t['id']} - OCR read: {json.dumps(t['src'], ensure_ascii=False)}"})
        if crop.size:
            out.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpg(crop, 360)}})
        out.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpg(full, 320)}})
    return out


def classify(proj, api_key, model="", only=None, progress=None, log=print):
    """Updates titles.json in place. Returns {kind: count}."""
    path = os.path.join(proj, "titles.json")
    T = vio.load_json(path)
    items = [t for t in T["titles"] if (only is None or t["id"] in only) and not t.get("from_series")
             # credits and known logos are already certain: no need to ask
             and t.get("auto_reason") not in ("קרדיטים", "לוגו של חברה")]
    # the same text again (a plate seen in 8 shots, a repeated caption): ask once, apply to all
    from .detect import norm
    reps, same = [], {}
    for t in items:
        k = (norm(t["src"]), t["group"]) if len(norm(t["src"])) > 0 else (t["id"], "")
        k2 = norm(t["src"])
        hit = next((r for r in reps if norm(r["src"]) == k2 and k2), None)
        if hit:
            same.setdefault(hit["id"], []).append(t["id"])
        else:
            reps.append(t)
    items = reps
    if not items:
        return {}
    res = {}
    step = 20                                  # 20 items (40 small pictures) per request
    for k in range(0, len(items), step):
        bt = items[k:k + step]
        body = {"model": MODEL, "max_tokens": 4000,
                "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": _content(proj, bt) + [{"type": "text", "text": "Classify the items above."}]}]}
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                     headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                              "content-type": "application/json"})
        r = call(req)
        text = "".join(b.get("text", "") for b in r.get("content", []))
        m = re.search(r"\{.*\}", text, re.S)
        for it in (json.loads(m.group(0)).get("items", []) if m else []):
            if it.get("kind") in KINDS:
                res[str(it.get("id"))] = it
                for other in same.get(str(it.get("id")), []):
                    res[other] = it
        if progress:
            progress(min(1.0, (k + step) / len(items)))
    count = {}
    T = vio.load_json(path)                    # re-read: the editor may have saved meanwhile
    for t in T["titles"]:
        it = res.get(t["id"])
        if not it:
            continue
        kind = it["kind"]; count[kind] = count.get(kind, 0) + 1
        t["ai_kind"] = kind
        if it.get("text") and it["text"].strip() and it["text"].strip() != t["src"]:
            t["ocr_src"] = t.get("ocr_src", t["src"]); t["src"] = it["text"].strip()
        if kind == "title":
            t["translate"] = True; t.pop("auto_reason", None)
            t["styled"] = bool(it.get("styled"))
            if t.get("kind") == "sub":
                t["kind"] = "text"; t.pop("mode", None)
        elif kind == "dialogue":
            t["translate"] = True; t["kind"] = "sub"; t["mode"] = "remove"; t.pop("auto_reason", None)
        else:
            t["translate"] = False; t["auto_reason"] = KINDS[kind]; t.pop("mode", None)
            if t.get("kind") == "sub":
                t["kind"] = "text"
    vio.save_json(path, T)
    return count
