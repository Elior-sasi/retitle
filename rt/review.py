"""Claude looks at every text the OCR found and says what it is: a title to translate, a logo,
credits, text that belongs to the scene (signs, plates, screens), burned-in dialogue, or OCR noise.
It also corrects the reading ("FROMTHE" -> "FROM THE"). Runs at the end of the analysis when a
Claude key is set; the rules in detect.py stay as the fallback without a key."""
import os, json, base64, re, urllib.request
import cv2
from . import vio
from .translate import call

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
Also give "text": the correct reading of the line (fix OCR mistakes and missing spaces), same language.
Return ONLY JSON: {"items": [{"id": "...", "kind": "...", "text": "...", "why": "few words"}]}"""


def _jpg(img, w):
    if img.shape[1] > w:
        img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(b.tobytes()).decode()


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
            out.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpg(crop, 640)}})
        out.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpg(full, 480)}})
    return out


def classify(proj, api_key, model="", only=None, progress=None, log=print):
    """Updates titles.json in place. Returns {kind: count}."""
    path = os.path.join(proj, "titles.json")
    T = vio.load_json(path)
    items = [t for t in T["titles"] if (only is None or t["id"] in only) and not t.get("from_series")]
    if not items:
        return {}
    res = {}
    step = 10                                  # 10 items = 20 pictures per request
    for k in range(0, len(items), step):
        bt = items[k:k + step]
        body = {"model": (model or "").strip() or "claude-sonnet-5-5", "max_tokens": 4000, "system": SYSTEM,
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
