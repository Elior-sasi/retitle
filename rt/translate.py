"""Optional automatic translation through the Anthropic API (key in settings). The user edits after."""
import json, re, urllib.request

NIQQUD = re.compile("[" + chr(0x0591) + "-" + chr(0x05BD) + chr(0x05BF) + chr(0x05C1) + chr(0x05C2) + chr(0x05C4) + chr(0x05C5) + chr(0x05C7) + "]")

CONTENT = {
    "trailer": "a film/series trailer - punchy, theatrical",
    "anime": "an anime film or series - keep honorific-free natural Hebrew, character names per glossary",
    "series": "a scripted TV series episode",
    "documentary": "a documentary - neutral, precise, on-screen text is often names/titles (lower thirds), places and dates",
    "reality": "a reality TV show - casual spoken Hebrew, on-screen text is often names and captions",
    "other": "a video",
}


def context(kind, what):
    base = CONTENT.get(kind or "other", CONTENT["other"])
    if what == "titles":
        return f"On-screen text (titles, captions, lower thirds, signs) in {base}."
    return f"Dialogue subtitles for {base}. Max 2 lines of 42 characters per item, condense like a subtitler."


RULES = """You are a professional Hebrew audiovisual localizer (subtitles and on-screen text).
Translate each item into natural, idiomatic Hebrew suited to the content described in the context.
Rules:
- No niqqud at all.
- Use plain double quotes (") and a regular hyphen (-), never an em dash.
- Keep the meaning and tone; on-screen text must be short enough to fit where the source was.
- Keep the split across items: each item is one line/caption and its Hebrew must stand on its own
  but read naturally together with its neighbours (items with the same "group" are one sentence).
- Names: use the glossary spellings exactly. Names not in the glossary: standard Hebrew transliteration.
- Return ONLY a JSON object {"items": [{"id": ..., "he": ...}, ...]} with the same ids."""


def translate(items, api_key, glossary="", context="", model="claude-sonnet-5-5", src_lang="", images=None):
    """items: [{"id", "src", "group"?}] -> {id: he}. images: {id: jpeg path} - the on-screen text as
    it looks; OCR text can be wrong (or unreadable for some languages), the image is the truth."""
    import base64
    if not items:
        return {}
    user = (f"Source language: {src_lang or 'auto'}\n"
            f"Glossary (source = Hebrew):\n{glossary or '-'}\n"
            f"Context: {context or '-'}\n\nItems:\n" + json.dumps(items, ensure_ascii=False, separators=(",", ":")))
    content = []
    seen_groups = set()
    for it in items:
        p = (images or {}).get(it["id"])
        g = it.get("group") or it["id"]
        if p and g not in seen_groups:             # one small picture per card is enough context
            seen_groups.add(g)
            data = small_jpeg(p, 360)
            if data:
                content.append({"type": "text", "text": f"Image of card {g} (item {it['id']}; read the text from it, the OCR 'src' may be wrong):"})
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
    content.append({"type": "text", "text": user})
    body = {"model": (model or "").strip() or "claude-sonnet-5-5", "max_tokens": 8000,
            "system": [{"type": "text", "text": RULES, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    res = call(req)
    text = "".join(b.get("text", "") for b in res.get("content", []))
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0)) if m else {"items": []}
    out = {}
    for it in data.get("items", []):
        he = it.get("he", "")
        he = NIQQUD.sub("", he).replace(chr(0x05BE), "-").replace(chr(0x2014), "-").replace(chr(0x2013), "-")
        he = he.replace(chr(0x201C), '"').replace(chr(0x201D), '"').replace(chr(0x05F4), '"')
        out[str(it["id"])] = he
    return out


def call(req):
    """POST to the Claude API; on an error, say what the API said (in Hebrew where it is a known case)."""
    import urllib.error
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            res = json.loads(r.read().decode())
        u = res.get("usage") or {}
        for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            USAGE[k] = USAGE.get(k, 0) + int(u.get(k) or 0)
        USAGE["calls"] = USAGE.get("calls", 0) + 1
        return res
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode()).get("error", {}).get("message", "")
        except Exception:
            msg = ""
        low = msg.lower()
        if "credit balance" in low:
            raise RuntimeError("אין יתרה בחשבון ה-API של Claude. צריך להוסיף קרדיט ב-platform.claude.com, בחלק של החיוב (Billing)") from None
        if e.code == 401 or "api key" in low or "authentication" in low:
            raise RuntimeError("מפתח ה-API לא תקין - בדוק שהעתקת אותו במלואו להעדפות") from None
        if "model" in low:
            raise RuntimeError(f"שם המודל בהעדפות לא מוכר ל-API ({msg}). אפשר להשאיר את השדה ריק") from None
        if e.code == 429 or e.code == 529:
            raise RuntimeError("שרת Claude עמוס כרגע - נסה שוב בעוד דקה") from None
        raise RuntimeError(f"Claude API {e.code}: {msg or e.reason}") from None


USAGE = {}          # tokens used since the job started (shown in the log)


def usage_line():
    if not USAGE.get("calls"):
        return ""
    i = USAGE.get("input_tokens", 0) + USAGE.get("cache_creation_input_tokens", 0)
    c = USAGE.get("cache_read_input_tokens", 0)
    return (f"Claude: {USAGE['calls']} קריאות, {i:,} טוקנים נכנסים" + (f" (+{c:,} מהמטמון, זול פי 10)" if c else "")
            + f", {USAGE.get('output_tokens', 0):,} יוצאים")


def small_jpeg(src, w, q=72):
    """base64 JPEG no wider than w (pictures are most of the tokens: keep them as small as still readable)."""
    import base64, cv2, numpy as np
    img = cv2.imread(src) if isinstance(src, str) else src
    if img is None:
        return None
    if img.shape[1] > w:
        img = cv2.resize(img, (w, max(1, int(img.shape[0] * w / img.shape[1]))), interpolation=cv2.INTER_AREA)
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(b.tobytes()).decode() if ok else None
