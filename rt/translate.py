"""Optional automatic translation through the Anthropic API (key in settings). The user edits after."""
import json, re, urllib.request

NIQQUD = re.compile("[" + chr(0x0591) + "-" + chr(0x05BD) + chr(0x05BF) + chr(0x05C1) + chr(0x05C2) + chr(0x05C4) + chr(0x05C5) + chr(0x05C7) + "]")

RULES = """You are a professional Hebrew localizer for film trailers.
Translate each item into natural, idiomatic Hebrew suited to a theatrical trailer.
Rules:
- No niqqud at all.
- Use plain double quotes (") and a regular hyphen (-), never an em dash.
- Keep the meaning and tone; on-screen taglines should be short and punchy.
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
            f"Context: {context or '-'}\n\nItems:\n" + json.dumps(items, ensure_ascii=False, indent=1))
    content = []
    for it in items:
        p = (images or {}).get(it["id"])
        if p:
            with open(p, "rb") as f:
                content.append({"type": "text", "text": f"Image of item {it['id']} (read the text from it; the OCR 'src' may be wrong):"})
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                             "data": base64.b64encode(f.read()).decode()}})
    content.append({"type": "text", "text": user})
    body = {"model": model, "max_tokens": 8000, "system": RULES,
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                 headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        res = json.loads(r.read().decode())
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
