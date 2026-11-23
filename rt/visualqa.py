"""Claude looks at the before/after sheets (check/<group>.jpg) and lists what a viewer would notice:
English left on screen, a smear or ghost where the source was erased, Hebrew cut off, misplaced,
wrongly timed or in a style that clashes with the source. Results: visual_qa.json, shown in QA."""
import os, json, re, base64, glob, urllib.request
import cv2
from . import vio
from .translate import call
from .extract import groups

SYSTEM = """You are the QA reviewer of a Hebrew localization of on-screen titles in a video.
Each sheet is one title card. Every row is one frame: LEFT = the original, RIGHT = the Hebrew version.
The row label says the frame number and whether it is the first (IN) or last (OUT) frame of the title,
or a frame just before/after it. Look for problems a viewer would notice in the RIGHT side:
- English/source text still visible (fully or partly), or Hebrew missing where the source had text;
- a smear, blur, ghost outline, patch or colour block where the source text was erased;
- Hebrew cut off, overlapping something, badly placed, too big/small compared with the source;
- Hebrew that appears too early/late (visible in the before/after frames, or missing at IN/OUT);
- style that clashes with the source (colour, weight, shadow, animation), or wrong reading order;
- a typo or broken word in the Hebrew.
Do not report things that look fine. Write each problem in short, plain Hebrew (no niqqud,
plain " quotes, hyphen not em dash).
Return ONLY JSON: {"issues": [{"sheet": "<sheet id>", "frame": <number or null>, "sev": "err" or "warn", "msg": "..."}]}"""


def _b64(path, w=1400):
    img = cv2.imread(path)
    if img is None:
        return None
    if img.shape[1] > w:
        img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(b.tobytes()).decode()


def run(proj, api_key, model="", progress=None):
    T = vio.load_json(os.path.join(proj, "titles.json"))
    G = groups(T)
    sheets = sorted(glob.glob(os.path.join(proj, "check", "g*.jpg")))
    issues = []
    for k in range(0, len(sheets), 4):
        content = []
        for p in sheets[k:k + 4]:
            gid = os.path.basename(p)[:-4]
            lines = [t for t in G.get(gid, []) if t.get("translate", True)]
            data = _b64(p)
            if not data or not lines:
                continue
            desc = "; ".join(f"{json.dumps(t['src'], ensure_ascii=False)} -> {json.dumps(t.get('he', ''), ensure_ascii=False)}"
                             + (" (erase only)" if t.get("mode") == "remove" else "") for t in lines)
            content.append({"type": "text", "text": f"Sheet {gid}: {desc}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}})
        if content:
            body = {"model": (model or "").strip() or "claude-sonnet-5-5", "max_tokens": 4000, "system": SYSTEM,
                    "messages": [{"role": "user", "content": content + [{"type": "text", "text": "Review the sheets above."}]}]}
            req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                                         headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                                  "content-type": "application/json"})
            r = call(req)
            text = "".join(b.get("text", "") for b in r.get("content", []))
            m = re.search(r"\{.*\}", text, re.S)
            for it in (json.loads(m.group(0)).get("issues", []) if m else []):
                gid = str(it.get("sheet", "")).replace("Sheet", "").strip()
                lines = [t for t in G.get(gid, []) if t.get("translate", True)]
                if not lines or not it.get("msg"):
                    continue
                f = it.get("frame")
                issues.append({"k": "t", "id": lines[0]["id"], "group": gid, "f": int(f) if isinstance(f, (int, float)) else None,
                               "sev": "err" if it.get("sev") == "err" else "warn", "msg": "Claude: " + str(it["msg"]).strip()})
        if progress:
            progress(min(1.0, (k + 4) / max(1, len(sheets))))
    vio.save_json(os.path.join(proj, "visual_qa.json"), {"issues": issues})
    return issues
