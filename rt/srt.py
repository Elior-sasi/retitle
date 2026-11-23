"""Dialogue: Whisper JSON -> cue list for translation -> Hebrew SRT with subtitle timing rules.

Timing rules (defaults, all overridable in project.json "srt"):
  min_dur 1.0s | out = speech end + 0.3s | snap in/out to shot cuts within 7/12 frames |
  2 frames between cues | gaps under 12 frames are closed (chaining) | max 2 lines x 42 chars."""
import os, re, json
from . import vio

DEF = {"min_dur": 1.0, "max_dur": 7.0, "tail": 0.3, "gap_frames": 2, "snap_in": 7, "snap_out": 12,
       "chain_frames": 12, "max_line": 42, "max_cps": 17, "pause_split": 0.6, "rle": True}

END = re.compile(r"[.!?。！？…]$")


def _words(wj):
    segs = wj["segments"] if isinstance(wj, dict) else wj
    out = []
    for s in segs:
        ws = s.get("words") or []
        if ws:
            for w in ws:
                out.append({"s": float(w["start"]), "e": float(w["end"]), "t": w.get("word", w.get("text", ""))})
        else:
            out.append({"s": float(s["start"]), "e": float(s["end"]), "t": s["text"]})
    return out


def prep(proj, whisper_json, lang_cjk=None, rng_s=None):
    P = vio.load_json(os.path.join(proj, "project.json"))
    cfg = {**DEF, **P.get("srt", {})}
    wj = vio.load_json(whisper_json)
    ws = [w for w in _words(wj) if w["t"].strip()]
    if rng_s:                                  # work range: only cues there are rebuilt
        ws = [w for w in ws if rng_s[0] <= w["s"] < rng_s[1]]
    cjk = lang_cjk if lang_cjk is not None else (wj.get("language") in ("ja", "zh", "ko") if isinstance(wj, dict) else False)
    maxc = 30 if cjk else 80
    cues, cur = [], []
    for i, w in enumerate(ws):
        if cur:
            gap = w["s"] - cur[-1]["e"]
            txt = "".join(x["t"] for x in cur) if cjk else " ".join(x["t"].strip() for x in cur)
            if gap > cfg["pause_split"] or w["e"] - cur[0]["s"] > cfg["max_dur"] or len(txt) > maxc \
                    or (END.search(cur[-1]["t"].strip()) and (len(txt) >= 20 or cur[-1]["e"] - cur[0]["s"] >= 0.8)):
                cues.append(cur); cur = []
        cur.append(w)
    if cur:
        cues.append(cur)
    res = []
    for k, c in enumerate(cues):
        txt = "".join(x["t"] for x in c).strip() if cjk else " ".join(x["t"].strip() for x in c)
        res.append({"n": k + 1, "start": round(c[0]["s"], 3), "end": round(c[-1]["e"], 3), "src": txt, "he": ""})
    if rng_s:                                  # cues outside the range keep their text and Hebrew
        old = (vio.load_json(os.path.join(proj, "dialog.json")) or {}).get("cues", [])
        res = sorted([c for c in old if c["end"] <= rng_s[0] or c["start"] >= rng_s[1]] + res, key=lambda c: c["start"])
        for k, c in enumerate(res):
            c["n"] = k + 1
    vio.save_json(os.path.join(proj, "dialog.json"), {"cues": res})
    return res


def wrap(t, n):
    t = " ".join(t.replace("\n", " ").split())
    if len(t) <= n:
        return t
    words = t.split(" ")
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) > n or len(b) > n:
            continue
        score = abs(len(a) - len(b)) - (12 if re.search(r"[,.!?:;\-]$", a) else 0) + (0 if len(a) <= len(b) + 6 else 3)
        if best is None or score < best[0]:
            best = (score, a + "\n" + b)
    return best[1] if best else t


def _fmt(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build(proj, out_name=None, rle=None):
    P = vio.load_json(os.path.join(proj, "project.json"))
    S = vio.load_json(os.path.join(proj, "shots.json"), {"cuts": []})
    D = vio.load_json(os.path.join(proj, "dialog.json"))
    cfg = {**DEF, **P.get("srt", {})}
    if rle is not None:
        cfg["rle"] = rle
    fps = P["info"]["fps"]; F = 1 / fps
    cuts = [c / fps for c in S["cuts"] if c > 0]
    cues = [dict(c) for c in D["cues"] if c.get("he", "").strip()]
    for c in cues:
        a = c["start"]; e = c["end"]
        if c.get("manual"):  # timed by hand in the editor: keep exactly
            c["a"], c["b"] = a, e
            continue
        near = [x for x in cuts if abs(x - a) <= cfg["snap_in"] * F]
        if near:
            a = min(near, key=lambda x: abs(x - a))
        b = e + cfg["tail"]
        near = [x for x in cuts if e - 2 * F <= x <= b + cfg["snap_out"] * F and abs(x - b) <= cfg["snap_out"] * F]
        if near:
            b = min(near, key=lambda x: abs(x - b))
        if b - a < cfg["min_dur"]:
            b = a + cfg["min_dur"]
        c["a"], c["b"] = a, b
    gap = cfg["gap_frames"] * F
    for i in range(len(cues) - 1):
        c, nx = cues[i], cues[i + 1]
        if nx["a"] - c["b"] < gap or (nx["a"] - c["b"] < cfg["chain_frames"] * F and not c.get("manual")):
            c["b"] = nx["a"] - gap
    warns = []
    out = []
    for i, c in enumerate(cues):
        a = round(c["a"] * fps) / fps; b = round(c["b"] * fps) / fps
        txt = wrap(c["he"], cfg["max_line"]) if "\n" not in c["he"] else c["he"]
        lines = txt.split("\n")
        if len(lines) > 2 or max(map(len, lines)) > cfg["max_line"]:
            warns.append(f"#{c['n']} long lines")
        chars = len(txt.replace("\n", ""))
        if b - a < cfg["min_dur"] - 1e-6:
            warns.append(f"#{c['n']} under {cfg['min_dur']}s (squeezed by next cue)")
        if chars / max(0.01, b - a) > cfg["max_cps"]:
            warns.append(f"#{c['n']} {chars / (b - a):.1f} cps")
        if cfg.get("rle", True):
            txt = "\n".join("\u202b" + l.lstrip("\u202b") for l in txt.split("\n"))
        out.append(f"{i + 1}\n{_fmt(a)} --> {_fmt(b)}\n{txt}\n")
    os.makedirs(os.path.join(proj, "out"), exist_ok=True)
    base = os.path.splitext(os.path.basename(P["video"]))[0]
    p = os.path.join(proj, "out", out_name or f"{base}_HEB.srt")
    with open(p, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(out))
    return p, warns


def transcribe(video, out_json, lang=None, model="large-v3", offset=0.0, rng_s=None):
    """Runs where faster-whisper is installed (the PC). word timestamps on."""
    from . import asr

    def go(m):
        segs, info = m.transcribe(video, language=lang, word_timestamps=True, vad_filter=True)
        res = {"language": info.language, "segments": []}
        for s in segs:
            res["segments"].append({"start": s.start, "end": s.end, "text": s.text,
                                    "words": [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])]})
        return res
    m = asr.load(model)
    try:
        res = go(m)
    except Exception:
        if getattr(m.model, "device", "cpu") == "cpu":
            raise
        from faster_whisper import WhisperModel     # GPU libraries missing: fall back to the CPU
        res = go(WhisperModel(asr.model_dir(model) if asr.ready(model) else model, device="cpu", compute_type="int8"))
    res["segments"] = [x for x in res["segments"] if not noise_line(x["text"])]
    if offset:                                 # the audio was a cut of the work range: back to video time
        for x in res["segments"]:
            x["start"] += offset; x["end"] += offset
            for w in x["words"]:
                w["start"] += offset; w["end"] += offset
    if rng_s and os.path.exists(out_json):     # keep what was transcribed outside the range
        old = vio.load_json(out_json) or {}
        res["segments"] = sorted([x for x in old.get("segments", []) if x["end"] <= rng_s[0] or x["start"] >= rng_s[1]]
                                 + res["segments"], key=lambda x: x["start"])
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    return out_json


def noise_line(text):
    """Whisper's sound tags and bare cries, not dialogue: "(buzzing)", "[MUSIC]", "*scream", "Waaaaah",
    "♪♪", "..." ."""
    import re
    t = text.strip()
    if not t:
        return True
    if re.fullmatch(r"[\[\(\*♪#].*", t) and not re.search(r"[\]\)\*♪#]\s*\w", t[1:]):
        return True                                  # the whole line is a tag
    words = re.findall(r"\w+", t)
    if not words:
        return True
    if len(words) == 1 and re.search(r"(\w)\1{3,}", words[0]):
        return True                                  # one stretched cry
    return False


def parse_srt(path):
    """Read an SRT (UTF-8 / UTF-8 BOM / UTF-16 / Windows-1255). Returns [{start, end, text}]."""
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16", "cp1255", "latin-1"):
        try:
            txt = raw.decode(enc)
            if enc == "utf-16" and not raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                continue
            break
        except UnicodeDecodeError:
            continue
    txt = txt.replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for blk in re.split(r"\n\s*\n", txt.strip()):
        lines = [l for l in blk.split("\n")]
        i = next((k for k, l in enumerate(lines) if "-->" in l), None)
        if i is None:
            continue
        m = re.match(r"\s*(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)", lines[i])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        body = "\n".join(lines[i + 1:]).strip()
        body = re.sub(r"<[^>]+>", "", body)            # <i>, <font> ...
        body = re.sub(r"\{\\[^}]*\}", "", body)        # {\an8} ...
        body = body.replace("‫", "").replace("‪", "").replace("‬", "").replace("‏", "").replace("‎", "")
        out.append({"start": round(a, 3), "end": round(b, 3), "text": body})
    return out


def import_srt(proj, path, as_hebrew):
    """as_hebrew: the file is already the Hebrew translation (goes to `he`); otherwise source text to translate."""
    cues = parse_srt(path)
    res = []
    for k, c in enumerate(cues):
        res.append({"n": k + 1, "start": c["start"], "end": c["end"], "manual": True,
                    "src": "" if as_hebrew else c["text"], "he": c["text"] if as_hebrew else ""})
    vio.save_json(os.path.join(proj, "dialog.json"), {"cues": res, "imported": os.path.basename(path)})
    return res


def from_hardsubs(proj):
    """Burned-in source subtitles found on screen -> dialogue cues (text by OCR, exact frame timing)."""
    P = vio.load_json(os.path.join(proj, "project.json"))
    T = vio.load_json(os.path.join(proj, "titles.json"))
    fps = P["info"]["fps"]
    groups = {}
    for t in T["titles"]:
        if t.get("kind") == "sub":
            groups.setdefault(t["group"], []).append(t)
    cues = []
    for gid, lines in sorted(groups.items(), key=lambda kv: min(t["in"] for t in kv[1])):
        lines.sort(key=lambda t: t["box"][1])
        a = min(t["in"] for t in lines); b = max(t["out"] for t in lines)
        cues.append({"start": round(a / fps, 3), "end": round((b + 1) / fps, 3), "manual": True,
                     "src": "\n".join(t["src"] for t in lines), "he": ""})
    for k, c in enumerate(cues):
        c["n"] = k + 1
    vio.save_json(os.path.join(proj, "dialog.json"), {"cues": cues, "imported": "hardsubs"})
    return cues
