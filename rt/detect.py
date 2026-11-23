"""Pass 2: OCR the sampled frames, link detections into on-screen text events."""
import os, glob, subprocess, csv, io, re, unicodedata
from difflib import SequenceMatcher
from multiprocessing import Pool
import cv2, numpy as np
from . import vio


_ENGINE = None


GPU = {"on": False}


def _gpu_patch():
    """Run the OCR networks on the graphics card when onnxruntime has DirectML (the Windows
    build ships onnxruntime-directml: any DX12 card - NVIDIA, AMD, Intel - nothing to install).
    Same models and same input, so the text found is the same; only faster."""
    if os.environ.get("RT_NO_GPU"):
        return False
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    if "DmlExecutionProvider" not in ort.get_available_providers() or getattr(ort, "_rt_dml", False):
        return getattr(ort, "_rt_dml", False)
    base = ort.InferenceSession

    class DmlSession(base):
        def __init__(self, path, sess_options=None, providers=None, **kw):
            so = sess_options or ort.SessionOptions()
            so.enable_mem_pattern = False                        # required by DirectML
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            try:
                super().__init__(path, so, providers=["DmlExecutionProvider", "CPUExecutionProvider"], **kw)
                GPU["on"] = GPU["on"] or "DmlExecutionProvider" in self.get_providers()
            except Exception:                                    # driver too old etc.: CPU as before
                super().__init__(path, so, providers=providers or ["CPUExecutionProvider"], **kw)
    ort.InferenceSession = DmlSession
    ort._rt_dml = True
    return True


def gpu_available():
    try:
        import onnxruntime as ort
        return "DmlExecutionProvider" in ort.get_available_providers() and not os.environ.get("RT_NO_GPU")
    except ImportError:
        return False


def _rapid(img):
    global _ENGINE
    if _ENGINE is None:
        _gpu_patch()
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    r, _ = _ENGINE(img)
    out = []
    for k, (pts, text, conf) in enumerate(r or []):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        out.append({"psm": "r", "b": k, "p": 0, "l": 0, "conf": float(conf) * 100, "text": text,
                    "x": min(xs), "y": min(ys), "w": max(xs) - min(xs), "h": max(ys) - min(ys)})
    return out


def _tess(img, langs):
    ok, png = cv2.imencode(".png", img)
    out = []
    for psm in ("11", "3"):
        r = subprocess.run(["tesseract", "stdin", "stdout", "-l", langs, "--psm", psm, "tsv"],
                           input=png.tobytes(), capture_output=True, **vio.NOWIN)
        rows = csv.DictReader(io.StringIO(r.stdout.decode("utf-8", "replace")), delimiter="\t",
                              quoting=csv.QUOTE_NONE)
        for row in rows:
            try:
                conf = float(row["conf"])
            except (TypeError, ValueError):
                continue
            t = (row.get("text") or "").strip()
            if conf < 55 or not t:
                continue
            out.append({"psm": psm, "b": int(row["block_num"]), "p": int(row["par_num"]),
                        "l": int(row["line_num"]), "conf": conf, "text": t,
                        "x": int(row["left"]), "y": int(row["top"]),
                        "w": int(row["width"]), "h": int(row["height"])})
    return out


def _ocr(args):
    """RapidOCR (PP-OCR det+rec, bundled ONNX - finds stylised text on busy backgrounds).
    For languages it cannot read (ja/ko/he...) the detected boxes are re-read with tesseract."""
    path, langs, scale = args
    img = cv2.imread(path)
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    try:
        words = _rapid(img)
        if langs not in ("eng", "chi_sim", "eng+chi_sim"):
            for wd in words:
                x0, y0 = int(max(0, wd["x"] - 6)), int(max(0, wd["y"] - 6))
                crop = img[y0:int(wd["y"] + wd["h"] + 6), x0:int(wd["x"] + wd["w"] + 6)]
                ok, png = cv2.imencode(".png", crop)
                try:
                    r = subprocess.run(["tesseract", "stdin", "stdout", "-l", langs, "--psm", "7"],
                                       input=png.tobytes(), capture_output=True, **vio.NOWIN)
                except FileNotFoundError:
                    break  # no tesseract: keep the boxes, text is read from the thumbnail
                t = r.stdout.decode("utf-8", "replace").strip()
                if t:
                    wd["text"] = t
    except ImportError:
        words = _tess(img, langs)
    for wd in words:
        for k in ("x", "y", "w", "h"):
            wd[k] = wd[k] / scale
    return path, words


def _good(t):
    alnum = sum(ch.isalnum() for ch in t)
    cjk = sum(unicodedata.east_asian_width(ch) in "WF" for ch in t)
    return cjk >= 1 or alnum >= 2


def _lines(words):
    """Merge tesseract words into lines (per psm, then dedupe across psm)."""
    groups = {}
    words = [w for w in words if _good(w["text"])]
    rap = [w for w in words if w["psm"] == "r"]
    # rapid: chain boxes on the same line (vertical overlap, small gap) into one line
    rap.sort(key=lambda a: a["x"])
    used = set()
    for i, a in enumerate(rap):
        if i in used:
            continue
        chain = [a]; used.add(i)
        for j, b in enumerate(rap):
            if j in used:
                continue
            c = chain[-1]
            ov = min(c["y"] + c["h"], b["y"] + b["h"]) - max(c["y"], b["y"])
            if ov > 0.6 * min(c["h"], b["h"]) and 0 <= b["x"] - (c["x"] + c["w"]) < 1.5 * max(c["h"], b["h"]):
                chain.append(b); used.add(j)
        groups[("r", i, 0, 0)] = chain
    for wd in words:
        if wd["psm"] == "r":
            continue
        groups.setdefault((wd["psm"], wd["b"], wd["p"], wd["l"]), []).append(wd)
    lines = []
    for ws in groups.values():
        ws.sort(key=lambda a: a["x"])
        # split a tesseract line where horizontal gaps are huge
        cur = [ws[0]]
        for a in ws[1:]:
            prev = cur[-1]
            if a["x"] - (prev["x"] + prev["w"]) > 2.5 * max(prev["h"], a["h"]):
                lines.append(cur); cur = [a]
            else:
                cur.append(a)
        lines.append(cur)
    res = []
    for ws in lines:
        x0 = min(a["x"] for a in ws); y0 = min(a["y"] for a in ws)
        x1 = max(a["x"] + a["w"] for a in ws); y1 = max(a["y"] + a["h"] for a in ws)
        res.append({"text": " ".join(a["text"] for a in ws), "box": [x0, y0, x1, y1],
                    "conf": float(np.mean([a["conf"] for a in ws])),
                    "words": [[a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]] for a in ws]})
    # dedupe overlapping lines from the two psm passes, keep higher conf*len
    res.sort(key=lambda r: -r["conf"] * len(r["text"]))
    keep = []
    for r in res:
        if all(iou(r["box"], k["box"]) < 0.4 for k in keep):
            keep.append(r)
    return keep


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0


def norm(t):
    return re.sub(r"\W+", "", t.lower())


def tsim(a, b):
    """Text similarity tolerant to OCR noise and to text that is still being revealed."""
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    sm = SequenceMatcher(None, a, b)
    m = sum(x.size for x in sm.get_matching_blocks())
    return max(sm.ratio(), m / min(len(a), len(b)) if min(len(a), len(b)) >= 4 else 0)


def _live(proj, f, words):
    from . import live
    if not live.due():
        return
    img = cv2.imread(f)
    if img is None:
        return
    sh = img.shape[1] / max(1.0, vio.load_json(os.path.join(proj, "project.json"))["info"]["width"])
    for w in words:
        x, y, ww, hh = (int(w[k] * sh) for k in ("x", "y", "w", "h"))
        cv2.rectangle(img, (x - 3, y - 3), (x + ww + 3, y + hh + 3), (60, 200, 255), 2)
    fr = int(os.path.basename(f)[:-4])
    live.show(proj, f"מחפש טקסט - פריים {fr}" + (f" - נמצא: {' '.join(w['text'] for w in words)[:40]}" if words else ""), img, width=960)


def run(proj, langs="eng", workers=None, scale=None, progress=None, rng=None):
    P = vio.load_json(os.path.join(proj, "project.json"))
    S = vio.load_json(os.path.join(proj, "shots.json"))
    files = sorted(glob.glob(os.path.join(proj, "samples", "*.jpg")))
    if rng:                                    # work range: only its samples (and 2 s around it)
        m = int(2 * P["info"]["fps"])
        files = [f for f in files if rng[0] - m <= int(os.path.basename(f)[:-4]) <= rng[1] + m]
    W = P["info"]["width"]
    if scale is None:  # detection works well at ~1280 px; faster for 4K / long material
        scale = min(1.0, 1280 / W)
    gpu = gpu_available()
    # on the graphics card fewer processes are better (they share one card); the CPU does the rest
    workers = workers or (2 if gpu else max(1, min(4, (os.cpu_count() or 2) - 1)))
    print("OCR:", "graphics card (DirectML)" if gpu else "CPU", workers, "processes")
    res = {}
    with Pool(workers) as pool:
        for k, (f, r) in enumerate(pool.imap_unordered(_ocr, [(f, langs, scale) for f in files], chunksize=4)):
            res[f] = r
            if progress and k % 10 == 0:
                progress(k / max(1, len(files)))
            if r or k % 8 == 0:
                _live(proj, f, r)
    every = S["every"]
    order = sorted(set(S.get("samples") or []) | {int(os.path.basename(f)[:-4]) for f in files})
    sidx = {fr: k for k, fr in enumerate(order)}
    dets = []
    for f in files:
        fr = int(os.path.basename(f)[:-4])
        for ln in _lines(res[f]):
            ln["frame"] = fr
            dets.append(ln)
    # link into events
    events = []
    for d in sorted(dets, key=lambda d: d["frame"]):
        best = None
        for e in events:
            last = e["dets"][-1]
            if sidx[d["frame"]] - sidx[last["frame"]] > 3 or d["frame"] == last["frame"]:
                continue
            sim = tsim(d["text"], last["text"])
            ov = iou(d["box"], last["box"])
            # same place but different words = a new title (e.g. consecutive subtitles)
            if sim > 0.6 and ov > 0.15:
                sc = sim + ov
                if best is None or sc > best[0]:
                    best = (sc, e)
        if best:
            best[1]["dets"].append(d)
        else:
            events.append({"dets": [d]})
    def _ib(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / max(1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))
    merged = True
    while merged:
        merged = False
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                A, B = events[i]["dets"], events[j]["dets"]
                if sidx[B[0]["frame"]] > sidx[A[-1]["frame"]] + 3 or sidx[A[0]["frame"]] > sidx[B[-1]["frame"]] + 3:
                    continue
                ba = [min(d["box"][0] for d in A), min(d["box"][1] for d in A), max(d["box"][2] for d in A), max(d["box"][3] for d in A)]
                bb = [min(d["box"][0] for d in B), min(d["box"][1] for d in B), max(d["box"][2] for d in B), max(d["box"][3] for d in B)]
                ta = max((x["text"] for x in A), key=len); tb = max((x["text"] for x in B), key=len)
                if _ib(ba, bb) > 0.5 and tsim(ta, tb) > 0.6:
                    events[i]["dets"] = sorted(A + B, key=lambda d: d["frame"])
                    del events[j]; merged = True; break
            if merged:
                break
    titles = []
    for e in events:
        ds = e["dets"]
        if len(ds) < 2 and ds[0]["conf"] < 65:
            continue  # one-off OCR noise
        texts = [d["text"] for d in ds]
        text = max(set(texts), key=lambda t: (texts.count(t) + len(norm(t)) / 4, len(t)))
        boxes = np.array([d["box"] for d in ds])
        # reference = the detection with the most complete text (fully in, not mid-animation)
        L = max(len(norm(d["text"])) for d in ds)
        centre = (ds[0]["frame"] + ds[-1]["frame"]) / 2
        mid = min([d for d in ds if len(norm(d["text"])) >= 0.85 * L], key=lambda d: abs(d["frame"] - centre))
        titles.append({
            "id": f"t{len(titles) + 1:02d}", "src": text,
            "sample_in": ds[0]["frame"], "sample_out": ds[-1]["frame"],
            # neighbouring samples without this text: the exact in/out lies between them
            "prev_sample": order[sidx[ds[0]["frame"]] - 1] if sidx[ds[0]["frame"]] > 0 else 0,
            "next_sample": order[sidx[ds[-1]["frame"]] + 1] if sidx[ds[-1]["frame"]] + 1 < len(order) else P.get("n_frames", ds[-1]["frame"] + every) - 1,
            "in": ds[0]["frame"], "out": ds[-1]["frame"], "ref": mid["frame"],
            "box": [int(boxes[:, 0].min()), int(boxes[:, 1].min()),
                    int(boxes[:, 2].max()), int(boxes[:, 3].max())],
            "ref_box": [int(v) for v in mid["box"]],
            "words": [[int(v) for v in w] for w in mid["words"]],
            "conf": round(float(np.mean([d["conf"] for d in ds])), 1),
            "translate": True, "he": "", "kind": "text"})
    # group lines that live together (same card)
    gid = 0
    for t in titles:
        t.setdefault("group", None)
    for i, a in enumerate(titles):
        if a["group"] is None:
            gid += 1; a["group"] = f"g{gid:02d}"
        for b in titles[i + 1:]:
            if b["group"] is None and abs(a["sample_in"] - b["sample_in"]) <= every * 6 \
                    and abs(a["sample_out"] - b["sample_out"]) <= every * 6:
                h = a["ref_box"][3] - a["ref_box"][1]
                if abs(b["ref_box"][1] - a["ref_box"][3]) < 3 * h or abs(a["ref_box"][1] - b["ref_box"][3]) < 3 * h:
                    b["group"] = a["group"]
    merge_cards(titles, every)
    for i, t in enumerate(titles):
        t["id"] = f"t{i + 1:02d}"
    classify_keep(titles, P, langs)
    classify_hardsubs([t for t in titles if not t.get("auto_reason")], P)
    if rng:
        titles = merge_range(proj, titles, rng)
    vio.save_json(os.path.join(proj, "titles.json"), {"titles": titles})
    sheet(proj, titles, P)
    return titles


def merge_range(proj, new, rng):
    """Work range: titles found in it replace the ones there before; everything outside the range
    (with its Hebrew, masks and style) stays as it was."""
    a, b = rng
    old = (vio.load_json(os.path.join(proj, "titles.json")) or {}).get("titles", [])
    keep = [t for t in old if t["sample_out"] < a or t["sample_in"] > b]
    new = [t for t in new if not (t["sample_out"] < a or t["sample_in"] > b)]
    num = lambda x, p: int(x[1:]) if x and x[0] == p and x[1:].isdigit() else 0
    ni = max([num(t["id"], "t") for t in old] + [0]); gi = max([num(t.get("group", ""), "g") for t in old] + [0])
    gmap = {}
    inside = [t for t in old if t not in keep]
    USER = ("he", "translate", "mode", "layer", "style", "style_locked", "auto_mask", "mask", "in_override",
            "out_override", "font_auto", "series_override", "from_series", "auto_reason", "logo_full", "reveal_dir")
    used = set()
    for t in new:
        # the same text at the same time as before: keep the user's work on it (Hebrew, choices, style)
        o = next((x for x in inside if x["id"] not in used and norm(x["src"]) == norm(t["src"])
                  and x["sample_in"] <= t["sample_out"] and t["sample_in"] <= x["sample_out"]), None)
        if o:
            used.add(o["id"])
            for k in USER:
                if k in o:
                    t[k] = o[k]
                else:
                    t.pop(k, None) if k in ("auto_reason",) else None
            t["id"] = o["id"]
            continue
        ni += 1; t["id"] = f"t{ni:02d}"
        if t["group"] not in gmap:
            gi += 1; gmap[t["group"]] = f"g{gi:02d}"
        t["group"] = gmap[t["group"]]
    return sorted(keep + new, key=lambda t: (t["sample_in"], t["box"][1]))


def classify_hardsubs(titles, P):
    """Burned-in dialogue subtitles of the source (e.g. English hardsubs on anime): many short
    lines, centred, in the bottom part of the frame. They are erased (mode "remove") instead of
    replaced - the Hebrew goes to the SRT - and can be imported as dialogue cues."""
    W, H, fps = P["info"]["width"], P["info"]["height"], P["info"]["fps"]
    cand = []
    for t in titles:
        x0, y0, x1, y1 = t["box"]
        cx = (x0 + x1) / 2
        dur = (t["sample_out"] - t["sample_in"]) / fps
        if abs(cx - W / 2) < 0.12 * W and y1 > 0.70 * H and y0 > 0.55 * H and dur <= 9:
            cand.append(t)
    if len(cand) >= 4:
        for t in cand:
            t["kind"] = "sub"; t["mode"] = "remove"
    return cand


def sheet(proj, titles, P, name="detect_sheet.jpg"):
    """Contact sheet: each detected title at its ref frame with box drawn, for review."""
    tiles = []
    for t in titles:
        f = os.path.join(proj, "samples", f"{t['ref']:06d}.jpg")
        img = cv2.imread(f)
        if img is None:
            continue
        x0, y0, x1, y1 = t["ref_box"]
        cv2.rectangle(img, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 0, 255), 3)
        img = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])))
        cv2.putText(img, f"{t['id']} {t['group']} f{t['sample_in']}-{t['sample_out']}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(img, f"{t['id']} {t['group']} f{t['sample_in']}-{t['sample_out']}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(img)
    if not tiles:
        return None
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    out = os.path.join(proj, name)
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


# ------------------------------------------------------------------ what should stay as it is
# Lines that are better left untouched by default. Each gets translate=False and a reason
# the user sees ("auto_reason"); one click brings it back.
STUDIOS = ["universal", "universalpictures", "comcast", "comcastcompany", "netflix", "disney", "pixar",
           "dreamworks", "warnerbros", "warner", "paramount", "sonypictures", "columbia", "lionsgate", "a24",
           "hbo", "hbomax", "amazonmgmstudios", "primevideo", "appletv", "illumination", "legendary",
           "neon", "focusfeatures", "searchlight", "20thcenturystudios", "newline", "mgm", "studiocanal",
           "toho", "toei", "toeianimation", "aniplex", "crunchyroll", "madhouse", "mappa", "ufotable",
           "kyotoanimation", "sunrise", "bonesstudio", "productioni.g", "funimation", "gkids", "sola",
           "solamedia", "likeaphoton", "likeaphotoncreative", "alceon", "screenqueensland", "screenaustralia",
           "nationalgeographic", "bbc", "bbcstudios", "discovery", "showtime", "starz", "peacock", "hulu",
           "skykids", "skyone", "skycinema", "cartoonnetwork", "boomerang", "nickelodeon", "nickjr",
           "disneychannel", "disneyjunior", "disneyxd", "cbbc", "cbeebies", "pbskids", "foxkids", "jetix",
           "zodiakkids", "zodiak", "marathonmedia", "banijay", "teletoon", "gulli", "kika", "rai", "zdf",
           "hopchannel", "logichannel"]
CREDIT_WORDS = ["directedby", "producedby", "screenplay", "writtenby", "executiveproducer", "producers",
                "coproducer", "musicby", "composition", "composer", "cinematography", "editedby", "castingby",
                "productiondesign", "costume", "visualeffects", "basedon", "pty", "ltd", "inc", "llc",
                "entertainment", "partners", "present", "presents", "inassociation", "allrightsreserved",
                "copyright", "studios"]
_PLATE = re.compile(r"^(?=.*\d)(?=.*[a-z])[a-z0-9]{4,9}$")


def _studio(text):
    k = norm(text)
    if len(k) < 3:
        return None
    for s in STUDIOS:
        if k == s or (len(k) >= 5 and len(s) >= 5 and SequenceMatcher(None, k, s).ratio() >= 0.75):
            return s
    return None


def classify_keep(titles, P, langs="eng"):
    W, H, fps = P["info"]["width"], P["info"]["height"], P["info"]["fps"]
    groups = {}
    for t in titles:
        groups.setdefault(t["group"], []).append(t)
    gsmall = lambda g: all(x["box"][3] - x["box"][1] < 0.06 * H for x in groups[g])

    def keep(t, why):
        if not t.get("auto_reason"):
            t["translate"] = False; t["auto_reason"] = why
            t.pop("mode", None); t["kind"] = "text" if t.get("kind") == "sub" else t.get("kind", "text")
    # OCR noise first: letters of the wrong script, and short letter strings that are no word
    for t in titles:
        why = junk_reason(t["src"], langs, len(groups[t["group"]]))
        if why:
            keep(t, why)
    # credits: a card of many small lines, or any card naming credit roles / companies
    for g, ls in groups.items():
        k = [norm(t["src"]) for t in ls]
        creditish = sum(any(w in x for w in CREDIT_WORDS) for x in k)
        small = all(t["box"][3] - t["box"][1] < 0.075 * H for t in ls)
        if creditish and (creditish >= 2 or len(ls) >= 4 or small) or (len(ls) >= 5 and small):
            for t in ls:
                keep(t, "קרדיטים")
    # credits that follow a credit card (logo strip under it, same time)
    cred = [t for t in titles if t.get("auto_reason") == "קרדיטים"]
    for t in titles:
        if not t.get("auto_reason") and any(c["sample_in"] - fps <= t["sample_in"] <= c["sample_out"] and
                                            t["sample_out"] <= c["sample_out"] + fps for c in cred) \
                and t["box"][3] - t["box"][1] < 0.1 * H:
            keep(t, "קרדיטים")
    for t in titles:
        if t.get("auto_reason"):
            continue
        k = norm(t["src"]); h = t["box"][3] - t["box"][1]
        if re.fullmatch(r"(an?)?(production|productions|presents|present|incollaborationwith|inassociationwith)", k):
            keep(t, "קרדיטים"); continue
        dur = (t["sample_out"] - t["sample_in"]) / fps
        alone = len(groups[t["group"]]) == 1
        if _studio(t["src"]):
            keep(t, "לוגו של חברה")
        elif _PLATE.match(k) and " " not in t["src"].strip():
            keep(t, "טקסט בתוך התמונה")          # licence plates, codes, serials
        elif gsmall(t["group"]) and dur < 1.5 and len(k) <= 10 and " " not in t["src"].strip():
            keep(t, "טקסט בתוך התמונה")          # signs / labels in the scene
        elif alone and h < 0.1 * H and dur < 0.6 and len(k) <= 5 and t["conf"] < 70:
            keep(t, "טקסט בתוך התמונה")
    return [t for t in titles if t.get("auto_reason")]


def merge_cards(titles, every):
    """Words of one animated card (a film title that builds up word by word) often get separate
    groups: they come in at different times but leave together. Join them into one card."""
    # the same word seen a moment earlier at a nearby place (still animating in): one title
    for b in list(titles):
        for a in titles:
            if a is b or b not in titles:
                continue
            if norm(a["src"]) == norm(b["src"]) and 0 <= a["sample_in"] - b["sample_out"] <= every * 8 \
                    and _near(a["box"], b["box"]):
                a["sample_in"], a["prev_sample"] = b["sample_in"], b["prev_sample"]
                a["in"] = min(a["in"], b["in"])
                a["box"] = [min(a["box"][0], b["box"][0]), min(a["box"][1], b["box"][1]),
                            max(a["box"][2], b["box"][2]), max(a["box"][3], b["box"][3])]
                titles.remove(b)
                break
    changed = True
    while changed:
        changed = False
        for a in titles:
            for b in titles:
                if a["group"] == b["group"]:
                    continue
                if abs(a["sample_out"] - b["sample_out"]) > every * 2:
                    continue
                ov = min(a["sample_out"], b["sample_out"]) - max(a["sample_in"], b["sample_in"])
                if ov < 0.4 * min(a["sample_out"] - a["sample_in"], b["sample_out"] - b["sample_in"]) or ov < every:
                    continue
                A, B = a["box"], b["box"]
                ha, hb = A[3] - A[1], B[3] - B[1]
                gap = max(B[1] - A[3], A[1] - B[3])
                hx = min(A[2], B[2]) - max(A[0], B[0])
                if gap < 0.5 * max(ha, hb) and hx > 0.3 * min(A[2] - A[0], B[2] - B[0]) and min(ha, hb) > 0.6 * max(ha, hb) / 3:
                    old, new = b["group"], a["group"]
                    for t in titles:
                        if t["group"] == old:
                            t["group"] = new
                    changed = True


def _near(A, B):
    h = max(A[3] - A[1], B[3] - B[1])
    return max(B[0] - A[2], A[0] - B[2]) < h and max(B[1] - A[3], A[1] - B[3]) < h


SHORT_WORDS = set("""a i an as at be by do go he if in is it me my no of oh ok on or so to up us we
all and are boy but can day did end for get got god guy had has her hey him his how its let man may new
not now off old one our out run say see she the too try two war was way who why yes yet you
it's i'm""".split())


def junk_reason(src, langs="eng", card_lines=1):
    """Why a detected line is OCR noise (not a title), or None.
    - a CJK character when the video's language is not Chinese/Japanese/Korean (the OCR model reads
      Chinese, so patterns and drawings sometimes come out as one or two Chinese characters);
    - a lone, very short letter string that is no word ("DD", "MOD" read off a drawing)."""
    import unicodedata
    t = src.strip()
    cjk_lang = any(x in (langs or "") for x in ("chi", "jpn", "kor"))
    cjk = sum(unicodedata.east_asian_width(ch) in "WF" for ch in t)
    if cjk and not cjk_lang:
        return "רעש של זיהוי"
    letters = re.sub(r"[^A-Za-z']", "", t)
    if card_lines == 1 and 0 < len(letters) <= 3 and letters.lower() not in SHORT_WORDS and not re.search(r"\d", t):
        return "רעש של זיהוי"
    return None
