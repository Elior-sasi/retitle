"""Pass 2: OCR the sampled frames, link detections into on-screen text events."""
import os, glob, subprocess, csv, io, re, unicodedata
from difflib import SequenceMatcher
from multiprocessing import Pool
import cv2, numpy as np
from . import vio


_ENGINE = None


def _rapid(img):
    global _ENGINE
    if _ENGINE is None:
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


def run(proj, langs="eng", workers=None, scale=None, progress=None):
    P = vio.load_json(os.path.join(proj, "project.json"))
    S = vio.load_json(os.path.join(proj, "shots.json"))
    files = sorted(glob.glob(os.path.join(proj, "samples", "*.jpg")))
    W = P["info"]["width"]
    if scale is None:  # detection works well at ~1280 px; faster for 4K / long material
        scale = min(1.0, 1280 / W)
    workers = workers or max(1, min(4, (os.cpu_count() or 2) - 1))
    res = {}
    with Pool(workers) as pool:
        for k, (f, r) in enumerate(pool.imap_unordered(_ocr, [(f, langs, scale) for f in files], chunksize=4)):
            res[f] = r
            if progress and k % 10 == 0:
                progress(k / max(1, len(files)))
    every = S["every"]
    order = S.get("samples") or sorted(int(os.path.basename(f)[:-4]) for f in files)
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
    classify_hardsubs(titles, P)
    vio.save_json(os.path.join(proj, "titles.json"), {"titles": titles})
    sheet(proj, titles, P)
    return titles


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
