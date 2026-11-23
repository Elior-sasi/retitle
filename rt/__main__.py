"""retitle CLI.  python -m rt <command> <project_dir> [options]

  init <proj> <video>          create project
  shots <proj>                 cuts + OCR samples
  detect <proj> --langs eng    OCR on-screen text -> titles.json + detect_sheet.jpg
  extract <proj>               cache crops, estimate mask/style, exact in/out
  masks <proj> [gid]           mask preview sheets (check before cleaning)
  clean <proj> [gid..]         remove source text, rebuild background
  track <proj>                 measure source animation per frame
  fontsheet <proj> <id> [fonts..]  source crop vs Hebrew in candidate fonts
  layers <proj>                render Hebrew layers
  preview <proj> <gid> [frames..]  result crops without encoding
  compose <proj>               write final video
  check <proj>                 before/after sheets + report.html from the encoded output
  fonts                        download Hebrew font set
  srt-prep <proj> <whisper.json>   cue list for translation -> dialog.json
  srt-build <proj>             Hebrew SRT with timing rules
  transcribe <video> <out.json> [lang]  faster-whisper (run on the PC)
"""
import os, sys
import cv2, numpy as np
from . import vio


def _p(proj, *a):
    return os.path.join(proj, *a)


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return
    cmd, rest = argv[0], argv[1:]
    opt = {}
    pos = []
    i = 0
    while i < len(rest):
        if rest[i].startswith("--"):
            opt[rest[i][2:]] = rest[i + 1]; i += 2
        else:
            pos.append(rest[i]); i += 1
    if cmd == "fonts":
        from . import fonts, render
        fonts.run(opt.get("dest", render.FONT_DIR)); return
    if cmd == "transcribe":
        from . import srt
        print(srt.transcribe(pos[0], pos[1], pos[2] if len(pos) > 2 else None, opt.get("model", "large-v3"))); return
    proj = pos[0]
    if cmd == "init":
        os.makedirs(proj, exist_ok=True)
        info = vio.probe(pos[1])
        vio.save_json(_p(proj, "project.json"), {"video": os.path.abspath(pos[1]), "info": info})
        print(info)
    elif cmd == "shots":
        from . import shots
        c = shots.run(proj, int(opt.get("every", 4)))
        print(len(c), "shots")
    elif cmd == "detect":
        from . import detect
        ts = detect.run(proj, opt.get("langs", "eng"), scale=float(opt.get("scale", 1)))
        fps = vio.load_json(_p(proj, "project.json"))["info"]["fps"]
        for t in ts:
            print(t["id"], t["group"], vio.tc(t["sample_in"], fps), "-", vio.tc(t["sample_out"], fps), t["conf"], t["src"])
    elif cmd == "extract":
        from . import extract
        extract.run(proj)
        T = vio.load_json(_p(proj, "titles.json"))
        for t in T["titles"]:
            print(t["id"], t.get("in"), t.get("out"), t.get("style", {}).get("fill"), t.get("warn", ""))
    elif cmd == "masks":
        mask_sheet(proj, pos[1:] or None)
    elif cmd == "clean":
        from . import clean
        clean.run(proj, pos[1:] or None)
    elif cmd == "track":
        from . import render
        render.track(proj)
        for t in vio.load_json(_p(proj, "titles.json"))["titles"]:
            print(t["id"], t.get("anim", {}).get("mode"))
    elif cmd == "fontsheet":
        font_sheet(proj, pos[1], pos[2:])
    elif cmd == "layers":
        from . import render
        render.build_layers(proj)
    elif cmd == "preview":
        print(preview(proj, pos[1], [int(x) for x in pos[2:]]))
    elif cmd == "compose":
        from . import compose
        print(compose.run(proj, crf=int(opt.get("crf", 14))))
    elif cmd == "check":
        from . import check
        for s in check.run(proj):
            print(s)
    elif cmd == "srt-prep":
        from . import srt
        for c in srt.prep(proj, pos[1]):
            print(c["n"], c["start"], c["end"], c["src"])
    elif cmd == "srt-build":
        from . import srt
        p, w = srt.build(proj)
        print(p); [print("WARN", x) for x in w]
    else:
        print(__doc__)


def mask_sheet(proj, only):
    from .extract import groups
    from .clean import frame_mask
    T = vio.load_json(_p(proj, "titles.json"))
    out = []
    for gid, lines in groups(T).items():
        act = [t for t in lines if t.get("translate", True) and "mask" in t]
        if not act or (only and gid not in only):
            continue
        a = min(t["in"] for t in act); b = max(t["out"] for t in act)
        rows = []
        for f in sorted({a, (a + b) // 2, b}):
            img = cv2.imread(_p(proj, "cache", gid, f"{f:06d}.png"))
            m = frame_mask(img, f, lines, act[0]["region"])
            ov = img.copy(); ov[m > 0] = (0, 0, 255)
            rows.append(np.hstack([img, cv2.addWeighted(img, 0.4, ov, 0.6, 0)]))
        p = _p(proj, f"mask_{gid}.jpg")
        im = np.vstack(rows)
        sc = min(1.0, 1400 / im.shape[1])
        cv2.imwrite(p, cv2.resize(im, None, fx=sc, fy=sc))
        out.append(p); print(p)
    return out


def font_sheet(proj, tid, fonts):
    from . import render
    from PIL import Image
    T = vio.load_json(_p(proj, "titles.json"))
    t = next(x for x in T["titles"] if x["id"] == tid)
    reg = t["region"]
    src = cv2.imread(_p(proj, "cache", t["group"], f"{t['ref']:06d}.png"))
    bx = t["box"]; h = bx[3] - bx[1]
    crop = src[max(0, bx[1] - reg[1] - h // 2): bx[3] - reg[1] + h // 2, max(0, bx[0] - reg[0] - h // 2): bx[2] - reg[0] + h // 2]
    cat = vio.load_json(os.path.join(render.FONT_DIR, "catalog.json"), {})
    fonts = fonts or list(cat)
    tiles = [crop]
    for fn in fonts:
        st = dict(t.get("style", {})); st["font"] = fn
        try:
            img, _, ink = render.text_layer(t.get("he") or "אבגדה הוזחט", st, 1)
        except Exception as e:
            print(fn, e); continue
        a = np.array(img)[..., [2, 1, 0, 3]].astype(np.float32) / 255
        bg = np.full(a.shape[:2] + (3,), 90, np.float32) / 255
        im = (a[..., :3] * a[..., 3:] + bg * (1 - a[..., 3:])) * 255
        im = im.astype(np.uint8)
        cv2.putText(im, fn, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(im)
    w = max(x.shape[1] for x in tiles)
    tiles = [cv2.copyMakeBorder(x, 4, 4, 0, w - x.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for x in tiles]
    p = _p(proj, f"fonts_{tid}.jpg")
    im = np.vstack(tiles)
    sc = min(1.0, 1200 / im.shape[1])
    cv2.imwrite(p, cv2.resize(im, None, fx=sc, fy=sc))
    print(p)


def preview(proj, gid, frames):
    from . import compose
    from .extract import groups
    P = vio.load_json(_p(proj, "project.json"))
    T = vio.load_json(_p(proj, "titles.json"))
    meta = vio.load_json(_p(proj, "layers", "meta.json"), {})
    W, H = P["info"]["width"], P["info"]["height"]
    lines = groups(T)[gid]
    reg = lines[0]["region"]
    if not frames:
        a = min(t["in"] for t in lines); b = max(t["out"] for t in lines)
        frames = sorted({a, a + 3, (a + b) // 2, b})
    rows = []
    cache = {}
    for f in frames:
        o = cv2.imread(_p(proj, "cache", gid, f"{f:06d}.png"))
        full = np.zeros((H, W, 3), np.uint8)
        full[reg[1]:reg[3], reg[0]:reg[2]] = o
        r = compose.frame_result(proj, f, full, T, meta, cache, W, H)[reg[1]:reg[3], reg[0]:reg[2]]
        row = np.hstack([o, np.full((o.shape[0], 6, 3), 40, np.uint8), r])
        cv2.putText(row, f"f{f}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        rows.append(row)
    im = np.vstack(rows)
    sc = min(1.0, 1500 / im.shape[1])
    p = _p(proj, f"preview_{gid}.jpg")
    cv2.imwrite(p, cv2.resize(im, None, fx=sc, fy=sc))
    return p


if __name__ == "__main__":
    main(sys.argv[1:])
