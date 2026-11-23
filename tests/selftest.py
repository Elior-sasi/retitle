"""End-to-end check: generated clip with 3 animated English titles -> Hebrew video + SRT."""
import os, sys, json, shutil, subprocess, tempfile
sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rt import vio, shots, detect, extract, clean, render, compose, check, srt


def main():
    d = tempfile.mkdtemp()
    clip = os.path.join(d, "clip.mp4")
    bgv = os.path.join(d, "bg.mp4")
    subprocess.run([sys.executable, os.path.join(ROOT, "tests", "make_clip.py"), clip, bgv], check=True)
    p = os.path.join(d, "p"); os.makedirs(p)
    vio.save_json(os.path.join(p, "project.json"), {"video": clip, "info": vio.probe(clip)})
    cuts = shots.run(p); assert cuts == [0, 72], cuts
    ts = detect.run(p, "eng")
    srcs = sorted(t["src"] for t in ts); print(srcs)
    assert len(ts) == 3, srcs
    extract.run(p)
    T = vio.load_json(os.path.join(p, "titles.json"))
    want = {"IT'S TIME": (24, 60), "FOR A TRUE HERO": (30, 66), "FROM THE STUDIO": (80, 130)}
    he = {"IT'S TIME": "הגיע הזמן", "FOR A TRUE HERO": "שגיבור אמיתי", "FROM THE STUDIO": "מהאולפן"}
    for t in T["titles"]:
        k = min(want, key=lambda w: abs(len(w) - len(t["src"])) + (0 if w.split()[0] in t["src"] else 9))
        a, b = want[k]
        print(t["id"], t["src"], t["in"], t["out"], "want", a, b)
        assert abs(t["in"] - a) <= 1 and abs(t["out"] - b) <= 1, (t["src"], t["in"], t["out"])
        t["he"] = he[k]
    vio.save_json(os.path.join(p, "titles.json"), T)
    clean.run(p)
    # the rebuilt background must match the true background under the removed text
    import cv2, numpy as np
    T = vio.load_json(os.path.join(p, "titles.json"))
    info = vio.probe(bgv); bgs = list(vio.frames(bgv, info["width"], info["height"]))
    worst = 0
    for t in T["titles"]:
        x0, y0, x1, y1 = t["region"]
        for f in range(t["in"], t["out"] + 1, 3):
            cimg = cv2.imread(os.path.join(p, "clean", t["group"], f"{f:06d}.png"))
            m = cv2.imread(os.path.join(p, "clean", t["group"], f"m{f:06d}.png"), 0) > 0
            if cimg is None or not m.any():
                continue
            err = float(np.abs(cimg.astype(int) - bgs[f][y0:y1, x0:x1].astype(int))[m].mean())
            worst = max(worst, err)
    print("clean error (mean abs, worst frame):", round(worst, 2))
    assert worst < 10, worst
    render.track(p); render.build_layers(p)
    out = compose.run(p); check.run(p)
    assert os.path.getsize(out) > 10000
    json.dump({"language": "en", "segments": [{"start": 0.2, "end": 1.1, "text": "Hello.", "words": []}]},
              open(os.path.join(p, "w.json"), "w"))
    srt.prep(p, os.path.join(p, "w.json"))
    D = vio.load_json(os.path.join(p, "dialog.json")); D["cues"][0]["he"] = "שלום."
    vio.save_json(os.path.join(p, "dialog.json"), D)
    sp, w = srt.build(p)
    print(open(sp, encoding="utf-8").read())
    static_test(d)
    hardsub_test(d)
    qa_test()
    print("SELFTEST OK")


def hardsub_test(d):
    """English hardsubs at the bottom: found as 6 separate subtitles with exact timing, marked
    for removal, importable as dialogue cues."""
    clip = os.path.join(d, "hs.mp4")
    subprocess.run([sys.executable, os.path.join(ROOT, "tests", "make_hardsubs.py"), clip], check=True)
    p = os.path.join(d, "hs"); os.makedirs(p)
    vio.save_json(os.path.join(p, "project.json"), {"video": clip, "info": vio.probe(clip)})
    shots.run(p); detect.run(p, "eng"); extract.run(p)
    T = vio.load_json(os.path.join(p, "titles.json"))
    got = sorted((t["in"], t["out"], t.get("kind"), t.get("mode")) for t in T["titles"])
    print("hardsubs:", got)
    assert len(got) == 6, got
    for k, (a, b, kind, mode) in enumerate(got):
        assert abs(a - 48 * k) <= 1 and abs(b - (48 * k + 43)) <= 1 and kind == "sub" and mode == "remove", got[k]
    cues = srt.from_hardsubs(p)
    assert len(cues) == 6 and "find her" in cues[1]["src"], cues


def qa_test():
    from rt.qa import text_issues
    assert text_issues("מה?\n-טוב", True) == []
    assert any("ניקוד" in m for _, m in text_issues("שָׁלוֹם", True))
    assert any("עם רווח" in m for _, m in text_issues("מה?\n- טוב", True))
    assert any("פיסוק" in m for _, m in text_issues("זה - לא", True))
    assert any("אנגלית" in m for _, m in text_issues("GPS שלי", True))
    print("qa: ok")


def static_test(d):
    """Locked-off interview shot, 20 s, fading lower third on a semi-transparent box:
    few samples, exact in/out, text removed with the true background seen through the box."""
    import cv2, numpy as np
    clip = os.path.join(d, "st.mp4"); bgv = os.path.join(d, "st_bg.mp4")
    subprocess.run([sys.executable, os.path.join(ROOT, "tests", "make_static.py"), clip, bgv], check=True)
    p = os.path.join(d, "sp"); os.makedirs(p)
    vio.save_json(os.path.join(p, "project.json"), {"video": clip, "info": vio.probe(clip)})
    shots.run(p)
    S = vio.load_json(os.path.join(p, "shots.json"))
    print("static: samples", len(S["samples"]), "of", S["n_frames"], "frames")
    assert len(S["samples"]) < 20, len(S["samples"])
    ts = detect.run(p, "eng")
    assert len(ts) == 2, [t["src"] for t in ts]
    extract.run(p)
    T = vio.load_json(os.path.join(p, "titles.json"))
    for t in T["titles"]:
        print("static:", t["src"], t["in"], t["out"])
        assert abs(t["in"] - 125) <= 1 and abs(t["out"] - 312) <= 1, (t["in"], t["out"])
    clean.run(p)
    T = vio.load_json(os.path.join(p, "titles.json"))
    info = vio.probe(bgv)
    x0, y0, x1, y1 = T["titles"][0]["region"]
    C = np.array([80, 30, 10], np.float32)
    worst = 0
    for f in (126, 140, 200, 300, 311):
        bg = vio.grab(bgv, f, info).astype(np.float32)
        al = min(1, (f - 125 + 1) / 6, (312 - f + 1) / 6) * 200 / 255
        bg[540:651, 60:561] = bg[540:651, 60:561] * (1 - al) + C * al
        out = cv2.imread(os.path.join(p, "clean", T["titles"][0]["group"], f"{f:06d}.png")).astype(np.float32)
        err = np.abs(out - bg[y0:y1, x0:x1]).max(2)[530 - y0:660 - y0, 50 - x0:570 - x0]
        print("static: frame", f, "mean", round(float(err.mean()), 2), "p99", round(float(np.percentile(err, 99)), 1))
        worst = max(worst, float(err.mean()))
    assert worst < 5, worst


if __name__ == "__main__":
    main()
