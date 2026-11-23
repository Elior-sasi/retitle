"""End-to-end check: generated clip with 3 animated English titles -> Hebrew video + SRT."""
import os, sys, json, shutil, subprocess, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rt import vio, shots, detect, extract, clean, render, compose, check, srt

d = tempfile.mkdtemp()
clip = os.path.join(d, "clip.mp4")
subprocess.run([sys.executable, os.path.join(ROOT, "tests", "make_clip.py"), clip], check=True)
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
clean.run(p); render.track(p); render.build_layers(p)
out = compose.run(p); check.run(p)
assert os.path.getsize(out) > 10000
json.dump({"language": "en", "segments": [{"start": 0.2, "end": 1.1, "text": "Hello.", "words": []}]},
          open(os.path.join(p, "w.json"), "w"))
srt.prep(p, os.path.join(p, "w.json"))
D = vio.load_json(os.path.join(p, "dialog.json")); D["cues"][0]["he"] = "שלום."
vio.save_json(os.path.join(p, "dialog.json"), D)
sp, w = srt.build(p)
print(open(sp, encoding="utf-8").read())
print("SELFTEST OK")
