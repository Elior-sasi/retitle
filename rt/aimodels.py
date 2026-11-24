"""Two more free models next to ProPainter, in the same private Python (%LOCALAPPDATA%/Retitle/propainter):

SAM 2 (github.com/facebookresearch/sam2, Apache 2.0) - exact masks of a title through all its frames: the box
  the detector found is the prompt, SAM 2 follows the letters (with their 3D side, bevel, glow and shadow) frame
  by frame. The erase then takes the whole letter, not only the part the colour model saw.
Lens (github.com/mixfont/lens, free for non-commercial use) - names the Latin font of the source among about
  1000 open fonts, and its network describes how any lettering looks (an embedding), so the source can be
  compared with every font on this computer and every Google font, Latin and Hebrew.

Both run in a helper process (ai_worker.py, written next to the models) so the app itself stays small."""
import os, sys, json, glob, shutil, subprocess, tempfile, zipfile
import numpy as np
import cv2
from . import vio, propaint

SAM_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM_ZIP = f"https://github.com/facebookresearch/sam2/archive/{SAM_COMMIT}.zip"
SAM_CKPT = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
LENS_COMMIT = "dcbc882de3c91d936a00ee94bff88e8fa754c148"
LENS_ZIP = f"https://github.com/mixfont/lens/archive/{LENS_COMMIT}.zip"
PKGS = ["hydra-core>=1.3.2", "iopath>=0.1.10", "tqdm"]

WORKER = r'''
import sys, os, json
job = json.load(open(sys.argv[1], encoding="utf-8"))
import numpy as np, torch
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = {}
if job["cmd"] == "sam":
    sys.path.insert(0, job["code"])
    from sam2.build_sam import build_sam2_video_predictor
    import cv2
    pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_s.yaml", job["ckpt"], device=dev)
    ac = torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)
    with torch.inference_mode(), ac:
        st = pred.init_state(job["frames"], offload_video_to_cpu=True, offload_state_to_cpu=dev == "cuda")
        for i, p in enumerate(job["prompts"]):
            pred.add_new_points_or_box(st, frame_idx=p["frame"], obj_id=i + 1, box=np.array(p["box"], np.float32))
        masks = {}
        n = len(os.listdir(job["frames"]))
        done = 0
        for rev in (False, True):
            for fi, ids, logits in pred.propagate_in_video(st, reverse=rev):
                m = (logits > 0).any(0).squeeze().cpu().numpy()
                masks[fi] = (masks[fi] | m) if fi in masks else m
                done += 1
                print(f"PROG {min(1.0, done / (2.0 * n)):.3f}", flush=True)
    os.makedirs(job["out"], exist_ok=True)
    for fi, m in masks.items():
        cv2.imwrite(os.path.join(job["out"], f"{fi:05d}.png"), m.astype(np.uint8) * 255)
    out = {"frames": len(masks), "device": dev}
else:
    sys.path.insert(0, job["code"])
    from pathlib import Path
    from PIL import Image
    from lens_inference import get_model_bundle, run_model
    b = get_model_bundle(Path(job["code"]) / "model" / "font_classifier.pt")
    if job["cmd"] == "lens":
        out = {p: run_model(Image.open(p).convert("RGB"), b, int(job.get("k", 8))) for p in job["images"]}
    else:                                    # embed: the network's view of the lettering, before the font classes
        fc = b.model.fc
        b.model.fc = torch.nn.Identity()
        vecs = []
        with torch.inference_mode():
            for i in range(0, len(job["images"]), 64):
                xs = [b.tensor_transform(b.preprocess(Image.open(p).convert("RGB"))) for p in job["images"][i:i + 64]]
                v = b.model(torch.stack(xs).to(b.device)).float()
                vecs.append(torch.nn.functional.normalize(v, dim=1).cpu().numpy())
                print(f"PROG {min(1.0, (i + 64) / max(1, len(job['images']))):.3f}", flush=True)
        b.model.fc = fc
        np.save(job["out"], np.concatenate(vecs) if vecs else np.zeros((0, 512), np.float32))
        out = {"n": len(job["images"])}
print("RESULT " + json.dumps(out), flush=True)
'''


def sam_dir():
    return os.path.join(propaint.home(), "sam2-" + SAM_COMMIT)


def lens_dir():
    return os.path.join(propaint.home(), "lens-" + LENS_COMMIT)


def _ckpt():
    return os.path.join(propaint.home(), "sam2.1_hiera_small.pt")


def sam_ready():
    return propaint.ready() and os.path.exists(os.path.join(sam_dir(), "sam2", "build_sam.py")) and \
        os.path.exists(_ckpt()) and os.path.exists(os.path.join(propaint.home(), "ok_ai"))


def lens_ready():
    return propaint.ready() and os.path.exists(os.path.join(lens_dir(), "model", "font_classifier.pt")) and \
        os.path.exists(os.path.join(propaint.home(), "ok_ai"))


def ready():
    return sam_ready() and lens_ready()


def install(part=(0.0, 1.0)):
    """SAM 2 and Lens into ProPainter's Python (which must be set up first). Finished parts are kept."""
    h = propaint.home()
    S = propaint.STATE

    def p(a):
        return part[0] + (part[1] - part[0]) * a
    uv = shutil.which("uv") or os.path.join(h, "uv.exe" if os.name == "nt" else "uv")
    propaint._run([uv, "pip", "install", "--python", propaint._py()] + PKGS, (p(0), p(0.1)), "מתקין ספריות ל-SAM 2")
    for url, d, a in ((SAM_ZIP, sam_dir(), 0.1), (LENS_ZIP, lens_dir(), 0.2)):
        if not os.path.isdir(d):
            S["what"] = "מוריד את " + ("SAM 2" if "sam2" in url else "Lens (זיהוי גופנים)")
            z = os.path.join(h, "ai.zip")
            propaint._get(url, z, (p(a), p(a + 0.1)))
            with zipfile.ZipFile(z) as zf:
                zf.extractall(h)
            os.remove(z)
    if not os.path.exists(_ckpt()):
        S["what"] = "מוריד את המודל של SAM 2"
        propaint._get(SAM_CKPT, _ckpt(), (p(0.3), p(1.0)))
    with open(os.path.join(h, "ai_worker.py"), "w", encoding="utf-8") as f:
        f.write(WORKER)
    open(os.path.join(h, "ok_ai"), "w").close()


def _work(job, progress=None, log=print, timeout=7200):
    h = propaint.home()
    w = os.path.join(h, "ai_worker.py")
    with open(w, "w", encoding="utf-8") as f:          # always the current version
        f.write(WORKER)
    fd, jp = tempfile.mkstemp(suffix=".json"); os.close(fd)
    vio.save_json(jp, job)
    try:
        pr = subprocess.Popen([propaint._py(), w, jp], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        tail, res = [], None
        for line in pr.stdout:
            if line.startswith("PROG ") and progress:
                try:
                    progress(float(line[5:]))
                except ValueError:
                    pass
            elif line.startswith("RESULT "):
                res = json.loads(line[7:])
            else:
                tail = (tail + [line])[-15:]
        pr.wait(timeout)
        if res is None:
            raise RuntimeError("".join(tail)[-400:])
        return res
    finally:
        os.remove(jp)


# ------------------------------------------------------------------ SAM 2
def sam_card(proj, gid, log=print, progress=None):
    """Masks of card gid's translated lines in every cached frame -> proj/sam/<gid>/<frame>.png (cache-crop
    coordinates, the same as clean's). A frame where SAM 2 took far more than the letters (the background of
    the box) is dropped, so a wrong mask never widens the erase."""
    from .extract import groups
    if not sam_ready():
        raise RuntimeError("SAM 2 עוד לא מותקן")
    T = vio.load_json(os.path.join(proj, "titles.json"))
    lines = [t for t in groups(T)[gid] if t.get("translate", True) and "cache" in t and "mask" in t]
    if not lines:
        return 0
    lo, hi = lines[0]["cache"]
    reg = lines[0]["region"]
    work = os.path.join(proj, "sam_work", gid)
    shutil.rmtree(work, ignore_errors=True)
    fr = os.path.join(work, "frames"); os.makedirs(fr)
    for i, f in enumerate(range(lo, hi + 1)):
        im = cv2.imread(os.path.join(proj, "cache", gid, f"{f:06d}.png"))
        if im is None:
            break
        cv2.imwrite(os.path.join(fr, f"{i:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    prompts, areas = [], []
    for t in lines:
        bx = t.get("ref_box") or t.get("text_box") or t["box"]
        pad = max(3, int(0.06 * (bx[3] - bx[1])))
        b = [bx[0] - reg[0] - pad, bx[1] - reg[1] - pad, bx[2] - reg[0] + pad, bx[3] - reg[1] + pad]
        prompts.append({"frame": int(t.get("ref", lo)) - lo, "box": b})
        areas.append((b[2] - b[0]) * (b[3] - b[1]))
    log(f"SAM 2 עוקב אחרי האותיות של {gid}")
    r = _work({"cmd": "sam", "code": sam_dir(), "ckpt": _ckpt(), "frames": fr, "prompts": prompts,
               "out": os.path.join(work, "out")}, progress=progress, log=log)
    d = os.path.join(proj, "sam", gid)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    limit = 1.6 * sum(areas)
    kept = 0
    for p in glob.glob(os.path.join(work, "out", "*.png")):
        i = int(os.path.basename(p)[:5])
        m = cv2.imread(p, 0)
        if m is None or not m.any() or int((m > 0).sum()) > limit:
            continue
        cv2.imwrite(os.path.join(d, f"{lo + i:06d}.png"), m)
        kept += 1
    shutil.rmtree(work, ignore_errors=True)
    for t in T["titles"]:
        if t.get("group") == gid:
            t["sam"] = kept > 0
            t["built"] = None
    vio.save_json(os.path.join(proj, "titles.json"), T)
    log(f"SAM 2: מסכות ל-{kept} פריימים ({r.get('device')})")
    return kept


def sam_mask(proj, gid, f):
    """SAM 2's mask of frame f (cache-crop size), or None."""
    p = os.path.join(proj, "sam", gid, f"{f:06d}.png")
    return cv2.imread(p, 0) if os.path.exists(p) else None


# ------------------------------------------------------------------ Lens
def lens(images, k=8):
    """{image path: [{"name": family, "score": 0..1}, ...]} - the Latin font of each image."""
    if not lens_ready():
        return {}
    r = _work({"cmd": "lens", "code": lens_dir(), "images": list(images), "k": k})
    return {p: [{"name": x["name"], "score": x["score"]} for x in v] for p, v in r.items()}


def embed(images, progress=None):
    """Lens's description of each image's lettering: unit vectors (n, 512); similar lettering -> close vectors."""
    if not lens_ready() or not images:
        return None
    fd, out = tempfile.mkstemp(suffix=".npy"); os.close(fd)
    try:
        _work({"cmd": "embed", "code": lens_dir(), "images": list(images), "out": out}, progress=progress)
        return np.load(out)
    finally:
        os.remove(out)
