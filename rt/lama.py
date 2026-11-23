"""LaMa inpainting (Suvorov et al. 2022, Apache-2.0; ONNX int8 build from OpenCV Zoo), run locally
with onnxruntime. Used for background that is never visible in any other frame."""
import os, sys
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "inpainting_lama_2025jan.onnx"
SHA256 = "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2"
URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/inpainting_lama/" + NAME
_SESS = None


def model_path():
    for d in (os.environ.get("RETITLE_MODELS"), os.path.join(os.path.dirname(HERE), "models"),
              os.path.join(getattr(sys, "_MEIPASS", ""), "models")):
        if d and os.path.exists(os.path.join(d, NAME)):
            return os.path.join(d, NAME)
    return None


def available():
    return model_path() is not None


def _session():
    global _SESS
    if _SESS is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = int(os.environ.get("RT_THREADS") or 0) or max(1, (os.cpu_count() or 2))
        _SESS = ort.InferenceSession(model_path(), so, providers=["CPUExecutionProvider"])
    return _SESS


def inpaint(img, hole):
    """img BGR uint8, hole bool. Fills hole pixels; context = a crop around the hole."""
    hole = hole.astype(bool)
    ys, xs = np.nonzero(hole)
    if len(ys) == 0:
        return img
    H, W = hole.shape
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    # square crop with generous context around the holes
    side = int(max(y1 - y0, x1 - x0) * 1.4 + 96)
    side = max(side, 256)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    X0, Y0 = max(0, min(W - side, cx - side // 2)), max(0, min(H - side, cy - side // 2))
    X1, Y1 = min(W, X0 + side), min(H, Y0 + side)
    crop = img[Y0:Y1, X0:X1]; m = hole[Y0:Y1, X0:X1]
    ch, cw = crop.shape[:2]
    x = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA if max(ch, cw) > 512 else cv2.INTER_CUBIC)
    mm = cv2.resize(m.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
    mm = cv2.dilate(mm, np.ones((3, 3), np.uint8))
    inp = (x[..., ::-1].astype(np.float32) / 255).transpose(2, 0, 1)[None]
    out = _session().run(None, {"image": inp, "mask": mm[None, None].astype(np.float32)})[0][0]
    out = out.transpose(1, 2, 0)
    if out.max() <= 1.5:
        out = out * 255
    out = np.clip(out, 0, 255).astype(np.uint8)[..., ::-1]
    out = cv2.resize(out, (cw, ch), interpolation=cv2.INTER_CUBIC)
    res = img.copy()
    sub = res[Y0:Y1, X0:X1]
    sub[m] = out[m]
    # outside the crop (very wide holes): fall back to diffusion
    rest = hole.copy(); rest[Y0:Y1, X0:X1] = False
    if rest.any():
        res = cv2.inpaint(res, rest.astype(np.uint8) * 255, 5, cv2.INPAINT_NS)
    return res


def fetch(dest_dir):
    """Download the model (used by the build)."""
    import hashlib, urllib.request
    os.makedirs(dest_dir, exist_ok=True)
    p = os.path.join(dest_dir, NAME)
    if not os.path.exists(p):
        urllib.request.urlretrieve(URL, p)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != SHA256:
        os.remove(p)
        raise RuntimeError("LaMa model checksum mismatch")
    return p
