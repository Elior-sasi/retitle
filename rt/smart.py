"""Smart export: only the stretches with titles are encoded again; everything else is copied
bit for bit from the source (no generation loss, and minutes become seconds on long episodes).
Every result is checked frame by frame against the source; on any doubt the caller falls back
to a full encode."""
import os, subprocess, hashlib, shutil
import cv2
import numpy as np
from . import vio

WHY = [""]


def _no(why):
    WHY[0] = why
    print("smart export skipped:", why)
    return False


def keyframes(src):
    """Frame indices (presentation order) of key frames, plus each frame's time in seconds."""
    out = subprocess.run([vio.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "packet=pts,flags", "-of", "csv=p=0", src], capture_output=True, text=True,
                         check=True, **vio.NOWIN).stdout
    tb = subprocess.run([vio.FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=time_base", "-of", "csv=p=0", src], capture_output=True, text=True,
                        check=True, **vio.NOWIN).stdout.strip()
    num, den = (int(x) for x in tb.split("/"))
    pk = []
    for ln in out.splitlines():
        p = ln.split(",")
        if len(p) >= 2 and p[0] not in ("", "N/A"):
            pk.append((int(p[0]), "K" in p[1]))
    pk.sort()
    times = [pts * num / den for pts, _ in pk]
    return [i for i, (_, k) in enumerate(pk) if k], times


def usable(info, scale):
    return not scale and info.get("codec") == "h264" and info.get("pix_fmt") == "yuv420p"


def plan(busy, n, keys):
    """busy: set of frame indices to rebuild -> [(kind, a, b)] covering 0..n, encode stretches
    widened to whole key-frame groups so the copied stretches start on a key frame."""
    import bisect
    ks = sorted(set(keys) | {0})
    if not busy:
        return [("copy", 0, n)]
    fr = sorted(busy)
    runs, s = [], fr[0]
    for x, y in zip(fr, fr[1:] + [None]):
        if y is None or y != x + 1:
            runs.append((s, x + 1)); s = y
    enc = []
    for a, b in runs:
        ka = ks[bisect.bisect_right(ks, a) - 1]
        j = bisect.bisect_left(ks, b)
        kb = ks[j] if j < len(ks) else n
        if enc and ka <= enc[-1][1]:
            enc[-1][1] = max(enc[-1][1], kb)
        else:
            enc.append([ka, kb])
    segs, pos = [], 0
    for a, b in enc:
        if a > pos:
            segs.append(("copy", pos, a))
        segs.append(("enc", a, b)); pos = b
    if pos < n:
        segs.append(("copy", pos, n))
    return segs


def _md5s(path, fr0=0, count=None):
    """md5 of every decoded frame (as ffmpeg decodes it) - the comparison is exact."""
    args = [vio.FFMPEG, "-v", "error", "-i", path, "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "framemd5", "-"]
    out = subprocess.run(args, capture_output=True, text=True, **vio.NOWIN).stdout
    return [ln.rsplit(",", 1)[1].strip() for ln in out.splitlines() if ln and not ln.startswith("#")]


def run(src, info, n, busy, render, out, crf=14, preset="veryfast", progress=None, live_cb=None):
    """render(f, frame) -> output frame for f in busy. Returns True when the smart file is written
    and verified; False means: do a full encode instead (nothing is left behind)."""
    keys, times = keyframes(src)
    if len(times) != n or not keys or keys[0] != 0:
        return _no(f"frames {len(times)} vs {n}, first key {keys[:1]}")
    segs = plan(busy, n, keys)
    tot_enc = sum(b - a for k, a, b in segs if k == "enc")
    if tot_enc > 0.7 * n:
        return _no("titles almost everywhere")  #: a plain encode is as fast
    W, H = info["width"], info["height"]
    fps = f"{info['fps_num']}/{info['fps_den']}"
    tmp = out + ".parts"
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    try:
        files, done, refs = [], 0, {}
        # the source cut at the stretch borders (all key frames) in one pass, packets untouched
        cuts = ",".join(str(a) for _, a, _ in segs[1:])
        r = subprocess.run([vio.FFMPEG, "-v", "error", "-y", "-i", src, "-map", "0:v:0", "-c", "copy",
                            "-bsf:v", "h264_mp4toannexb,dump_extra=freq=keyframe", "-f", "segment", "-segment_format", "mpegts",
                            *(["-segment_frames", cuts] if cuts else []), "-reset_timestamps", "0",
                            os.path.join(tmp, "src%04d.ts")], capture_output=True, **vio.NOWIN)
        if r.returncode or not all(os.path.exists(os.path.join(tmp, f"src{k:04d}.ts")) for k in range(len(segs))):
            return _no("split failed")
        for k, (kind, a, b) in enumerate(segs):
            pth = os.path.join(tmp, f"src{k:04d}.ts") if kind == "copy" else os.path.join(tmp, f"{k:04d}.ts")
            if kind == "copy":
                pass
            else:
                # decode the stretch from its own cut (starts on a key frame): no seeking, frame exact
                dec = subprocess.Popen([vio.FFMPEG, "-v", "error", "-i", os.path.join(tmp, f"src{k:04d}.ts"), "-map", "0:v:0", "-frames:v", str(b - a),
                                        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **vio.NOWIN)
                enc = subprocess.Popen([vio.FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                                        "-r", fps, "-i", "-", "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
                                        "-pix_fmt", "yuv420p", "-x264-params", "open-gop=0", "-f", "mpegts", pth],
                                       stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, **vio.NOWIN)
                nb = W * H * 3
                try:
                    for f in range(a, b):
                        buf = dec.stdout.read(nb)
                        if len(buf) < nb:
                            raise RuntimeError("short read")
                        fr = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
                        res = render(f, fr)
                        refs.setdefault((a, b), []).append(cv2.resize(cv2.cvtColor(np.ascontiguousarray(res), cv2.COLOR_BGR2GRAY),
                                                                         (480, 270), interpolation=cv2.INTER_AREA).astype(np.float32))
                        if live_cb:
                            live_cb(f, res, fr)
                        enc.stdin.write(np.ascontiguousarray(res).tobytes())
                        done += 1
                        if progress and done % 12 == 0:
                            progress(0.85 * done / max(1, tot_enc))
                except BaseException:
                    enc.kill(); dec.kill(); enc.wait(); dec.wait()
                    raise
                enc.stdin.close(); enc.wait(); dec.stdout.close(); dec.wait()
                if enc.returncode:
                    return _no("encode failed")
            files.append(pth)
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w", encoding="utf-8") as fh:
            for p in files:
                fh.write("file '" + p.replace("\\", "/").replace("'", "'\\''") + "'\n")
        tmpout = out + ".tmp.mp4"
        r = subprocess.run([vio.FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", src,
                            "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-r", fps, "-movflags", "+faststart", tmpout],
                           capture_output=True, **vio.NOWIN)
        if r.returncode:
            return _no("join failed")
        if progress:
            progress(0.9)
        # check: every copied frame decodes exactly like the source, and the frame count is right
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as ex:
            fa, fb = ex.submit(_md5s, src), ex.submit(_md5s, tmpout)
            a5, b5 = fa.result(), fb.result()
        if len(a5) != n or len(b5) != n:
            _rm(tmpout); return _no(f"frame count {len(a5)} {len(b5)} {n}")
        for kind, a, b in segs:
            if kind == "copy" and a5[a:b] != b5[a:b]:
                _rm(tmpout); return _no(f"copied frames differ {a}-{b}")
        for kind, a, b in segs:
            if kind == "enc" and _psnr_ref(tmpout, a, b, refs[(a, b)]) < 30:   # against what was rendered
                _rm(tmpout); return _no(f"encoded stretch {a}-{b} does not match")
        os.replace(tmpout, out)
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _rm(p):
    try:
        os.remove(p)
    except OSError:
        pass


def _gray(path, a, b, w=480, h=270):
    r = subprocess.run([vio.FFMPEG, "-v", "error", "-i", path, "-map", "0:v:0", "-vf",
                        f"trim=start_frame={a}:end_frame={b},scale={w}:{h}", "-fps_mode", "passthrough",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, **vio.NOWIN)
    d = np.frombuffer(r.stdout, np.uint8)
    return d[: len(d) // (w * h) * w * h].reshape(-1, h, w).astype(np.float32)


def _psnr(x, y, a, b):
    """Mean PSNR of frames a..b-1 between two files (the titles change only part of the picture)."""
    A, B = _gray(x, a, b), _gray(y, a, b)
    if len(A) != b - a or len(B) != b - a:
        return 0.0
    mse = ((A - B) ** 2).mean(axis=(1, 2))
    return float(np.mean(10 * np.log10(255 ** 2 / np.maximum(mse, 1e-3))))


def _psnr_ref(path, a, b, ref):
    B = _gray(path, a, b, 480, 270)
    if len(B) != len(ref):
        return 0.0
    A = np.stack(ref)
    mse = ((A - B) ** 2).mean(axis=(1, 2))
    return float(np.mean(10 * np.log10(255 ** 2 / np.maximum(mse, 1e-3))))
