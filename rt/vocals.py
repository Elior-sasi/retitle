"""Speech apart from music before transcription. MDX-Net vocal model (Kim_Vocal_2, from the
Ultimate Vocal Remover model set), run locally with onnxruntime - on the graphics card when the
DirectML build is present. Whisper then hears the dialogue without the score and effects,
which helps most on trailers. The timing is untouched: the output has exactly the input length."""
import os, sys, subprocess, hashlib, urllib.request, shutil
import numpy as np
from . import vio

NAME = "Kim_Vocal_2.onnx"
URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/" + NAME
SHA256 = "ce74ef3b6a6024ce44211a07be9cf8bc6d87728cc852a68ab34eb8e58cde9c8b"
N_FFT, HOP, DIM_F, DIM_T, COMP = 7680, 1024, 3072, 256, 1.009
SR = 44100
STATE = {"state": "idle", "progress": 0.0}
_SESS = None


def model_dir():
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "Retitle", "models")


def model_path():
    for d in (os.environ.get("RETITLE_MODELS"), model_dir()):
        if d and os.path.exists(os.path.join(d, NAME)):
            return os.path.join(d, NAME)
    return None


def ready():
    return model_path() is not None


def fetch(progress=None):
    """Download once (about 66 MB), checksum verified."""
    if ready():
        return model_path()
    d = model_dir(); os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, NAME + ".part")
    STATE.update(state="downloading", progress=0.0)
    try:
        h = hashlib.sha256()
        with urllib.request.urlopen(URL, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0); got = 0
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b); h.update(b); got += len(b)
                if total:
                    STATE["progress"] = got / total
                    if progress:
                        progress(got / total)
        if h.hexdigest() != SHA256:
            raise RuntimeError("bad checksum")
        os.replace(tmp, os.path.join(d, NAME))
        STATE.update(state="ready", progress=1.0)
        return os.path.join(d, NAME)
    except Exception:
        STATE.update(state="error")
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _session():
    global _SESS
    if _SESS is None:
        import onnxruntime as ort
        so = ort.SessionOptions(); so.log_severity_level = 3
        prov = ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in ort.get_available_providers() and not os.environ.get("RT_NO_GPU"):
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            prov = ["DmlExecutionProvider", "CPUExecutionProvider"]
        _SESS = ort.InferenceSession(model_path(), so, providers=prov)
    return _SESS


_WIN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)   # periodic Hann


def _stft(x):
    """x (2, L) -> (2, N_FFT//2+1, T) complex, centred with reflect padding (as torch.stft)."""
    p = N_FFT // 2
    x = np.pad(x, ((0, 0), (p, p)), mode="reflect")
    T = 1 + (x.shape[1] - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(T)[:, None]
    fr = x[:, idx] * _WIN                        # (2, T, N_FFT)
    return np.fft.rfft(fr, axis=-1).transpose(0, 2, 1)


def _istft(S, length):
    fr = np.fft.irfft(S.transpose(0, 2, 1), n=N_FFT, axis=-1).astype(np.float32) * _WIN
    T = fr.shape[1]
    L = N_FFT + HOP * (T - 1)
    out = np.zeros((2, L), np.float32); wsum = np.zeros(L, np.float32)
    for t in range(T):
        out[:, t * HOP:t * HOP + N_FFT] += fr[:, t]
        wsum[t * HOP:t * HOP + N_FFT] += _WIN ** 2
    out /= np.maximum(wsum, 1e-8)
    p = N_FFT // 2
    return out[:, p:p + length]


OVERLAP = 0.5        # windows overlap and are blended: no seams at the window borders
SOFT = 0.25          # the rest of the soundtrack is kept, 12 dB down: Whisper hears clean speech
                     # without the artefacts a hard cut leaves (those turned words into other words)


def _window(mix, a, chunk):
    n = mix.shape[1]
    w = np.zeros((2, chunk), np.float32)
    lo, hi = max(0, a), min(n, a + chunk)
    if hi > lo:
        w[:, lo - a:hi - a] = mix[:, lo:hi]
    return w


def _infer(sess, w, chunk, trim):
    S = _stft(w)
    spec = np.stack([S.real, S.imag], 1).reshape(4, S.shape[1], S.shape[2])[:, :DIM_F]
    y = sess.run(None, {"input": spec[None].astype(np.float32)})[0][0]
    full = np.zeros((4, N_FFT // 2 + 1, DIM_T), np.float32); full[:, :DIM_F] = y
    full = full.reshape(2, 2, N_FFT // 2 + 1, DIM_T)
    return _istft(full[:, 0] + 1j * full[:, 1], chunk)[:, trim:-trim] * COMP


def chunks(mix, progress=None, overlap=OVERLAP):
    """mix (2, n) float32 at 44.1 kHz -> yields (start, vocals piece) that tile 0..n in order.
    Overlapping windows are cross-faded; memory stays small even for an hour of audio."""
    n = mix.shape[1]
    chunk = HOP * (DIM_T - 1)
    trim = N_FFT // 2
    gen = chunk - 2 * trim
    step = max(1, int(gen * (1 - overlap)))
    wgt = np.maximum(np.hanning(gen + 2)[1:-1], 1e-3).astype(np.float32) if overlap > 0 else np.ones(gen, np.float32)
    sess = _session()
    starts = list(range(-(gen - step) if overlap > 0 else 0, n, step))
    acc = np.zeros((2, 0), np.float32); wacc = np.zeros(0, np.float32); base = starts[0]; emitted = 0
    for k, s in enumerate(starts):
        v = _infer(sess, _window(mix, s - trim, chunk), chunk, trim)
        need = s + gen - base
        if acc.shape[1] < need:
            acc = np.concatenate([acc, np.zeros((2, need - acc.shape[1]), np.float32)], 1)
            wacc = np.concatenate([wacc, np.zeros(need - len(wacc), np.float32)])
        acc[:, s - base:s - base + gen] += v * wgt
        wacc[s - base:s - base + gen] += wgt
        upto = min(n, starts[k + 1] if k + 1 < len(starts) else n)      # no later window touches these
        if upto > emitted:
            a0, a1 = emitted - base, upto - base
            yield emitted, acc[:, a0:a1] / wacc[a0:a1]
            acc, wacc, base = acc[:, a1:], wacc[a1:], upto
            emitted = upto
        if progress:
            progress((k + 1) / len(starts))


def separate(mix, progress=None, overlap=OVERLAP):
    return np.concatenate([c for _, c in chunks(mix, progress, overlap)], 1)


def speech_wav(video, out_wav, progress=None, soft=SOFT, start=None, dur=None):
    """Write a 16 kHz mono WAV of the dialogue only (same length as the video's audio)."""
    cut = (["-ss", f"{start:.3f}"] if start else []) + (["-t", f"{dur:.3f}"] if dur else [])
    r = subprocess.run([vio.FFMPEG, "-v", "error", *cut, "-i", video, "-map", "0:a:0", "-ac", "2", "-ar", str(SR),
                        "-f", "f32le", "-"], capture_output=True, check=True, **vio.NOWIN)
    mix = np.frombuffer(r.stdout, np.float32).reshape(-1, 2).T.copy()
    del r
    p = subprocess.Popen([vio.FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "2", "-i", "-",
                          "-ac", "1", "-ar", "16000", out_wav], stdin=subprocess.PIPE, **vio.NOWIN)
    try:
        for i, c in chunks(mix, progress):
            out = c + soft * (mix[:, i:i + c.shape[1]] - c)
            p.stdin.write(np.ascontiguousarray(out.T).astype(np.float32).tobytes())
    except BaseException:
        p.kill(); p.wait()
        raise
    p.stdin.close(); p.wait()
    if p.returncode:
        raise RuntimeError("could not write the speech track")
    return out_wav
