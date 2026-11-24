"""DictaLM 3.0 (Dicta, Apache 2.0) - a Hebrew model on this computer, as a second translation engine next to Claude.

The 24B model (unsloth/DictaLM-3.0-24B-Thinking-GGUF; Q4_K_M 14.3 GB, Q6_K 19.3 GB or Q8_0 25.1 GB) runs in llama.cpp's server
(github.com/ggml-org/llama.cpp, MIT): as many layers as fit on the NVIDIA card, the rest from RAM. Both are
downloaded once into %LOCALAPPDATA%/Retitle/dicta. It is free and does not use the subscription; it is slow
(a laptop card holds only part of the model), so it is used where the user picks it, and in the side by side
comparison with Claude."""
import os, re, json, time, glob, shutil, socket, subprocess, threading, zipfile, urllib.request
from . import vio

GGUF_REPO = "unsloth/DictaLM-3.0-24B-Thinking-GGUF"
QUANTS = {"Q4_K_M": 14.3, "Q6_K": 19.3, "Q8_0": 25.1}     # size in GB; all are the full 24B model, less or more compressed


def quant():
    s = vio.load_json(os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/.config"), "Retitle", "settings.json"), {})
    q = s.get("dicta_quant") or "Q4_K_M"
    return q if q in QUANTS else "Q4_K_M"


def _gguf():
    return f"DictaLM-3.0-24B-Thinking-{quant()}.gguf"
LAYERS = 40                                   # Mistral-Small-3.1 24B
STATE = {"state": "idle", "progress": None, "what": "", "error": ""}
_T = [None]
_SRV = {"proc": None, "port": 0}
_LOCK = threading.Lock()


def home():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    return os.path.join(base, "Retitle", "dicta")


def _server():
    for p in glob.glob(os.path.join(home(), "llama", "**", "llama-server.exe" if os.name == "nt" else "llama-server"),
                       recursive=True):
        return p
    return shutil.which("llama-server") or ""


def _model():
    return os.path.join(home(), _gguf())


def ready():
    return bool(_server()) and os.path.exists(_model()) and os.path.getsize(_model()) > 0.9e9 * QUANTS[quant()]


def status():
    if ready() and STATE["state"] != "installing":
        STATE.update(state="ready")
    return dict(STATE, ready=ready())


def _get(url, dst, part):
    """Download with resume (a 14 GB file over a home connection may be cut)."""
    tmp = dst + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Retitle", **({"Range": f"bytes={have}-"} if have else {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        if have and r.status != 206:
            have = 0
        total = have + int(r.headers.get("Content-Length") or 0)
        with open(tmp, "ab" if have else "wb") as f:
            done = have
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b); done += len(b)
                if total:
                    STATE["progress"] = part[0] + (part[1] - part[0]) * done / total
    os.replace(tmp, dst)


def _llama_assets():
    """Download links of llama.cpp's latest Windows build: CUDA (+ its runtime) with an NVIDIA card, else CPU."""
    from .propaint import gpu
    with urllib.request.urlopen(urllib.request.Request("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
                                                       headers={"User-Agent": "Retitle"}), timeout=60) as r:
        rel = json.load(r)
    names = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
    if gpu():
        main = [n for n in names if re.search(r"bin-win-cuda-12[\d.]*-x64\.zip$", n)]
        rt = [n for n in names if re.search(r"^cudart-.*win-cuda-12[\d.]*-x64\.zip$", n)]
        if main:
            main.sort(); rt.sort()
            return [names[main[-1]]] + ([names[rt[-1]]] if rt else [])
    cpu = [n for n in names if re.search(r"bin-win-(cpu|avx2)-x64\.zip$", n)]
    if not cpu:
        raise RuntimeError("לא נמצאה גרסת Windows של llama.cpp")
    return [names[sorted(cpu)[-1]]]


def install():
    os.makedirs(home(), exist_ok=True)
    STATE.update(state="installing", progress=0.0, error="")
    try:
        if not _server():
            STATE["what"] = "מוריד את llama.cpp"
            d = os.path.join(home(), "llama"); os.makedirs(d, exist_ok=True)
            for i, u in enumerate(_llama_assets()):
                z = os.path.join(home(), f"l{i}.zip")
                _get(u, z, (0.0, 0.04))
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(d)
                os.remove(z)
        if not os.path.exists(_model()):
            STATE["what"] = f"מוריד את DictaLM 3.0 ({QUANTS[quant()]}GB)"
            _get(f"https://huggingface.co/{GGUF_REPO}/resolve/main/{_gguf()}", _model(), (0.04, 1.0))
        STATE.update(state="ready", progress=1.0, what="")
    except Exception as e:
        STATE.update(state="error", error=str(e)[:300], what="")
        raise


def install_async():
    if _T[0] and _T[0].is_alive():
        return
    def go():
        try:
            install()
        except Exception:
            pass
    _T[0] = threading.Thread(target=go, daemon=True)
    _T[0].start()


def _gpu_layers():
    from .propaint import gpu, vram_gb
    if not gpu():
        return 0
    per = QUANTS[quant()] / LAYERS            # GB per layer
    return max(0, min(LAYERS, int((vram_gb() - 1.4) / per)))


def start(log=print):
    """The model in llama-server (once per app session; it stays loaded for the next requests)."""
    with _LOCK:
        p = _SRV["proc"]
        if p and p.poll() is None:
            return _SRV["port"]
        if not ready():
            raise RuntimeError("DictaLM עוד לא הותקן - מתקינים אותו בהעדפות")
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
        ngl = _gpu_layers()
        log(f"טוען את DictaLM ({ngl} מתוך {LAYERS} שכבות על כרטיס המסך) - בפעם הראשונה זה לוקח דקה-שתיים")
        cmd = [_server(), "-m", _model(), "--port", str(port), "--host", "127.0.0.1", "-c", "8192",
               "-ngl", str(ngl), "--reasoning-budget", "0", "--jinja"]
        _SRV["proc"] = subprocess.Popen(cmd, cwd=os.path.dirname(_server()), stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _SRV["port"] = port
        t0 = time.time()
        while time.time() - t0 < 600:
            if _SRV["proc"].poll() is not None:
                raise RuntimeError("DictaLM לא עלה (אולי חסר זיכרון)")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                    if r.status == 200:
                        return port
            except Exception:
                pass
            time.sleep(2)
        raise RuntimeError("DictaLM לא עלה בזמן")


def stop():
    p = _SRV["proc"]
    if p and p.poll() is None:
        p.terminate()


def chat(system, user, max_tokens=4096, log=print):
    port = start(log)
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.3, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.load(r)
    text = res["choices"][0]["message"].get("content") or ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def translate(items, glossary="", context="", src_lang="", log=print):
    """The same contract as translate.translate: items [{"id","src","group"}] -> {id: he}."""
    from .translate import RULES, NIQQUD
    from .house import with_house
    if not items:
        return {}
    out = {}
    for i in range(0, len(items), 25):            # small batches: the model's context is short and it is slow
        part = items[i:i + 25]
        user = (f"Source language: {src_lang or 'auto'}\nGlossary (source = Hebrew):\n{glossary or '-'}\n"
                f"Context: {context or '-'}\n\nItems:\n" + json.dumps(part, ensure_ascii=False, separators=(",", ":")) +
                '\n\nReply ONLY with JSON: {"items":[{"id":"...","he":"..."}]}')
        text = chat(with_house(RULES), user, log=log)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(m.group(0)) if m else {"items": []}
        except ValueError:
            data = {"items": []}
        for it in data.get("items", []):
            he = NIQQUD.sub("", str(it.get("he", ""))).replace(chr(0x05BE), "-").replace(chr(0x2014), "-").replace(chr(0x2013), "-")
            he = he.replace(chr(0x201C), '"').replace(chr(0x201D), '"').replace(chr(0x05F4), '"')
            out[str(it.get("id"))] = he
    return out
