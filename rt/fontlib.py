"""Every Hebrew font you can use, from five sources:

  built-in  - the set shipped with the app (fonts/catalog.json)
  google    - every Google Fonts family with Hebrew (rt/gfonts_he.json), downloaded on first use (OFL / Apache)
  system    - fonts installed in Windows, including Adobe Fonts activated from Creative Cloud
  mine      - TTF/OTF files you import (kept in the sync folder when one is set, so other computers get them)

Only fonts that actually contain Hebrew letters are listed. Names are unique; a font is referred to by name.
"""
import os, json, shutil, threading, urllib.request, urllib.parse, time
from . import vio

HERE = os.path.dirname(os.path.abspath(__file__))
BUILTIN = os.environ.get("RETITLE_FONTS", os.path.join(os.path.dirname(HERE), "fonts"))
GOOGLE = os.path.join(HERE, "gfonts_he.json")
GH_RAW = "https://raw.githubusercontent.com/google/fonts/main/"
CAT_HE = {"sans_serif": "סנס", "serif": "סריף", "display": "תצוגה", "handwriting": "כתב יד", "monospace": "מונוספייס"}
EXTS = (".ttf", ".otf", ".ttc")
SRC_HE = {"system": "מערכת", "adobe": "Adobe", "mine": "שלי"}
_LOCK = threading.Lock()
_CACHE = {"t": 0, "cat": None}


def _appdata():
    return os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~/.config"), "Retitle")


def user_dir():
    """Imported + downloaded fonts. In the sync folder when one is set."""
    d = vio.load_json(os.path.join(_appdata(), "settings.json"), {}).get("sync_dir", "")
    base = os.path.join(d, "Retitle") if d and os.path.isdir(d) else _appdata()
    return os.path.join(base, "fonts")


def _has_hebrew(path, index=0):
    from fontTools.ttLib import TTFont
    try:
        f = TTFont(path, fontNumber=index, lazy=True)
        cmap = f.getBestCmap() or {}
        if not all(c in cmap for c in (0x05D0, 0x05E9, 0x05EA)):     # alef, shin, tav
            return None
        nm = f["name"]
        full = nm.getDebugName(4) or nm.getDebugName(1) or os.path.splitext(os.path.basename(path))[0]
        fam = nm.getDebugName(16) or nm.getDebugName(1) or full
        return {"name": full.strip(), "family": fam.strip(), "variable": "fvar" in f}
    except Exception:
        return None


def system_dirs():
    out = []
    if os.name == "nt":
        win = os.environ.get("WINDIR", r"C:\Windows")
        la = os.environ.get("LOCALAPPDATA", "")
        out += [(os.path.join(win, "Fonts"), "system"), (os.path.join(la, "Microsoft", "Windows", "Fonts"), "system"),
                (os.path.join(os.environ.get("APPDATA", ""), "Adobe", "CoreSync", "plugins", "livetype", "r"), "adobe")]
    else:
        out += [("/usr/share/fonts", "system"), (os.path.expanduser("~/.local/share/fonts"), "system"),
                (os.path.expanduser("~/Library/Fonts"), "system"), ("/Library/Fonts", "system")]
    return [(d, s) for d, s in out if d and os.path.isdir(d)]


def _scan(dirs):
    """Hebrew fonts in dirs; results cached by path + mtime (system font folders hold thousands of files)."""
    cache_p = os.path.join(_appdata(), "fontscan.json")
    cache = vio.load_json(cache_p, {})
    out, seen = [], {}
    for d, src in dirs:
        for root, _, files in os.walk(d):
            for fn in files:
                p = os.path.join(root, fn)
                ext = os.path.splitext(fn)[1].lower()
                if src != "adobe" and ext not in EXTS:
                    continue                      # Adobe Fonts files have no extension
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                k = f"{p}|{int(st.st_mtime)}|{st.st_size}"
                if k not in cache:
                    cache[k] = _has_hebrew(p) or {}
                seen[k] = cache[k]
                if cache[k]:
                    out.append(dict(cache[k], file=p, source=src))
    os.makedirs(_appdata(), exist_ok=True)
    vio.save_json(cache_p, seen)
    return out


def catalog(refresh=False):
    """name -> {file, source, he, family, installed, ...}. Google families not downloaded yet have installed False."""
    with _LOCK:
        if _CACHE["cat"] is not None and not refresh and time.time() - _CACHE["t"] < 300:
            return _CACHE["cat"]
        cat = {}
        for n, v in vio.load_json(os.path.join(BUILTIN, "catalog.json"), {}).items():
            cat[n] = {"file": os.path.join(BUILTIN, v["file"]), "source": "built-in", "he": v.get("he", ""), "installed": True}
        ud = user_dir()
        for n, g in vio.load_json(GOOGLE, {}).items():
            for fn, w in g["files"]:
                nm = n if (g["variable"] or w == 400 or len(g["files"]) == 1) else f"{n} {_wname(w)}"
                if nm in cat:
                    continue
                local = os.path.join(ud, "google", fn)
                cat[nm] = {"file": local, "source": "google", "he": CAT_HE.get(g["cat"], "") + (", כל המשקלים" if g["variable"] else ""),
                           "installed": os.path.exists(local), "url": GH_RAW + g["dir"] + "/" + urllib.parse.quote(fn)}
        mine = [(os.path.join(ud, "mine"), "mine")] if os.path.isdir(os.path.join(ud, "mine")) else []
        for f in _scan(mine + system_dirs()):
            nm = f["name"]
            if nm in cat and cat[nm]["source"] in ("built-in", "google") and f["source"] != "mine":
                continue                          # same font already listed (e.g. Heebo installed in Windows)
            if nm in cat:
                nm = nm + " (" + SRC_HE[f["source"]] + ")"
            cat[nm] = {"file": f["file"], "source": f["source"], "he": "", "family": f["family"], "installed": True}
        _CACHE.update(t=time.time(), cat=cat)
        return cat


def _wname(w):
    return {100: "Thin", 200: "ExtraLight", 300: "Light", 500: "Medium", 600: "SemiBold", 700: "Bold", 800: "ExtraBold", 900: "Black"}.get(w, str(w))


def ensure(name):
    """Path of a usable font file for name; downloads a Google font on first use."""
    cat = catalog()
    e = cat.get(name)
    if not e:
        return None
    if not e["installed"] and e.get("url"):
        os.makedirs(os.path.dirname(e["file"]), exist_ok=True)
        tmp = e["file"] + ".part"
        with urllib.request.urlopen(e["url"], timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if not _has_hebrew(tmp):
            os.remove(tmp)
            raise RuntimeError(f"הגופן {name} לא נטען כראוי")
        os.replace(tmp, e["file"])
        e["installed"] = True
    return e["file"] if os.path.exists(e["file"]) else None


def import_files(paths):
    """Copy TTF/OTF files into the user font folder. Returns (added names, rejected file names)."""
    d = os.path.join(user_dir(), "mine")
    os.makedirs(d, exist_ok=True)
    added, bad = [], []
    for p in paths:
        info = _has_hebrew(p)
        if not info:
            bad.append(os.path.basename(p)); continue
        dst = os.path.join(d, os.path.basename(p))
        shutil.copyfile(p, dst)
        added.append(info["name"])
    with _LOCK:
        _CACHE["cat"] = None
    return added, bad
