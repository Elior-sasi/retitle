"""Series memory: graphics continuity between episodes.

A series remembers, per source text, what you did with a title (Hebrew, keep/remove, chosen font/size/colour,
PNG logo). A new episode in the same series gets those automatically; such titles are marked from_series.
Changing one in a later episode stays local (series_override) until you push it to the whole series.
Stored in %APPDATA%/Retitle/series/<name>.json (+ logos/), so every project on this computer can use it.
"""
import os, re, shutil, time, hashlib
from . import vio
from .detect import norm

STYLE_KEYS = ("font", "weight", "size_ratio", "fill", "align", "nudge_x", "nudge_y")


def root():
    base = os.environ.get("APPDATA") or os.path.expanduser("~/.config")
    return os.path.join(base, "Retitle", "series")


def _file(name):
    slug = re.sub(r'[\\/:*?"<>|]+', "_", name.strip())[:80] or "series"
    return os.path.join(root(), slug + ".json")


def list_all():
    out = []
    if os.path.isdir(root()):
        for f in sorted(os.listdir(root())):
            if f.endswith(".json"):
                d = vio.load_json(os.path.join(root(), f), {})
                out.append({"name": d.get("name", f[:-5]), "count": len(d.get("titles", {})), "updated": d.get("updated", "")})
    return out


def load(name):
    return vio.load_json(_file(name), {"name": name, "titles": {}})


def _save(doc):
    os.makedirs(root(), exist_ok=True)
    doc["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = _file(doc["name"]) + ".tmp"
    vio.save_json(tmp, doc)
    os.replace(tmp, _file(doc["name"]))


def key(t):
    if t.get("kind") == "sub":            # burned-in dialogue subtitles are not graphics
        return ""
    k = norm(t.get("src", ""))
    return k if len(k) >= 2 else ""


def _meaningful(t):
    lay = t.get("layer") or {}
    return (not t.get("translate", True)) or t.get("mode") == "remove" or bool(t.get("he")) or lay.get("type") == "image"


def _entry(t, proj, name):
    e = {"src": t.get("src", ""), "he": t.get("he", ""), "translate": t.get("translate", True),
         "mode": t.get("mode", "replace"), "origin": os.path.normcase(os.path.abspath(proj)),
         "style": {k: (t.get("style") or {})[k] for k in t.get("style_locked", []) if k in STYLE_KEYS and k in (t.get("style") or {})}}
    lay = t.get("layer") or {}
    if lay.get("type") == "image":
        src = os.path.join(proj, lay["path"])
        if os.path.isfile(src):
            with open(src, "rb") as f:
                h = hashlib.sha1(f.read()).hexdigest()[:16]
            d = os.path.join(root(), os.path.splitext(os.path.basename(_file(name)))[0] + "_logos")
            os.makedirs(d, exist_ok=True)
            dst = os.path.join(d, h + ".png")
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
            e["logo"] = dst
    return e


def learn(name, proj, titles, force_ids=()):
    """Remember titles of this episode. New source texts are added; an existing entry is updated only from the
    episode that created it, or when forced ("update the whole series")."""
    if not name:
        return 0
    doc = load(name)
    me = os.path.normcase(os.path.abspath(proj))
    n = 0
    for t in titles:
        k = key(t)
        if not k or not _meaningful(t):
            continue
        cur = doc["titles"].get(k)
        if t["id"] in force_ids or cur is None or (cur.get("origin") == me and not t.get("from_series")):
            e = _entry(t, proj, name)
            if cur and t["id"] in force_ids:
                # pushed to the whole series: the episode that first had it now follows the series too
                e["adopt"] = sorted((set(cur.get("adopt", [])) | {cur.get("origin", "")}) - {me, ""})
            if cur != e:
                doc["titles"][k] = e; n += 1
    if n:
        _save(doc)
    return n


def apply(name, proj, T, only=None):
    """Fill this episode's titles from the series. Returns ids whose Hebrew layer must be re-rendered."""
    if not name:
        return set()
    doc = load(name)
    me = os.path.normcase(os.path.abspath(proj))
    changed = set()
    for t in T["titles"]:
        if only is not None and t["id"] not in only:
            continue
        k = key(t)
        e = doc["titles"].get(k) if k else None
        if not e or e.get("origin") == me or t.get("series_override"):
            continue
        if t.get("he") and not t.get("from_series") and me not in e.get("adopt", []):
            continue                          # typed here before joining the series: keep it
        before = (t.get("he"), t.get("translate", True), t.get("mode", "replace"), t.get("layer"), dict(t.get("style") or {}))
        t["he"] = e.get("he", "")
        t["translate"] = e.get("translate", True)
        t["mode"] = e.get("mode", "replace")
        st = t.setdefault("style", {})
        for sk, v in (e.get("style") or {}).items():
            st[sk] = v
        if e.get("style"):
            t["style_locked"] = sorted(set(t.get("style_locked", [])) | set(e["style"]))
        if e.get("logo") and os.path.isfile(e["logo"]):
            os.makedirs(os.path.join(proj, "logos"), exist_ok=True)
            shutil.copyfile(e["logo"], os.path.join(proj, "logos", f"{t['id']}.png"))
            t["layer"] = {"type": "image", "path": f"logos/{t['id']}.png", "t": os.path.getmtime(e["logo"])}
        elif (t.get("layer") or {}).get("type") == "image" and t.get("from_series"):
            t.pop("layer", None)
        t["from_series"] = True
        if before != (t.get("he"), t.get("translate", True), t.get("mode", "replace"), t.get("layer"), dict(t.get("style") or {})):
            changed.add(t["id"])
    return changed
