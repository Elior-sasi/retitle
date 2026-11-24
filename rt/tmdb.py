"""Official Hebrew names from TMDB (themoviedb.org): the film or series of the project, by the video's file name
and the logo text, in he-IL - so the translation uses the title the Israeli distributor gave it. Needs the
user's free TMDB key (API key or read access token). TMDB is a community database: the hint is passed to the
translator as such, not as certain truth.

Without a key of its own, Retitle asks the user's Unwatch server (unwatch-one.vercel.app/api/search), which holds
his TMDB key and answers in Hebrew - the key itself never leaves Vercel."""
import os, re, json, urllib.request, urllib.parse
from . import vio

JUNK = r"\b(official|trailer|teaser|clip|promo|tv spot|spot|hd|4k|uhd|1080p|720p|full episode|episode|ep\d*|s\d+e\d+|season \d+|final|new|\d{4})\b"


def _get(path, key, **params):
    params = {k: v for k, v in params.items() if v is not None}
    headers = {"accept": "application/json", "User-Agent": "Retitle"}
    if len(key) > 40:                          # v4 read access token
        headers["Authorization"] = f"Bearer {key}"
    else:
        params["api_key"] = key
    url = f"https://api.themoviedb.org/3{path}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
        return json.load(r)


def guess_query(proj):
    P = vio.load_json(os.path.join(proj, "project.json"))
    name = os.path.splitext(os.path.basename(P["video"]))[0]
    name = re.sub(r"[_\.\[\]\(\)|@#]+", " ", name)
    name = re.split(r"\s[-–—:]\s", name)[0]
    name = re.sub(JUNK, " ", name, flags=re.I)
    return re.sub(r"\s+", " ", name).strip()


def lookup(proj, key, log=print):
    """{"he": ..., "original": ..., "kind": "movie"|"tv", "year": ...} or {}. Cached in project.json."""
    P = vio.load_json(os.path.join(proj, "project.json"))
    q = (P.get("tmdb_query") or guess_query(proj)).strip()
    if P.get("tmdb") and P["tmdb"].get("query") == q:
        return P["tmdb"]
    if not q:
        return {}
    if not key:
        return _via_unwatch(proj, P, q, log)
    try:
        res = _get("/search/multi", key, query=q, language="he-IL", include_adult="false").get("results", [])
    except Exception as e:
        log(f"TMDB לא זמין ({str(e)[:80]})")
        return {}
    res = [r for r in res if r.get("media_type") in ("movie", "tv")]
    if not res:
        out = {"query": q}
    else:
        r = res[0]
        kind = r["media_type"]
        out = {"query": q, "kind": kind, "id": r["id"],
               "he": r.get("title") or r.get("name") or "",
               "original": r.get("original_title") or r.get("original_name") or "",
               "year": (r.get("release_date") or r.get("first_air_date") or "")[:4]}
        # TMDB falls back to the original name when no Hebrew one was entered: then there is no Hebrew title
        if out["he"] == out["original"] or not re.search(r"[֐-׿]", out["he"]):
            out["he"] = ""
    P["tmdb"] = out
    vio.save_json(os.path.join(proj, "project.json"), P)
    return out


UNWATCH = "https://unwatch-one.vercel.app/api/search?q="


def _via_unwatch(proj, P, q, log):
    try:
        req = urllib.request.Request(UNWATCH + urllib.parse.quote(q), headers={"User-Agent": "Retitle", "accept": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            res = json.load(r).get("results", [])
    except Exception as e:
        log(f"TMDB דרך Unwatch לא זמין ({str(e)[:80]})")
        return {}
    out = {"query": q}
    if res:
        r = res[0]
        he = r.get("title") or ""
        out.update(kind=r.get("media"), id=r.get("tmdb"), original=r.get("orig") or "", year=r.get("year") or "",
                   he=he if re.search(r"[\u0590-\u05FF]", he) else "")
    P["tmdb"] = out
    vio.save_json(os.path.join(proj, "project.json"), P)
    return out


def hint(proj, key, log=print):
    t = lookup(proj, key, log)
    if not t.get("he"):
        return ""
    kind = "film" if t.get("kind") == "movie" else "series"
    log(f"TMDB: השם העברי של ה{'סרט' if kind == 'film' else 'סדרה'} - {t['he']}")
    return (f"TMDB lists the Hebrew title of this {kind} ({t['original']}, {t.get('year', '')}) as \"{t['he']}\" "
            f"(community data - use it for the title unless the image clearly shows otherwise).")
