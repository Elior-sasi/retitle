"""Retitle project files (.retitle), like a Premiere .prproj.

The .retitle file is small JSON next to the video: it points at the video and at the data folder
(<name>_retitle) that holds titles, cues, frame caches and rebuilt backgrounds. Edits are saved
continuously into the data folder; the .retitle file records the project and when it was saved.
"""
import os, shutil, time
from . import vio

EXT = ".retitle"
FORMAT = "retitle-project"


def data_dir_for(rfile):
    return os.path.splitext(rfile)[0] + "_retitle"


def for_video(video):
    """The project of a video: <stem>.retitle next to it (created if missing; an older data folder is kept)."""
    video = os.path.abspath(video)
    rfile = os.path.splitext(video)[0] + EXT
    if os.path.exists(rfile):
        return rfile
    data = data_dir_for(rfile)
    os.makedirs(data, exist_ok=True)
    if not os.path.exists(os.path.join(data, "project.json")):
        vio.save_json(os.path.join(data, "project.json"), {"video": video, "info": vio.probe(video)})
    write(rfile, data, video)
    return rfile


def write(rfile, data, video, created=None):
    base = os.path.dirname(os.path.abspath(rfile))
    old = vio.load_json(rfile, {}) if os.path.exists(rfile) else {}
    try:
        from ._version import BUILD
    except Exception:
        BUILD = 0
    doc = {"format": FORMAT, "version": 1,
           "name": os.path.splitext(os.path.basename(rfile))[0],
           "video": os.path.abspath(video),
           "video_rel": os.path.relpath(os.path.abspath(video), base) if _same_drive(video, base) else "",
           "data": os.path.relpath(os.path.abspath(data), base) if _same_drive(data, base) else os.path.abspath(data),
           "created": created or old.get("created") or time.strftime("%Y-%m-%d %H:%M:%S"),
           "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
           "app_build": BUILD}
    tmp = rfile + ".tmp"
    vio.save_json(tmp, doc)
    os.replace(tmp, rfile)
    return doc


def _same_drive(a, b):
    return os.path.splitdrive(os.path.abspath(a))[0].lower() == os.path.splitdrive(os.path.abspath(b))[0].lower()


def read(rfile):
    """Returns (data_dir, video_path or None if the video can't be found)."""
    doc = vio.load_json(rfile)
    if doc.get("format") != FORMAT:
        raise ValueError("זה לא קובץ פרויקט של Retitle")
    base = os.path.dirname(os.path.abspath(rfile))
    data = doc["data"] if os.path.isabs(doc["data"]) else os.path.normpath(os.path.join(base, doc["data"]))
    if not os.path.isdir(data):
        raise FileNotFoundError(f"תיקיית הנתונים של הפרויקט חסרה: {data}")
    pj = vio.load_json(os.path.join(data, "project.json"), {})
    video = None
    for cand in [doc.get("video"), os.path.join(base, doc["video_rel"]) if doc.get("video_rel") else None, pj.get("video")]:
        if cand and os.path.isfile(cand):
            video = os.path.abspath(cand)
            break
    if video and pj.get("video") != video:          # moved together with the project: follow it
        pj["video"] = video
        vio.save_json(os.path.join(data, "project.json"), pj)
    return data, video


def relink(rfile, data, video):
    """Point the project at a moved/renamed video. Refuses a different video (other length or size)."""
    pj = vio.load_json(os.path.join(data, "project.json"))
    info = vio.probe(video)
    old = pj.get("info", {})
    if old and (info["width"], info["height"]) != (old.get("width"), old.get("height")) \
            or old and abs(float(info["duration"]) - float(old.get("duration", info["duration"]))) > 0.5:
        raise ValueError("זה לא אותו סרטון (אורך או גודל שונים)")
    pj["video"] = os.path.abspath(video)
    vio.save_json(os.path.join(data, "project.json"), pj)
    write(rfile, data, video)


SKIP = {"out", "check", "proxy.mp4"}


def save_as(rfile, data, new_rfile):
    """Copy the project to a new .retitle (its own data folder), so versions can diverge."""
    if not new_rfile.lower().endswith(EXT):
        new_rfile += EXT
    new_data = data_dir_for(new_rfile)
    if os.path.abspath(new_data) != os.path.abspath(data):
        if os.path.exists(new_data):
            shutil.rmtree(new_data)
        shutil.copytree(data, new_data, ignore=lambda d, names: [n for n in names if d == data and n in SKIP])
    pj = vio.load_json(os.path.join(new_data, "project.json"))
    write(new_rfile, new_data, pj["video"], created=time.strftime("%Y-%m-%d %H:%M:%S"))
    return new_rfile, new_data
