"""Write manifest.json into the built app folder: every file with its size and SHA-256.
The updater compares it with the new release's manifest and fetches only the files that changed."""
import os, sys, json, hashlib

root, build = sys.argv[1], int(sys.argv[2])
files = {}
for d, _, fs in os.walk(root):
    for f in fs:
        p = os.path.join(d, f)
        rel = os.path.relpath(p, root).replace("\\", "/")
        if rel == "manifest.json":
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        files[rel] = [os.path.getsize(p), h.hexdigest()]
with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump({"build": build, "files": files}, f, indent=0, sort_keys=True)
print(len(files), "files,", round(sum(v[0] for v in files.values()) / 1e6), "MB")
