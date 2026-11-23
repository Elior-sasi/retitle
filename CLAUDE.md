# Retitle - notes for Claude

Retitle localizes any video to Hebrew (trailers, episodes, documentaries, reality, anime), not only trailers.
Engine: `rt/` (CLI `python -m rt --help`). App: `rt/app.py` + `rt/ui/index.html`, packaged by `.github/workflows/build.yml` (PyInstaller, Windows, release per push to main). `tests/selftest.py` must pass.

When localizing a video in chat with this repo: init -> shots -> detect (look at detect_sheet.jpg, fix/merge titles in titles.json, set translate=false for credits/logos/English names, write `he`) -> extract -> masks (look) -> clean -> track -> layers -> preview (look) -> compose -> check (look at every check/*.jpg). Rules: no niqqud, plain " quotes, hyphen not em dash, user's name glossary.
User-facing text in Hebrew, accessible UI (large text, labels, RTL, contrast, reduced motion).
