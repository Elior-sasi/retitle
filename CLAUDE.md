# Retitle - notes for Claude

Retitle localizes any video to Hebrew (trailers, episodes, documentaries, reality, anime), not only trailers.
Engine: `rt/` (CLI `python -m rt --help`). App: `rt/app.py` + `rt/ui/index.html`, packaged by `.github/workflows/build.yml` (PyInstaller, Windows, release per push to main). `tests/selftest.py` must pass.

When localizing a video in chat with this repo: init -> shots -> detect (look at detect_sheet.jpg, fix/merge titles in titles.json, set translate=false for credits/logos/English names, write `he`) -> extract -> masks (look) -> clean -> track -> layers -> preview (look) -> compose -> check (look at every check/*.jpg). Rules: no niqqud, plain " quotes, hyphen not em dash, user's name glossary.
User-facing text in Hebrew, accessible UI (large text, labels, RTL, contrast, reduced motion).

Releases: CI publishes to this private repo and, with the repo secret RELEASES_TOKEN, also to the public Elior-sasi/retitle-releases, which installs and self-updates read (no token on the user's side).

Google backup (rt/gdrive.py): CI writes rt/_google.py from the repo secrets GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET (Desktop OAuth client, scope drive.appdata). Without them the backup shows "not available".

Full mode (rt/agent.py, job "agent"): one Claude Code session (user's subscription) reviews detect_sheet and every check sheet round after round, edits titles.json, runs the engine itself through `<proj>/agent/rt.cmd` (calls /api/agent_run in the running app: `python -m rt` commands, or `--py script.py` for code it writes) and asks the app for fixes, until done.

SAM 2 + Lens (rt/aimodels.py, in ProPainter's private Python, installed with it): SAM 2 makes exact per-frame masks (sam/<gid>/) from the detector's box - auto for logos/styled/strong cards and cards the erase self-check found leftovers in; clean.clean_frame unions them into M0. Lens names the source's Latin font and embeds lettering; fontmatch.match compares the source with every Latin font on the PC and every Hebrew candidate (PC, Adobe, imported, all Google Hebrew fonts - downloaded once in _warm), puts the Hebrew version of the source family first, and passes it as a hint to Claude.
