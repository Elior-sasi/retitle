# Retitle - notes for Claude

Retitle localizes any video to Hebrew (trailers, episodes, documentaries, reality, anime), not only trailers.
Engine: `rt/` (CLI `python -m rt --help`). App: `rt/app.py` + `rt/ui/index.html`, packaged by `.github/workflows/build.yml` (PyInstaller, Windows, release per push to main). `tests/selftest.py` must pass.

When localizing a video in chat with this repo: init -> shots -> detect (look at detect_sheet.jpg, fix/merge titles in titles.json, set translate=false for credits/logos/English names, write `he`) -> extract -> masks (look) -> clean -> track -> layers -> preview (look) -> compose -> check (look at every check/*.jpg). Rules: no niqqud, plain " quotes, hyphen not em dash, user's name glossary.
User-facing text in Hebrew, accessible UI (large text, labels, RTL, contrast, reduced motion).

Releases: CI publishes to this private repo and, with the repo secret RELEASES_TOKEN, also to the public Elior-sasi/retitle-releases, which installs and self-updates read (no token on the user's side).

Google backup (rt/gdrive.py): CI writes rt/_google.py from the repo secrets GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET (Desktop OAuth client, scope drive.appdata). Without them the backup shows "not available".

Full mode (rt/agent.py, job "agent"): one Claude Code session (user's subscription) reviews detect_sheet and every check sheet round after round, edits titles.json, runs the engine itself through `<proj>/agent/rt.cmd` (calls /api/agent_run in the running app: `python -m rt` commands, or `--py script.py` for code it writes) and asks the app for fixes, until done.

SAM 2 + Lens (rt/aimodels.py, in ProPainter's private Python, installed with it): SAM 2 makes exact per-frame masks (sam/<gid>/) from the detector's box - auto for logos/styled/strong cards and cards the erase self-check found leftovers in; clean.clean_frame unions them into M0. Lens names the source's Latin font and embeds lettering; fontmatch.match compares the source with every Latin font on the PC and every Hebrew candidate (PC, Adobe, imported, all Google Hebrew fonts - downloaded once in _warm), puts the Hebrew version of the source family first, and passes it as a hint to Claude.

rea-video itself (rt/reavideo.py, github.com/JuneYaooo/rea-video pinned, Apache 2.0): installed with ProPainter into the private Python, with its Claude Code skill. Full mode runs `rea-video decompose` into <proj>/rea first, copies the skill into <proj>/.claude/skills, and Claude Code gets the `rea-video` command on PATH.

DictaLM 3.0 (rt/dicta.py): optional local translation engine (settings translate_engine=dicta). 24B Thinking GGUF from unsloth (dicta_quant Q4_K_M/Q6_K/Q8_0) in llama.cpp's llama-server (latest Windows CUDA build from GitHub), layers split GPU/RAM, thinking off. Job translate_compare writes compare.json (Claude vs DictaLM per title); the UI dialog picks per line (/api/compare/pick).

QC and guards: CueLint (MIT, rt/ui/cuelint.web.js, runs in the page) checks the Hebrew SRT against Netflix/BBC/base or the user's client profiles (settings qc_clients), Hebrew messages, never edits text. check.guard() compares source vs result outside every title region (~1 fps) -> check/guard.json + guard_<frame>.jpg. qc-video skill (MIT, rt/skills) is copied into full-mode projects; run via RT --py. TMDB (rt/tmdb.py): Hebrew official title as a translation hint, own key (tmdb_key) or via the user's Unwatch server /api/search. No commercial-mode switch: the user wants ProPainter and Lens used.

Notifications (rt/notify.py): when a job of 20s+ ends or fails while the window is not in front (the page reports focus with each /api/status poll), a Windows toast through PowerShell's WinRT bridge (AUMID Elior.Retitle, set on the installer's shortcuts) plus a flashing taskbar button. Setting "notify", test button in preferences. Full mode streams Claude Code (stream-json): the card shows each tool call ("Claude: מסתכל על ..."), the live image is the picture Claude reads, and no ETA is shown during a Claude turn.

Translation approval (option "review" in the one-click dialog, both modes): wait_review() pauses the job before design/export (full mode: after step 1) with status waiting="tr_review"; the UI opens d-trv, /api/tr_review saves the edits and marks lines he_locked; agent.settle restores he/translate of locked lines if Claude changes them. Log dialog has a copy button.
