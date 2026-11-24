# Video/Audio QC Detection Rules

One entry per defect. Each gives: **what it is**, the **detector command**, the
**threshold meaning**, **how to read a hit**, and the **false-positive traps**
that make naive checks lie. Substitute `IN` for the input path.

Set once:
```bash
IN="/path/to/video.mp4"
```

## Table of contents
1. [Probe the file (always first)](#0-probe-the-file-always-first)
2. [Black frames / black segments](#1-black-frames--black-segments)
3. [Freeze / stuck frames](#2-freeze--stuck-frames)
4. [Dropped & duplicate frames](#3-dropped--duplicate-frames)
5. [Interlacing / combing / field order](#4-interlacing--combing--field-order)
6. [Illegal broadcast levels](#5-illegal-broadcast-levels)
7. [Letterbox / pillarbox / wrong aspect](#6-letterbox--pillarbox--wrong-aspect)
8. [Flash frames / flicker](#7-flash-frames--flicker)
9. [Blockiness / compression artifacts](#8-blockiness--compression-artifacts)
   · [Blur / out of focus](#8b-blur--out-of-focus)
10. [Audio silence / dropouts](#9-audio-silence--dropouts)
11. [Audio clipping](#10-audio-clipping)
12. [Loudness compliance (EBU R128)](#11-loudness-compliance-ebu-r128)
13. [Stereo phase problems](#12-stereo-phase-problems)
14. [A/V sync drift](#13-av-sync-drift)
15. [Container corruption / decode errors](#14-container-corruption--decode-errors)
16. [Mitigation (only when asked)](#mitigation-only-when-asked)

---

## 0. Probe the file (always first)

Everything downstream depends on knowing the declared framerate, duration, and
stream layout. A "wrong frame count" is only wrong relative to `fps × duration`.

```bash
ffprobe -v error -show_entries \
  format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,sample_rate,channels \
  -of default=noprint_wrappers=1 "$IN"
```

Record: duration, `r_frame_rate` (nominal) vs `avg_frame_rate` (actual — a gap
between them already hints at drops/VFR), resolution, and whether audio exists.

---

## 1. Black frames / black segments

**What:** Stretches of fully (or near) black picture — missing content, bad
splices, unintentional gaps.

```bash
ffmpeg -hide_banner -i "$IN" -vf "blackdetect=d=0.1:pic_th=0.98:pix_th=0.10" \
  -an -f null - 2>&1 | grep blackdetect
```
- `d` = minimum black duration in seconds to report (0.1 ≈ a few frames).
- `pic_th` = fraction of the picture that must be black (0.98 = 98%).
- `pix_th` = per-pixel luminance threshold below which a pixel counts as black.

**Reading a hit:** `black_start:… black_end:… black_duration:…` in seconds.
A single black frame at a hard cut is usually fine; multi-second black mid-shot
is a real defect.

**False-positive traps:** Fade-to-black transitions are legitimate — check
whether the black sits at a scene boundary. Very dark night footage can trip a
loose `pix_th`; lower `pix_th` (e.g. 0.05) or raise `pic_th` for dark sources.

---

## 2. Freeze / stuck frames

**What:** The same frame held over time — encoder stall, capture glitch, or a
source hold. Often what people loosely call a "dropped frame."

```bash
ffmpeg -hide_banner -i "$IN" -vf "freezedetect=n=0.003:d=0.5" \
  -map 0:v -f null - 2>&1 | grep -i lavfi.freezedetect
```
- `n` = noise tolerance (how different consecutive frames may be and still count
  as frozen). 0.003 is sensitive; raise it for noisy/grainy sources.
- `d` = minimum freeze duration in seconds.

**Reading a hit:** `freeze_start`, `freeze_duration`, `freeze_end` (seconds).

**False-positive traps:** Static graphics, slates, color bars, letterboxed
still title cards, and legitimate held shots read as freezes. Cross-check
against content. Grain-free CG animation with a static background can also trip
it — raise `n`.

---

## 3. Dropped & duplicate frames

**What:** Actual frames missing or repeated relative to the timeline. Two angles:

**(a) Count mismatch** — decoded frames vs. expected:
```bash
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames,r_frame_rate -of default=nw=1 "$IN"
# expected ≈ r_frame_rate × duration; compare to nb_read_frames
```

**(b) Duplicate detection** — how many frames ffmpeg considers droppable dupes:
```bash
ffmpeg -hide_banner -i "$IN" -vf mpdecimate -loglevel debug -f null - 2>&1 \
  | grep -c 'drop_count'
```

**Reading a hit:** A large gap between expected and actual count, or a high dupe
count, indicates real timeline problems.

**False-positive traps:** **Telecine / pulldown** (24p film in a 30i/60i
container) *legitimately* repeats fields/frames — dupes there are expected, not
a defect. Variable-frame-rate (VFR) source makes the `fps × duration` estimate
unreliable; note VFR when `avg_frame_rate ≠ r_frame_rate`. Animation on "twos/
threes" holds frames by design.

---

## 4. Interlacing / combing / field order

**What:** Interlaced fields showing as horizontal comb teeth on motion, or a
declared field order that doesn't match the actual one.

```bash
ffmpeg -hide_banner -i "$IN" -vf idet -frames:v 2000 -an -f null - 2>&1 \
  | grep -Ei 'idet|Single|Multi|Repeated'
```

**Reading a hit:** `idet` reports counts for TFF (top field first), BFF (bottom
field first), Progressive, and Undetermined, across Single-frame and Multi-frame
detection. Mostly-TFF or mostly-BFF ⇒ interlaced content; mostly-Progressive ⇒
progressive. A mix of TFF and BFF suggests inconsistent/broken field order.

**False-positive traps:** A short sample can misclassify — scan a few thousand
frames, not a handful. Progressive content carried in an interlaced-flagged
container is common and not itself a defect; report the mismatch, don't assume
combing.

---

## 5. Illegal broadcast levels

**What:** Luma/chroma outside the broadcast-safe range — super-blacks below 16,
whites above 235 (8-bit), or out-of-gamut chroma. Fails broadcast QC.

```bash
ffmpeg -hide_banner -i "$IN" -vf "signalstats,metadata=print:key=lavfi.signalstats.YMIN" \
  -frames:v 300 -an -f null - 2>&1 | grep YMIN | head
```
For a fuller read, use `signalstats` YMIN/YMAX/YAVG/SATMAX. YMIN well below 16
or YMAX well above 235 (on a 0–255 scale) signals illegal levels.

**Reading a hit:** Persistent YMIN < 16 or YMAX > 235 across many frames ⇒
levels problem. Occasional single-frame spikes are usually noise.

**False-positive traps:** Full-range (JPEG-range, 0–255) content is legal by its
own standard — confirm the intended range before flagging. Specular highlights
briefly clipping is often acceptable creative intent.

---

## 6. Letterbox / pillarbox / wrong aspect

**What:** Unexpected black bars (letterbox = top/bottom, pillarbox = sides), or
picture cropped to the wrong aspect ratio.

```bash
ffmpeg -hide_banner -i "$IN" -vf cropdetect=round=2 -frames:v 500 -an -f null - 2>&1 \
  | grep -o 'crop=[0-9:]*' | sort | uniq -c | sort -rn | head
```

**Reading a hit:** If the dominant `crop=` value is smaller than the full frame,
there are bars. Compare the implied aspect (e.g. detected 1920×816 inside a
1920×1080 frame ⇒ ~2.35:1 letterbox).

**False-positive traps:** Intentional cinematic letterboxing is a creative
choice, not a defect — report it, don't condemn it. Dark scenes can fool
`cropdetect` into seeing bars where there are none; sample across the whole file,
not just one dark stretch.

---

## 7. Flash frames / flicker

**What:** Abrupt one-frame brightness spikes (editing errors, bad splices) or
rapid luminance flicker (which can also be an accessibility/photosensitivity
concern).

```bash
ffmpeg -hide_banner -i "$IN" \
  -vf "signalstats,metadata=print:key=lavfi.signalstats.YAVG" \
  -frames:v 1000 -an -f null - 2>&1 | grep YAVG
```
Post-process the YAVG series: a single frame whose average luma jumps far from
both neighbors is a flash frame; oscillation across many frames is flicker.

**False-positive traps:** Legitimate flashes (lightning, camera strobes,
explosions) look identical to error flashes — flag for human review rather than
asserting a defect.

> **In the scanner:** `detect_flash()` implements this with numpy — it pulls the
> YAVG series from ffmpeg, then flags frames whose luma jumps far above/below
> **both** neighbours (adaptive threshold from the median frame-to-frame delta).
> Requires the venv (numpy); skipped otherwise. Runs on ffmpeg-extracted stats,
> so it adds no extra decode cost.

---

## 8. Blockiness / compression artifacts

**What:** Visible macroblock structure from heavy compression or a corrupted
stream.

No single clean ffmpeg detector; use these signals:
```bash
# entropy/uniformity via signalstats over time, and visual spot-checks:
ffmpeg -hide_banner -i "$IN" -vf "signalstats,metadata=print:key=lavfi.signalstats.YDIF" \
  -frames:v 500 -an -f null - 2>&1 | grep YDIF | head
```
Or export frames at suspected timestamps and inspect visually. `qcli` with the
QCTools GUI is genuinely better here — its graphs make macroblocking obvious.

**Reading a hit:** Sustained low detail with sharp block edges. This is often a
judgment call — recommend visual confirmation.

**False-positive traps:** Low-detail scenes (fog, sky, blur) naturally have low
variance and aren't artifacts. This detector is the least deterministic; be
honest about that in the report.

---

## 8b. Blur / out of focus

**What:** Soft or out-of-focus frames — bad focus pulls, motion smear, or a
defocused source.

There's no ffmpeg filter for this; the scanner uses OpenCV's **variance of the
Laplacian** (a standard sharpness metric — lower variance = softer image),
sampling ~1 frame/second so a batch stays fast.

> **In the scanner:** `detect_blur()` requires the venv (opencv-python + numpy);
> skipped otherwise. It flags frames far softer than the file's *own* median
> sharpness (and below an absolute floor), which adapts to each source instead
> of assuming a universal "sharp."

**Reading a hit:** `sharpness N vs median M` — a run of low-sharpness frames is a
real soft passage; isolated ones are often motion blur.

**False-positive traps:** Intentional shallow depth-of-field, defocus
transitions, fog, and genuinely dark frames (a black frame scores ~0) all read
as "soft." Treat as a pointer for visual review, not a hard fail. Since it's
sampled, it localizes *roughly* — confirm the exact frame visually if it matters.

---

## 9. Audio silence / dropouts

**What:** Unexpected dead air — a drop where audio should be present.

```bash
ffmpeg -hide_banner -i "$IN" -af "silencedetect=n=-50dB:d=0.5" \
  -f null - 2>&1 | grep silence
```
- `n` = level below which it counts as silence (-50 dB is a reasonable floor).
- `d` = minimum silent duration in seconds.

**Reading a hit:** `silence_start`, `silence_end`, `silence_duration`. Silence
over program content is a defect; silence in intended pauses is not.

**False-positive traps:** Intentional silence (before music starts, dramatic
pauses, room tone near the floor) trips it. Multi-track/multi-language files may
have a legitimately empty channel — check per stream.

---

## 10. Audio clipping

**What:** Samples pinned at or near full scale (0 dBFS) ⇒ distortion.

```bash
ffmpeg -hide_banner -i "$IN" -af astats \
  -f null - 2>&1 | grep -Ei 'Peak level|Flat factor|Peak count'
```

**Reading a hit:** `Peak level` at/above ~-0.1 dBFS combined with a high
`Flat factor` / large `Peak count` indicates clipping (flat-topped waveform).

**False-positive traps:** A single sample at peak isn't audible clipping — it's
the *sustained flat tops* (high flat factor) that matter. Heavily limited/
mastered music legitimately rides near 0 dBFS.

---

## 11. Loudness compliance (EBU R128)

**What:** Integrated loudness outside the delivery target (broadcast commonly
-23 LUFS ±1 for EBU R128; -24 LKFS for US ATSC A/85; streaming platforms vary,
often -14 LUFS).

```bash
ffmpeg -hide_banner -i "$IN" -af ebur128=peak=true -f null - 2>&1 | tail -20
```

**Reading a hit:** Read the `Integrated loudness`, `Loudness range`, and
`True peak` summary at the end. Compare integrated LUFS to the delivery spec the
user is targeting; flag true-peak above the spec ceiling (e.g. -1 dBTP).

**False-positive traps:** There is no universal target — always ask/confirm the
delivery spec before calling it a fail. Short clips give unstable integrated
readings.

---

## 12. Stereo phase problems

**What:** Out-of-phase stereo that partially cancels when summed to mono.

```bash
ffmpeg -hide_banner -i "$IN" -af aphasemeter=video=0 -f null - 2>&1 \
  | grep -i phase | head
```

**Reading a hit:** Phase values persistently near -1 indicate anti-phase
content that will collapse in mono.

**False-positive traps:** Wide stereo/ambience legitimately dips negative
briefly. Mono files carried as dual-mono read as perfectly correlated (+1) — not
a defect. Only sustained strong negative correlation is a real problem.

---

## 13. A/V sync drift

**What:** Audio and video progressively (or constantly) misaligned.

ffmpeg can't "hear" lip-sync, but it can surface structural causes: mismatched
stream start times and durations.
```bash
ffprobe -v error -show_entries stream=codec_type,start_time,duration \
  -of default=nw=1 "$IN"
```

**Reading a hit:** A meaningful difference in `start_time` between the audio and
video streams offsets sync from frame one. Different `duration` values hint at
drift accumulating over the file.

**False-positive traps:** Small start-time offsets (a few ms) are normal.
Perceptual lip-sync requires human/visual verification — report structural
evidence and recommend a spot-check, don't assert lip-sync is broken from
metadata alone.

---

## 14. Container corruption / decode errors

**What:** Broken packets, truncated files, or a stream that won't fully decode.

```bash
ffmpeg -hide_banner -v error -i "$IN" -f null - 2>&1 | head -40
```
A clean decode prints nothing. Any lines here (e.g. `error while decoding`,
`Invalid data`, `moov atom not found`) are real problems.

**Reading a hit:** Note the timestamps in the error lines — they localize the
corruption. Many errors clustered at the end often mean a truncated/incomplete
file.

**False-positive traps:** A handful of non-fatal warnings on an otherwise
playable file may be benign; distinguish `error`-level from `warning`-level.

---

## Mitigation (only when asked)

The user must explicitly request a fix. **Always write to a new file. Never
overwrite the source.** Be explicit about what each operation restores versus
what is permanently gone.

| Defect | Recipe | What it does / doesn't do |
|--------|--------|---------------------------|
| Black/freeze/silence segment | trim with `-ss`/`-to`, or `select`/`aselect` filters | Removes the bad span. Does not recover missing content. |
| Duplicate frames | `-vf mpdecimate,setpts=N/FRAME_RATE/TB` | Drops repeated frames. Wrong on true telecine — deinterlace/IVTC instead. |
| Interlacing | `-vf yadif` (or `bwdif`) | Deinterlaces. Softens the image slightly; can't undo field loss. |
| Letterbox bars | `-vf cropdetect` → `-vf crop=W:H:X:Y` | Crops bars. Verify the detected values before committing. |
| Loudness off-target | `-af loudnorm=I=-23:TP=-1:LRA=11` | Normalizes to target. Two-pass loudnorm is more accurate than one-pass. |
| Illegal levels | `-vf scale=in_range=full:out_range=tv` (or `colorlevels`) | Remaps range. Confirm the source range first or you'll crush legal footage. |

Compression artifacts, real dropped content, focus loss, and detail lost to
prior encoding are **not recoverable** — say so plainly rather than implying a
filter will restore them.
