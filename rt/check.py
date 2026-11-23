"""Verification: for every card, frames before-in / in / mid / out / after-out, source vs result,
read back from the encoded output (what the viewer gets)."""
import os, html
import cv2, numpy as np
from . import vio
from .extract import groups


def _grab(path, frames, W, H):
    want = set(frames); got = {}
    for i, f in enumerate(vio.frames(path, W, H)):
        if i in want:
            got[i] = f.copy()
        if i >= max(want):
            break
    return got


def run(proj, video_out=None):
    P = vio.load_json(os.path.join(proj, "project.json"))
    T = vio.load_json(os.path.join(proj, "titles.json"))
    stats = vio.load_json(os.path.join(proj, "clean_stats.json"), {})
    W, H, n, fps = P["info"]["width"], P["info"]["height"], P["n_frames"], P["info"]["fps"]
    base = os.path.splitext(os.path.basename(P["video"]))[0]
    video_out = video_out or os.path.join(proj, "out", f"{base}_HEB.mp4")
    plan = {}
    for gid, lines in groups(T).items():
        act = [t for t in lines if t.get("translate", True) and "region" in t]
        if not act:
            continue
        a = min(t["in"] for t in act); b = max(t["out"] for t in act)
        fs = sorted({max(0, a - 1), a, min(a + 2, b), (a + b) // 2, max(a, b - 2), b, min(n - 1, b + 1)})
        plan[gid] = (fs, act)
    allf = sorted({f for fs, _ in plan.values() for f in fs})
    if not allf:
        return []
    src = _grab(P["video"], allf, W, H)
    dst = _grab(video_out, allf, W, H)
    d = os.path.join(proj, "check"); os.makedirs(d, exist_ok=True)
    sheets = []
    for gid, (fs, act) in plan.items():
        x0, y0, x1, y1 = act[0]["region"]
        rows = []
        for f in fs:
            if f not in src or f not in dst:
                continue
            a = src[f][y0:y1, x0:x1]; b = dst[f][y0:y1, x0:x1]
            sc = min(1.0, 900 / a.shape[1])
            a = cv2.resize(a, None, fx=sc, fy=sc); b = cv2.resize(b, None, fx=sc, fy=sc)
            bar = np.full((a.shape[0], 8, 3), 40, np.uint8)
            row = np.hstack([a, bar, b])
            lab = f"f{f} {vio.tc(f, fps)}" + ("  IN" if f == min(t['in'] for t in act) else "") + \
                  ("  OUT" if f == max(t['out'] for t in act) else "") + \
                  ("  (before)" if f < min(t['in'] for t in act) else "") + \
                  ("  (after)" if f > max(t['out'] for t in act) else "")
            cv2.putText(row, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
            cv2.putText(row, lab, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            rows.append(row)
            rows.append(np.full((6, row.shape[1], 3), 40, np.uint8))
        if rows:
            p = os.path.join(d, f"{gid}.jpg")
            cv2.imwrite(p, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
            sheets.append(p)
    # accessible Hebrew report
    rows = []
    for gid, (fs, act) in plan.items():
        for t in act:
            st = stats.get(gid, {})
            rows.append(f"<tr><td>{t['id']}</td><td dir='ltr'>{vio.tc(t['in'], fps)} - {vio.tc(t['out'], fps)}</td>"
                        f"<td dir='ltr' lang='und'>{html.escape(t['src'])}</td><td>{html.escape(t.get('he', ''))}</td>"
                        f"<td>{t.get('anim', {}).get('mode', '')}</td><td>{int(100 * st.get('synth_max', 0))}%</td></tr>")
        rows.append(f"<tr><td colspan='6'><img src='{gid}.jpg' alt='כרטיס {gid}: מקור משמאל, עברית מימין, פריים לפני הכניסה עד אחרי היציאה'></td></tr>")
    page = f"""<!DOCTYPE html><html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>בדיקת כותרות</title>
<style>body{{font-family:Arial,sans-serif;font-size:1.15rem;background:#111;color:#f2f2f2;padding:1rem}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #666;padding:.5rem}}img{{max-width:100%}}</style>
</head><body><main><h1>בדיקת כותרות</h1><p>בכל תמונה: משמאל המקור, מימין התוצאה. העמודה "סינתזה" היא החלק של הרקע שלא נמצא באף פריים שכן והושלם.</p>
<table><thead><tr><th scope="col">מזהה</th><th scope="col">זמן</th><th scope="col">מקור</th><th scope="col">עברית</th><th scope="col">אנימציה</th><th scope="col">סינתזה</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></main></body></html>"""
    with open(os.path.join(d, "report.html"), "w", encoding="utf-8") as f:
        f.write(page)
    return sheets
