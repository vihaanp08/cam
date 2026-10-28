"""Score an end-to-end simulator run (Documents/e2e_<name>/log.csv written by the app's test mode).
Usage: python e2e_eval.py LOG.csv OUT.json [GT.json|-] [press,press,...] [harness_metrics.json]

The last argument (optional) is a harness metrics file of the same build, for where the fake is
opaque (bleed is then scored the same way as in trace_eval.py)."""
import csv, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalgeom import align, area, clip, inner  # noqa: E402

log_path, out_path = sys.argv[1], sys.argv[2]
gt_all = json.load(open(sys.argv[3])) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
presses = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 and sys.argv[4] else []
hm = json.load(open(sys.argv[5])) if len(sys.argv) > 5 and os.path.exists(sys.argv[5]) else {}
op = hm.get("opaque_inset", [0.020, 0.0163])
INK = (0.034, 0.043)
raw_rows = list(csv.DictReader(open(log_path)))
# One row per frame, by the frame's own index (with a head start the screen shows older frames and
# can repeat or skip one while catching up).
order = [int(r["i"]) for r in raw_rows]
seen = {}
for r in raw_rows:
    seen.setdefault(int(r["i"]), r)
rows = [seen[i] for i in sorted(seen)]
idx = [int(r["i"]) for r in rows]
drew = [r["drew"] == "1" for r in rows]
armed = [r["armed"] == "1" for r in rows]
m = {"frames": len(rows), "drawn_frames": sum(drew),
     "first_drawn": next((idx[n] for n, d in enumerate(drew) if d), None),
     "flickers": sum(1 for i in range(1, len(drew)) if drew[i - 1] and not drew[i]),
     # how the picture was paced: frames shown twice, frames never shown
     "repeated_frames": len(order) - len(set(order)),
     "skipped_frames": (max(order) - min(order) + 1 - len(set(order))) if order else 0,
     "reasons": {}}
for r in rows:
    m["reasons"][r["reason"]] = m["reasons"].get(r["reason"], 0) + 1
# the volume-button presses: armed state a few frames after each
m["armed_after_presses"] = [(p, armed[min(range(len(idx)), key=lambda n: abs(idx[n] - p - 5))]) for p in presses if idx]
m["frames_shown"] = len(idx)

if gt_all:
    frames = gt_all["frames"]
    FW, FH = gt_all["width"], gt_all["height"]
    FRAME = [(0, 0), (FW, 0), (FW, FH), (0, FH)]
    shown, ghost, errs, bleeds = [], [], [], []
    for n, r in enumerate(rows):
        i = idx[n]
        if i >= len(frames):
            break
        f = frames[i]
        drew_i, armed_i = drew[n], armed[n]
        inf = f.get("inframe", 1 if f["visible"] else 0)
        if armed_i and inf >= 0.2 and not drew_i:
            shown.append(i)
        if drew_i and inf < 0.05:
            ghost.append(i)
        if drew_i and inf > 0.02 and r["x0"]:
            h = float(r["h"])
            q = [(float(r["x%d" % k]), h - float(r["y%d" % k])) for k in range(4)]
            t = [tuple(c) for c in f["corners"]]
            e, qa = align(q, t)
            if f["visible"]:
                errs.append(e)
            ink = clip(inner(t, *INK), FRAME)
            a_ink = area(ink) if len(ink) >= 3 else 0
            if a_ink > 1:
                cov = clip(ink, inner(qa, *op))
                bleeds.append((max(0.0, 1 - (area(cov) if len(cov) >= 3 else 0) / a_ink), i))
    errs.sort()
    first_in = next((f["i"] for f in frames if f.get("inframe", 0) >= 0.2), None)
    b = sorted(x[0] for x in bleeds)
    m.update({"real_card_shown_frames": len(shown), "real_card_shown_list": shown[:40],
              "ghost_frames": len(ghost), "ghost_list": ghost[:40],
              "lock_delay_frames": (m["first_drawn"] - first_in) if (m["first_drawn"] is not None and first_in is not None) else None,
              "corner_err_mean": round(sum(errs) / len(errs), 2) if errs else -1,
              "corner_err_p95": round(errs[int(0.95 * (len(errs) - 1))], 2) if errs else -1,
              "bleed_pct_mean": round(100 * sum(b) / len(b), 2) if b else -1,
              "bleed_pct_p95": round(100 * b[int(0.95 * (len(b) - 1))], 2) if b else -1,
              "bleed_frames_over_0.5pct": sum(1 for x in b if x > 0.005),
              "worst_bleed": [(i, round(100 * v, 1)) for v, i in sorted(bleeds)[-6:][::-1]]})
json.dump(m, open(out_path, "w"), indent=1)
print(json.dumps(m, indent=1))
