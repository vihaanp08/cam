"""Score an end-to-end simulator run (Documents/e2e_<name>/log.csv written by the app's test mode).
Usage: python e2e_eval.py LOG.csv OUT.json [GT.json] [press,press,...]"""
import csv, json, math, sys

log_path, out_path = sys.argv[1], sys.argv[2]
gt = json.load(open(sys.argv[3])) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
presses = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 and sys.argv[4] else []
rows = list(csv.DictReader(open(log_path)))
# One row per frame, by the frame's own index (with a head start the screen shows older frames and
# can repeat or skip one while catching up).
seen = {}
for r in rows:
    seen.setdefault(int(r["i"]), r)
rows = [seen[i] for i in sorted(seen)]
idx = [int(r["i"]) for r in rows]
drew = [r["drew"] == "1" for r in rows]
armed = [r["armed"] == "1" for r in rows]
m = {"frames": len(rows), "drawn_frames": sum(drew),
     "first_drawn": next((idx[n] for n, d in enumerate(drew) if d), None),
     "flickers": sum(1 for i in range(1, len(drew)) if drew[i - 1] and not drew[i]),
     "reasons": {}}
for r in rows:
    m["reasons"][r["reason"]] = m["reasons"].get(r["reason"], 0) + 1
# the volume-button presses: armed state a few frames after each
m["armed_after_presses"] = [(p, armed[min(range(len(idx)), key=lambda n: abs(idx[n] - p - 5))]) for p in presses if idx]
m["frames_shown"] = len(idx)

if gt:
    frames = gt["frames"]
    shown, ghost, errs = [], [], []
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
        if drew_i and f["visible"] and r["x0"]:
            h = float(r["h"])
            q = [(float(r["x%d" % k]), h - float(r["y%d" % k])) for k in range(4)]
            t = f["corners"]
            best = min(sum(math.dist(q[(k + s) % 4], t[k]) for k in range(4)) / 4 for s in range(4))
            errs.append(best)
    errs.sort()
    m.update({"real_card_shown_frames": len(shown), "real_card_shown_list": shown[:40],
              "ghost_frames": len(ghost), "ghost_list": ghost[:40],
              "corner_err_mean": round(sum(errs) / len(errs), 2) if errs else -1,
              "corner_err_p95": round(errs[int(0.95 * (len(errs) - 1))], 2) if errs else -1})
json.dump(m, open(out_path, "w"), indent=1)
print(json.dumps(m, indent=1))
