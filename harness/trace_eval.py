"""Scores a harness trace against the synthetic ground truth, in the terms the audience would see.

Usage: python trace_eval.py GT.json TRACE.csv METRICS.json OUT.json [main|close|court]

For every frame where the fake is drawn:
  - bleed: share of the real card's printed area (inside its white border) that is still visible
    because the fake's opaque area doesn't cover it (misplaced, too small, lagging behind)
  - spill: fake painted outside the real card, as a share of the card's area
  - corner error of the drawn outline and of the raw (unsmoothed) measurement
And per motion segment of the script, so it is clear where it goes wrong.
"""
import csv, json, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalgeom import align, area, clip, inner  # noqa: E402

gt_all = json.load(open(sys.argv[1]))
gt = gt_all["frames"]
FW, FH = gt_all["width"], gt_all["height"]
rows = list(csv.DictReader(open(sys.argv[2])))
metrics = json.load(open(sys.argv[3]))
out_path = sys.argv[4]
script = sys.argv[5] if len(sys.argv) > 5 else "main"

# Where the fake is fully opaque / where it ends, as fractions of the card's width and height
# from each edge (reported by the harness for the code being tested).
op = metrics.get("opaque_inset", [0.046, 0.035])
paint = metrics.get("paint_inset", [0.032, 0.025])
# The performer's deck: printing starts this far in from the edges.
INK = (0.034, 0.043)


FRAME = [(0, 0), (FW, 0), (FW, FH), (0, FH)]


def quad_from(r, prefix):
    try:
        v = [float(r[f"{prefix}{a}{k}"]) for k in range(4) for a in "xy"]
    except (KeyError, ValueError):
        return None
    return [(v[0], v[1]), (v[2], v[3]), (v[4], v[5]), (v[6], v[7])]


SEGS = {
    "main": [(0, 21, "enter"), (21, 90, "drift"), (90, 126, "still"), (126, 132, "drift2"), (132, 162, "tilt"),
             (162, 165, "-"), (165, 180, "swipe"), (180, 225, "drift3"), (225, 246, "exit/return"), (246, 300, "end")],
    "close": [(0, 21, "enter"), (21, 60, "drift"), (60, 135, "close-up"), (135, 150, "drift2"), (150, 216, "angles"),
              (216, 225, "drift3"), (225, 285, "turn"), (285, 300, "end")],
    "court": [(0, 21, "enter"), (21, 90, "drift"), (90, 120, "still"), (120, 165, "tilt"), (165, 210, "closer"),
              (210, 216, "drift2"), (216, 246, "out/back"), (246, 300, "end")],
}[script]

per = {}
seen = set()
for r in rows:
    i = int(r["i"])
    if i in seen or i >= len(gt):
        continue
    seen.add(i)
    f = gt[i]
    t = [tuple(c) for c in f["corners"]]
    inf = f.get("inframe", 1.0 if f["visible"] else 0.0)
    drew = r["drew"] == "1"
    d = {"drew": drew, "inframe": inf, "reason": r.get("reason", "")}
    q = quad_from(r, "")
    raw = quad_from(r, "r")
    if drew and q and inf > 0.02:
        e, qa = align(q, t)
        d["err"] = e
        ink = clip(inner(t, *INK), FRAME)
        a_ink = area(ink) if len(ink) >= 3 else 0
        if a_ink > 1:
            cov = clip(ink, inner(qa, *op))
            d["bleed"] = max(0.0, 1 - (area(cov) if len(cov) >= 3 else 0) / a_ink)
        painted = clip(inner(qa, *paint), FRAME)
        a_p = area(painted) if len(painted) >= 3 else 0
        on_card = clip(painted, t) if a_p > 0 else []
        d["spill"] = (a_p - (area(on_card) if len(on_card) >= 3 else 0)) / max(1.0, area(t))
    if raw and inf > 0.02:
        d["raw_err"] = align(raw, t)[0]
    per[i] = d


def stats(v):
    v = sorted(v)
    if not v:
        return None
    return {"mean": round(sum(v) / len(v), 4), "p50": round(v[len(v) // 2], 4),
            "p95": round(v[int(0.95 * (len(v) - 1))], 4), "max": round(v[-1], 4), "n": len(v)}


def summary(idx):
    ds = [per[i] for i in idx if i in per]
    exposed = [i for i in idx if i in per and per[i]["inframe"] >= 0.2 and not per[i]["drew"]]
    ghost = [i for i in idx if i in per and per[i]["inframe"] < 0.05 and per[i]["drew"]]
    s = {"exposed": len(exposed), "ghost": len(ghost),
         "err": stats([d["err"] for d in ds if "err" in d]),
         "raw_err": stats([d["raw_err"] for d in ds if "raw_err" in d]),
         "bleed_pct": stats([100 * d["bleed"] for d in ds if "bleed" in d]),
         "spill_pct": stats([100 * d["spill"] for d in ds if "spill" in d]),
         "bleed_frames_over_0.5pct": sum(1 for d in ds if d.get("bleed", 0) > 0.005)}
    if exposed:
        s["exposed_list"] = exposed[:30]
    if ghost:
        s["ghost_list"] = ghost[:30]
    return s


res = {"script": script, "opaque_inset": op, "paint_inset": paint, "all": summary(range(len(gt)))}
res["segments"] = {name: summary(range(a, b)) for a, b, name in SEGS}
worst = sorted((d.get("bleed", 0), i) for i, d in per.items())[-8:]
res["worst_bleed"] = [(i, round(100 * float(b), 2), per[i]["reason"]) for b, i in reversed(worst) if b > 0]
json.dump(res, open(out_path, "w"), indent=1)


def short(s):
    if not s:
        return "-"
    return f"{s['mean']:.2f}/{s['p95']:.2f}/{s['max']:.2f}"


print(f"{'segment':12s} exposed ghost  err(mean/p95/max)   raw_err            bleed%             spill%")
for name, s in [("ALL", res["all"])] + list(res["segments"].items()):
    print(f"{name:12s} {s['exposed']:7d} {s['ghost']:5d}  {short(s['err']):18s} {short(s['raw_err']):18s} "
          f"{short(s['bleed_pct']):18s} {short(s['spill_pct'])}")
print("worst bleed:", res["worst_bleed"])
