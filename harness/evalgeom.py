"""Polygon helpers shared by the evaluation scripts (image coordinates)."""
import math


def homography(src, dst):
    import numpy as np
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return [float(x) for x in h] + [1.0]


def apply(H, p):
    x, y = p
    w = H[6] * x + H[7] * y + H[8]
    return ((H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w)


UNIT = [(0, 0), (1, 0), (1, 1), (0, 1)]


def inner(quad, mx, my):
    """The part of the card from mx, my (fractions) in from each edge, in image coordinates."""
    H = homography(UNIT, quad)
    return [apply(H, p) for p in [(mx, my), (1 - mx, my), (1 - mx, 1 - my), (mx, 1 - my)]]


def area(poly):
    a = 0.0
    for k in range(len(poly)):
        x0, y0 = poly[k]; x1, y1 = poly[(k + 1) % len(poly)]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2


def ccw(poly):
    a = 0.0
    for k in range(len(poly)):
        x0, y0 = poly[k]; x1, y1 = poly[(k + 1) % len(poly)]
        a += x0 * y1 - x1 * y0
    return poly if a > 0 else poly[::-1]


def clip(subject, clipper):
    """Sutherland-Hodgman: subject polygon clipped by a convex polygon."""
    out = subject
    cp = ccw(clipper)
    for k in range(len(cp)):
        a, b = cp[k], cp[(k + 1) % len(cp)]
        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0
        def cross(p, q):
            x1, y1 = p; x2, y2 = q; x3, y3 = a; x4, y4 = b
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < 1e-12:
                return q
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        inp, out = out, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            if inside(e):
                if not inside(s):
                    out.append(cross(s, e))
                out.append(e)
            elif inside(s):
                out.append(cross(s, e))
            s = e
    return out


def align(q, t):
    """The labelling of q (any rotation, either direction) that best matches t."""
    best = None
    for rev in (False, True):
        qq = q[::-1] if rev else q
        for s in range(4):
            r = [qq[(k + s) % 4] for k in range(4)]
            e = sum(math.dist(r[k], t[k]) for k in range(4)) / 4
            if best is None or e < best[0]:
                best = (e, r)
    return best
