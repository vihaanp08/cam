"""Synthetic test video for the card tracker, with exact ground-truth corners.

Mimics a hand-held card in front of a phone camera: bright, low-contrast background (the hardest
case: white card against a white ceiling), a palm behind the card with fingers and a thumb over its
edges, smooth motion with rotation and tilt, a still hold, a fast swipe, the card leaving and
coming back, lighting drift, blur and sensor noise.

Usage: python gen_synthetic.py OUT_DIR [frames]
Writes OUT_DIR/frame_00000.jpg ... and OUT_DIR/gt.json
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cards import card  # noqa: E402

FW, FH = 720, 1280
FPS = 30


def homography(src, dst):
    """3x3 H with dst ~ H * src."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    b = [c for p in dst for c in p]
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1).reshape(3, 3)


def pil_perspective(card_img, quad):
    """Warp card_img onto the frame so its corners land on quad (TL, TR, BR, BL)."""
    w, h = card_img.size
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    Hm = homography(quad, src)          # frame -> card (PIL wants the inverse map)
    coeffs = (Hm / Hm[2, 2]).flatten()[:8]
    return card_img.transform((FW, FH), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC)


def background(t):
    """Bright off-white ceiling with soft beams, plus a dark blob (hair) at one edge."""
    y, x = np.mgrid[0:FH, 0:FW].astype(np.float32)
    base = 228 + 12 * np.sin((x * 0.6 + y) / 260.0 + 0.3 * t)
    beams = 14 * (np.sin((x - 0.35 * y) / 70.0) > 0.6)
    g = base - beams
    rgb = np.stack([g * 1.0, g * 0.97, g * 0.9], -1)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    d.ellipse((-160, 300 + 20 * math.sin(t), 120, 700), fill=(38, 30, 26))
    return img.filter(ImageFilter.GaussianBlur(2))


def quad_for(cx, cy, size, rot, tilt_x, tilt_y):
    """Card corners for a card centred at (cx, cy), about `size` px wide, rotated and tilted in 3D,
    seen by a pinhole camera like a phone's selfie camera (about 70 degrees across the short side,
    principal point at the image centre)."""
    f = 0.75 * FW
    w_mm, h_mm = 63.0, 88.0
    Z = f * w_mm / size                       # distance that makes the card `size` px wide
    X = (cx - FW / 2) * Z / f
    Y = (cy - FH / 2) * Z / f
    pts = []
    for (px, py) in [(-w_mm / 2, -h_mm / 2), (w_mm / 2, -h_mm / 2), (w_mm / 2, h_mm / 2), (-w_mm / 2, h_mm / 2)]:
        # tilt around the x and y axes, then rotate in plane
        y1 = py * math.cos(tilt_x); z1 = py * math.sin(tilt_x)
        x2 = px * math.cos(tilt_y) + z1 * math.sin(tilt_y); z2 = -px * math.sin(tilt_y) + z1 * math.cos(tilt_y)
        xr = x2 * math.cos(rot) - y1 * math.sin(rot)
        yr = x2 * math.sin(rot) + y1 * math.cos(rot)
        pts.append((FW / 2 + f * (X + xr) / (Z + z2), FH / 2 + f * (Y + yr) / (Z + z2)))
    return pts


def inframe_fraction(quad):
    """Fraction of the card's area that is inside the frame."""
    m = Image.new("L", (FW * 3, FH * 3), 0)
    ImageDraw.Draw(m).polygon([(x + FW, y + FH) for x, y in quad], fill=1)
    a = np.asarray(m)
    total = a.sum()
    return float(a[FH:2 * FH, FW:2 * FW].sum()) / total if total else 0.0


def real_card():
    """The performer's real card (5 of spades traced from a photo of his deck), with rounded corners."""
    from render_face import render
    im = render("7_clubs", paper=(246, 246, 243), scale_ink=1.0).convert("RGBA")
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, im.size[0] - 1, im.size[1] - 1), radius=26, fill=255)
    im.putalpha(mask)
    return im


def court_card():
    """A king: corner indices, and a printed frame round a colourful picture (the frame is a strong,
    card-shaped rectangle inside the card: the classic way to end up with a card inside a card)."""
    im = card(13, 0, rounded=False)
    d = ImageDraw.Draw(im)
    W, H = im.size
    x0, y0, x1, y1 = int(0.16 * W), int(0.12 * H), int(0.84 * W), int(0.88 * H)
    rng = np.random.default_rng(7)
    cols = [(204, 18, 33, 255), (30, 60, 160, 255), (240, 190, 40, 255), (36, 36, 36, 255)]
    for k in range(40):        # the picture: overlapping blocks of colour, mirrored top to bottom
        cx, cy = rng.uniform(x0, x1), rng.uniform(y0, (y0 + y1) / 2)
        rw, rh = rng.uniform(15, 70), rng.uniform(15, 60)
        c = cols[k % 4]
        for (yy, flip) in ((cy, False), (y0 + y1 - cy, True)):
            xx = cx if not flip else x0 + x1 - cx
            d.ellipse((xx - rw / 2, yy - rh / 2, xx + rw / 2, yy + rh / 2), fill=c)
    d.line(((x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2)), fill=(36, 36, 36, 255), width=3)
    d.rectangle((x0, y0, x1, y1), outline=(36, 36, 36, 255), width=4)
    im = im.convert("RGB").convert("RGBA")
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, W - 1, H - 1), radius=26, fill=255)
    im.putalpha(mask)
    # paper slightly off-white like the real deck
    a = np.asarray(im).astype(np.float32)
    a[..., :3] *= np.array([246, 246, 243]) / 255.0
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")


def wall(t):
    """A plain wall with a sheet of paper pinned to it (white, printed, nearly a card's shape: the
    kind of thing a fake card must never jump onto) and a dark picture frame."""
    y, x = np.mgrid[0:FH, 0:FW].astype(np.float32)
    g = 214 + 10 * np.sin((x * 0.4 + y) / 300.0 + 0.2 * t)
    rgb = np.stack([g * 1.0, g * 0.98, g * 0.94], -1)
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    px, py, pw, ph = 470, 120, 190, 269            # A4-ish, 0.707
    d.rectangle((px, py, px + pw, py + ph), fill=(242, 241, 236))
    for k in range(14):                              # lines of print
        ly = py + 24 + k * 16
        d.rectangle((px + 18, ly, px + pw - 18 - (k % 3) * 30, ly + 5), fill=(70, 70, 75))
    d.rectangle((px + 18, py + ph - 70, px + 90, py + ph - 20), fill=(180, 40, 40))
    fx, fy = 30, 980                                 # picture frame
    d.rectangle((fx, fy, fx + 220, fy + 160), fill=(60, 45, 35))
    d.rectangle((fx + 18, fy + 18, fx + 202, fy + 142), fill=(120, 150, 170))
    return img.filter(ImageFilter.GaussianBlur(1.6))


def pose_court(i, n):
    """Third script: a court card in front of a wall with a printed sheet on it. Comes in, drifts,
    holds still, tilts, comes closer, goes out to the side and back."""
    seg = i / n
    t = i / FPS
    cx = FW * 0.42 + 70 * math.sin(t * 0.8)
    cy = FH * 0.55 + 90 * math.sin(t * 0.6 + 1)
    rot = math.radians(10 * math.sin(t * 0.7))
    tx = math.radians(8 * math.sin(t * 0.5))
    ty = math.radians(8 * math.sin(t * 0.6 + 0.5))
    size = 300.0
    if seg < 0.07:
        k = 1 - seg / 0.07
        cy += 900 * k * k
    if 0.30 <= seg < 0.40:            # held still
        t0 = 0.30 * n / FPS
        cx = FW * 0.42 + 70 * math.sin(t0 * 0.8); cy = FH * 0.55 + 90 * math.sin(t0 * 0.6 + 1)
        rot = math.radians(10 * math.sin(t0 * 0.7)); tx = math.radians(8 * math.sin(t0 * 0.5)); ty = math.radians(8 * math.sin(t0 * 0.6 + 0.5))
    if 0.40 <= seg < 0.55:            # tipped back and to the side
        k = (seg - 0.40) / 0.15
        tx += math.radians(35) * math.sin(k * 2 * math.pi)
        ty += math.radians(30) * math.sin(k * math.pi)
    if 0.55 <= seg < 0.70:            # brought closer
        k = (seg - 0.55) / 0.15
        size += 260 * math.sin(k * math.pi)
    if 0.72 <= seg < 0.82:            # out to the right and back
        k = (seg - 0.72) / 0.10
        cx += 700 * math.sin(k * math.pi)
    return cx, cy, size, rot, tx, ty


def pose(i, n):
    """Scripted motion: drift, still hold, fast swipe, leave and return."""
    seg = i / n
    # time freezes during the still hold, then carries on smoothly (no teleport)
    h0, h1 = int(0.30 * n), int(0.42 * n)
    t = (min(i, h0) + max(0, i - h1)) / FPS
    cx = FW / 2 + 120 * math.sin(t * 0.9)
    cy = FH * 0.46 + 140 * math.sin(t * 0.7 + 1)
    rot = math.radians(14 * math.sin(t * 0.8))
    tx = math.radians(12 * math.sin(t * 0.5))
    ty = math.radians(14 * math.sin(t * 0.6 + 0.5))
    size = 300 + 30 * math.sin(t * 0.4)
    if seg < 0.07:                    # comes up into view from below at the start
        k = 1 - seg / 0.07
        cy += 900 * k * k
    if 0.44 <= seg < 0.54:            # strong tilt back and forth (card held at an angle)
        k = (seg - 0.44) / 0.10
        tx += math.radians(38) * math.sin(k * math.pi)
        ty += math.radians(30) * math.sin(2 * k * math.pi)
    if 0.55 <= seg < 0.60:            # fast swipe to the right and back
        k = (seg - 0.55) / 0.05
        cx += 260 * math.sin(k * math.pi)
    if 0.75 <= seg < 0.82:            # leaves the frame downward, then returns
        k = (seg - 0.75) / 0.07
        cy += 1100 * math.sin(k * math.pi)
    return cx, cy, size, rot, tx, ty


def pose_close(i, n):
    """Second script: close-ups and angles. Comes in, then is brought right up to the camera (much
    bigger than the picture), then held at steep angles, then turned round a long way."""
    seg = i / n
    t = i / FPS
    cx = FW / 2 + 60 * math.sin(t * 0.9)
    cy = FH * 0.48 + 70 * math.sin(t * 0.7 + 1)
    rot = math.radians(8 * math.sin(t * 0.8))
    tx = math.radians(6 * math.sin(t * 0.5))
    ty = math.radians(6 * math.sin(t * 0.6 + 0.5))
    size = 300.0
    if seg < 0.07:                    # comes up into view from below
        k = 1 - seg / 0.07
        cy += 900 * k * k
    if 0.20 <= seg < 0.45:            # brought right up to the camera and back
        k = (seg - 0.20) / 0.25
        size += 700 * math.sin(k * math.pi)
        cy += 120 * math.sin(k * math.pi)
    if 0.50 <= seg < 0.72:            # steep angles: tipped back, forward, to each side
        k = (seg - 0.50) / 0.22
        tx += math.radians(48) * math.sin(k * 2 * math.pi)
        ty += math.radians(42) * math.sin(k * math.pi)
    if 0.75 <= seg < 0.95:            # turned a long way round, and back
        k = (seg - 0.75) / 0.20
        rot += math.radians(70) * math.sin(k * math.pi)
    return cx, cy, size, rot, tx, ty


def _approach(t_in, start, end, speed, ease=6.0):
    """Position `t_in` frames into a straight move from `start` to `end`: `speed` px a frame, slowing
    to a stop over the last `ease` frames."""
    D = abs(end - start)
    s = 1.0 if end >= start else -1.0
    lin = max(0.0, D - speed * ease / 2)
    t_lin = lin / speed
    if t_in <= t_lin:
        d = speed * t_in
    else:
        u = min(1.0, (t_in - t_lin) / ease)
        d = lin + speed * ease * (u - u * u / 2)
    return start + s * min(d, D)


def _approach_frames(start, end, speed, ease=6.0):
    return max(0.0, abs(end - start) - speed * ease / 2) / speed + ease


def _leave(t_out, start, end, speed, ease=4.0):
    """The other way: speeding up from a stop at `start` to `speed` px a frame, towards `end`."""
    D = abs(end - start)
    s = 1.0 if end >= start else -1.0
    d = speed * t_out * t_out / (2 * ease) if t_out < ease else speed * ease / 2 + speed * (t_out - ease)
    return start + s * min(d, D)


# Fourth script: the card brought into view again and again. Each: which way it comes in ("x" across,
# "y" up), from, to, speed (px a frame), in-plane turn (degrees), where it is held on the other axis,
# where it leaves to and how fast.
ENTRIES = [
    ("x", -420.0, FW * 0.5, 55.0, 0.0, FH * 0.5, 1150.0, 60.0),       # swept in fast from the left
    ("y", FH + 210.0, FH * 0.77, 16.0, 15.0, FW * 0.5, FH + 600.0, 40.0),  # raised slowly from below, at an angle
    ("x", 1150.0, FW * 0.5, 35.0, 90.0, FH * 0.45, -500.0, 50.0),     # held sideways, in from the right
]
ENTRY_LEAD, ENTRY_HOLD, ENTRY_GAP = 4, 24, 80


def entries_schedule():
    """[(first frame, frames coming in, frames held, frames leaving)] and the segment names."""
    out, segs, f = [], [], ENTRY_LEAD
    for e, (axis, a, b, v, rot, other, away, vout) in enumerate(ENTRIES):
        n_in = int(math.ceil(_approach_frames(a, b, v)))
        n_out = int(math.ceil((abs(away - b) - vout * 2) / vout + 4))
        out.append((f, n_in, ENTRY_HOLD, n_out))
        segs += [(f, f + n_in, f"enter{e + 1}"), (f + n_in, f + n_in + ENTRY_HOLD, f"held{e + 1}"),
                 (f + n_in + ENTRY_HOLD, f + n_in + ENTRY_HOLD + n_out, f"leave{e + 1}"),
                 (f + n_in + ENTRY_HOLD + n_out, f + n_in + ENTRY_HOLD + n_out + ENTRY_GAP, f"away{e + 1}")]
        f += n_in + ENTRY_HOLD + n_out + ENTRY_GAP
    return out, segs, f


def pose_entries(i, n):
    """Fourth script: the card brought into view the way a performer brings it up, again and again,
    each time after long enough out of view that nothing is known about it (no size, no picture):
    swept in fast from the left, raised slowly from below at an angle, brought in held sideways from
    the right. Each is held for a moment, then taken out of view again."""
    t = i / FPS
    tx = math.radians(6 * math.sin(t * 0.5))
    ty = math.radians(6 * math.sin(t * 0.6 + 0.5))
    size = 280.0
    sched, _, _ = entries_schedule()
    k = 0
    while k + 1 < len(sched) and i >= sched[k + 1][0]:
        k += 1
    axis, a, b, v, rot, other, away, vout = ENTRIES[k]
    f0, n_in, n_hold, n_out = sched[k]
    j = i - f0
    if j < 0:
        p = a
    elif j < n_in:
        p = _approach(j, a, b, v)
    elif j < n_in + n_hold:
        p = b
    else:
        p = _leave(j - n_in - n_hold, b, away, vout)
    # (a little drift while held, as a hand does)
    h = max(0, min(j - n_in, n_hold)) if j >= n_in else 0
    wob = math.sin(h * 0.25) * (1 if n_in <= j < n_in + n_hold else 0)
    if axis == "x":
        cx, cy = p, other + 15 * wob
    else:
        cx, cy = other + 15 * wob, p
    return cx, cy, size, math.radians(rot + 3 * wob), tx, ty


# Fifth script: a court card shown at an ordinary distance, then brought back close to the camera,
# running off two edges of the picture (left and top), twice; the second time long enough after that
# the card's size is no longer known. Then shown at an ordinary distance again and brought straight
# back much closer, off the bottom edge. (start, held at, size, turn, leaves to, speed in, speed out,
# frames held, frames away after)
CLOSE_EVENTS = [
    ((1000.0, 860.0), (380.0, 860.0), 260.0, 4.0, (380.0, 2000.0), 60.0, 60.0, 30, 30),
    ((760.0, -560.0), (230.0, 300.0), 600.0, -3.0, (-300.0, 1700.0), 70.0, 70.0, 30, 100),
    ((-500.0, 1700.0), (180.0, 1000.0), 560.0, 6.0, (-420.0, 1900.0), 70.0, 70.0, 30, 20),
    # shown at an ordinary distance, then straight back up from below much closer, its bottom out of
    # view: its whole top side is in view, and that side taken for the long one makes a sideways card
    # about the size it was just seen at
    ((900.0, 500.0), (360.0, 520.0), 260.0, 2.0, (360.0, 1700.0), 60.0, 70.0, 30, 8),
    ((360.0, 1900.0), (360.0, 1130.0), 420.0, -2.0, (360.0, 2000.0), 50.0, 60.0, 40, 20),
]
CLOSE_LEAD = 6


def close_schedule():
    """[(first frame, frames coming in, frames held, frames leaving, frames away)] and segments."""
    out, segs, f = [], [], CLOSE_LEAD
    for e, (a, b, size, rot, away, vin, vout, hold, gap) in enumerate(CLOSE_EVENTS):
        d_in = math.hypot(b[0] - a[0], b[1] - a[1])
        d_out = math.hypot(away[0] - b[0], away[1] - b[1])
        n_in = int(math.ceil(_approach_frames(0.0, d_in, vin)))
        n_out = int(math.ceil((d_out - vout * 2) / vout + 4))
        out.append((f, n_in, hold, n_out, gap))
        segs += [(f, f + n_in, f"enter{e + 1}"), (f + n_in, f + n_in + hold, f"held{e + 1}"),
                 (f + n_in + hold, f + n_in + hold + n_out, f"leave{e + 1}"),
                 (f + n_in + hold + n_out, f + n_in + hold + n_out + gap, f"away{e + 1}")]
        f += n_in + hold + n_out + gap
    return out, segs, f


def pose_closecourt(i, n):
    """Fifth script (see CLOSE_EVENTS)."""
    t = i / FPS
    tx = math.radians(5 * math.sin(t * 0.5))
    ty = math.radians(5 * math.sin(t * 0.6 + 0.5))
    sched, _, _ = close_schedule()
    k = 0
    while k + 1 < len(sched) and i >= sched[k + 1][0]:
        k += 1
    a, b, size, rot, away, vin, vout, hold, gap = CLOSE_EVENTS[k]
    f0, n_in, n_hold, n_out, _ = sched[k]
    j = i - f0
    d_in = math.hypot(b[0] - a[0], b[1] - a[1])
    d_out = math.hypot(away[0] - b[0], away[1] - b[1])
    if j < 0:
        x, y = a
    elif j < n_in:
        f = _approach(j, 0.0, d_in, vin) / d_in
        x, y = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
    elif j < n_in + n_hold:
        h = j - n_in
        x, y = b[0] + 18 * math.sin(h * 0.21), b[1] + 14 * math.sin(h * 0.17 + 1)
    else:
        f = _leave(j - n_in - n_hold, 0.0, d_out, vout) / d_out
        x, y = b[0] + f * (away[0] - b[0]), b[1] + f * (away[1] - b[1])
    return x, y, size, math.radians(rot), tx, ty


WARM_LAMP = (0.88, 0.58, 0.36)


def lamp(layer, tint):
    """An RGBA layer under a coloured light (r, g, b multipliers)."""
    if tint == (1.0, 1.0, 1.0):
        return layer
    a = np.asarray(layer).astype(np.float32)
    for k in range(3):
        a[..., k] *= tint[k]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")


def hand_layer(quad, behind):
    """Skin-coloured palm behind the card, or fingers/thumb in front of it."""
    layer = Image.new("RGBA", (FW, FH), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    skin = (214, 160, 128, 255)
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = quad
    cx, cy = (x0 + x1 + x2 + x3) / 4, (y0 + y1 + y2 + y3) / 4
    w = math.hypot(x1 - x0, y1 - y0)
    if behind:
        # palm: a big blob centred a bit below the card, wider than it
        d.ellipse((cx - w * 0.9, cy - w * 0.4, cx + w * 0.9, cy + w * 1.6), fill=skin)
        for k in range(4):  # fingers sticking out above the card
            fx = cx - w * 0.45 + k * w * 0.3
            d.rounded_rectangle((fx - w * 0.11, cy - w * 1.05, fx + w * 0.11, cy), radius=int(w * 0.1), fill=skin)
    else:
        # thumb over the middle of the left long edge, fingertip over the right edge
        lx, ly = (x0 + x3) / 2, (y0 + y3) / 2
        d.ellipse((lx - w * 0.22, ly - w * 0.12, lx + w * 0.08, ly + w * 0.16), fill=skin)
        rx, ry = (x1 + x2) / 2, (y1 + y2) / 2
        d.ellipse((rx - w * 0.06, ry + w * 0.05, rx + w * 0.16, ry + w * 0.22), fill=skin)
    return layer.filter(ImageFilter.GaussianBlur(1.5))


def main():
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    kind = sys.argv[3] if len(sys.argv) > 3 else "main"
    script = {"close": pose_close, "court": pose_court, "entries": pose_entries, "closecourt": pose_closecourt}.get(kind, pose)
    if kind == "entries":
        n = entries_schedule()[2]
    if kind == "closecourt":
        n = close_schedule()[2]
    os.makedirs(out, exist_ok=True)
    face = court_card() if kind in ("court", "closecourt") else real_card()
    # Sixth script: the main one with the card and the hand under a warm lamp (orange: the paper
    # comes out about R 200, G 130, B 75, as measured on a real card under a desk lamp) while the
    # room behind stays white.
    tint = WARM_LAMP if kind == "warm" else (1.0, 1.0, 1.0)
    # (the hand in the lamp's light too, darker than the paper by as much as on the real one: the palm
    # came out at about half the card's brightness)
    hand_tint = tuple(0.7 * c for c in tint) if kind == "warm" else tint
    still = {"court": (0.30, 0.40), "entries": (2.0, 2.0), "closecourt": (2.0, 2.0)}.get(kind, (0.30, 0.42))
    rng = np.random.default_rng(1)
    gts = []
    prev = None
    for i in range(n):
        t = i / FPS
        cx, cy, size, rot, tx, ty = script(i, n)
        quad = quad_for(cx, cy, size, rot, tx, ty)
        frame = (wall(t) if kind == "court" else background(t)).convert("RGBA")
        frame.alpha_composite(lamp(hand_layer(quad, behind=True), hand_tint))
        # lighting across the card: brighter at one corner, warm cast, slow global drift
        gain = 0.92 + 0.06 * math.sin(t * 0.7)
        lit = np.asarray(face).astype(np.float32)
        yy, xx = np.mgrid[0:lit.shape[0], 0:lit.shape[1]]
        grad = 0.85 + 0.15 * (xx / lit.shape[1]) * (1 - yy / lit.shape[0] * 0.5)
        lit[..., 0] *= grad * gain * 1.0 * tint[0]
        lit[..., 1] *= grad * gain * 0.97 * tint[1]
        lit[..., 2] *= grad * gain * 0.9 * tint[2]
        lit_img = Image.fromarray(np.clip(lit, 0, 255).astype(np.uint8), "RGBA")
        frame.alpha_composite(pil_perspective(lit_img, quad))
        frame.alpha_composite(lamp(hand_layer(quad, behind=False), hand_tint))
        frame = frame.convert("RGB")
        speed = 0 if prev is None else math.hypot(cx - prev[0], cy - prev[1])
        # (out of view, the card's jumps from where it left to where it comes in next blur nothing)
        if kind in ("entries", "closecourt") and inframe_fraction(quad) == 0:
            speed = 0
        prev = (cx, cy)
        frame = frame.filter(ImageFilter.GaussianBlur(0.7 + min(3.0, speed / 25)))
        arr = np.asarray(frame).astype(np.float32) + rng.normal(0, 3.0, (FH, FW, 3))
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(os.path.join(out, f"frame_{i:05d}.jpg"), quality=90)
        inside = all(0 <= x < FW and 0 <= y < FH for x, y in quad)
        gts.append({"i": i, "visible": inside, "corners": [[round(x, 2), round(y, 2)] for x, y in quad],
                    "still": still[0] <= i / n < still[1], "inframe": round(inframe_fraction(quad), 3)})
    meta = {"width": FW, "height": FH, "fps": FPS, "frames": gts}
    if kind == "entries":
        meta["segments"] = entries_schedule()[1]
    if kind == "closecourt":
        meta["segments"] = close_schedule()[1]
    json.dump(meta, open(os.path.join(out, "gt.json"), "w"))
    print("wrote", n, "frames to", out)


if __name__ == "__main__":
    main()
