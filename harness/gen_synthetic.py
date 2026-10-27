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
    """Card corners for a card centred at (cx, cy), `size` px wide, rotated and tilted in 3D."""
    hw, hh = size / 2, size / 2 * 88 / 63
    pts = []
    f = 900.0
    for (px, py) in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
        # tilt around x and y axes, then rotate in plane, then perspective-project
        z = px * math.sin(tilt_y) + py * math.sin(tilt_x)
        x = px * math.cos(tilt_y)
        yv = py * math.cos(tilt_x)
        xr = x * math.cos(rot) - yv * math.sin(rot)
        yr = x * math.sin(rot) + yv * math.cos(rot)
        s = f / (f + z)
        pts.append((cx + xr * s, cy + yr * s))
    return pts


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
    if 0.55 <= seg < 0.60:            # fast swipe to the right and back
        k = (seg - 0.55) / 0.05
        cx += 260 * math.sin(k * math.pi)
    if 0.75 <= seg < 0.82:            # leaves the frame downward, then returns
        k = (seg - 0.75) / 0.07
        cy += 1100 * math.sin(k * math.pi)
    return cx, cy, size, rot, tx, ty


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
    os.makedirs(out, exist_ok=True)
    face = card(5, 0)
    rng = np.random.default_rng(1)
    gts = []
    prev = None
    for i in range(n):
        t = i / FPS
        cx, cy, size, rot, tx, ty = pose(i, n)
        quad = quad_for(cx, cy, size, rot, tx, ty)
        frame = background(t).convert("RGBA")
        frame.alpha_composite(hand_layer(quad, behind=True))
        # lighting across the card: brighter at one corner, warm cast, slow global drift
        gain = 0.92 + 0.06 * math.sin(t * 0.7)
        lit = np.asarray(face).astype(np.float32)
        yy, xx = np.mgrid[0:lit.shape[0], 0:lit.shape[1]]
        grad = 0.85 + 0.15 * (xx / lit.shape[1]) * (1 - yy / lit.shape[0] * 0.5)
        lit[..., 0] *= grad * gain * 1.0
        lit[..., 1] *= grad * gain * 0.97
        lit[..., 2] *= grad * gain * 0.9
        lit_img = Image.fromarray(np.clip(lit, 0, 255).astype(np.uint8), "RGBA")
        frame.alpha_composite(pil_perspective(lit_img, quad))
        frame.alpha_composite(hand_layer(quad, behind=False))
        frame = frame.convert("RGB")
        speed = 0 if prev is None else math.hypot(cx - prev[0], cy - prev[1])
        prev = (cx, cy)
        frame = frame.filter(ImageFilter.GaussianBlur(0.7 + min(3.0, speed / 25)))
        arr = np.asarray(frame).astype(np.float32) + rng.normal(0, 3.0, (FH, FW, 3))
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(os.path.join(out, f"frame_{i:05d}.jpg"), quality=90)
        inside = all(0 <= x < FW and 0 <= y < FH for x, y in quad)
        gts.append({"i": i, "visible": inside, "corners": [[round(x, 2), round(y, 2)] for x, y in quad],
                    "still": 0.30 <= i / n < 0.42})
    json.dump({"width": FW, "height": FH, "fps": FPS, "frames": gts}, open(os.path.join(out, "gt.json"), "w"))
    print("wrote", n, "frames to", out)


if __name__ == "__main__":
    main()
