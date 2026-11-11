"""Make a copy of a real clip as if the phone had been tilted while filming.

The camera turning about its own centre moves every point of the picture by one homography,
H = K R K^-1, so a clip filmed with the phone level can be re-shot "tilted" exactly: the card
then appears at the angle it would have with the phone held off level. The tilt sways
smoothly (forward/back up to `pitch` degrees, side to side up to `yaw`, plus a little roll),
the way a hand drifts while performing. The picture is shifted back after each tilt so the
point the camera was looking at stays in place (the performer keeps the card in view), which
leaves the card seen at the tilted angle.

usage: tilt_clip.py in.mov out.mp4 [pitch_deg] [yaw_deg] [roll_deg]
"""
import math
import sys

import cv2
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
pitch = float(sys.argv[3]) if len(sys.argv) > 3 else 25
yaw = float(sys.argv[4]) if len(sys.argv) > 4 else 20
roll = float(sys.argv[5]) if len(sys.argv) > 5 else 10

cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
out = None
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    h0, w0 = frame.shape[:2]
    # portrait, at most 720 wide (the app works on frames about this size)
    if w0 > h0:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h0, w0 = frame.shape[:2]
    s = min(1.0, 720 / w0)
    w, h = int(w0 * s) // 2 * 2, int(h0 * s) // 2 * 2
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    if out is None:
        out = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    t = i / fps
    a = math.radians(pitch) * math.sin(2 * math.pi * t / 4.0)
    b = math.radians(yaw) * math.sin(2 * math.pi * t / 5.3 + 1.0)
    c = math.radians(roll) * math.sin(2 * math.pi * t / 6.7 + 2.0)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    f = 0.9 * max(w, h)   # selfie camera, roughly 75 degrees across the long side
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    H = K @ Rz @ Ry @ Rx @ np.linalg.inv(K)
    # (the performer keeps the card in view as the phone tips: shift the picture back so the point
    # the camera was looking at stays put, leaving only the change of angle)
    c0 = H @ np.array([w / 2, h / 2, 1.0])
    T = np.array([[1, 0, w / 2 - c0[0] / c0[2]], [0, 1, h / 2 - c0[1] / c0[2]], [0, 0, 1]])
    H = T @ H
    warped = cv2.warpPerspective(frame, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out.write(warped)
    i += 1
if out is not None:
    out.release()
print(f"{dst}: {i} frames, tilt up to {pitch}/{yaw}/{roll} degrees")
