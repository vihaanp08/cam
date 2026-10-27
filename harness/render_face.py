"""Render a traced face (harness/faces.json) as albedo PNG (paper white), like the app does.
Usage: python render_face.py KEY OUT.png   (KEY e.g. 9_hearts)"""
import json, os, sys
from PIL import Image, ImageDraw

S = 4  # supersampling for smooth edges


def render(key, paper=(255, 255, 255), scale_ink=255 / 250):
    d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces.json")))[key]
    ink = tuple(min(255, int(round(c * scale_ink))) for c in d["ink"])
    mask = Image.new("L", (504 * S, 704 * S), 0)
    dr = ImageDraw.Draw(mask)
    # even-odd: XOR each polygon in
    import numpy as np
    acc = np.zeros((704 * S, 504 * S), bool)
    for p in d["polys"]:
        m = Image.new("1", (504 * S, 704 * S), 0)
        ImageDraw.Draw(m).polygon([((x + 0.5) * S, (y + 0.5) * S) for x, y in p], fill=1)
        acc ^= np.asarray(m, bool)
    mask = Image.fromarray((acc * 255).astype("uint8")).resize((504, 704), Image.LANCZOS)
    img = Image.new("RGB", (504, 704), paper)
    img.paste(Image.new("RGB", (504, 704), ink), (0, 0), mask)
    return img


if __name__ == "__main__":
    render(sys.argv[1]).save(sys.argv[2])
