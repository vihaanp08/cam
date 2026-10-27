"""Draws simple playing-card faces (504 x 704 px, 8 px per mm) with PIL, for the test harness."""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 504, 704
SYM = {0: "♠", 1: "♥", 2: "♣", 3: "♦"}
NAMES = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

SERIF = [
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "/Library/Fonts/Times New Roman Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
]
SYMF = [
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _font(paths, size):
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _suit(img, s, color, cx, cy, h, flip):
    t = Image.new("RGBA", (int(h * 1.6), int(h * 1.6)), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    f = _font(SYMF, int(h * 1.25))
    bb = d.textbbox((0, 0), SYM[s], font=f)
    d.text(((t.width - (bb[2] - bb[0])) / 2 - bb[0], (t.height - (bb[3] - bb[1])) / 2 - bb[1]), SYM[s], font=f, fill=color)
    if flip:
        t = t.rotate(180)
    img.alpha_composite(t, (int(cx - t.width / 2), int(cy - t.height / 2)))


def card(rank, s, rounded=True):
    """RGBA card face. Paper is pure white (albedo 1)."""
    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    c = (204, 18, 33, 255) if s in (1, 3) else (36, 36, 36, 255)

    def index(im):
        d = ImageDraw.Draw(im)
        f = _font(SERIF, 70)
        lab = NAMES[rank - 1]
        bb = d.textbbox((0, 0), lab, font=f)
        tw = bb[2] - bb[0]
        t = Image.new("RGBA", (tw + 20, 100), (0, 0, 0, 0))
        ImageDraw.Draw(t).text((10 - bb[0], 0), lab, font=f, fill=c)
        sq = 0.62 if lab == "10" else 0.8
        t = t.resize((max(1, int(t.width * sq)), t.height))
        im.alpha_composite(t, (int(46 - t.width / 2), 22))
        _suit(im, s, c, 46, 22 + 80 + 18, 38, False)

    index(img)
    r = img.rotate(180)
    index(r)
    img = r.rotate(180)
    L, M, R = .31, .5, .69
    top, bot = .19, .81
    lay = {2: [(M, top), (M, bot)], 3: [(M, top), (M, .5), (M, bot)],
           4: [(L, top), (R, top), (L, bot), (R, bot)],
           5: [(L, top), (R, top), (M, .5), (L, bot), (R, bot)],
           6: [(L, top), (R, top), (L, .5), (R, .5), (L, bot), (R, bot)],
           7: [(L, top), (R, top), (M, .345), (L, .5), (R, .5), (L, bot), (R, bot)],
           8: [(L, top), (R, top), (M, .345), (L, .5), (R, .5), (M, .655), (L, bot), (R, bot)],
           9: [(L, top), (R, top), (L, .397), (R, .397), (M, .5), (L, .603), (R, .603), (L, bot), (R, bot)],
           10: [(L, top), (R, top), (M, .295), (L, .397), (R, .397), (L, .603), (R, .603), (M, .705), (L, bot), (R, bot)]}
    if rank == 1:
        _suit(img, s, c, W / 2, H / 2, 150, False)
    elif rank <= 10:
        for x, y in lay[rank]:
            _suit(img, s, c, x * W, y * H, 84, y > .5)
    if rounded:
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, W - 1, H - 1), radius=26, fill=255)
        img.putalpha(mask)
    return img


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    card(7, 3, rounded=False).convert("RGB").save(os.path.join(out, "fake_7d.png"))
    card(5, 0).save(os.path.join(out, "real_5s.png"))
