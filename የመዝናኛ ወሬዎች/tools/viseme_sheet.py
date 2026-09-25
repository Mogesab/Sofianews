"""
Contact sheet of every mouth shape the lip-sync can produce, drawn on the current presenter photo.
Use it to check the shapes look right after you change the presenter or tweak the renderer.

    python tools/viseme_sheet.py            ->  output/viseme_sheet.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
import numpy as np
from studio import config, presenters
from studio.face import FaceAnimator
from datetime import date

SHAPES = [
    ("rest / closed", {}),
    ("m b p  (lips shut)", dict(mpress=1.0)),
    ("f v  (lip under teeth)", dict(mtuck=1.0, mteeth=1.0, mopen=0.12, mspread=0.1)),
    ("th  (tongue on teeth)", dict(mtongue=1.0, mteeth=0.9, mopen=0.15, mspread=0.1)),
    ("t d n l", dict(mtongue=0.85, mteeth=0.5, mopen=0.14)),
    ("s z", dict(mteeth=1.0, mopen=0.08, mspread=0.35)),
    ("sh ch j", dict(mround=0.75, mspread=-0.25, mopen=0.16, mteeth=0.6)),
    ("r", dict(mround=0.55, mspread=-0.2, mopen=0.24)),
    ("oo  w", dict(mround=1.0, mspread=-0.75, mopen=0.2)),
    ("aw  oh", dict(mround=0.55, mspread=-0.3, mopen=0.8)),
    ("ah  (father)", dict(mopen=1.0, mteeth=0.35)),
    ("uh  (but)", dict(mopen=0.6, mteeth=0.25)),
    ("a  (cat)", dict(mopen=0.85, mspread=0.4, mteeth=0.5)),
    ("eh  (bed)", dict(mopen=0.58, mspread=0.3, mteeth=0.4)),
    ("ih  (sit)", dict(mopen=0.32, mspread=0.4, mteeth=0.5)),
    ("ee  (see)", dict(mopen=0.2, mspread=0.9, mteeth=0.75)),
]


def main():
    cfg = config.load()
    img, lm = presenters.resolve(cfg, config.ASSETS, date.today())
    fa = FaceAnimator(Path(img), Path(lm), int(cfg["width"]), int(cfg["height"]))
    x0, y0, x1, y1 = fa.m_box
    cx = int(fa.m_c[0])
    hw = int(fa.m_a * 2.0)
    top, bot = int(fa.m_c[1] - fa.m_a * 1.2), int(fa.m_c[1] + fa.m_a * 2.6)
    tiles = []
    for name, p in SHAPES:
        f = fa.render(dict(p))
        c = f[top:bot, cx - hw:cx + hw]
        c = cv2.resize(c, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)
        bar = np.full((34, c.shape[1], 3), 24, np.uint8)
        cv2.putText(bar, name, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, c]))
    cols = 4
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    sheet = np.vstack(rows)
    config.OUTPUT.mkdir(exist_ok=True)
    out = config.OUTPUT / "viseme_sheet.png"
    cv2.imwrite(str(out), sheet)
    print("wrote", out)


if __name__ == "__main__":
    main()
