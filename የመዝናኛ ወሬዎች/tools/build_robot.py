"""
Prepares the robot anchor photo for animation.  Run once (already done for
assets/Robot.png), or again after replacing the photo:

    python tools/build_robot.py [path/to/Robot.png]

Writes into assets/presenters/:
    robot.png              the photo
    robot.json             landmarks (jaw plate, eyes, LEDs, hands, laptop, globe ...)
    robot.matte.png        alpha matte of the robot (so it can move over the studio)
    robot.plate.png        the studio with the robot and the globe lettering removed
    robot.globe_text.png   the "GLOBAL NEWS LIVE" lettering as an RGBA sprite

All landmark numbers are pixels on the native 941x1672 photo.  If you swap in a
different robot photo, re-measure them in LANDMARKS below.
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "presenters"

LANDMARKS = {
    "_comment": "Robot anchor landmarks, pixels on the native photo. Made by tools/build_robot.py.",
    "kind": "robot",
    "image": "robot.png",
    "size": [941, 1672],
    "layers": {"matte": "robot.matte.png", "plate": "robot.plate.png", "globe_text": "robot.globe_text.png"},
    "sprite_box": [8, 248, 780, 1100],
    # graphics printed on the photo, erased from the studio plate (the app draws its own, animated)
    "erase": [[58, 94, 212, 150], [0, 1350, 941, 1538]],
    "head": {"center": [410, 418], "radii": [152, 176], "pivot": [410, 588], "neck_fade": [566, 618]},
    # lower face plate that drops like a mechanical jaw (between the two cheek seams)
    "jaw": {
        "poly": [[380, 518], [395, 519.5], [410, 520.5], [425, 519.5], [445, 518], [462, 535], [455, 551],
                 [448, 558], [435, 566], [415, 569], [395, 566], [378, 560], [367, 551], [362, 535]],
        "mouth": [[380, 518], [445, 518]],
        "drop": 9.5,
    },
    "eyes": [{"center": [364, 424], "half_width": 27, "iris_r": 12},
             {"center": [466, 426], "half_width": 27, "iris_r": 12}],
    "headphones": [[266, 370, 312, 474], [508, 370, 554, 474]],
    "torso": {"pivot": [410, 1065]},
    "hands": {"box": [160, 928, 562, 1084], "center": [360, 1012], "fade": 42},
    "desk_anchor": [990, 1070],
    "reflection": {"box": [150, 1080, 580, 1225]},
    "laptop": {"poly": [[720, 879], [941, 859], [941, 1114], [700, 1119], [642, 1122], [496, 1091],
                        [496, 1078], [661, 1075]]},
    "neck": {"poly": [[332, 532], [492, 532], [482, 628], [344, 628]]},
    "tie": {"poly": [[382, 621], [436, 621], [428, 660], [422, 675], [428, 700], [434, 800], [437, 880], [429, 925],
                     [418, 966], [410, 980], [400, 966], [386, 925], [372, 880], [377, 800], [391, 700], [396, 675],
                     [390, 660]]},
    "hands_poly": [[208, 972], [300, 930], [430, 930], [512, 972], [516, 1062], [470, 1080], [250, 1080],
                   [205, 1062]],
    "globe": {
        "center": [815, 468], "radius": 282, "y_range": [190, 712], "speed_deg": 9.0,
        "banner": [[618, 457], [941, 449], [941, 543], [610, 551]],
        "letters": [[648, 396, 856, 452], [768, 554, 870, 602]],
        # per-row limits of clean globe texture (x_min over rows the headphone covers)
        "clean_x_min": [[370, 474, 556]],
    },
}


def _poly(shape, pts, val=255):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], val)
    return m


def matte(img: np.ndarray, lm: dict) -> np.ndarray:
    H, W = img.shape[:2]
    mask = np.full((H, W), cv2.GC_BGD, np.uint8)
    pr = np.zeros((H, W), np.uint8)
    cv2.ellipse(pr, (410, 420), (140, 165), 0, 0, 360, 255, -1)
    body = [(330, 560), (490, 560), (600, 585), (700, 640), (745, 720), (752, 860), (745, 1080), (560, 1080),
            (170, 1080), (40, 1060), (22, 960), (30, 850), (70, 740), (120, 640), (250, 580)]
    cv2.fillPoly(pr, [np.array(body, np.int32)], 255)
    big = cv2.dilate(pr, np.ones((41, 41), np.uint8))
    small = cv2.erode(pr, np.ones((41, 41), np.uint8))
    mask[big > 0] = cv2.GC_PR_BGD
    mask[pr > 0] = cv2.GC_PR_FGD
    mask[small > 0] = cv2.GC_FGD
    for (cx, cy) in ((292, 422), (527, 424)):      # headphones incl. their glowing rims
        hp = np.zeros((H, W), np.uint8)
        cv2.ellipse(hp, (cx, cy), (22, 50), 0, 0, 360, 255, -1)
        mask[hp > 0] = cv2.GC_FGD
        cv2.ellipse(hp, (cx, cy), (30, 58), 0, 0, 360, 255, -1)
        mask[(hp > 0) & (mask != cv2.GC_FGD)] = cv2.GC_PR_FGD
    mask[1086:, :] = cv2.GC_BGD
    mask[690:1090, 756:] = cv2.GC_BGD
    mask[_poly((H, W), lm["laptop"]["poly"]) > 0] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, None, bgd, fgd, 8, cv2.GC_INIT_WITH_MASK)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # keep the main body, fill holes
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = np.where(lab == keep, 255, 0).astype(np.uint8)
    inv = cv2.bitwise_not(fg)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 4000 and x > 0 and y > 0 and x + w < W and y + h < H:
            fg[lab == i] = 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # soft 1-2 px edge
    a = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 1.1)
    return np.clip(a, 0, 255).astype(np.uint8)


def globe_text(img: np.ndarray, lm: dict):
    H, W = img.shape[:2]
    g = lm["globe"]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V, S = hsv[..., 2].astype(np.float32), hsv[..., 1].astype(np.float32)
    letters = np.zeros((H, W), np.float32)
    for x0, y0, x1, y1 in g["letters"]:
        box = np.clip((V[y0:y1, x0:x1] - 150) / 70, 0, 1) * np.clip((150 - S[y0:y1, x0:x1]) / 60, 0, 1)
        letters[y0:y1, x0:x1] = np.maximum(letters[y0:y1, x0:x1], box)
    banner = cv2.GaussianBlur(_poly((H, W), g["banner"]).astype(np.float32), (0, 0), 0.8) / 255.0
    alpha = np.clip(np.maximum(letters, banner), 0, 1)
    sprite = np.dstack([img, (alpha * 255).astype(np.uint8)])
    # remove the lettering from the studio plate so the globe can turn under it
    hole = ((letters > 0.08) | (banner > 0.02)).astype(np.uint8) * 255
    hole = cv2.dilate(hole, np.ones((5, 5), np.uint8))
    return sprite, hole


def smooth_fill(img: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Fill the hole with a soft, streak-free guess of the background: inpaint at
    quarter size (no radial smears), scale back up, blend in with a feathered edge.
    Only thin slivers of it are ever seen, next to the moving robot."""
    H, W = img.shape[:2]
    f = 4
    small = cv2.resize(img, (W // f, H // f), interpolation=cv2.INTER_AREA)
    hs = cv2.resize(hole, (W // f, H // f), interpolation=cv2.INTER_NEAREST)
    hs = cv2.dilate(hs, np.ones((3, 3), np.uint8))
    fill = cv2.inpaint(small, hs, 6, cv2.INPAINT_NS)
    fill = cv2.GaussianBlur(fill, (0, 0), 1.2)
    fill = cv2.resize(fill, (W, H), interpolation=cv2.INTER_CUBIC)
    a = cv2.GaussianBlur(hole.astype(np.float32) / 255.0, (0, 0), 2.0)[..., None]
    a = np.maximum(a, (hole > 0)[..., None])
    return np.clip(img * (1 - a) + fill * a, 0, 255).astype(np.uint8)


def main(src: Path):
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"cannot read {src}")
    if tuple(img.shape[1::-1]) != tuple(LANDMARKS["size"]):
        print(f"WARNING: photo is {img.shape[1]}x{img.shape[0]}, landmarks were measured on "
              f"{LANDMARKS['size'][0]}x{LANDMARKS['size'][1]} - re-measure LANDMARKS in this script.")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, OUT / "robot.png")
    print("cutting the robot out of the studio …")
    a = matte(img, LANDMARKS)
    cv2.imwrite(str(OUT / "robot.matte.png"), a)
    sprite, hole = globe_text(img, LANDMARKS)
    cv2.imwrite(str(OUT / "robot.globe_text.png"), sprite)
    print("building the empty studio plate …")
    rob = cv2.dilate((a > 20).astype(np.uint8) * 255, np.ones((13, 13), np.uint8))
    for x0, y0, x1, y1 in LANDMARKS.get("erase", []):
        rob[y0:y1, x0:x1] = 255
    plate = smooth_fill(img, rob)
    plate = cv2.inpaint(plate, hole, 5, cv2.INPAINT_TELEA)
    cv2.imwrite(str(OUT / "robot.plate.png"), plate)
    (OUT / "robot.json").write_text(json.dumps(LANDMARKS, indent=1), encoding="utf-8")
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "assets" / "Robot.png")
