"""
Add a new presenter photo (a different suit, or a different but similarly-posed
person) to the daily rotation.

    python tools/add_presenter.py path/to/new_photo.jpg tuesday-navy-suit

What it does:
  1. Detects the two eyes in the new photo.
  2. Works out the rotation/scale/position that maps the REFERENCE photo's eyes
     onto this photo's eyes, and applies that same transform to every landmark
     in assets/face.json (mouth, brows, head circle, hands box, torso box...).
  3. Saves assets/presenters/<name>.png and assets/presenters/<name>.json.
  4. Renders assets/presenters/<name>_check.png with the landmarks drawn on
     top, so you can SEE whether they line up before trusting it.

This only works well when the new photo is framed like the reference one:
similar zoom, shoulders-up, facing the camera, similar lighting. It is not a
full face-tracking model - it is one rotation+scale transform derived from the
eyes, so if the new photo is a very different crop or angle the overlay will
show it (mouth box off the mouth, etc.) and you should retake/recrop the photo
rather than trust it.

THE EASIER ALTERNATIVE: if you just want a different colour suit on the SAME
person, it is much more reliable to edit the ORIGINAL photo (change only the
suit/tie colour, keep the face and crop identical - any photo editor or an
AI image-edit tool can do this) and use it as-is with the existing
assets/face.json. No landmark work needed, and it can never be misaligned.
Use this tool only when you actually have a different photo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from studio import config

REF_IMAGE = config.ASSETS / "journalist.png"
REF_LM = config.ASSETS / "face.json"


def _detect_eyes(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    face_c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    faces = face_c.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])          # largest face
    roi = gray[y:y + h, x:x + w]
    eyes = eye_c.detectMultiScale(roi, 1.1, 5, minSize=(int(w * 0.08), int(w * 0.08)))
    if len(eyes) < 2:
        return None
    eyes = sorted(eyes, key=lambda e: -(e[2] * e[3]))[:2]        # two biggest
    centers = [np.array([x + ex + ew / 2.0, y + ey + eh / 2.0]) for ex, ey, ew, eh in eyes]
    centers.sort(key=lambda c: c[0])                             # left, right (image coords)
    return centers[0], centers[1]


def _similarity(src_a, src_b, dst_a, dst_b):
    """2x2 rotation+scale matrix R and translation t such that dst ~= R @ src + t,
    fit from the two eye pairs (a Procrustes fit on two points)."""
    sv, dv = src_b - src_a, dst_b - dst_a
    s_len, d_len = np.linalg.norm(sv), np.linalg.norm(dv)
    scale = d_len / s_len
    ang = np.arctan2(dv[1], dv[0]) - np.arctan2(sv[1], sv[0])
    c, s = np.cos(ang), np.sin(ang)
    R = scale * np.array([[c, -s], [s, c]])
    t = dst_a - R @ src_a
    return R, t, np.degrees(ang), scale


def _tf_point(R, t, p):
    return (R @ np.array(p, float) + t).tolist()


def _tf_box(R, t, box):
    x0, y0, x1, y1 = box
    pts = [_tf_point(R, t, (x0, y0)), _tf_point(R, t, (x1, y0)),
           _tf_point(R, t, (x1, y1)), _tf_point(R, t, (x0, y1))]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def transform_landmarks(ref: dict, R, t, ang_delta: float, scale: float, new_size) -> dict:
    lm = json.loads(json.dumps(ref))                             # deep copy
    lm["size"] = list(new_size)
    lm["image"] = None

    m = lm["mouth"]
    m["center"] = _tf_point(R, t, m["center"])
    m["half_width"] *= scale
    m["sag"] *= scale
    m["angle_deg"] += ang_delta

    for e in lm["eyes"]:
        e["center"] = _tf_point(R, t, e["center"])
        e["iris"] = _tf_point(R, t, e["iris"])
        e["half_width"] *= scale
        e["iris_r"] *= scale
        e["angle_deg"] += ang_delta

    lm["brows"] = [_tf_point(R, t, b) for b in lm["brows"]]

    h = lm["head"]
    h["center"] = _tf_point(R, t, h["center"])
    h["radii"] = [r * scale for r in h["radii"]]
    h["pivot"] = _tf_point(R, t, h["pivot"])

    lm["torso"]["box"] = _tf_box(R, t, ref["torso"]["box"])
    lm["hands"]["box"] = _tf_box(R, t, ref["hands"]["box"])
    lm["hands"]["fade"] = [f * scale for f in ref["hands"]["fade"]]
    if "globe" in ref:
        lm["globe"]["box"] = _tf_box(R, t, ref["globe"]["box"])
        lm["globe"]["fade"] = ref["globe"]["fade"] * scale
    lm["rois"]["head"] = _tf_box(R, t, ref["rois"]["head"])

    ref_head_y = ref["head"]["center"][1]
    new_head_y = lm["head"]["center"][1]
    shift = new_head_y - ref_head_y * scale
    lm["head"]["neck_fade"] = [nf * scale + shift for nf in ref["head"]["neck_fade"]]
    return lm


def draw_check(img: np.ndarray, lm: dict) -> np.ndarray:
    out = img.copy()
    def box(b, col):
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
    box(lm["torso"]["box"], (255, 180, 0))
    box(lm["hands"]["box"], (0, 200, 255))
    box(lm["rois"]["head"], (0, 255, 0))
    m = lm["mouth"]
    cv2.ellipse(out, (int(m["center"][0]), int(m["center"][1])),
                (int(m["half_width"]), int(m["half_width"] * 0.45)), m["angle_deg"], 0, 360, (0, 0, 255), 2)
    for e in lm["eyes"]:
        cv2.circle(out, (int(e["center"][0]), int(e["center"][1])), int(e["half_width"]), (255, 0, 255), 2)
        cv2.circle(out, (int(e["iris"][0]), int(e["iris"][1])), 3, (255, 255, 255), -1)
    for b in lm["brows"]:
        cv2.circle(out, (int(b[0]), int(b[1])), 4, (0, 255, 255), -1)
    h = lm["head"]
    cv2.ellipse(out, (int(h["center"][0]), int(h["center"][1])),
                (int(h["radii"][0]), int(h["radii"][1])), 0, 0, 360, (0, 255, 0), 2)
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("usage: python tools/add_presenter.py <photo> <name>")
        return
    photo, name = Path(sys.argv[1]), sys.argv[2]
    if not photo.exists():
        print(f"Not found: {photo}")
        return

    ref_img = cv2.imread(str(REF_IMAGE))
    ref_lm = json.loads(REF_LM.read_text())
    new_img = cv2.imread(str(photo))
    if new_img is None:
        print("Could not read that image (bad format?).")
        return

    ref_eyes = _detect_eyes(cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY))
    new_eyes = _detect_eyes(cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY))
    if ref_eyes is None or new_eyes is None:
        print("Could not detect two eyes in " + ("the new photo." if ref_eyes else "the reference photo (unexpected)."))
        print("Use a clear, front-facing, well-lit headshot, similar in framing to assets/journalist.png.")
        return

    R, t, ang_delta, scale = _similarity(*ref_eyes, *new_eyes)
    lm = transform_landmarks(ref_lm, R, t, ang_delta, scale, (new_img.shape[1], new_img.shape[0]))

    out_dir = config.ASSETS / "presenters"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / f"{name}{photo.suffix.lower()}"
    lm_out = out_dir / f"{name}.json"
    cv2.imwrite(str(img_out), new_img)
    lm_out.write_text(json.dumps(lm, indent=1))

    check = draw_check(new_img, lm)
    check_path = out_dir / f"{name}_check.png"
    cv2.imwrite(str(check_path), check)

    print(f"Saved {img_out.name} + {lm_out.name} to assets/presenters/")
    print(f"-> LOOK AT {check_path} FIRST. Green circle = head, red = mouth, magenta = eyes,")
    print("   yellow dots = brows, orange box = torso, cyan box = hands.")
    print("   If those line up with the photo, it's good to use. If not, delete the two files")
    print("   above (or just don't add more) and try a photo framed more like the original -")
    print("   forward-facing, shoulders up, similar distance from the camera.")


if __name__ == "__main__":
    main()
