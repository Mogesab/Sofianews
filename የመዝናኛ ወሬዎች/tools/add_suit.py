"""
Add a new photo of the SAME presenter (only the suit/tie colour changed, face
and framing identical) in one command - no landmark work needed.

    python tools/add_suit.py path/to/new_photo.png tuesday-navy-suit
    python tools/add_suit.py path/to/new_photo.png tuesday-navy-suit --today

What it does:
  1. Checks the new photo is the same pixel size as the reference photo
     (assets/journalist.png). If it isn't, this tool refuses - a size
     mismatch means the crop changed and the landmarks would be wrong
     (use tools/add_presenter.py instead for a genuinely different photo).
  2. Copies the photo to assets/presenters/<name>.<ext> and copies the
     reference landmarks (assets/face.json) alongside it as <name>.json,
     since the face never moved, those landmarks are still exactly right.
  3. With --today, also sets "presenter_today" in config.json to <name>, so
     the very next broadcast uses it. Without --today, just drop more suits
     in and they join the automatic daily rotation.

This is the fast path for "same reporter, different suit". If you're using
an AI image editor, ask it to change ONLY the suit/tie colour and keep the
face, crop and framing pixel-for-pixel identical - that is what makes this
shortcut safe.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from studio import config

REF_IMAGE = config.ASSETS / "journalist.png"
REF_LM = config.ASSETS / "face.json"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    set_today = "--today" in sys.argv
    if len(args) < 2:
        print(__doc__)
        print("usage: python tools/add_suit.py <photo> <name> [--today]")
        return
    photo, name = Path(args[0]), args[1]
    if not photo.exists():
        print(f"Not found: {photo}")
        return

    ref = cv2.imread(str(REF_IMAGE))
    new = cv2.imread(str(photo))
    if new is None:
        print("Could not read that image (bad format?).")
        return
    if ref.shape[:2] != new.shape[:2]:
        rw, rh = ref.shape[1], ref.shape[0]
        nw, nh = new.shape[1], new.shape[0]
        print(f"That photo is {nw}x{nh}, but the reference is {rw}x{rh}.")
        print("Same-suit-shortcut only works when the crop is pixel-for-pixel")
        print("identical to assets/journalist.png (only colours changed).")
        print("If this is genuinely a different photo/crop, use instead:")
        print(f"    python tools/add_presenter.py {photo} {name}")
        return

    out_dir = config.ASSETS / "presenters"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_out = out_dir / f"{name}{photo.suffix.lower()}"
    lm_out = out_dir / f"{name}.json"
    shutil.copyfile(photo, img_out)
    lm = json.loads(REF_LM.read_text(encoding="utf-8"))
    lm["image"] = img_out.name
    lm_out.write_text(json.dumps(lm, indent=1), encoding="utf-8")
    print(f"Saved {img_out.name} + {lm_out.name} to assets/presenters/")

    if set_today:
        cfg_path = config.ROOT / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        cfg["presenter_today"] = name
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f'Set "presenter_today": "{name}" in config.json - the next broadcast uses this suit.')
    else:
        print("Added to the rotation folder. Run again with --today to force it for the next broadcast,")
        print('or set "presenter_today" in config.json yourself.')


if __name__ == "__main__":
    main()
