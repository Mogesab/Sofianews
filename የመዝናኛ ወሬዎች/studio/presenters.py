"""
Picks which presenter photo (and its matching landmarks) to use for today.

Three modes, chosen automatically by what is set / on disk:

  * config.json has "presenter_today": "<name>" and that pair exists in
    assets/presenters/ -> always use that one photo, until you change the
    setting. This is the easiest way to hand-pick "today's suit" yourself.
  * assets/presenters/ has one or more <name>.png + <name>.json pairs (and no
    presenter_today override) -> rotate through them, one per day (see
    add_presenter.py to create pairs)
  * assets/presenters/ is empty or missing
    -> the original single assets/<presenter_image> + assets/face.json,
       exactly as before. Nothing changes for anyone who does not use this.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def _pairs(folder: Path) -> list[tuple[Path, Path]]:
    if not folder.is_dir():
        return []
    out = []
    for img in sorted(folder.iterdir()):
        if img.suffix.lower() not in IMG_EXT:
            continue
        j = img.with_suffix(".json")
        if j.exists():
            out.append((img, j))
    return out


def _find(folder: Path, name: str) -> tuple[Path, Path] | None:
    for ext in IMG_EXT:
        img = folder / f"{name}{ext}"
        j = folder / f"{name}.json"
        if img.exists() and j.exists():
            return img, j
    return None


def resolve(cfg: dict, assets: Path, today: date | None = None) -> tuple[Path, Path]:
    """Returns (image_path, landmarks_path) to render with today."""
    today_name = str(cfg.get("presenter_today") or "").strip()
    if today_name:
        hit = _find(assets / "presenters", today_name)
        if hit:
            return hit
    pairs = _pairs(assets / "presenters") if cfg.get("presenter_rotation", True) else []
    if not pairs:
        return assets / cfg["presenter_image"], assets / "face.json"
    today = today or date.today()
    i = today.toordinal() % len(pairs)
    return pairs[i]


def kind(landmarks: Path) -> str:
    """'robot' for the animated robot anchor (tools/build_robot.py), else 'human'."""
    import json
    try:
        return str(json.loads(Path(landmarks).read_text(encoding="utf-8")).get("kind", "human"))
    except Exception:
        return "human"


def list_presenters(assets: Path) -> list[str]:
    return [img.stem for img, _ in _pairs(assets / "presenters")]
