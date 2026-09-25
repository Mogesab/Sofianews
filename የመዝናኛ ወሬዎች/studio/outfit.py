"""
Daily wardrobe for the robot anchor: a different suit colour + cloth pattern
and a different tie colour + pattern every day.

The suit and tie are re-dyed on the photo itself, keeping every fold, shadow
and highlight of the original cloth (the new colour is scaled by the original
brightness), then a woven pattern is laid on top. Because this happens on the
still photo before any animation, the new clothes move exactly like the old.

    config.json  "outfit_today": ""          -> automatic, changes every day
                 "outfit_today": "7"         -> force outfit number 7
                 "outfit_rotation": false    -> always the original navy suit + blue tie
"""
from __future__ import annotations

from datetime import date

import cv2
import numpy as np

# BGR colours
SUITS = [
    ("charcoal", (58, 56, 54)),
    ("navy", (74, 38, 18)),
    ("black", (30, 28, 27)),
    ("mid grey", (118, 114, 110)),
    ("burgundy", (38, 26, 92)),
    ("royal blue", (132, 62, 26)),
    ("forest green", (44, 66, 34)),
    ("chocolate brown", (32, 50, 78)),
    ("sand beige", (122, 160, 186)),
    ("plum", (78, 34, 62)),
    ("slate blue", (112, 86, 64)),
]
SUIT_PATTERNS = ["solid", "pinstripe", "chalk stripe", "windowpane", "glen check", "solid", "birdseye"]

TIES = [
    ("crimson", (40, 30, 190)),
    ("gold", (40, 160, 214)),
    ("silver", (190, 188, 184)),
    ("emerald", (90, 140, 30)),
    ("burgundy", (48, 26, 118)),
    ("sky blue", (220, 160, 90)),
    ("purple", (140, 50, 100)),
    ("orange", (30, 110, 225)),
    ("pink", (170, 120, 225)),
    ("teal", (140, 130, 20)),
    ("navy", (96, 40, 16)),
    ("black", (34, 32, 32)),
    ("mustard", (30, 150, 190)),
]
TIE_PATTERNS = ["solid", "stripes", "dots", "micro check", "stripes", "solid", "wide stripes"]


def _lum(bgr) -> float:
    b, g, r = bgr
    return 0.114 * b + 0.587 * g + 0.299 * r


def _dist(a, b) -> float:
    return float(np.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def pick(cfg: dict, today: date | None = None) -> dict | None:
    """Returns today's outfit, or None to keep the photo's own clothes."""
    if not cfg.get("outfit_rotation", True):
        return None
    today = today or date.today()
    forced = str(cfg.get("outfit_today") or "").strip()
    d = int(forced) if forced.lstrip("-").isdigit() else today.toordinal()
    # coprime cycle lengths -> the same combination does not come back for years
    si, sp = d % len(SUITS), (d // 2) % len(SUIT_PATTERNS)
    ti, tp = (d * 5) % len(TIES), (d * 3) % len(TIE_PATTERNS)
    suit, tie = SUITS[si], TIES[ti]
    # never a tie that disappears into the suit, or a tie that is lighter-on-light
    for k in range(len(TIES)):
        tie = TIES[(ti + k) % len(TIES)]
        if _dist(tie[1], suit[1]) > 90 and abs(_lum(tie[1]) - _lum(suit[1])) > 22:
            break
    tie2 = TIES[(TIES.index(tie) + 4) % len(TIES)]      # accent colour for stripes / dots
    if _dist(tie2[1], tie[1]) < 80:
        tie2 = ("silver", (190, 188, 184)) if _lum(tie[1]) < 120 else ("navy", (96, 40, 16))
    suit_pat = SUIT_PATTERNS[sp] if suit[0] not in ("sand beige",) else "solid"
    tie_pat = TIE_PATTERNS[tp]
    name = f"{suit[0]} {suit_pat if suit_pat != 'solid' else ''} suit, {tie[0]} {tie_pat if tie_pat != 'solid' else ''} tie"
    return dict(name=" ".join(name.split()), suit=suit[1], suit_pattern=suit_pat,
                tie=tie[1], tie_accent=tie2[1], tie_pattern=tie_pat)


# ------------------------------------------------------------------------ masks
def _poly(shape, pts):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 255)
    return m > 0


def masks(img: np.ndarray, matte: np.ndarray, lm: dict) -> tuple[np.ndarray, np.ndarray]:
    """Soft (0..1) masks of the suit and the tie on the native photo."""
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    tie_zone = cv2.dilate(_poly((H, W), lm["tie"]["poly"]).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    tie = (h >= 94) & (h <= 130) & (s > 45) & (v > 36) & tie_zone
    tie = cv2.morphologyEx(tie.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    body = matte > 128
    body[: int(lm["head"]["neck_fade"][0]), :] = False
    bluish = (h >= 95) & (h <= 135) & ((s > 35) | (v < 48))
    suit = body & bluish & (v < 140) & ~tie
    hands = _poly((H, W), lm["hands_poly"])
    suit &= ~hands | (s > 60)                      # black robot fingers are neutral; cloth between them is not
    suit &= ~_poly((H, W), lm["neck"]["poly"]) & ~_poly((H, W), lm["laptop"]["poly"])
    suit = cv2.morphologyEx(suit.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    suit = cv2.morphologyEx(suit, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    soft = lambda m: cv2.GaussianBlur(m.astype(np.float32) / 255.0 if m.dtype == np.uint8 else m.astype(np.float32), (0, 0), 0.9)
    suit_m = soft(suit) * (matte.astype(np.float32) / 255.0)
    tie_m = soft(tie.astype(np.uint8) * 255)
    return np.clip(suit_m - tie_m, 0, 1), np.clip(tie_m, 0, 1)


# --------------------------------------------------------------------- patterns
def _lines(coord, period, width):
    d = np.abs(((coord % period) + period) % period - period / 2.0)
    return np.clip((width / 2.0 + 0.5) - (period / 2.0 - d), 0, 1)


def _suit_pattern(kind: str, H: int, W: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if kind == "pinstripe":
        return 0.85 * _lines(xx, 13.0, 1.0)
    if kind == "chalk stripe":
        return 0.55 * cv2.GaussianBlur(_lines(xx, 21.0, 2.0), (0, 0), 0.9)
    if kind == "windowpane":
        return 0.6 * np.maximum(_lines(xx, 38.0, 1.2), _lines(yy, 38.0, 1.2))
    if kind == "glen check":
        small = ((np.floor(xx / 3) + np.floor(yy / 3)) % 2) * 0.22
        return np.clip(small + 0.5 * np.maximum(_lines(xx, 30.0, 1.0), _lines(yy, 30.0, 1.0)), 0, 1)
    if kind == "birdseye":
        return 0.35 * ((np.floor(xx / 2) % 2 == 0) & (np.floor(yy / 2) % 2 == 0)).astype(np.float32)
    return np.zeros((H, W), np.float32)


def _tie_pattern(kind: str, H: int, W: int) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    diag = (xx + yy) / np.sqrt(2.0)
    if kind == "stripes":
        return _lines(diag, 15.0, 4.0)
    if kind == "wide stripes":
        return _lines(diag, 26.0, 8.0)
    if kind == "dots":
        gx, gy = xx % 12.0 - 6.0, (yy + 6.0 * (np.floor(xx / 12.0) % 2)) % 12.0 - 6.0
        return np.clip(2.6 - np.sqrt(gx * gx + gy * gy), 0, 1)
    if kind == "micro check":
        return 0.7 * ((np.floor(xx / 5) + np.floor(yy / 5)) % 2)
    return np.zeros((H, W), np.float32)


def _dye(img: np.ndarray, mask: np.ndarray, colour, pattern: np.ndarray, accent=None) -> np.ndarray:
    m = mask > 0.2
    if not m.any():
        return img
    f = img.astype(np.float32)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref = float(np.median(L[m])) + 1e-3
    shade = np.clip(L / ref, 0.0, 3.0) ** 0.85                     # keep folds, shadows and sheen
    col = np.array(colour, np.float32)[None, None, :]
    new = col * shade[..., None]
    if accent is not None:                                         # tie: second colour for the pattern
        acc = np.array(accent, np.float32)[None, None, :] * shade[..., None]
        new = new * (1 - pattern[..., None]) + acc * pattern[..., None]
    else:                                                          # suit: lighter woven thread
        thread = new * 1.45 + 26.0 * shade[..., None]
        new = new * (1 - pattern[..., None]) + thread * pattern[..., None]
    a = mask[..., None]
    return np.clip(f * (1 - a) + new * a, 0, 255).astype(np.uint8)


def apply(img: np.ndarray, matte: np.ndarray, lm: dict, outfit: dict | None) -> np.ndarray:
    """Re-dresses the native photo. img/matte are the native-size photo and alpha."""
    if not outfit:
        return img
    H, W = img.shape[:2]
    suit_m, tie_m = masks(img, matte, lm)
    out = _dye(img, suit_m, outfit["suit"], _suit_pattern(outfit["suit_pattern"], H, W))
    out = _dye(out, tie_m, outfit["tie"], _tie_pattern(outfit["tie_pattern"], H, W), accent=outfit["tie_accent"])
    return out
