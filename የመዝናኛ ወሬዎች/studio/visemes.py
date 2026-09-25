"""
Timed phonemes -> mouth-shape curves (one value per video frame).

Every sound has a *target* for a handful of articulator parameters and a
*dominance* (how strongly, and over how long, it insists on that target).
The mouth at any instant is the dominance-weighted average of all sounds nearby
(Cohen & Massaro's coarticulation model). That single idea produces most of what
makes speech look right:

  * m / b / p  -> lips pressed shut (dominant), while the *width/rounding* of the
                  neighbouring vowel bleeds through ("moo" is rounded from the m)
  * f / v      -> lower lip tucked under the upper teeth
  * oo / w / sh / r -> rounded, protruded lips
  * ee / s     -> spread lips, teeth showing
  * t d n l th -> tongue up behind / between the teeth, jaw barely moves
  * k g h      -> almost no shape of their own; the vowels around them decide
  * vowels     -> jaw opening in proportion to how open the vowel is

Parameters (all per frame):
  jaw    0..1     how far the jaw drops
  wide  -1..1     lips pursed (-) .. spread (+)
  round  0..1     lip protrusion / rounding
  press  0..1     lips pressed together (bilabials)
  tuck   0..1     lower lip tucked behind upper teeth (f, v)
  tongue 0..1     tongue tip visible / raised behind the teeth
  teeth  0..1     upper teeth exposed
"""
from __future__ import annotations

import numpy as np

from .align import Seg
from .phonemes import VOWELS

PARAMS = ["jaw", "wide", "round", "press", "tuck", "tongue", "teeth"]
PI = {p: i for i, p in enumerate(PARAMS)}
NP = len(PARAMS)

# vowel targets: (jaw, wide, round, teeth)   [a diphthong lists two]
_VOW = {
    "AA": [(1.00, 0.00, 0.00, 0.35)],
    "AE": [(0.85, 0.40, 0.00, 0.50)],
    "AH": [(0.60, 0.05, 0.00, 0.25)],
    "AO": [(0.80, -0.30, 0.55, 0.20)],
    "EH": [(0.58, 0.30, 0.00, 0.40)],
    "ER": [(0.32, -0.05, 0.35, 0.15)],
    "IH": [(0.32, 0.40, 0.00, 0.50)],
    "IY": [(0.20, 0.90, 0.00, 0.75)],
    "UH": [(0.32, -0.35, 0.60, 0.15)],
    "UW": [(0.20, -0.75, 1.00, 0.05)],
    "OW": [(0.55, -0.40, 0.75, 0.10), (0.32, -0.70, 1.00, 0.05)],
    "AW": [(0.95, 0.00, 0.00, 0.30), (0.32, -0.60, 0.90, 0.05)],
    "AY": [(0.95, 0.05, 0.00, 0.30), (0.30, 0.60, 0.00, 0.60)],
    "EY": [(0.55, 0.30, 0.00, 0.40), (0.25, 0.75, 0.00, 0.65)],
    "OY": [(0.75, -0.30, 0.60, 0.20), (0.30, 0.50, 0.00, 0.50)],
}

# consonants: param -> (target, alpha, theta)   -- theta = decay per second outside the sound
def _c(**kw):
    return {k: tuple(v) for k, v in kw.items()}

_CONS = {
    # bilabials: closure is what the eye looks for; vowel supplies width/rounding
    "P": _c(press=(1.0, 3.2, 32), jaw=(0.08, 0.55, 16), wide=(0, 0.12, 14), round=(0, 0.12, 14), tongue=(0, 0.1, 12), teeth=(0, 0.6, 20)),
    "B": _c(press=(1.0, 3.2, 32), jaw=(0.08, 0.55, 16), wide=(0, 0.12, 14), round=(0, 0.12, 14), tongue=(0, 0.1, 12), teeth=(0, 0.6, 20)),
    "M": _c(press=(1.0, 3.2, 32), jaw=(0.06, 0.55, 16), wide=(0, 0.12, 14), round=(0, 0.12, 14), tongue=(0, 0.1, 12), teeth=(0, 0.6, 20)),
    # labiodentals
    "F": _c(tuck=(1.0, 3.0, 30), teeth=(1.0, 1.2, 24), jaw=(0.12, 0.8, 16), wide=(0.10, 0.30, 14), round=(0, 0.25, 14)),
    "V": _c(tuck=(1.0, 3.0, 30), teeth=(1.0, 1.2, 24), jaw=(0.12, 0.8, 16), wide=(0.10, 0.30, 14), round=(0, 0.25, 14)),
    # interdentals
    "TH": _c(tongue=(1.0, 1.4, 26), teeth=(0.9, 1.0, 22), jaw=(0.15, 0.7, 16), wide=(0.10, 0.25, 14), round=(0, 0.25, 14)),
    "DH": _c(tongue=(1.0, 1.4, 26), teeth=(0.9, 1.0, 22), jaw=(0.15, 0.7, 16), wide=(0.10, 0.25, 14), round=(0, 0.25, 14)),
    # alveolars
    "T": _c(tongue=(0.85, 1.0, 22), teeth=(0.5, 0.6, 18), jaw=(0.14, 0.5, 15), wide=(0, 0.18, 14), round=(0, 0.18, 14)),
    "D": _c(tongue=(0.85, 1.0, 22), teeth=(0.5, 0.6, 18), jaw=(0.14, 0.5, 15), wide=(0, 0.18, 14), round=(0, 0.18, 14)),
    "N": _c(tongue=(0.85, 1.0, 22), teeth=(0.5, 0.6, 18), jaw=(0.14, 0.5, 15), wide=(0, 0.18, 14), round=(0, 0.18, 14)),
    "L": _c(tongue=(0.90, 1.0, 22), teeth=(0.5, 0.6, 18), jaw=(0.20, 0.5, 15), wide=(0, 0.18, 14), round=(0, 0.18, 14)),
    # sibilants: teeth together, lips a little spread
    "S": _c(teeth=(1.0, 1.3, 26), jaw=(0.08, 1.0, 18), wide=(0.35, 0.55, 14), round=(0, 0.35, 14), tongue=(0.15, 0.4, 14)),
    "Z": _c(teeth=(1.0, 1.3, 26), jaw=(0.08, 1.0, 18), wide=(0.35, 0.55, 14), round=(0, 0.35, 14), tongue=(0.15, 0.4, 14)),
    # post-alveolars: lips pushed forward
    "SH": _c(round=(0.75, 1.2, 20), wide=(-0.25, 0.9, 16), jaw=(0.16, 0.8, 16), teeth=(0.6, 0.8, 18), tongue=(0.2, 0.3, 14)),
    "ZH": _c(round=(0.75, 1.2, 20), wide=(-0.25, 0.9, 16), jaw=(0.16, 0.8, 16), teeth=(0.6, 0.8, 18), tongue=(0.2, 0.3, 14)),
    "CH": _c(round=(0.70, 1.2, 20), wide=(-0.20, 0.9, 16), jaw=(0.16, 0.8, 16), teeth=(0.6, 0.8, 18), tongue=(0.3, 0.4, 14)),
    "JH": _c(round=(0.70, 1.2, 20), wide=(-0.20, 0.9, 16), jaw=(0.16, 0.8, 16), teeth=(0.6, 0.8, 18), tongue=(0.3, 0.4, 14)),
    # approximants
    "R": _c(round=(0.55, 1.0, 16), wide=(-0.20, 0.85, 14), jaw=(0.24, 0.5, 14), teeth=(0.1, 0.4, 14)),
    "W": _c(round=(1.0, 1.5, 20), wide=(-0.85, 1.3, 18), jaw=(0.14, 0.9, 16), teeth=(0, 0.6, 14)),
    "Y": _c(wide=(0.60, 0.9, 16), jaw=(0.16, 0.6, 15), teeth=(0.5, 0.6, 14), round=(0, 0.4, 14)),
    # velars / glottal: almost no shape of their own
    "K": _c(jaw=(0.22, 0.30, 14), tongue=(0.05, 0.1, 12), wide=(0, 0.06, 12), round=(0, 0.06, 12), teeth=(0.2, 0.2, 12)),
    "G": _c(jaw=(0.22, 0.30, 14), tongue=(0.05, 0.1, 12), wide=(0, 0.06, 12), round=(0, 0.06, 12), teeth=(0.2, 0.2, 12)),
    "NG": _c(jaw=(0.20, 0.30, 14), tongue=(0.05, 0.1, 12), wide=(0, 0.06, 12), round=(0, 0.06, 12), teeth=(0.2, 0.2, 12)),
    "HH": _c(jaw=(0.30, 0.06, 12), wide=(0, 0.04, 12), round=(0, 0.04, 12)),
}
# rest position while silent
_REST = dict(jaw=(0.0, 0.7, 16), wide=(0.0, 0.7, 14), round=(0.0, 0.7, 14), press=(0.0, 0.5, 14),
             tuck=(0.0, 0.6, 14), tongue=(0.0, 0.6, 14), teeth=(0.0, 0.7, 14))
_INHALE = dict(jaw=(0.16, 0.9, 18), wide=(0.0, 0.5, 14), round=(0.0, 0.5, 14), press=(0.0, 0.8, 14),
               tuck=(0.0, 0.6, 14), tongue=(0.0, 0.3, 14), teeth=(0.0, 0.5, 14))


def _vowel_params(sym: str, stress: int, gain: float, intensity: float):
    """-> list of dicts param -> (target, alpha, theta), one per vowel sub-target"""
    outs = []
    red = 1.0 if stress else 0.74                          # unstressed vowels are smaller
    for (jaw, wide, rnd, teeth) in _VOW[sym]:
        g = gain * red
        outs.append(dict(
            jaw=(min(1.0, jaw * g * intensity), 1.0, 13),
            wide=(wide * (0.9 + 0.1 * g) * (0.85 if not stress else 1.0), 1.0, 12),
            round=(min(1.0, rnd * (0.9 + 0.1 * g)), 1.0, 12),
            press=(0.0, 0.5, 32), tuck=(0.0, 0.5, 30),
            tongue=(0.0, 0.5, 12), teeth=(teeth, 0.6, 12)))
    return outs


def _expand(segs: list[Seg], intensity: float, total: float):
    """Segments -> list of (t0, t1, {param: (target, alpha, theta)})"""
    out = []
    last_end = 0.0
    sp_levels = [s.level for s in segs if s.kind == "ph" and s.sym in VOWELS]
    ref_level = float(np.median(sp_levels)) if sp_levels else -6.0
    nsegs = len(segs)
    for k, s in enumerate(segs):
        t0, t1 = s.start, s.end
        if t0 - last_end > 0.03:                            # fill gaps with rest
            out.append((last_end, t0, _REST))
        if s.kind == "pause":
            out.append((t0, t1, _REST))
            dur = t1 - t0
            nxt = segs[k + 1] if k + 1 < nsegs else None
            if nxt is not None and nxt.kind == "ph" and dur >= 0.28:      # a small breath-in before speaking again
                out.append((max(t0, t1 - 0.13), t1, _INHALE))
        else:
            gain = float(np.clip(1.0 + 0.030 * (s.level - ref_level), 0.82, 1.18))
            if s.sym in VOWELS:
                subs = _vowel_params(s.sym, s.stress, gain, intensity)
                if len(subs) == 1:
                    out.append((t0, t1, subs[0]))
                else:
                    mid = t0 + (t1 - t0) * 0.5
                    out.append((t0, mid, subs[0]))
                    out.append((mid, t1, subs[1]))
            else:
                tab = dict(_CONS.get(s.sym) or _REST)
                if "jaw" in tab:
                    tab["jaw"] = (tab["jaw"][0] * intensity, tab["jaw"][1], tab["jaw"][2])
                full = dict(_REST)
                full.update({k2: v for k2, v in tab.items()})
                # anything the consonant does not mention gets a very weak "rest" claim so it never
                # overrides a neighbouring vowel
                for k2 in full:
                    if k2 not in tab:
                        full[k2] = (full[k2][0], 0.06, full[k2][2])
                out.append((t0, t1, full))
        last_end = max(last_end, t1)
    if total - last_end > 0.03:
        out.append((last_end, total, _REST))
    return out


def curves(segs: list[Seg], total: float, fps: int, n_frames: int, lead: float = 0.025,
           intensity: float = 1.0, sub: int = 4) -> dict[str, np.ndarray]:
    """Mouth-parameter curves per video frame."""
    rate = fps * sub
    N = int((total + 1.2) * rate) + 2
    off = int(0.5 * rate)                                   # time origin shift so windows never go negative
    num = np.zeros((NP, N + 2 * off), np.float32)
    den = np.zeros((NP, N + 2 * off), np.float32)
    reach = int(0.32 * rate)
    parts = _expand(segs, intensity, total)
    for (t0, t1, tab) in parts:
        a, b = int(t0 * rate) + off, max(int(t1 * rate) + off, int(t0 * rate) + off + 1)
        lo, hi = max(0, a - reach), min(num.shape[1], b + reach)
        idx = np.arange(lo, hi)
        dist = np.where(idx < a, (a - idx), np.where(idx >= b, idx - b + 1, 0)) / rate
        for name, (tgt, alpha, theta) in tab.items():
            p = PI[name]
            d = alpha * np.exp(-theta * dist)
            num[p, lo:hi] += d * tgt
            den[p, lo:hi] += d
    val = num / np.maximum(den, 1e-6)
    val[den < 1e-4] = 0.0

    # articulator inertia: the jaw and lips are not weightless
    def lp(x, tau):
        a = 1 - np.exp(-1.0 / (rate * tau))
        y = np.empty_like(x)
        acc = float(x[0])
        for i in range(len(x)):
            acc += a * (x[i] - acc)
            y[i] = acc
        return y
    val[PI["jaw"]] = lp(val[PI["jaw"]], 0.012)
    val[PI["wide"]] = lp(val[PI["wide"]], 0.014)
    val[PI["round"]] = lp(val[PI["round"]], 0.014)
    val[PI["tongue"]] = lp(val[PI["tongue"]], 0.012)
    val[PI["teeth"]] = lp(val[PI["teeth"]], 0.012)

    # sample at video-frame centres (mouth leads the sound slightly); closures are max-pooled so a
    # 1-2 frame lip closure is never skipped
    out = {}
    for name in PARAMS:
        p = PI[name]
        arr = np.zeros(n_frames, np.float32)
        for i in range(n_frames):
            c = (i + 0.5) / fps + lead
            a = int((c - 0.5 / fps) * rate) + off
            b = int((c + 0.5 / fps) * rate) + off + 1
            a, b = max(0, a), min(val.shape[1], max(b, a + 1))
            seg = val[p, a:b]
            arr[i] = seg.max() if name in ("press", "tuck") else seg.mean()
        out[name] = arr
    out["jaw"] = np.clip(out["jaw"], 0, 1)
    out["wide"] = np.clip(out["wide"], -1, 1)
    for k in ("round", "press", "tuck", "tongue", "teeth"):
        out[k] = np.clip(out[k], 0, 1)
    return out
