"""
Presenter behaviour over time: head sway, nods on stress, blinks, gaze
saccades, the odd glance down at the script, brow lifts, breathing.

The curves are built from models of how a real person actually moves rather
than from plain sine waves:

* posture   : slow filtered noise (a random walk that always drifts back to
              centre) plus deliberate "settle into a new position" shifts at
              story boundaries -- no repeating wobble
* nods      : each stressed syllable fires a critically-damped spring, so the
              head drops and springs back the way a real nod does
* gaze      : fixation / saccade model -- the eyes sit still, then flick in a
              few frames with a small overshoot, never drifting smoothly
* blinks    : biased toward sentence ends and pauses, where people really blink
* brows     : attack/decay envelope on emphasis, plus a brow flash on new stories
* breathing : varying rate, with a deeper inhale during pauses

All values are pre-computed per frame (deterministic), so every render worker
produces identical results for its slice of the timeline.
"""
from __future__ import annotations

import numpy as np


def _bell(x):
    x = np.clip(x, 0.0, 1.0)
    return np.sin(np.pi * x) ** 2


def _ou(rng, n: int, dt: float, tau: float, sigma: float) -> np.ndarray:
    """Ornstein-Uhlenbeck noise: wanders like a person shifting weight, but
    always pulled gently back to the middle. Never repeats like a sine."""
    a = float(np.exp(-dt / tau))
    step = rng.standard_normal(n).astype(np.float32) * (sigma * np.sqrt(1 - a * a))
    out = np.empty(n, np.float32)
    acc = 0.0
    for i in range(n):
        acc = a * acc + step[i]
        out[i] = acc
    return out


def _drift(rng, n: int, dt: float, sigma: float) -> np.ndarray:
    """Layered OU at several time scales -> natural 1/f body sway."""
    return (_ou(rng, n, dt, 6.5, sigma)
            + 0.55 * _ou(rng, n, dt, 2.4, sigma)
            + 0.28 * _ou(rng, n, dt, 0.9, sigma)).astype(np.float32)


def _spring(x: np.ndarray, dt: float, f0: float, zeta: float) -> np.ndarray:
    """Push a drive signal through a damped spring. Gives real body dynamics:
    overshoot, settle, and a bit of follow-through instead of instant snapping."""
    w = 2 * np.pi * f0
    y = np.empty_like(x, dtype=np.float32)
    pos = vel = 0.0
    for i, tgt in enumerate(x):
        acc = w * w * (tgt - pos) - 2 * zeta * w * vel
        vel += acc * dt
        pos += vel * dt
        y[i] = pos
    return y


def _env(x: np.ndarray, dt: float, attack: float, release: float) -> np.ndarray:
    """Asymmetric envelope follower (fast up, slow down), like muscle tone."""
    au = 1 - np.exp(-dt / max(attack, 1e-3))
    ad = 1 - np.exp(-dt / max(release, 1e-3))
    y = np.empty_like(x, dtype=np.float32)
    acc = 0.0
    for i, v in enumerate(x):
        acc += (au if v > acc else ad) * (v - acc)
        y[i] = acc
    return y


def _peaks(x: np.ndarray, thresh: float, min_gap: int):
    """Local maxima of the emphasis curve = stressed syllables to nod on."""
    out, last = [], -10 ** 9
    for i in range(1, len(x) - 1):
        if x[i] >= thresh and x[i] >= x[i - 1] and x[i] > x[i + 1] and i - last >= min_gap:
            out.append(i)
            last = i
    return out


def _blink_shape(t, s0, d=0.17):
    x = (t - s0) / d
    m = (x >= 0) & (x <= 1)
    prof = np.where(x < 0.35, np.sin(np.pi / 2 * x / 0.35), np.cos(np.pi / 2 * (x - 0.35) / 0.65))
    return np.where(m, np.clip(prof, 0, 1), 0).astype(np.float32)


def build(n: int, fps: int, lip: dict, seed: int = 7, lead_in: float = 3.5, tail: float = 3.0,
          story_starts: list[float] | None = None, cue_starts: list[float] | None = None,
          cue_ends: list[float] | None = None) -> dict:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    dt = 1.0 / fps
    dur = n / fps
    story_starts = story_starts or []
    cue_starts = cue_starts or []
    cue_ends = cue_ends or []

    speaking = np.clip(lip["loud"] * 4.0, 0, 1)
    k80 = max(1, int(fps * 0.8))
    speak_s = np.convolve(speaking, np.ones(k80) / k80, mode="same").astype(np.float32)
    live = 0.5 + 0.5 * speak_s
    emph = lip["emph"]

    # ---------------------------------------------------------------- posture
    # slow, non-repeating sway (px at 1080p / degrees), a touch larger while talking
    tx = _drift(rng, n, dt, 0.80) * live
    ty = _drift(rng, n, dt, 0.62) * live
    rot = _drift(rng, n, dt, 0.22) * live
    # real heads roll a little into the direction they lean
    rot = rot - 0.10 * tx

    # deliberate posture resets: a new story, or just every so often, the anchor
    # settles into a slightly different position instead of orbiting one point
    pose_x = np.zeros(n, np.float32)
    pose_y = np.zeros(n, np.float32)
    pose_r = np.zeros(n, np.float32)
    marks = sorted(set([lead_in + 1.0] + list(story_starts)))
    extra = lead_in + rng.uniform(9, 16)
    while extra < dur - tail:
        marks.append(extra)
        extra += rng.uniform(11, 20)
    marks = sorted(marks)
    cur = np.array([0.0, 0.0, 0.0], np.float32)
    for j, ts in enumerate(marks):
        i0 = int(np.clip(ts * fps, 0, n - 1))
        i1 = int(np.clip(marks[j + 1] * fps, 0, n)) if j + 1 < len(marks) else n
        cur = np.array([rng.uniform(-1.2, 1.2), rng.uniform(-0.8, 0.8), rng.uniform(-0.26, 0.26)], np.float32)
        pose_x[i0:i1], pose_y[i0:i1], pose_r[i0:i1] = cur
    # ease into each new posture (0.9 Hz spring, slightly underdamped = human)
    tx = tx + _spring(pose_x, dt, 0.9, 0.95)
    ty = ty + _spring(pose_y, dt, 0.9, 0.95)
    rot = rot + _spring(pose_r, dt, 0.8, 0.95)

    # ---------------------------------------------------------------- nodding
    # one impulse per stressed syllable, each shaped by a spring, so the nod has
    # weight (down, back up, small settle) instead of tracking the audio directly
    drive_y = np.zeros(n, np.float32)
    drive_r = np.zeros(n, np.float32)
    idx = _peaks(emph, 0.30, int(fps * 0.34))
    for i in idx:
        amp = 0.55 + 0.9 * float(emph[i])
        L = max(2, int(fps * 0.10))
        drive_y[i:i + L] += amp
        if rng.random() < 0.45:                        # some beats are a tilt, not a nod
            drive_r[i:i + L] += amp * rng.uniform(-0.30, 0.30)
    ty = ty + 1.9 * _spring(drive_y, dt, 1.55, 0.58)
    rot = rot + 0.55 * _spring(drive_r, dt, 1.3, 0.62)

    # a slightly bigger "lean into it" at the start of each new story
    lean = np.zeros(n, np.float32)
    for ts in story_starts:
        lean = np.maximum(lean, _bell((t - (ts - 0.35)) / 1.5)).astype(np.float32)
    ty = ty + 1.1 * lean

    # ------------------------------------------------------------- glances down
    glance = np.zeros(n, np.float32)
    tg = lead_in + rng.uniform(4, 9)
    glances = []
    while tg < dur - tail - 2:
        d = rng.uniform(0.7, 1.05)
        glance = np.maximum(glance, _bell((t - tg) / d)).astype(np.float32)
        glances.append(tg + d * 0.5)
        tg += rng.uniform(13, 24)

    # ------------------------------------------------------------------ blinks
    # people blink where the speech breaks: sentence ends and pauses. We seed
    # blinks there first, then fill the gaps so there is never a long stare.
    blink = np.zeros(n, np.float32)
    times = []
    for ce in cue_ends:
        if rng.random() < 0.45 and lead_in * 0.5 < ce < dur - 0.6:
            times.append(ce + rng.uniform(0.02, 0.22))
    times += [g + 0.55 for g in glances]               # blink on looking back up
    times.sort()
    filled, last = [], 0.9
    for ts in times + [dur]:
        while ts - last > rng.uniform(3.8, 6.0):       # no staring contests
            last += rng.uniform(3.0, 5.2)
            filled.append(last)
        if ts < dur - 0.5:
            filled.append(ts)
        last = max(last, ts)
    for ts in filled:
        if not (0.4 < ts < dur - 0.4):
            continue
        blink = np.maximum(blink, _blink_shape(t, ts))
        if rng.random() < 0.13:                        # the occasional double blink
            blink = np.maximum(blink, _blink_shape(t, ts + 0.30))

    # -------------------------------------------------------------------- gaze
    def saccades(amp, lo, hi, settle=0.055):
        """Fixation / saccade: hold still, then flick to the next spot in a few
        frames with a touch of overshoot. This is what real eyes do -- the old
        smooth drift is what made the stare look artificial."""
        tgt = np.zeros(n, np.float32)
        i, cur = 0, 0.0
        while i < n:
            L = max(2, int(rng.uniform(lo, hi) * fps))
            cur = float(np.clip(cur + rng.normal(0, amp * 0.8), -amp, amp))
            tgt[i:i + L] = cur
            i += L
        return _spring(tgt, dt, 1.0 / (2 * np.pi * settle), 0.72)

    gx = saccades(0.85, 0.7, 2.6)
    gy = saccades(0.5, 0.8, 2.8) + 1.9 * glance
    # ocular micro-tremor: eyes are never perfectly still, even when "locked on"
    kk = max(1, int(fps * 0.07))
    gx = gx + np.convolve(rng.standard_normal(n).astype(np.float32), np.ones(kk) / kk, mode="same") * 0.13
    gy = gy + np.convolve(rng.standard_normal(n).astype(np.float32), np.ones(kk) / kk, mode="same") * 0.09
    # the head follows the eyes a little, slightly behind them (vestibular lag)
    tx = tx + _spring(gx * 0.7, dt, 0.7, 1.0)
    ty = ty + _spring(gy * 0.35, dt, 0.7, 1.0) + 2.0 * glance
    lid = 0.20 * glance

    # ------------------------------------------------------------------- brows
    brow = 1.7 * _env(emph, dt, 0.06, 0.34)
    for ts in story_starts:                             # small brow flash on a new story
        brow = np.maximum(brow, 0.9 * _bell((t - (ts - 0.2)) / 0.7)).astype(np.float32)
    brow = brow + 0.25 * _drift(rng, n, dt, 0.35) * speak_s
    brow = np.maximum(brow, 0.0).astype(np.float32)

    # --------------------------------------------------------------- breathing
    # rate drifts, and there is a deeper inhale when the voice stops for a breath
    rate = 0.25 + 0.035 * _ou(rng, n, dt, 9.0, 1.0)     # ~15 breaths/min, wandering
    phase = np.cumsum(2 * np.pi * rate * dt).astype(np.float32) + rng.uniform(0, 2 * np.pi)
    pause = np.clip(1.0 - speak_s * 1.6, 0, 1)
    depth = 1.0 + 0.55 * _env(pause, dt, 0.5, 1.2)
    breath = (np.sin(phase) * depth).astype(np.float32)

    # settle at the very start / end
    ramp = np.clip(np.minimum(t / 1.2, (dur - t) / 1.2), 0, 1).astype(np.float32)
    tx, ty, rot = tx * ramp, ty * ramp, rot * ramp

    # ------------------------------------------------------- hands / paperwork
    handshift = (_drift(rng, n, dt, 0.42) * (0.4 + 0.6 * speak_s) * ramp).astype(np.float32)
    pageflip = np.zeros(n, np.float32)
    for ts in story_starts:
        pageflip = np.maximum(pageflip, _bell((t - (ts - 0.15)) / 0.85)).astype(np.float32)

    return dict(tx=tx.astype(np.float32), ty=ty.astype(np.float32), rot=rot.astype(np.float32),
                blink=blink.astype(np.float32), gx=gx.astype(np.float32), gy=gy.astype(np.float32),
                lid=lid.astype(np.float32), brow=brow, breath=breath,
                handshift=handshift, pageflip=pageflip.astype(np.float32),
                t=t.astype(np.float32))
