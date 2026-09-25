"""
Robot anchor behaviour over time.

A person drifts continuously; a robot moves like a machine driven by servos:
it holds a pose perfectly still, then snaps to the next one with a quick,
slightly overshooting move and settles. Every curve here is built that way
(step targets pushed through a stiff, under-damped spring), while the timing
follows the speech like a TV anchor's would:

* head   : re-poses (tilt / small turn / shift) at sentence starts and every
           couple of seconds, squares up to camera on a new story, and gives a
           short mechanical nod on stressed words
* torso  : leans into each new story, shifts weight between stories
* hands  : the clasped hands lift and tap down on the desk on emphasis beats,
           tilting and shifting side to side, with a bigger "here's the next
           story" lift on a story change
* eyes   : LED blinks (lights dim for a moment), stepping gaze, glow that rises
           while speaking and flashes on a new story
* jaw    : phoneme-aligned mouth opening (studio/lipsync.py), quantised into
           mechanical steps so it reads as a machine, not a rubber mouth
* camera : slow push-in during a story, cut between wide and closer shot on
           story changes (config "camera_moves")

Everything is precomputed per frame and deterministic.
"""
from __future__ import annotations

import numpy as np

from .motion import _bell, _peaks


def _servo(target: np.ndarray, dt: float, f0: float, zeta: float) -> np.ndarray:
    """Stiff spring (a servo) chasing the target. Integrated in sub-steps so even
    very fast servos stay numerically stable at 30 fps."""
    w = 2 * np.pi * f0
    sub = max(1, int(np.ceil(w * dt / 0.25)))
    h = dt / sub
    y = np.empty_like(target, dtype=np.float32)
    pos = vel = 0.0
    for i, tgt in enumerate(target):
        tgt = float(tgt)
        for _ in range(sub):
            vel += (w * w * (tgt - pos) - 2 * zeta * w * vel) * h
            pos += vel * h
        y[i] = pos
    return y


def _steps(n: int, fps: int, times: list[float], values: list) -> np.ndarray:
    """Piecewise-constant target: values[k] from times[k] until the next time."""
    out = np.zeros(n, np.float32)
    order = np.argsort(times)
    for j, k in enumerate(order):
        i0 = int(np.clip(times[k] * fps, 0, n))
        i1 = int(np.clip(times[order[j + 1]] * fps, 0, n)) if j + 1 < len(order) else n
        out[i0:i1] = values[k]
    return out


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _quantise(x: np.ndarray, levels: int, hyst: float = 0.55) -> np.ndarray:
    out = np.empty_like(x)
    cur = 0.0
    for i, v in enumerate(x * levels):
        if abs(v - cur) > hyst:
            cur = float(np.round(v))
        out[i] = cur / levels
    return out


def build(n: int, fps: int, lip: dict, seed: int = 11, story_starts=None, cue_starts=None, cue_ends=None,
          camera: bool = True, lead_in: float = 0.6) -> dict:
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    t = np.arange(n) / fps
    dur = n / fps
    story_starts = list(story_starts or [])
    cue_starts = list(cue_starts or [])
    cue_ends = list(cue_ends or [])
    zeros = np.zeros(n, np.float32)

    loud = np.asarray(lip.get("loud", zeros), np.float32)
    emph = np.asarray(lip.get("emph", zeros), np.float32)
    speaking = np.clip(loud * 4.0, 0, 1)
    k = max(1, int(fps * 0.6))
    speak_s = np.convolve(speaking, np.ones(k) / k, mode="same").astype(np.float32)
    ramp = np.clip(np.minimum((t - 0.2) / 0.6, (dur - t) / 0.8), 0, 1).astype(np.float32)

    # ------------------------------------------------------------------ jaw
    jaw = np.asarray(lip.get("mopen", zeros), np.float32)
    press = _smoothstep(0.3, 0.7, np.asarray(lip.get("mpress", zeros), np.float32))
    teeth = np.asarray(lip.get("mteeth", zeros), np.float32)
    tuck = np.asarray(lip.get("mtuck", zeros), np.float32)
    rnd = np.asarray(lip.get("mround", zeros), np.float32)
    wide = np.asarray(lip.get("mspread", zeros), np.float32)
    opening = np.maximum.reduce([jaw, 0.20 * _smoothstep(0.35, 0.85, teeth), 0.22 * tuck]) * (1.0 - press)
    opening = np.clip(opening * 1.08, 0, 1)
    jaw_q = _quantise(opening, 7)                                   # mechanical steps
    jaw_w = np.clip(1.0 + 0.16 * np.maximum(wide, 0) - 0.34 * rnd, 0.6, 1.2).astype(np.float32)

    # ------------------------------------------------------------------ head
    times, vals = [0.0], [(0.0, 0.0, 0.0, 0.0)]
    story_set = set(round(s, 2) for s in story_starts)
    last = 0.0
    for cs in cue_starts:
        if round(cs, 2) in story_set:
            continue
        if rng.random() < 0.7 and cs - last > 0.9:
            times.append(cs - 0.08)
            vals.append((rng.uniform(-3.2, 3.2), rng.uniform(-4, 4), rng.uniform(-1.2, 2.0), rng.uniform(-7, 7)))
            last = cs
    for cs, ce in zip(cue_starts, cue_ends):                        # extra re-poses inside long sentences
        tt = cs + rng.uniform(1.6, 2.6)
        while tt < ce - 0.8:
            times.append(tt)
            vals.append((rng.uniform(-2.6, 2.6), rng.uniform(-3, 3), rng.uniform(-1, 1.5), rng.uniform(-5, 5)))
            tt += rng.uniform(1.8, 3.2)
    for ss in story_starts:                                         # square up to camera for a new story
        times.append(ss - 0.25)
        vals.append((0.0, 0.0, 0.8, 0.0))
    arr = np.array(vals, np.float32)
    h_rot = _servo(_steps(n, fps, times, list(arr[:, 0])), dt, 2.6, 0.58)
    h_tx = _servo(_steps(n, fps, times, list(arr[:, 1])), dt, 2.6, 0.58)
    h_ty = _servo(_steps(n, fps, times, list(arr[:, 2])), dt, 2.8, 0.6)
    h_yaw = _servo(_steps(n, fps, times, list(arr[:, 3])), dt, 2.4, 0.62)
    # short mechanical nods on stressed syllables
    drive = np.zeros(n, np.float32)
    for i in _peaks(emph, 0.42, int(fps * 0.55)):
        drive[i:i + max(2, int(fps * 0.1))] += 1.0 + 0.8 * float(emph[i])
    nod = _servo(drive, dt, 4.2, 0.5)
    h_ty = h_ty + 2.6 * nod
    h_rot = h_rot + 0.35 * nod * np.sign(np.sin(t * 0.7))

    # ----------------------------------------------------------------- torso
    tt_, bv = [0.0], [(0.0, 0.0)]
    tt = lead_in + rng.uniform(2.5, 4.5)
    while tt < dur:
        tt_.append(tt)
        bv.append((rng.uniform(-0.9, 0.9), rng.uniform(-5, 5)))
        tt += rng.uniform(2.8, 5.5)
    bv = np.array(bv, np.float32)
    b_rot = _servo(_steps(n, fps, tt_, list(bv[:, 0])), dt, 1.2, 0.75)
    b_tx = _servo(_steps(n, fps, tt_, list(bv[:, 1])), dt, 1.2, 0.75)
    lean = np.zeros(n, np.float32)
    for ss in story_starts:
        lean = np.maximum(lean, _bell((t - (ss - 0.35)) / 2.8)).astype(np.float32)
    lean = _servo(lean, dt, 1.5, 0.7)
    b_zoom = 0.014 * lean
    b_ty = 3.5 * lean

    # ----------------------------------------------------------------- hands
    lift_t = np.zeros(n, np.float32)
    rot_times, rot_vals, x_vals = [0.0], [0.0], [0.0]
    side = 1.0
    last = -10.0
    for i in _peaks(emph, 0.3, int(fps * 0.4)):
        ti = i / fps
        if ti - last < 1.1 or speak_s[i] < 0.25 or rng.random() > 0.78:
            continue
        last = ti
        L = max(2, int(fps * rng.uniform(0.16, 0.26)))
        lift_t[max(0, i - int(fps * 0.08)):i - int(fps * 0.08) + L] = -rng.uniform(7, 13)
        side = -side
        rot_times.append(ti - 0.06)
        rot_vals.append(side * rng.uniform(1.8, 4.0))
        x_vals.append(side * rng.uniform(2.5, 7.0))
    for ss in story_starts:                                         # "and now..." presentation lift
        i0 = int(np.clip((ss - 0.3) * fps, 0, n))
        lift_t[i0:i0 + int(fps * 0.45)] = -16.0
        rot_times.append(ss - 0.3)
        rot_vals.append(0.0)
        x_vals.append(0.0)
    hd_ty = _servo(lift_t, dt, 3.3, 0.48)
    hd_ty = np.minimum(hd_ty, 0.8)                                  # never push through the desk
    hd_rot = _servo(_steps(n, fps, rot_times, rot_vals), dt, 2.2, 0.6)
    hd_tx = _servo(_steps(n, fps, rot_times, x_vals), dt, 2.0, 0.65)
    b_ty = b_ty + 0.12 * hd_ty                                      # shoulders ride along with the hands

    # ------------------------------------------------------------------ eyes
    bl_times = [ce + rng.uniform(0.03, 0.2) for ce in cue_ends if rng.random() < 0.5]
    tt = rng.uniform(1.5, 3.0)
    while tt < dur:
        if not any(abs(tt - b) < 1.6 for b in bl_times):
            bl_times.append(tt)
        tt += rng.uniform(2.6, 5.2)
    blink = np.zeros(n, np.float32)
    for b in bl_times:
        x = t - b
        shape = np.where(x < 0, 0, np.where(x < 0.03, x / 0.03, np.where(x < 0.1, 1.0, np.where(x < 0.16, 1 - (x - 0.1) / 0.06, 0))))
        blink = np.maximum(blink, shape.astype(np.float32))
    g_times = [0.0]
    g_x, g_y = [0.0], [0.0]
    tt = rng.uniform(0.6, 1.5)
    while tt < dur:
        g_times.append(tt)
        g_x.append(rng.uniform(-1.6, 1.6))
        g_y.append(rng.uniform(-0.6, 0.9))
        tt += rng.uniform(0.8, 2.4)
    gx = _servo(_steps(n, fps, g_times, g_x), dt, 7.0, 0.7)
    gy = _servo(_steps(n, fps, g_times, g_y), dt, 7.0, 0.7)
    flash = np.zeros(n, np.float32)
    for ss in story_starts:
        flash = np.maximum(flash, _bell((t - (ss - 0.1)) / 0.5)).astype(np.float32)
    glow = (1.0 + 0.16 * speak_s + 0.35 * flash + 0.04 * np.sin(2 * np.pi * t / 3.1)).astype(np.float32)

    # ---------------------------------------------------------------- camera
    cam_z = np.ones(n, np.float32)
    cam_x = np.full(n, 0.47, np.float32)
    cam_y = np.full(n, 0.33, np.float32)
    if camera:
        marks = sorted(story_starts)
        bounds = [0.0] + marks + [dur]
        for j in range(len(bounds) - 1):
            i0, i1 = int(bounds[j] * fps), min(n, int(bounds[j + 1] * fps))
            if i1 <= i0:
                continue
            close = (j % 2 == 1)
            z0 = 1.075 if close else 1.0
            prog = np.linspace(0, 1, i1 - i0, dtype=np.float32)
            cam_z[i0:i1] = z0 + 0.018 * prog                        # slow push-in during the story
        # the last seconds (sign-off) go back to the wide shot
        i_end = int(max(0, (dur - 3.0)) * fps)
        cam_z[i_end:] = np.minimum(cam_z[i_end:], 1.0 + 0.01 * np.linspace(0, 1, n - i_end))

    R = lambda a: (np.asarray(a, np.float32) * ramp).astype(np.float32)
    return dict(t=t.astype(np.float32), jaw=jaw_q.astype(np.float32), jaw_w=jaw_w, loud=loud,
                h_rot=R(h_rot), h_tx=R(h_tx), h_ty=R(h_ty), h_yaw=R(h_yaw),
                b_rot=R(b_rot), b_tx=R(b_tx), b_ty=R(b_ty), b_zoom=R(b_zoom),
                hd_tx=R(hd_tx), hd_ty=R(hd_ty), hd_rot=R(hd_rot),
                blink=blink, gx=gx.astype(np.float32), gy=gy.astype(np.float32), glow=glow,
                cam_z=cam_z, cam_x=cam_x, cam_y=cam_y)
