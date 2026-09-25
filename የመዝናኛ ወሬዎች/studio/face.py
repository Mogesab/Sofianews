"""
FaceAnimator - brings a single still photo of a presenter to life.

Everything is done with numpy + OpenCV (no GPU, no neural nets):

* mouth   : jaw-drop warp + painted mouth interior (dark cavity, upper teeth,
            tongue) driven by the audio (openness / lip spread)
* eyes    : blinks (skin-coloured eyelid + lash line), tiny gaze drift, and the
            occasional glance down at the script
* brows   : small lifts on emphasis
* head    : soft-masked rigid motion (nod / tilt / sway) pivoting at the neck
* torso   : 1-pixel breathing

All landmark numbers live in assets/face.json (pixel positions on the photo).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _smooth(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class FaceAnimator:
    def __init__(self, image_path: Path, landmarks_path: Path, width: int, height: int):
        lm = json.loads(Path(landmarks_path).read_text(encoding="utf-8"))
        from . import media
        img = media.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(image_path)
        self.W, self.H = width, height
        self.base = cv2.resize(img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        self.s = width / float(lm["size"][0])
        s = self.s

        # ---- mouth ------------------------------------------------------------
        # Everything is expressed in the mouth's own frame: u along the lips, v across them
        # (positive = down), origin at the middle of the closed lip seam.
        m = lm["mouth"]
        self.m_c = (m["center"][0] * s, m["center"][1] * s)
        self.m_a = m["half_width"] * s
        ang = np.deg2rad(m["angle_deg"])
        self.m_cos, self.m_sin = float(np.cos(ang)), float(np.sin(ang))
        self.m_sag = m["sag"] * s
        A = self.m_a
        self.m_box = self._box(self.m_c[0] - 2.0 * A, self.m_c[1] - 1.25 * A,
                               self.m_c[0] + 2.0 * A, self.m_c[1] + 2.95 * A)
        x0, y0, x1, y1 = self.m_box
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = gx - self.m_c[0], gy - self.m_c[1]
        self.m_u = (dx * self.m_cos + dy * self.m_sin).astype(np.float32)
        self.m_v = (-dx * self.m_sin + dy * self.m_cos).astype(np.float32)
        self.m_gx, self.m_gy = gx, gy
        self.m_D = 18.0 * s                     # jaw drop for a fully open vowel (px)
        bxx = np.minimum(gx - x0, (x1 - 1) - gx)
        byy = np.minimum(gy - y0, (y1 - 1) - gy)
        self.m_win = (_smooth(0.0, 0.9 * A, bxx) * _smooth(0.0, 0.9 * A, byy)).astype(np.float32)
        self.m_u2A = ((self.m_u / A) ** 2).astype(np.float32)

        # ---- eyes ------------------------------------------------------------
        self.eyes = []
        for e in lm["eyes"]:
            a = np.deg2rad(e["angle_deg"])
            cx, cy = e["center"][0] * s, e["center"][1] * s
            A_e = e["half_width"] * s
            box = self._box(cx - A_e * 1.3, cy - 14 * s, cx + A_e * 1.3, cy + 12 * s)
            bx0, by0, bx1, by1 = box
            gy_, gx_ = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
            ddx, ddy = gx_ - cx, gy_ - cy
            cosa, sina = float(np.cos(a)), float(np.sin(a))
            u_ = ddx * cosa + ddy * sina
            v_ = -ddx * sina + ddy * cosa
            # skin colour for the closed lid: skin just above the eye, blended with cheek below
            samples = [self._median(cx - 0.55 * A_e, cy + 10.5 * s), self._median(cx, cy + 11.5 * s),
                       self._median(cx + 0.55 * A_e, cy + 10.5 * s)]
            skin = np.mean(samples, axis=0) * 0.965          # closed lid: cheek tone, a hair darker
            self.eyes.append(dict(box=box, gx=gx_, gy=gy_, u=u_.astype(np.float32), v=v_.astype(np.float32),
                                  A=A_e, skin=skin.astype(np.float32),
                                  iris=(e["iris"][0] * s, e["iris"][1] * s), iris_r=e["iris_r"] * s))
        # ---- brows ------------------------------------------------------------
        self.brows = []
        for bx, by in lm["brows"]:
            cx, cy = bx * s, by * s
            box = self._box(cx - 40 * s, cy - 24 * s, cx + 40 * s, cy + 16 * s)
            x0_, y0_, x1_, y1_ = box
            gy_, gx_ = np.mgrid[y0_:y1_, x0_:x1_].astype(np.float32)
            w = np.exp(-0.5 * (((gx_ - cx) / (24 * s)) ** 2 + ((gy_ - cy) / (9 * s)) ** 2)).astype(np.float32)
            self.brows.append(dict(box=box, gx=gx_, gy=gy_, w=w))

        # ---- head ROI + weights ------------------------------------------------
        h = lm["head"]
        rx0, ry0, rx1, ry1 = lm["rois"]["head"]
        self.h_box = self._box(rx0 * s, ry0 * s, rx1 * s, ry1 * s)
        x0, y0, x1, y1 = self.h_box
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        cx, cy = h["center"][0] * s, h["center"][1] * s
        rx, ry = h["radii"][0] * s, h["radii"][1] * s
        r = np.sqrt(((gx - cx) / rx) ** 2 + ((gy - cy) / ry) ** 2)
        w = 1.0 - _smooth(0.98, 1.24, r)
        w *= 1.0 - _smooth(h["neck_fade"][0] * s, h["neck_fade"][1] * s, gy)
        self.h_w = w.astype(np.float32)
        self.h_gx, self.h_gy = gx, gy
        self.h_X = (gx - h["pivot"][0] * s).astype(np.float32)
        self.h_Y = (gy - h["pivot"][1] * s).astype(np.float32)

        # ---- hands / papers (subtle life: idle fidget + page-turn lift) --------
        if "hands" in lm:
            hb = lm["hands"]
            hx0, hy0, hx1, hy1 = hb["box"]
            self.hd_box = self._box(hx0 * s, hy0 * s, hx1 * s, hy1 * s)
            x0, y0, x1, y1 = self.hd_box
            gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            ft, fb, fs = hb.get("fade", [30, 30, 20])
            wy = _smooth(y0, y0 + ft * s, gy) * (1 - _smooth(y1 - fb * s, y1, gy))
            wx = _smooth(x0, x0 + fs * s, gx) * (1 - _smooth(x1 - fs * s, x1, gx))
            self.hd_w = (wy * wx).astype(np.float32)
            self.hd_gx, self.hd_gy = gx, gy
        else:
            self.hd_box = None

        # ---- background screens: any panel that should feel "alive" behind the
        # presenter (world-map globe, the video wall of smaller monitors...).
        # Each gets: slow drift + zoom (as if the camera/scene breathes), pulsing
        # node/pixel lights, and a soft light sweep, driven off its own pixels so
        # it never needs per-photo hand tuning of colours.
        self.bg_screens = []
        for key, drift_amp, drift_period, zoom_amp, seed in (
            ("globe", 9.0, 34.0, 0.028, 3),
            ("screens", 5.5, 22.0, 0.034, 11),
        ):
            if key not in lm:
                continue
            bx0, by0, bx1, by1 = lm[key]["box"]
            box = self._box(bx0 * s, by0 * s, bx1 * s, by1 * s)
            x0, y0, x1, y1 = box
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            gy_, gx_ = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            fade = lm[key].get("fade", 45) * s
            wx = _smooth(x0, x0 + fade, gx_) * (1 - _smooth(x1 - fade, x1, gx_))
            wy = _smooth(y0, y0 + fade, gy_) * (1 - _smooth(y1 - fade, y1, gy_))
            w = (wx * wy).astype(np.float32)
            patch = self.base[y0:y1, x0:x1].astype(np.float32)
            lum = patch.mean(axis=2)
            lo, hi = np.percentile(lum, 55), np.percentile(lum, 97)
            bright = (np.clip((lum - lo) / (hi - lo + 1e-6), 0, 1) * w).astype(np.float32)
            ph_h, ph_w = 10, 12
            rng = np.random.default_rng(seed)
            self.bg_screens.append(dict(
                box=box, gx=gx_, gy=gy_, w=w, bright=bright, shape=(y1 - y0, x1 - x0),
                phase=rng.uniform(0, 2 * np.pi, (ph_h, ph_w)).astype(np.float32),
                speed=rng.uniform(0.22, 0.8, (ph_h, ph_w)).astype(np.float32),
                cx=(x0 + x1) / 2.0, cy=(y0 + y1) / 2.0,
                drift_amp=drift_amp * s, drift_period=drift_period, zoom_amp=zoom_amp,
                sweep_period=8.0 + seed * 0.6, phase0=rng.uniform(0, 2 * np.pi),
            ))


        # ---- torso (breathing) --------------------------------------------------
        tx0, ty0, tx1, ty1 = lm["torso"]["box"]
        self.t_box = self._box(tx0 * s, ty0 * s, tx1 * s, ty1 * s)
        x0, y0, x1, y1 = self.t_box
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        wy = _smooth(ty0 * s, ty0 * s + 90 * s, gy) * (1 - _smooth(ty1 * s - 130 * s, ty1 * s, gy))
        wx = _smooth(tx0 * s, tx0 * s + 60 * s, gx) * (1 - _smooth(tx1 * s - 60 * s, tx1 * s, gx))
        self.t_w = (wy * wx).astype(np.float32)
        self.t_gx, self.t_gy = gx, gy

    # ------------------------------------------------------------------ helpers
    def _box(self, x0, y0, x1, y1):
        return (max(0, int(x0)), max(0, int(y0)), min(self.W, int(x1)), min(self.H, int(y1)))

    def _median(self, cx, cy, r=3):
        x, y = int(cx), int(cy)
        patch = self.base[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3)
        return np.median(patch, axis=0).astype(np.float32)

    @staticmethod
    def _remap(patch, mapx, mapy):
        return cv2.remap(np.ascontiguousarray(patch), np.ascontiguousarray(mapx, dtype=np.float32),
                         np.ascontiguousarray(mapy, dtype=np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    # -------------------------------------------------------------------- parts
    def _brow(self, frame, lift):
        if abs(lift) < 0.05:
            return
        for b in self.brows:
            x0, y0, x1, y1 = b["box"]
            src = frame[y0:y1, x0:x1]
            frame[y0:y1, x0:x1] = self._remap(src, b["gx"] - x0, b["gy"] - y0 + lift * self.s * b["w"])

    def _eye(self, frame, e, gx, gy, cover):
        # gaze: shift iris/pupil inside its own radius
        s = self.s
        if abs(gx) + abs(gy) > 0.05:
            ix, iy = e["iris"]
            R = e["iris_r"]
            x0, y0, x1, y1 = self._box(ix - 2.4 * R, iy - 2.4 * R, ix + 2.4 * R, iy + 2.4 * R)
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            d = np.sqrt((xx - ix) ** 2 + (yy - iy) ** 2)
            w = 1.0 - _smooth(1.0 * R, 2.0 * R, d)
            frame[y0:y1, x0:x1] = self._remap(frame[y0:y1, x0:x1],
                                              xx - x0 - gx * s * w, yy - y0 - gy * s * w)
        if cover < 0.03:
            return
        x0, y0, x1, y1 = e["box"]
        u, v, A = e["u"], e["v"], e["A"]
        prof = np.clip(1 - (u / A) ** 2, 0, 1) ** 0.75
        v_up = -4.6 * s * prof
        v_lo = 4.0 * s * prof
        edge = v_up + cover * (v_lo - v_up)
        top = v_up - 3.6 * s
        inside = _smooth(0.0, 0.22 * A, A * 0.98 - np.abs(u))
        lid = np.clip(edge - v + 0.6, 0, 1) * _smooth(top, v_up + 0.3 * s, v) * inside
        patch = frame[y0:y1, x0:x1].astype(np.float32)
        skin = e["skin"]
        # subtle shading: slightly darker toward the lash line
        shade = 1.0 - 0.07 * np.clip((v - top) / (edge - top + 1e-3), 0, 1)
        col = skin[None, None, :] * shade[..., None]
        a = lid[..., None]
        patch = patch * (1 - a) + col * a
        if cover > 0.3:
            lash = np.clip(1.0 - np.abs(v - edge - 0.2 * s) / (1.15 * s), 0, 1) * inside * min(1.0, (cover - 0.3) * 2.2) * 0.85
            lash_col = np.array([28, 22, 26], np.float32)
            b = lash[..., None]
            patch = patch * (1 - b) + lash_col * b
        frame[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)

    def _mouth(self, frame, jaw, wide=0.0, rnd=0.0, press=0.0, tuck=0.0, tongue=0.0, teeth=0.0):
        """Draw the mouth for one frame from articulator parameters (see studio/visemes.py)."""
        s, A = self.s, self.m_a
        closed = float(_smooth(0.30, 0.70, press))
        d_jaw = jaw * (1.0 - closed) * self.m_D
        teeth_v = float(_smooth(0.35, 0.85, teeth)) * (1.0 - closed) * (1.0 - 0.6 * rnd)
        d_teeth = 2.7 * s * teeth_v                       # lips part a little to show teeth (ee, s)
        d_tuck = 3.6 * s * tuck * (1.0 - closed)          # f / v: room for the upper teeth
        d = max(d_jaw, d_teeth, d_tuck)
        sx = float(np.clip(1.0 + 0.15 * max(wide, 0.0) + 0.27 * min(wide, 0.0) - 0.15 * rnd, 0.60, 1.2))
        if d < 0.5 and abs(sx - 1.0) < 0.012 and tuck < 0.02:
            return
        x0, y0, x1, y1 = self.m_box
        u, v = self.m_u, self.m_v
        Ax = A * sx
        line = np.where(np.abs(u) < Ax, self.m_sag * (1.0 - (u / Ax) ** 2), 0.0).astype(np.float32)

        # --- shape of the opening: how much of the mouth width parts, and how round it is
        open_frac = 0.68 + 0.24 * float(np.clip(d / (0.62 * A), 0.0, 1.0))
        open_frac = max(open_frac, 0.80 * teeth_v * (1.0 - 0.7 * rnd))
        if tuck > 0.3:
            open_frac = max(open_frac, 0.72)
        W_o = max(Ax * open_frac * (1.0 - 0.32 * rnd), 1.0)
        pw = 0.74 + 0.28 * rnd
        shape = np.clip(1.0 - (u / W_o) ** 2, 0.0, 1.0) ** pw           # 1 in the middle, 0 at the ends
        L = (d * shape).astype(np.float32)                              # lower-lip drop along the mouth

        # --- displacement field (source offset = src - dst) --------------------------------
        vr = v - line
        ramp = _smooth(-0.5 * s, 1.8 * s, vr)                           # only the lower lip / chin follows the jaw
        blend = _smooth(0.25 * A, 0.95 * A, vr)                         # lips follow the lens, chin follows the jaw
        lat_j = np.exp(-0.5 * (u / (1.55 * Ax)) ** 2)
        chin = d * lat_j * (1.0 - _smooth(1.55 * A, 2.75 * A, vr))
        dj = ramp * ((1.0 - blend) * L + blend * chin)
        if tuck > 0.02:                                                 # lower lip rides up behind the upper teeth
            lw = _smooth(-0.3 * s, 1.2 * s, vr) * (1.0 - _smooth(0.30 * A, 0.62 * A, vr)) * np.exp(-0.5 * (u / (0.75 * A)) ** 2)
            dj = dj - tuck * (1.0 - closed) * 1.9 * s * lw
        U = 0.24 * d                                                    # upper lip lifts a little
        uw = _smooth(-0.55 * A, -1.0 * s, vr) * (1.0 - _smooth(-0.6 * s, 0.6 * s, vr)) * np.clip(1.0 - (u / (1.2 * W_o)) ** 2, 0.0, 1.0) ** 0.8
        off_v = -dj + U * uw
        if rnd > 0.02:                                                  # protruded lips look fuller
            k = 1.0 + 0.30 * rnd * np.exp(-0.5 * (vr / (0.55 * A)) ** 2) * np.exp(-0.5 * (u / (0.9 * Ax)) ** 2)
            off_v = off_v + vr * (1.0 / k - 1.0)
        off_u = 0.0
        if abs(sx - 1.0) > 0.004:                                       # narrower (oo) / wider (ee) mouth
            wh = np.exp(-0.5 * (u / (1.15 * A)) ** 2) * np.exp(-0.5 * ((v - 0.15 * A) / (0.95 * A)) ** 2)
            off_u = (1.0 / sx - 1.0) * u * wh
            if wide > 0.05:                                             # corners lift a touch in a smile-ish spread
                cw = np.exp(-0.5 * ((np.abs(u) - Ax) / (0.32 * A)) ** 2) * np.exp(-0.5 * (vr / (0.5 * A)) ** 2)
                off_v = off_v + 1.3 * s * wide * cw
        off_u = off_u * self.m_win
        off_v = off_v * self.m_win
        ox = off_u * self.m_cos - off_v * self.m_sin
        oy = off_u * self.m_sin + off_v * self.m_cos
        patch = self._remap(frame[y0:y1, x0:x1], self.m_gx - x0 + ox, self.m_gy - y0 + oy)

        # --- interior of the mouth ---------------------------------------------------------
        gap_px = L * 0.97
        if float(gap_px.max()) < 0.9:
            frame[y0:y1, x0:x1] = patch
            return
        lens = np.clip(gap_px / (1.0 * s), 0, 1)                       # 0 outside the opening, 1 well inside it
        v_top = line - (U + 0.3 * s) * lens
        v_bot = line + gap_px
        gap = np.maximum(v_bot - v_top, 1e-3)
        cov = np.clip(v - v_top + 0.5, 0, 1) * np.clip(v_bot - v + 0.5, 0, 1) * np.clip((v_bot - v_top) / 1.2, 0, 1) * lens
        if float(cov.max()) <= 0.0:
            frame[y0:y1, x0:x1] = patch
            return
        t = np.clip((v - v_top) / gap, 0, 1)
        edge = np.clip(1.0 - np.abs(u) / W_o, 0, 1)
        big = float(np.clip((d - 4.0 * s) / (12.0 * s), 0.0, 1.0))
        cav = np.empty(patch.shape, np.float32)
        cav[...] = (38, 30, 78)                                          # BGR warm dark red-brown
        cav *= (0.55 + 0.45 * edge ** 0.6)[..., None]                    # darker toward the corners
        cav *= (1.0 - 0.28 * big * (1.0 - t))[..., None]                 # deeper toward the back/top
        # upper teeth
        tfrac = float(np.clip(0.30 + 0.55 * teeth + 0.9 * tuck, 0.28, 1.0))
        th = np.minimum(gap * tfrac, 4.8 * s)
        if tuck > 0.4:                                                  # f / v: teeth fill a slim gap, but never a wide one
            th = np.where(gap < 5.5 * s, gap, th)
        in_th = np.clip((th - (v - v_top)) / (1.0 * s) + 0.5, 0, 1)
        vis_t = float(np.clip((d - 1.0 * s) / (1.8 * s), 0, 1))
        in_th = in_th * vis_t * np.clip((0.90 * W_o - np.abs(u)) / (0.22 * W_o + 1e-3), 0, 1)
        rel = np.clip((v - v_top) / (th + 1e-3), 0, 1)
        shade = (0.62 + 0.38 * np.clip(1.0 - (u / W_o) ** 2, 0, 1)) * (0.74 + 0.30 * np.sin(np.pi * np.clip(rel * 0.9 + 0.05, 0, 1)) ** 0.7)
        tcol = np.array([196, 207, 220], np.float32)[None, None, :] * shade[..., None]
        # tongue (behind the teeth, rising toward them for t/d/n/l/th)
        space = np.maximum(gap - th, 0.0)
        h_t = space * np.clip(0.30 + 0.62 * tongue + 0.20 * big, 0.0, 1.0)
        tprof = np.clip(1.0 - (u / (0.78 * W_o + 1e-3)) ** 2, 0, 1) ** 0.6
        tong_a = np.clip((v - (v_bot - h_t * tprof)) / (1.2 * s) + 0.5, 0, 1) * np.clip(space / (2.2 * s), 0, 1) * (tprof > 0.02)
        tong_col = np.array([102, 92, 186], np.float32)[None, None, :] * (0.72 + 0.28 * np.clip((v - v_top) / gap, 0, 1))[..., None]
        cav = cav * (1 - tong_a[..., None] * 0.85) + tong_col * (tong_a[..., None] * 0.85)
        cav = cav * (1 - in_th[..., None]) + tcol * in_th[..., None]
        # lower teeth just peeking up when the mouth is wide open
        low_vis = float(np.clip((d - 9.0 * s) / (6.0 * s), 0, 1))
        if low_vis > 0.05:
            lt_h = np.minimum(0.16 * gap, 2.6 * s) * low_vis
            lt = np.clip((v - (v_bot - lt_h)) / (0.9 * s) + 0.5, 0, 1) * np.clip((0.55 * W_o - np.abs(u)) / (0.2 * W_o), 0, 1)
            cav = cav * (1 - lt[..., None] * 0.8) + np.array([182, 194, 208], np.float32)[None, None, :] * 0.85 * (lt[..., None] * 0.8)
        # soft shadow under the upper lip, and a darker rim where the lower lip's inner edge begins
        sh = np.clip(1.0 - (v - v_top) / (1.5 * s), 0, 1) * 0.30
        cav *= (1.0 - sh)[..., None]
        rim = np.clip(1.0 - (v_bot - v) / (1.2 * s), 0, 1) * 0.18
        cav *= (1.0 - rim)[..., None]
        a = cov[..., None]
        patch = np.clip(patch.astype(np.float32) * (1 - a) + cav * a, 0, 255).astype(np.uint8)
        frame[y0:y1, x0:x1] = patch

    def _hands(self, frame, shift, lift):
        if self.hd_box is None or (abs(shift) < 0.03 and lift < 0.03):
            return
        s = self.s
        x0, y0, x1, y1 = self.hd_box
        w = self.hd_w
        dy = -(1.1 * lift) * s * w                 # papers lift a touch as a page turns
        dx = (0.7 * shift + 0.6 * lift) * s * w     # tiny sideways shuffle
        frame[y0:y1, x0:x1] = self._remap(frame[y0:y1, x0:x1], self.hd_gx - x0 - dx, self.hd_gy - y0 - dy)

    def _bg_life(self, frame, t):
        for sc in self.bg_screens:
            x0, y0, x1, y1 = sc["box"]
            w = sc["w"]
            # slow drift + a very slight zoom, as if the graphic is gently
            # parallaxing / breathing on the video wall behind him
            dx = sc["drift_amp"] * np.sin(2 * np.pi * t / sc["drift_period"] + sc["phase0"]) * w
            zoom = 1.0 + sc["zoom_amp"] * np.sin(2 * np.pi * t / (sc["drift_period"] * 1.7) + sc["phase0"])
            gx = sc["cx"] + (sc["gx"] - sc["cx"]) / zoom - dx
            gy = sc["cy"] + (sc["gy"] - sc["cy"]) / zoom
            patch = self._remap(frame[y0:y1, x0:x1], gx - x0, gy - y0)
            # blinking network nodes / pixels: a coarse random-phase grid upsampled
            # to per-pixel brightness, masked to only the bright parts of the graphic
            cell = 0.5 + 0.5 * np.sin(sc["phase"] + t * sc["speed"])
            cell_up = cv2.resize(cell, (sc["shape"][1], sc["shape"][0]), interpolation=cv2.INTER_LINEAR)
            glow = (0.55 + 0.85 * cell_up) * sc["bright"]
            # slow diagonal scan sweep, like a live video-wall panel
            u = (sc["gx"] - x0) / max(1, (x1 - x0)) + (sc["gy"] - y0) / max(1, (y1 - y0)) * 0.4
            pos = (t % sc["sweep_period"]) / sc["sweep_period"] * 1.8 - 0.4
            sweep = np.clip(1 - np.abs(u - pos) / 0.22, 0, 1) ** 2 * w * 0.30
            # cool LED-wall tint: the added light leans blue/cyan rather than
            # flat white, so it reads as screen glow rather than exposure noise
            boost = (glow * 46.0) + sweep * 110.0
            tint = np.array([1.22, 1.05, 0.72], np.float32)  # BGR: push blue, hold green, cut red
            add = boost[..., None] * tint[None, None, :]
            frame[y0:y1, x0:x1] = np.clip(patch.astype(np.float32) + add, 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------- frame
    def render(self, p: dict) -> np.ndarray:
        frame = self.base.copy()
        s = self.s
        self._bg_life(frame, float(p.get("t", 0.0)))
        self._brow(frame, p.get("brow", 0.0))
        cover = float(np.clip(p.get("blink", 0.0) + p.get("lid", 0.0), 0, 1))
        for i, e in enumerate(self.eyes):
            self._eye(frame, e, p.get("gx", 0.0), p.get("gy", 0.0), cover)
        self._mouth(frame, float(p.get("mopen", 0.0)), float(p.get("mspread", 0.0)), float(p.get("mround", 0.0)),
                    float(p.get("mpress", 0.0)), float(p.get("mtuck", 0.0)), float(p.get("mtongue", 0.0)),
                    float(p.get("mteeth", 0.0)))

        # torso breathing
        breath = p.get("breath", 0.0) * s
        if abs(breath) > 0.02:
            x0, y0, x1, y1 = self.t_box
            frame[y0:y1, x0:x1] = self._remap(frame[y0:y1, x0:x1], self.t_gx - x0,
                                              self.t_gy - y0 - breath * self.t_w)
        # hands / papers: idle fidget + the odd small page-turn lift
        self._hands(frame, float(p.get("handshift", 0.0)), float(p.get("pageflip", 0.0)))
        # head motion (rotate about the neck, then translate)
        th = np.deg2rad(p.get("rot", 0.0))
        tx, ty = p.get("tx", 0.0) * s, p.get("ty", 0.0) * s
        if abs(th) > 1e-5 or abs(tx) > 0.02 or abs(ty) > 0.02:
            x0, y0, x1, y1 = self.h_box
            c, sn = np.cos(th), np.sin(th)
            X, Y, w = self.h_X, self.h_Y, self.h_w
            dx = w * ((c - 1) * X - sn * Y + tx)
            dy = w * (sn * X + (c - 1) * Y + ty)
            frame[y0:y1, x0:x1] = self._remap(frame[y0:y1, x0:x1], self.h_gx - x0 - dx, self.h_gy - y0 - dy)
        return frame
