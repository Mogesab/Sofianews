"""
RobotAnimator - brings the robot news anchor photo to life (numpy + OpenCV, no GPU).

The photo is split into layers once (tools/build_robot.py): the empty studio,
the robot (with an alpha matte), the laptop in front of him and the
"GLOBAL NEWS LIVE" lettering. Every frame is then composed as

    studio plate
      + rotating globe (the photographed globe, re-projected onto a sphere and spun)
      + desk reflection that follows the hands
      + robot:   mechanical jaw with an LED voice grille, LED eyes (blink = lights dim,
                 gaze = pupils step), headphone LEDs that pulse with the voice,
                 then the whole body warped by a small "skeleton"
                 (torso lean/sway, head tilt/turn/nod, hands lift/tap/tilt)
      + laptop (stays in front of his arm)
      + shining "GLOBAL NEWS LIVE" lettering
    and an optional TV camera push-in / cut to a closer shot.

Motion curves come from studio/robot_motion.py; the mouth is driven by the
same phoneme-aligned lip-sync as the human presenter (studio/lipsync.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from . import outfit as wardrobe


def _smooth(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _remap(img, mapx, mapy, border=cv2.BORDER_REPLICATE):
    return cv2.remap(np.ascontiguousarray(img), np.ascontiguousarray(mapx, dtype=np.float32),
                     np.ascontiguousarray(mapy, dtype=np.float32), cv2.INTER_LINEAR, borderMode=border)


class _Globe:
    """Spins the photographed globe. Each pixel of the disc is mapped to
    latitude/longitude; the longitude is advanced over time and sampled back
    from the still photo. Only part of the sphere is visible (it runs off the
    right edge and hides behind the banner), so the texture is mirrored at the
    edge of the visible part, which keeps it seamless while it flows round."""

    def __init__(self, plate, g, sx, sy, text_rgba, W, H):
        self.plate = plate
        cx, cy, r = g["center"][0] * sx, g["center"][1] * sy, g["radius"] * sx
        self.speed = np.deg2rad(float(g.get("speed_deg", 9.0)))
        ya, yb = g["y_range"]
        x0, y0 = int(max(0, cx - r)), int(max(0, ya * sy))
        x1, y1 = W, int(min(H, yb * sy))
        self.box = (x0, y0, x1, y1)
        gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        v = np.clip((gy - cy) / r, -0.999, 0.999)
        c = np.sqrt(1.0 - v * v)
        u = (gx - cx) / (r * c)
        inside = np.abs(u) < 1.0
        lon = np.arcsin(np.clip(u, -1, 1))
        # clean-texture limits per row
        rows = np.arange(y0, y1, dtype=np.float32)
        rc = r * np.sqrt(1.0 - np.clip((rows - cy) / r, -0.999, 0.999) ** 2)
        xmin = cx - rc + 3.0
        xmax = np.minimum(W - 3.0, cx + rc - 3.0)
        bx = np.array(g["banner"], np.float32) * [sx, sy]
        top, bot = bx[:, 1].min(), bx[:, 1].max()
        left_top, left_bot = bx[0], bx[3]
        for i, yy in enumerate(rows):
            if top - 2 <= yy <= bot + 2:
                f = np.clip((yy - left_top[1]) / (left_bot[1] - left_top[1] + 1e-6), 0, 1)
                xmax[i] = min(xmax[i], left_top[0] + f * (left_bot[0] - left_top[0]) - 7.0)
        for ra, rb, xm in g.get("clean_x_min", []):
            sel = (rows >= ra * sy) & (rows <= rb * sy)
            xmin[sel] = np.maximum(xmin[sel], xm * sx)
        lo = np.arcsin(np.clip((xmin - cx) / rc, -1, 1))
        hi = np.arcsin(np.clip((xmax - cx) / rc, -1, 1))
        ok = (hi - lo) > 0.06
        self.lo = np.where(ok, lo, 0)[:, None].astype(np.float32)
        self.span = np.where(ok, hi - lo, 1)[:, None].astype(np.float32)
        rn = np.sqrt(((gx - cx) / r) ** 2 + ((gy - cy) / r) ** 2)
        w = (1.0 - _smooth(0.965, 1.0, rn)) * inside * ok[:, None]
        w *= _smooth(y0, y0 + 10, gy) * (1.0 - _smooth(y1 - 10, y1, gy))
        self.w = w.astype(np.float32)[..., None]
        self.lon, self.rc, self.cx = lon.astype(np.float32), (r * c).astype(np.float32), cx
        self.gy = gy
        self.src = plate[y0:y1, x0:x1].astype(np.float32)
        # ---- lettering sprite + shine
        t = cv2.resize(text_rgba, (W, H), interpolation=cv2.INTER_AREA)
        a = t[..., 3].astype(np.float32) / 255.0
        ys, xs = np.where(a > 0.02)
        if len(xs):
            tx0, ty0, tx1, ty1 = xs.min() - 14, ys.min() - 14, min(W, xs.max() + 15), min(H, ys.max() + 15)
        else:
            tx0 = ty0 = 0
            tx1, ty1 = 1, 1
        self.tbox = (tx0, ty0, tx1, ty1)
        self.t_rgb = t[ty0:ty1, tx0:tx1, :3].astype(np.float32)
        self.t_a = a[ty0:ty1, tx0:tx1][..., None]
        lum = self.t_rgb.mean(axis=2)
        self.t_letters = (np.clip((lum - 150) / 70, 0, 1) * self.t_a[..., 0])[..., None]
        halo = cv2.GaussianBlur(self.t_letters[..., 0], (0, 0), 6)
        self.t_halo = (halo / (halo.max() + 1e-6))[..., None] * (1 - self.t_a)
        hy, hx = np.mgrid[0:ty1 - ty0, 0:tx1 - tx0].astype(np.float32)
        self.t_diag = (hx + 0.55 * hy) / max(1.0, (tx1 - tx0) + 0.55 * (ty1 - ty0))

    def draw(self, frame, t):
        x0, y0, x1, y1 = self.box
        L = self.lon - self.speed * t
        q = np.mod(L - self.lo, 2 * self.span)
        src_lon = self.lo + np.where(q <= self.span, q, 2 * self.span - q)
        mapx = self.cx + self.rc * np.sin(src_lon) - x0
        mapy = self.gy - y0
        spun = _remap(self.src, mapx, mapy)
        reg = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = (reg + (spun - reg) * self.w).astype(np.uint8)
        # lettering: slow breathing glow + a light sweep across it every few seconds
        tx0, ty0, tx1, ty1 = self.tbox
        reg = frame[ty0:ty1, tx0:tx1].astype(np.float32)
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * t / 2.6)
        halo = self.t_halo * (0.35 + 0.35 * pulse) * np.array([255, 215, 150], np.float32)
        reg = reg + halo * (1 - reg / 255.0)
        rgb = self.t_rgb * (0.96 + 0.08 * pulse)
        period = 3.6
        pos = (t % period) / period * 1.8 - 0.4
        band = np.exp(-((self.t_diag - pos) / 0.07) ** 2)[..., None]
        rgb = rgb + band * (0.55 * self.t_a + 0.45 * self.t_letters) * 150.0
        glint = np.exp(-((self.t_diag - pos) / 0.018) ** 2)[..., None] * self.t_letters * 90.0
        rgb = rgb + glint
        out = reg * (1 - self.t_a) + rgb * self.t_a
        frame[ty0:ty1, tx0:tx1] = np.clip(out, 0, 255).astype(np.uint8)


class RobotAnimator:
    def __init__(self, image_path: Path, landmarks_path: Path, width: int, height: int, outfit: dict | None = None):
        lm = json.loads(Path(landmarks_path).read_text(encoding="utf-8"))
        folder = Path(landmarks_path).parent
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(image_path)
        lay = lm["layers"]
        matte = cv2.imread(str(folder / lay["matte"]), cv2.IMREAD_GRAYSCALE)
        plate = cv2.imread(str(folder / lay["plate"]), cv2.IMREAD_COLOR)
        gtext = cv2.imread(str(folder / lay["globe_text"]), cv2.IMREAD_UNCHANGED)
        if matte is None or plate is None or gtext is None:
            raise FileNotFoundError("robot layers missing - run: python tools/build_robot.py")
        img = wardrobe.apply(img, matte, lm, outfit)

        W, H = width, height
        self.W, self.H = W, H
        nw, nh = lm["size"]
        sx, sy = W / float(nw), H / float(nh)
        self.s = s = sx
        P = lambda x, y: (x * sx, y * sy)
        self.P = P
        rs = lambda a, interp=cv2.INTER_LANCZOS4: cv2.resize(a, (W, H), interpolation=interp)
        self.base = rs(img)
        self.plate = rs(plate)
        alpha = rs(matte, cv2.INTER_LINEAR).astype(np.float32) / 255.0

        # ---- robot sprite (everything that moves), in its own box
        bx0, by0, bx1, by1 = lm["sprite_box"]
        x0, y0 = P(bx0, by0)
        x1, y1 = P(bx1, by1)
        self.sb = sb = (int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1)))
        X0, Y0, X1, Y1 = sb
        self.spr = self.base[Y0:Y1, X0:X1].copy()
        self.spr_a = alpha[Y0:Y1, X0:X1].copy()
        gy, gx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        self.gx, self.gy = gx, gy

        # skeleton weights ---------------------------------------------------------
        h = lm["head"]
        hcx, hcy = P(*h["center"])
        hrx, hry = h["radii"][0] * sx, h["radii"][1] * sy
        rn = np.sqrt(((gx - hcx) / hrx) ** 2 + ((gy - hcy) / hry) ** 2)
        w_head = (1.0 - _smooth(0.98, 1.2, rn)) * (1.0 - _smooth(h["neck_fade"][0] * sy, h["neck_fade"][1] * sy, gy))
        hb = lm["hands"]
        hx0, hy0 = P(hb["box"][0], hb["box"][1])
        hx1, hy1 = P(hb["box"][2], hb["box"][3])
        fd = hb.get("fade", 40) * s
        w_hands = (_smooth(hx0, hx0 + fd, gx) * (1 - _smooth(hx1 - fd, hx1, gx))
                   * _smooth(hy0, hy0 + fd, gy))
        da, db = lm["desk_anchor"]
        w_anchor = _smooth(da * sy, db * sy, gy) * (1.0 - w_hands)
        self.w_body = np.clip(1.0 - w_hands - w_anchor, 0, 1).astype(np.float32)
        # head / hands only need their own sub-boxes
        self.hbox = self._sub(np.where(w_head > 1e-3))
        self.dbox = self._sub(np.where(w_hands > 1e-3))
        a0, b0, a1, b1 = self.hbox
        self.w_head = w_head[b0:b1, a0:a1].astype(np.float32)
        a0, b0, a1, b1 = self.dbox
        self.w_hands = w_hands[b0:b1, a0:a1].astype(np.float32)

        self.pv_t = P(*lm["torso"]["pivot"])
        self.pv_h = P(*h["pivot"])
        self.pv_d = P(*hb["center"])
        self.h_c, self.h_rx = (hcx, hcy), hrx

        # jaw ------------------------------------------------------------------------
        j = lm["jaw"]
        poly = np.array([P(x, y) for x, y in j["poly"]], np.float32)
        self.jaw_drop = float(j.get("drop", 9.0)) * sy
        pad = int(self.jaw_drop) + 6
        jx0, jy0 = int(poly[:, 0].min()) - 4, int(poly[:, 1].min()) - 3
        jx1, jy1 = int(np.ceil(poly[:, 0].max())) + 5, int(np.ceil(poly[:, 1].max())) + pad
        self.jbox = (jx0 - X0, jy0 - Y0, jx1 - X0, jy1 - Y0)        # sprite-local
        m = np.zeros((jy1 - jy0, jx1 - jx0), np.uint8)
        sh = 8
        cv2.fillPoly(m, [np.round((poly - [jx0, jy0]) * (1 << sh)).astype(np.int32)], 255, cv2.LINE_AA, sh)
        self.jmask = (m.astype(np.float32) / 255.0)
        self.jpatch = self.spr[jy0 - Y0:jy1 - Y0, jx0 - X0:jx1 - X0].astype(np.float32)
        (mlx, mly), (mrx, mry) = P(*j["mouth"][0]), P(*j["mouth"][1])
        self.m_l, self.m_r, self.m_y = mlx - jx0, mrx - jx0, 0.5 * (mly + mry) - jy0
        jyy, jxx = np.mgrid[0:jy1 - jy0, 0:jx1 - jx0].astype(np.float32)
        self.jxx, self.jyy = jxx, jyy

        # eyes (LEDs) -------------------------------------------------------------
        self.eyes = []
        for e in lm["eyes"]:
            ex, ey = P(*e["center"])
            A = e["half_width"] * s
            ir = e["iris_r"] * s
            box = (int(ex - A * 1.35) - X0, int(ey - A * 0.8) - Y0, int(ex + A * 1.35) - X0, int(ey + A * 0.8) - Y0)
            a0, b0, a1, b1 = box
            patch = self.spr[b0:b1, a0:a1].astype(np.float32)
            B, G, R = patch[..., 0], patch[..., 1], patch[..., 2]
            glow = np.clip((B - 110) / 90, 0, 1) * np.clip((B - R - 40) / 60, 0, 1)
            yy, xx = np.mgrid[b0:b1, a0:a1].astype(np.float32)
            d = np.sqrt((xx - (ex - X0)) ** 2 + (yy - (ey - Y0)) ** 2)
            iris_w = (1.0 - _smooth(1.1 * ir, 1.9 * ir, d)).astype(np.float32)
            self.eyes.append(dict(box=box, glow=glow.astype(np.float32)[..., None], iris_w=iris_w,
                                  xx=xx - a0, yy=yy - b0))
        # headphone LEDs ---------------------------------------------------------
        self.leds = []
        for bx in lm.get("headphones", []):
            a0, b0 = P(bx[0], bx[1])
            a1, b1 = P(bx[2], bx[3])
            box = (int(a0) - X0, int(b0) - Y0, int(a1) - X0, int(b1) - Y0)
            p = self.spr[box[1]:box[3], box[0]:box[2]].astype(np.float32)
            g = np.clip((p[..., 0] - 120) / 90, 0, 1) * np.clip((p[..., 0] - p[..., 2] - 50) / 60, 0, 1)
            self.leds.append((box, g.astype(np.float32)[..., None]))

        # laptop (in front of his arm) ------------------------------------------------
        lp = np.array([P(x, y) for x, y in lm["laptop"]["poly"]], np.float32)
        lx0, ly0 = int(lp[:, 0].min()) - 2, int(lp[:, 1].min()) - 2
        lx1, ly1 = min(W, int(np.ceil(lp[:, 0].max())) + 2), min(H, int(np.ceil(lp[:, 1].max())) + 2)
        m = np.zeros((ly1 - ly0, lx1 - lx0), np.uint8)
        cv2.fillPoly(m, [np.round((lp - [lx0, ly0]) * 256).astype(np.int32)], 255, cv2.LINE_AA, 8)
        self.lbox = (lx0, ly0, lx1, ly1)
        self.l_a = (m.astype(np.float32) / 255.0)[..., None]
        self.l_rgb = self.base[ly0:ly1, lx0:lx1].astype(np.float32)

        # desk reflection of the hands --------------------------------------------------
        rb = lm["reflection"]["box"]
        r0x, r0y = P(rb[0], rb[1])
        r1x, r1y = P(rb[2], rb[3])
        self.rbox = (int(r0x), int(r0y), int(r1x), min(H, int(r1y)))
        a0, b0, a1, b1 = self.rbox
        ryy, rxx = np.mgrid[b0:b1, a0:a1].astype(np.float32)
        f = 30 * s
        self.r_w = (_smooth(a0, a0 + f, rxx) * (1 - _smooth(a1 - f, a1, rxx)) * (1 - _smooth(b0 + (b1 - b0) * 0.45, b1, ryy))).astype(np.float32)
        self.r_xx, self.r_yy = rxx - a0, ryy - b0

        self.globe = _Globe(self.plate, lm["globe"], sx, sy, gtext, W, H) if "globe" in lm else None

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _sub(idx):
        ys, xs = idx
        if len(xs) == 0:
            return (0, 0, 1, 1)
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    # -------------------------------------------------------------------- parts
    def _jaw(self, spr, open_, width, level, seed):
        d = float(open_) * self.jaw_drop
        if d < 0.35:
            return
        a0, b0, a1, b1 = self.jbox
        reg = spr[b0:b1, a0:a1].astype(np.float32)
        hgt, wid = reg.shape[:2]
        jm = self.jmask[..., None]
        # 1) the space behind the plate: dark machinery
        yy = self.jyy
        depth = np.clip((yy - self.m_y) / (self.jaw_drop + 1e-3), 0, 1)[..., None]
        cav = np.array([22, 17, 19], np.float32) * (1.0 - 0.35 * depth) + np.array([10, 4, 0], np.float32)
        reg = reg * (1 - jm) + cav * jm
        # 2) LED voice grille inside the opening
        cx = 0.5 * (self.m_l + self.m_r)
        half = 0.5 * (self.m_r - self.m_l) * float(np.clip(width, 0.6, 1.2)) * 0.86
        n = 9
        rng = np.random.default_rng(seed)
        xs = np.linspace(cx - half, cx + half, n)
        prof = np.exp(-0.5 * ((xs - cx) / (half * (0.45 + 0.5 * np.clip(width - 0.6, 0, 1)) + 1e-3)) ** 2)
        lv = np.clip(level * (0.55 + 0.45 * prof) * (0.75 + 0.5 * rng.random(n)), 0.12, 1.0)
        slot_top, slot_bot = self.m_y + 0.8, self.m_y + d - 0.6
        mid = 0.5 * (slot_top + slot_bot)
        half_h = 0.5 * max(0.0, slot_bot - slot_top)
        bw = max(1.2, (2 * half) / n * 0.55)
        led = np.zeros((hgt, wid), np.float32)
        for x, l in zip(xs, lv):
            hh = max(0.8, half_h * l)
            bar = np.clip(bw / 2 + 0.5 - np.abs(self.jxx - x), 0, 1) * np.clip(hh + 0.5 - np.abs(yy - mid), 0, 1)
            led = np.maximum(led, bar)
        led *= jm[..., 0]
        bloom = cv2.GaussianBlur(led, (0, 0), 2.2 * self.s)
        col = np.array([255, 190, 70], np.float32)
        reg = reg + (led[..., None] * 1.0 + bloom[..., None] * 0.9) * col * (0.55 + 0.45 * level)
        # 3) the jaw plate itself, dropped by d (rigid - it's a machine)
        M = np.float32([[1, 0, 0], [0, 1, d]])
        pm = cv2.warpAffine(self.jmask, M, (wid, hgt), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
        pp = cv2.warpAffine(self.jpatch, M, (wid, hgt), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # a thin shadow line on the plate's top edge sells the mechanism
        edge = np.clip(1.0 - np.abs(yy - (self.m_y + d + 0.8)) / 1.6, 0, 1)[..., None] * pm * 0.45
        pp = pp * (1 - edge)
        reg = reg * (1 - pm) + pp * pm
        # soft blue spill from the grille onto the lips
        spill = cv2.GaussianBlur(led, (0, 0), 5.0 * self.s)[..., None] * 0.5 * (1 - jm) * (1 - pm)
        reg = reg + spill * col * 0.5
        spr[b0:b1, a0:a1] = np.clip(reg, 0, 255).astype(np.uint8)

    def _eyes(self, spr, dim, gx, gy, glow):
        for e in self.eyes:
            a0, b0, a1, b1 = e["box"]
            reg = spr[b0:b1, a0:a1]
            if abs(gx) + abs(gy) > 0.05:
                w = e["iris_w"]
                reg = _remap(reg, e["xx"] - gx * self.s * w, e["yy"] - gy * self.s * w)
            f = reg.astype(np.float32)
            g = e["glow"]
            k = float(np.clip(glow, 0.5, 1.6)) * (1.0 - 0.9 * float(np.clip(dim, 0, 1)))
            f = f * (1 - g) + (f * k + (k - 1) * 30.0) * g
            spr[b0:b1, a0:a1] = np.clip(f, 0, 255).astype(np.uint8)

    def _leds(self, spr, level):
        k = 0.7 + 0.75 * float(np.clip(level, 0, 1))
        for (a0, b0, a1, b1), g in self.leds:
            f = spr[b0:b1, a0:a1].astype(np.float32)
            f = f * (1 - g) + np.clip(f * k, 0, 255) * g
            spr[b0:b1, a0:a1] = f.astype(np.uint8)

    @staticmethod
    def _rigid(X, Y, rot_deg, scale, tx, ty):
        th = np.deg2rad(rot_deg)
        c, s_ = np.cos(th) * scale, np.sin(th) * scale
        return (c - 1) * X - s_ * Y + tx, s_ * X + (c - 1) * Y + ty

    # -------------------------------------------------------------------- frame
    def render(self, p: dict) -> np.ndarray:
        s = self.s
        t = float(p.get("t", 0.0))
        frame = self.plate.copy()
        if self.globe is not None:
            self.globe.draw(frame, t)

        # desk reflection follows the hands (mirrored)
        hd_tx, hd_ty = p.get("hd_tx", 0.0) * s, p.get("hd_ty", 0.0) * s
        if abs(hd_tx) + abs(hd_ty) > 0.05:
            a0, b0, a1, b1 = self.rbox
            frame[b0:b1, a0:a1] = _remap(frame[b0:b1, a0:a1], self.r_xx - hd_tx * self.r_w, self.r_yy + hd_ty * self.r_w)

        # --- face details on the (still unmoved) sprite
        spr = self.spr.copy()
        level = float(p.get("loud", 0.0))
        self._leds(spr, level)
        self._eyes(spr, p.get("blink", 0.0), p.get("gx", 0.0), p.get("gy", 0.0), p.get("glow", 1.0))
        self._jaw(spr, p.get("jaw", 0.0), p.get("jaw_w", 1.0), level, int(t * 1000))

        # --- skeleton: body, head, hands
        X0, Y0, X1, Y1 = self.sb
        gx, gy = self.gx, self.gy
        dx, dy = self._rigid(gx - self.pv_t[0], gy - self.pv_t[1], p.get("b_rot", 0.0), 1.0 + p.get("b_zoom", 0.0),
                             p.get("b_tx", 0.0) * s, p.get("b_ty", 0.0) * s)
        dx, dy = dx * self.w_body, dy * self.w_body
        a0, b0, a1, b1 = self.hbox
        sgx, sgy = gx[b0:b1, a0:a1], gy[b0:b1, a0:a1]
        hx, hy = self._rigid(sgx - self.pv_h[0], sgy - self.pv_h[1], p.get("h_rot", 0.0), 1.0,
                             p.get("h_tx", 0.0) * s, p.get("h_ty", 0.0) * s)
        yaw = p.get("h_yaw", 0.0) * s
        if abs(yaw) > 0.02:
            u = (sgx - self.h_c[0]) / self.h_rx
            hx = hx + yaw * np.clip(1.0 - u * u, 0, 1)
        dx[b0:b1, a0:a1] += hx * self.w_head
        dy[b0:b1, a0:a1] += hy * self.w_head
        a0, b0, a1, b1 = self.dbox
        sgx, sgy = gx[b0:b1, a0:a1], gy[b0:b1, a0:a1]
        kx, ky = self._rigid(sgx - self.pv_d[0], sgy - self.pv_d[1], p.get("hd_rot", 0.0), 1.0, hd_tx, hd_ty)
        dx[b0:b1, a0:a1] += kx * self.w_hands
        dy[b0:b1, a0:a1] += ky * self.w_hands
        mx, my = gx - X0 - dx, gy - Y0 - dy
        rgb = _remap(spr, mx, my)
        a = _remap(self.spr_a, mx, my, cv2.BORDER_CONSTANT)[..., None]
        reg = frame[Y0:Y1, X0:X1].astype(np.float32)
        frame[Y0:Y1, X0:X1] = (reg + (rgb.astype(np.float32) - reg) * a).astype(np.uint8)

        # laptop stays in front of the arm
        lx0, ly0, lx1, ly1 = self.lbox
        reg = frame[ly0:ly1, lx0:lx1].astype(np.float32)
        frame[ly0:ly1, lx0:lx1] = (reg + (self.l_rgb - reg) * self.l_a).astype(np.uint8)

        # TV camera: slow push-in, and cuts to a closer shot between stories
        z = float(p.get("cam_z", 1.0))
        if z > 1.0005:
            cx, cy = float(p.get("cam_x", 0.5)) * self.W, float(p.get("cam_y", 0.4)) * self.H
            M = np.float32([[z, 0, cx - z * cx], [0, z, cy - z * cy]])
            frame = cv2.warpAffine(frame, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return frame
