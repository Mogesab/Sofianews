"""
The over-the-shoulder screen.

A framed monitor sitting above the presenter's shoulder that plays story-matched
footage while he reads. The footage itself comes from broll.build_strip(), which
has already cropped it to exactly this box and laid it out on the show's
timeline, so here we only have to read frame N and dress it up:

  * rounded bezel with a blue rim light and a soft drop shadow
  * a caption strip along the bottom of the screen with the story label
  * a slight glass sheen + vignette so it reads as a monitor, not a pasted video
  * it slides/scales in when a story starts and dips out between stories

Each render worker opens its own handle on the strip. Chunks are contiguous, so
we read sequentially and only seek when a worker jumps to a new chunk.
"""
from __future__ import annotations

import bisect
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .overlays import RED, NAVY, _bgra, blend


def _ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


class InsetScreen:
    # geometry as a fraction of the frame: upper right, clear of the presenter's
    # head and below the date pill. Height is derived so the screen is true 16:9
    # and the footage is not over-cropped.
    BOX = (0.642, 0.108, 0.968)               # x0, y0, x1

    @staticmethod
    def box_size(W: int, H: int):
        """(x0, y0, w, h) of the picture area. Used by the renderer and by the
        b-roll builder, so the strip is prepared at exactly the right size."""
        x0 = int(InsetScreen.BOX[0] * W)
        y0 = int(InsetScreen.BOX[1] * H)
        w = (int(InsetScreen.BOX[2] * W) - x0) // 2 * 2
        h = int(round(w * 9 / 16)) // 2 * 2
        return x0, y0, w, h

    def __init__(self, strip_path: str, spans: list, W: int, H: int, fonts_dir: Path,
                 timeline: dict | None = None, fps: int = 30):
        self.W, self.H, self.fps = W, H, fps
        self.k = W / 1920.0
        self.x0, self.y0, self.bw, self.bh = self.box_size(W, H)
        self.spans = spans or []
        self.starts = [s[0] for s in self.spans]
        self.bold = str(Path(fonts_dir) / "LiberationSans-Bold.ttf")
        self.cap = cv2.VideoCapture(strip_path) if strip_path else None
        self.next_i = -1
        self.last = None
        self._mask()
        self._chrome()
        self._captions(timeline or {})

    def _px(self, v):
        return max(1, int(round(v * self.k)))

    # ------------------------------------------------------------------ pieces
    def _mask(self):
        """Rounded-corner alpha for the picture area, plus a glass sheen."""
        r = self._px(12)
        m = Image.new("L", (self.bw, self.bh), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, self.bw - 1, self.bh - 1), radius=r, fill=255)
        self.mask = (np.asarray(m).astype(np.float32) / 255.0)[..., None]

        yy, xx = np.mgrid[0:self.bh, 0:self.bw].astype(np.float32)
        u = xx / self.bw - 0.5
        v = yy / self.bh - 0.5
        vig = np.clip(1.0 - 1.15 * (u * u + v * v * 0.8), 0.55, 1.0)          # screen vignette
        sheen = np.clip(1.0 - np.abs((u * 1.4 + v * 2.2) - 0.55) / 0.30, 0, 1) ** 2 * 0.055
        self.vig = vig[..., None].astype(np.float32)
        self.sheen = sheen[..., None].astype(np.float32)

    def _chrome(self):
        """Bezel + shadow, drawn once as an RGBA layer with a hole for the picture."""
        b = self._px(7)                                   # bezel thickness
        pad = self._px(26)                                # room for the shadow
        w, h = self.bw + 2 * b + 2 * pad, self.bh + 2 * b + 2 * pad
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle(
            (pad + self._px(4), pad + self._px(8), pad + self.bw + 2 * b, pad + self.bh + 2 * b + self._px(8)),
            radius=self._px(18), fill=(0, 0, 0, 150))
        img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(self._px(11))))

        d = ImageDraw.Draw(img)
        x0, y0 = pad, pad
        x1, y1 = pad + self.bw + 2 * b, pad + self.bh + 2 * b
        d.rounded_rectangle((x0, y0, x1, y1), radius=self._px(17), fill=(14, 26, 58, 255))
        d.rounded_rectangle((x0, y0, x1, y1), radius=self._px(17), outline=(70, 140, 235, 255), width=self._px(2))
        d.rounded_rectangle((x0 + b - 1, y0 + b - 1, x1 - b + 1, y1 - b + 1), radius=self._px(11),
                            outline=(6, 14, 34, 255), width=self._px(2))
        # cut the picture hole back out
        hole = Image.new("L", (w, h), 255)
        ImageDraw.Draw(hole).rounded_rectangle((x0 + b, y0 + b, x1 - b, y1 - b), radius=self._px(12), fill=0)
        img.putalpha(Image.fromarray(np.minimum(np.asarray(img.getchannel("A")), np.asarray(hole))))

        self.chrome = _bgra(img)
        self.chrome_off = (pad + b, pad + b)              # picture origin inside the layer

        # tag on the top bezel. It says what the screen is actually showing -
        # real footage is labelled as such, a generated backdrop is not passed
        # off as file footage.
        self.tags = {"clip": self._make_tag("FILE FOOTAGE", RED),
                     "generated": self._make_tag("STUDIO GRAPHIC", (40, 90, 190))}

    def _make_tag(self, text: str, colour):
        f = ImageFont.truetype(self.bold, self._px(15))
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        tw2, th = int(tmp.textlength(text, font=f)) + self._px(34), self._px(24)
        t = Image.new("RGBA", (tw2, th), (0, 0, 0, 0))
        td = ImageDraw.Draw(t)
        td.rounded_rectangle((0, 0, tw2 - 1, th - 1), radius=self._px(4), fill=tuple(colour) + (240,))
        td.ellipse((self._px(9), th // 2 - self._px(4), self._px(17), th // 2 + self._px(4)), fill=(255, 255, 255, 235))
        td.text((self._px(23), th // 2 + 1), text, font=f, fill="white", anchor="lm")
        return _bgra(t)

    def _captions(self, timeline: dict):
        """A small label strip inside the screen naming the story."""
        self.caps, self.cap_starts = [], []
        f = ImageFont.truetype(self.bold, self._px(19))
        seen = None
        for st in timeline.get("stories", []):
            label = str(st.get("headline", "")).upper()
            if not label or label == seen:
                continue
            seen = label
            h = self._px(34)
            img = Image.new("RGBA", (self.bw, h), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rectangle((0, 0, self.bw, h), fill=NAVY + (205,))
            d.rectangle((0, 0, self._px(5), h), fill=RED + (255,))
            txt = label
            while d.textlength(txt, font=f) > self.bw - self._px(26) and len(txt) > 6:
                txt = txt[:-2].rstrip()
            d.text((self._px(15), h // 2 + 1), txt, font=f, fill="white", anchor="lm")
            self.caps.append(_bgra(img))
            self.cap_starts.append(float(st.get("start", 0.0)))

    # ------------------------------------------------------------- visibility
    def _span(self, t: float):
        i = max(0, bisect.bisect_right(self.starts, t) - 1)
        sp = self.spans[i]
        return (sp + ("clip",))[:4] if len(sp) < 4 else sp      # older 3-tuples still work

    def alpha(self, t: float) -> float:
        """0 when there is no footage for this moment, easing in/out at the edges."""
        if not self.spans:
            return 0.0
        s0, s1, ok, _ = self._span(t)
        if not ok:
            return 0.0
        fade = 0.45
        return _ease(min((t - s0) / fade, (s1 - t) / fade, 1.0))

    # ------------------------------------------------------------------ frames
    def seek(self, i: int) -> None:
        if self.cap is not None and i != self.next_i:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.next_i = i

    def _read(self, i: int):
        if self.cap is None:
            return None
        if i != self.next_i:
            self.seek(i)
        ok, fr = self.cap.read()
        if not ok:
            return self.last
        self.next_i = i + 1
        if fr.shape[1] != self.bw or fr.shape[0] != self.bh:
            fr = cv2.resize(fr, (self.bw, self.bh), interpolation=cv2.INTER_AREA)
        self.last = fr
        return fr

    def draw(self, frame: np.ndarray, i: int, t: float) -> None:
        a = self.alpha(t)
        if a <= 0.004:
            if self.cap is not None:             # keep the strip in step while hidden
                if i != self.next_i:
                    self.seek(i)
                self.cap.grab()
                self.next_i = i + 1
            return
        pic = self._read(i)
        if pic is None:
            return

        # picture treatment: vignette + faint glass sheen
        p = pic.astype(np.float32) * self.vig
        p = np.clip(p + 255.0 * self.sheen, 0, 255)

        # the box grows in slightly as it appears, like a DVE wipe
        scale = 0.965 + 0.035 * a
        bw2, bh2 = max(2, int(self.bw * scale)), max(2, int(self.bh * scale))
        cx = self.x0 + self.bw // 2
        cy = self.y0 + self.bh // 2

        layer = np.dstack([p, self.mask[..., 0] * 255.0]).astype(np.uint8)
        # caption strip inside the screen
        if self.caps:
            j = max(0, bisect.bisect_right(self.cap_starts, t) - 1)
            capimg = self.caps[j]
            ch = capimg.shape[0]
            ca = _ease((t - self.cap_starts[j] - 0.35) / 0.5)
            if ca > 0:
                sub = layer[self.bh - ch - self._px(6):self.bh - self._px(6)]
                al = capimg[..., 3:4].astype(np.float32) / 255.0 * ca
                sub[..., :3] = (sub[..., :3].astype(np.float32) * (1 - al)
                                + capimg[..., :3].astype(np.float32) * al).astype(np.uint8)

        if scale != 1.0:
            layer = cv2.resize(layer, (bw2, bh2), interpolation=cv2.INTER_LINEAR)
        px, py = cx - bw2 // 2, cy - bh2 // 2

        ox, oy = self.chrome_off
        blend(frame, self.chrome, px - ox - (self.bw - bw2) // 2, py - oy - (self.bh - bh2) // 2, a)
        blend(frame, layer, px, py, a)
        tag = self.tags.get(self._span(t)[3], self.tags["clip"])
        blend(frame, tag, self.x0 + self._px(12), self.y0 - self._px(15), a)
