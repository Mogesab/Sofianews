"""
Broadcast graphics: studio bug, date, animated lower third (headline tab +
key-point bar that changes with what is being said) and a scrolling ticker.
Drawn with Pillow once per state, then alpha-blended per frame with numpy.
"""
from __future__ import annotations

import bisect
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

NAVY = (8, 22, 60)
NAVY_TXT = (11, 31, 75)
RED = (215, 38, 61)
ICE = (150, 195, 255)


def _ease(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


def _bgra(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGBA"))
    return np.ascontiguousarray(a[..., [2, 1, 0, 3]])


def blend(frame: np.ndarray, layer: np.ndarray, x: int, y: int, alpha: float = 1.0) -> None:
    H, W = frame.shape[:2]
    h, w = layer.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0 or alpha <= 0:
        return
    sub = layer[y0 - y:y1 - y, x0 - x:x1 - x]
    a = sub[..., 3:4].astype(np.float32) * (alpha / 255.0)
    reg = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (reg + (sub[..., :3].astype(np.float32) - reg) * a).astype(np.uint8)


class Overlays:
    def __init__(self, fonts_dir: Path, W: int, H: int, timeline: dict, studio_name: str,
                 date_label: str, show_ai_label: bool = True, topic_label: str = "WORLD NEWS",
                 show_countdown: bool = True, countdown_seconds: float = 0.0,
                 font_bold: str | None = None, font_regular: str | None = None, labels: dict | None = None):
        self.W, self.H = W, H
        self.k = W / 1920.0
        self.bold = font_bold or str(Path(fonts_dir) / "LiberationSans-Bold.ttf")
        self.reg = font_regular or str(Path(fonts_dir) / "LiberationSans-Regular.ttf")
        self.labels = dict(live="LIVE", time_left="TIME LEFT", story="STORY {i} / {n}",
                           ai="AI-GENERATED PRESENTER")
        self.labels.update(labels or {})
        self.studio = studio_name
        self.topic = topic_label or "NEWS"
        self.date_label = date_label
        self.show_ai = show_ai_label
        self.show_countdown = show_countdown
        self.countdown_total = float(countdown_seconds or timeline.get("duration") or 60.0)
        self._cd_ref = f"{int(self.countdown_total) // 60}:00"   # widest text this clock will ever show, for a stable badge width
        self.states = self._states(timeline)
        self.starts = [s["start"] for s in self.states]
        self.bug = self._make_bug()
        self.datepill = self._make_date()
        self._build_ticker(timeline)

    # ------------------------------------------------------------------ fonts
    def _f(self, size, bold=True):
        return ImageFont.truetype(self.bold if bold else self.reg, max(8, int(round(size * self.k))))

    def _px(self, v):
        return int(round(v * self.k))

    @staticmethod
    def _spaced(text: str) -> str:
        """'WORLD NEWS' -> 'W O R L D   N E W S' (matches the bug's letter-spaced style).
        Ethiopic text is left as it is - letter-spacing breaks up its syllables."""
        if any("ሀ" <= ch <= "፿" for ch in text):
            return text
        return "   ".join(" ".join(word) for word in text.split())

    # ----------------------------------------------------------------- states
    @staticmethod
    def _states(timeline):
        out = []
        for c in timeline["cues"]:
            key = (c["story"], c["highlight"])
            if out and out[-1]["key"] == key:
                out[-1]["end"] = c["end"]
                continue
            out.append(dict(key=key, start=c["start"], end=c["end"], headline=c["headline"], highlight=c["highlight"]))
        return out

    def _state_at(self, t):
        i = bisect.bisect_right(self.starts, t) - 1
        return max(i, 0)

    # -------------------------------------------------------------------- bug
    def _make_bug(self):
        k = self.k
        w, h = self._px(470), self._px(160)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        bh = self._px(64)
        f1, f2 = self._f(30), self._f(16)
        name = self.studio.upper()
        tw = d.textlength(name, font=f1)
        pw = int(self._px(64) + self._px(18) + tw + self._px(26))
        d.rounded_rectangle((0, 0, pw, bh), radius=self._px(6), fill=NAVY + (215,))
        d.rounded_rectangle((0, 0, self._px(64), bh), radius=self._px(6), fill=RED + (255,))
        fm = self._f(34)
        first = self.studio.split()[0] if self.studio.split() else "MA"
        initials = first.upper() if len(first) <= 3 else "".join(wd[0] for wd in self.studio.split()[:2]).upper()
        d.text((self._px(32), bh // 2), initials, font=fm, fill="white", anchor="mm")
        d.text((self._px(64 + 18), self._px(9)), name, font=f1, fill="white")
        d.text((self._px(64 + 19), self._px(43)), self._spaced(self.topic), font=f2, fill=ICE)
        if self.show_ai:
            f3 = self._f(14)
            label = "AI-GENERATED PRESENTER"
            lw = d.textlength(label, font=f3)
            y = bh + self._px(8)
            d.rounded_rectangle((0, y, lw + self._px(22), y + self._px(24)), radius=self._px(4), fill=(0, 0, 0, 120))
            d.text((self._px(11), y + self._px(12)), label, font=f3, fill=(255, 255, 255, 225), anchor="lm")
        return _bgra(img)

    def _make_date(self):
        f = self._f(22)
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        tw = int(tmp.textlength(self.date_label, font=f))
        w, h = tw + self._px(36), self._px(44)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=self._px(6), fill=NAVY + (205,))
        d.rectangle((0, 0, self._px(6), h), fill=RED + (255,))
        d.text((self._px(22), h // 2), self.date_label, font=f, fill="white", anchor="lm")
        return _bgra(img)

    # ------------------------------------------------------------ lower third
    @staticmethod
    def _best_split(words, font, tmp):
        """Split words into two lines so the two lines come out as even as possible."""
        best, best_cost = None, None
        for i in range(1, len(words)):
            a, b = " ".join(words[:i]), " ".join(words[i:])
            wa, wb = tmp.textlength(a, font=font), tmp.textlength(b, font=font)
            cost = max(wa, wb) * 1000 + abs(wa - wb)      # fit first, balance second
            if best_cost is None or cost < best_cost:
                best, best_cost = (a, b), cost
        return best

    @lru_cache(maxsize=24)
    def _panel(self, headline: str, highlight: str):
        pw = self.W - 2 * self._px(72)
        tab_h = self._px(42)
        pad = self._px(14)
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

        # ---- headline tab: shrink the font rather than let it run off the panel
        htxt = headline.upper()
        hsize = 25
        cut = self._px(18)
        while hsize > 15 and tmp.textlength(htxt, font=self._f(hsize)) + self._px(56) + cut > pw * 0.72:
            hsize -= 1
        hf = self._f(hsize)
        tab_w = int(tmp.textlength(htxt, font=hf)) + self._px(56)

        # ---- key-point text: fit on one line, else wrap onto two. Never cut a word off.
        maxw = pw - self._px(130)
        words = highlight.split()
        lines, f = None, None
        for size in range(42, 27, -2):                       # one line, big type
            fnt = self._f(size)
            if tmp.textlength(highlight, font=fnt) <= maxw:
                lines, f = [highlight], fnt
                break
        if lines is None and len(words) > 1:
            for size in range(36, 21, -2):                   # two lines, slightly smaller
                fnt = self._f(size)
                a, b = self._best_split(words, fnt, tmp)
                if max(tmp.textlength(a, font=fnt), tmp.textlength(b, font=fnt)) <= maxw:
                    lines, f = [a, b], fnt
                    break
        if lines is None:                                    # last resort only
            f = self._f(22)
            a, b = self._best_split(words, f, tmp) if len(words) > 1 else (highlight, "")
            out = []
            for ln in ([a, b] if b else [a]):
                while ln and tmp.textlength(ln, font=f) > maxw and len(ln) > 8:
                    ln = ln[:-2].rstrip()
                out.append(ln)
            out[-1] = out[-1] + "…"
            lines = out

        line_h = int(f.size * 1.16)
        bar_h = max(self._px(74), line_h * len(lines) + self._px(30))
        txt_w = max(tmp.textlength(ln, font=f) for ln in lines)
        bw = int(min(pw, max(tab_w + cut + self._px(60), txt_w + self._px(110))))
        total_w = max(bw, tab_w + cut)

        img = Image.new("RGBA", (total_w + 2 * pad, tab_h + bar_h + 2 * pad), (0, 0, 0, 0))
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rectangle((pad + 4, pad + 6, pad + total_w, pad + tab_h + bar_h + 6), fill=(0, 0, 0, 120))
        img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(self._px(7))))
        d = ImageDraw.Draw(img)

        d.polygon([(pad, pad), (pad + tab_w + cut, pad), (pad + tab_w, pad + tab_h), (pad, pad + tab_h)], fill=RED + (255,))
        d.text((pad + self._px(22), pad + tab_h // 2 + 1), htxt, font=hf, fill="white", anchor="lm")
        y0 = pad + tab_h
        d.rectangle((pad, y0, pad + bw, y0 + bar_h), fill=(244, 247, 252, 246))
        d.rectangle((pad, y0, pad + self._px(12), y0 + bar_h), fill=NAVY_TXT + (255,))
        d.rectangle((pad, y0 + bar_h - self._px(4), pad + bw, y0 + bar_h), fill=(30, 100, 220, 255))
        cy = y0 + (bar_h - self._px(4)) // 2
        first = cy - (len(lines) - 1) * line_h // 2
        for j, ln in enumerate(lines):
            d.text((pad + self._px(36), first + j * line_h), ln, font=f, fill=NAVY_TXT, anchor="lm")
        return _bgra(img), pad

    # ----------------------------------------------------------------- ticker
    def _build_ticker(self, timeline):
        self.t_h = self._px(48)
        tmp0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        label_size = 26
        while label_size > 15 and tmp0.textlength(self.topic, font=self._f(label_size)) > self._px(190):
            label_size -= 1
        self.t_label_font = self._f(label_size)
        self.t_label_w = int(tmp0.textlength(self.topic, font=self.t_label_font)) + self._px(40)
        self.t_speed = self._px(120)                       # px / second
        items = [i for i in timeline.get("ticker", []) if i] or [self.studio.upper() + " " + self.topic]
        f = self._f(25)
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        gap = self._px(70)
        win_w = self.W - self.t_label_w
        seq = list(items)
        total = lambda s: sum(tmp.textlength(x.upper(), font=f) + gap for x in s)
        while total(seq) < win_w + self._px(200):
            seq += items
        one = int(total(seq))
        strip = Image.new("RGBA", (one * 2, self.t_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(strip)
        x = 0
        for rep in range(2):
            for it in seq:
                txt = it.upper()
                d.text((x, self.t_h // 2 + 1), txt, font=f, fill="white", anchor="lm")
                x += int(tmp.textlength(txt, font=f))
                cx = x + gap // 2
                s = self._px(6)
                d.polygon([(cx, self.t_h // 2 - s), (cx + s, self.t_h // 2), (cx, self.t_h // 2 + s), (cx - s, self.t_h // 2)], fill=RED + (255,))
                x += gap
        self.t_strip = _bgra(strip)
        self.t_one = one
        bg = Image.new("RGBA", (self.W, self.t_h), NAVY + (238,))
        bd = ImageDraw.Draw(bg)
        bd.rectangle((0, 0, self.t_label_w, self.t_h), fill=RED + (255,))
        bd.text((self.t_label_w // 2, self.t_h // 2 + 1), self.topic, font=self.t_label_font, fill="white", anchor="mm")
        bd.rectangle((0, 0, self.W, self._px(3)), fill=(30, 100, 220, 255))
        self.t_bg = _bgra(bg)

    # --------------------------------------------------------------- countdown
    @lru_cache(maxsize=512)
    def _countdown_badge(self, remaining: int):
        urgent = remaining <= 10
        mm, ss = divmod(remaining, 60)
        txt = f"{mm}:{ss:02d}"
        f = self._f(32)
        sub = self._f(13)
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        tw = tmp.textlength(self._cd_ref, font=f)  # fixed reference width so the badge never jitters
        sub_txt = "TIME LEFT"
        sw = tmp.textlength(sub_txt, font=sub)
        pad_x, pad_y = self._px(22), self._px(10)
        w = int(max(tw, sw)) + pad_x * 2
        h = self._px(68)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        bgcol = (RED if urgent else NAVY) + (225,)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=self._px(10), fill=bgcol)
        if not urgent:
            d.rectangle((0, 0, w - 1, self._px(4)), fill=RED + (255,))
        d.text((w / 2, h * 0.40), txt, font=f, fill="white", anchor="mm")
        d.text((w / 2, h * 0.80), sub_txt, font=sub, fill=(255, 225, 225) if urgent else ICE, anchor="mm")
        return _bgra(img)

    # ------------------------------------------------------------------ frame
    def draw(self, frame: np.ndarray, t: float) -> None:
        k = self.k
        # studio bug + date fade in at the start
        a = _ease((t - 0.2) / 0.8)
        blend(frame, self.bug, self._px(56), self._px(40), a)
        blend(frame, self.datepill, self.W - self._px(56) - self.datepill.shape[1], self._px(40), a)

        # countdown clock, top center, timed to the actual video length
        if self.show_countdown:
            remaining = max(0, int(self.countdown_total - t))
            badge = self._countdown_badge(remaining)
            bx = (self.W - badge.shape[1]) // 2
            blend(frame, badge, bx, self._px(40), a)

        # ticker slides up
        e = _ease((t - 0.4) / 0.8)
        ty = self.H - self._px(16) - self.t_h + int((1 - e) * self._px(90))
        blend(frame, self.t_bg, 0, ty, 1.0)
        off = int(t * self.t_speed) % self.t_one
        win = self.W - self.t_label_w
        blend(frame, self.t_strip[:, off:off + win], self.t_label_w, ty, 1.0)
        # (label block stays on top of the crawl)
        blend(frame, self.t_bg[:, :self.t_label_w], 0, ty, 1.0)

        # lower third: changes with each key point
        i = self._state_at(t)
        st = self.states[i]
        e2 = _ease((t - st["start"]) / 0.32) if i > 0 else _ease((t - 1.0) / 0.6)
        if e2 <= 0:
            return
        panel, pad = self._panel(st["headline"], st["highlight"])
        px = self._px(72) - pad + int((1 - e2) * self._px(46))
        py = ty - self._px(12) - (panel.shape[0] - pad) + 0
        blend(frame, panel, px, py, e2)


class PortraitOverlays(Overlays):
    """Graphics laid out for a vertical 1080x1920 Short.

    Sizes are designed on a 1080-wide canvas (the landscape layout scaled them
    from 1920, which made everything tiny on a phone). The footer is one solid
    full-width block - story tab, a tall key-point bar with big type, and the
    ticker - so it also cleanly covers any lower third printed on the studio
    photo itself."""

    M = 36                     # side margin
    TOP = 100                  # top row
    F_TOP = 1538               # footer: tab row
    TAB_H = 62
    BAR_H = 176                # key-point bar
    TICK_H = 68                # ticker

    def __init__(self, fonts_dir: Path, W: int, H: int, timeline: dict, studio_name: str,
                 date_label: str, show_ai_label: bool = True, topic_label: str = "WORLD NEWS",
                 show_countdown: bool = True, countdown_seconds: float = 0.0,
                 font_bold: str | None = None, font_regular: str | None = None, labels: dict | None = None):
        super().__init__(fonts_dir, W, H, timeline, studio_name, date_label, show_ai_label, topic_label,
                         show_countdown, countdown_seconds, font_bold, font_regular, labels)
        # rebuild everything at the portrait scale
        self.k = W / 1080.0
        self.n_stories = max(1, len(timeline.get("stories", [])) - 2)
        self.bug = self._make_portrait_bug()
        self.datepill = self._make_portrait_date()
        self.ailabel = self._make_ai()
        self._build_portrait_ticker(timeline)
        self.footer_bg = self._make_footer_bg()
        self.dot = self._make_dot()

    # ------------------------------------------------------------------ top row
    def _make_portrait_bug(self):
        h = self._px(76)
        live_w = self._px(150)
        f1, f2 = self._f(34), self._f(22)
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        name = self.studio.upper()
        sub = self._spaced(self.topic)
        tw = max(tmp.textlength(name, font=f1), tmp.textlength(sub, font=f2))
        w = int(live_w + self._px(22) + tw + self._px(26))
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=self._px(8), fill=NAVY + (232,))
        d.rounded_rectangle((0, 0, live_w, h - 1), radius=self._px(8), fill=RED + (255,))
        d.rectangle((live_w - self._px(10), 0, live_w, h - 1), fill=RED + (255,))
        d.text((self._px(96), h // 2 + 1), self.labels["live"], font=self._f(30), fill="white", anchor="mm")
        d.text((live_w + self._px(22), self._px(10)), name, font=f1, fill="white")
        d.text((live_w + self._px(23), self._px(58)), sub, font=f2, fill=ICE, anchor="lm")
        return _bgra(img)

    def _make_dot(self):
        r = self._px(10)
        img = Image.new("RGBA", (4 * r, 4 * r), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((r, r, 3 * r, 3 * r), fill=(255, 255, 255, 255))
        glow = img.filter(ImageFilter.GaussianBlur(r * 0.6))
        return _bgra(Image.alpha_composite(glow, img))

    def _make_portrait_date(self):
        f = self._f(24)
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        tw = int(tmp.textlength(self.date_label, font=f))
        w, h = tw + self._px(40), self._px(48)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=self._px(7), fill=NAVY + (215,))
        d.rectangle((0, 0, self._px(7), h), fill=RED + (255,))
        d.text((self._px(24), h // 2), self.date_label, font=f, fill="white", anchor="lm")
        return _bgra(img)

    def _make_ai(self):
        if not self.show_ai:
            return None
        f = self._f(18)
        label = self.labels["ai"]
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        w, h = int(tmp.textlength(label, font=f)) + self._px(26), self._px(34)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=self._px(5), fill=(0, 0, 0, 140))
        d.text((self._px(13), h // 2), label, font=f, fill=(255, 255, 255, 230), anchor="lm")
        return _bgra(img)

    @lru_cache(maxsize=512)
    def _portrait_countdown(self, remaining: int):
        urgent = remaining <= 10
        mm, ss = divmod(remaining, 60)
        f, sub = self._f(40), self._f(19)
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        w = int(max(tmp.textlength(self._cd_ref, font=f), tmp.textlength(self.labels["time_left"], font=sub))) + self._px(44)
        h = self._px(76)
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=self._px(10), fill=(RED if urgent else NAVY) + (228,))
        if not urgent:
            d.rectangle((self._px(8), 0, w - self._px(8), self._px(5)), fill=RED + (255,))
        d.text((w / 2, h * 0.41), f"{mm}:{ss:02d}", font=f, fill="white", anchor="mm")
        d.text((w / 2, h * 0.80), self.labels["time_left"], font=sub, fill=(255, 225, 225) if urgent else ICE, anchor="mm")
        return _bgra(img)

    # ------------------------------------------------------------------ footer
    def _make_footer_bg(self):
        W = self.W
        th, bh = self._px(self.TAB_H), self._px(self.BAR_H)
        img = Image.new("RGBA", (W, th + bh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, W, th), fill=NAVY + (242,))
        d.rectangle((0, th, W, th + bh), fill=(245, 247, 252, 255))
        d.rectangle((0, th, self._px(16), th + bh), fill=NAVY_TXT + (255,))
        d.rectangle((0, th + bh - self._px(6), W, th + bh), fill=(30, 100, 220, 255))
        return _bgra(img)

    @lru_cache(maxsize=32)
    def _tab(self, headline: str, story: int):
        th = self._px(self.TAB_H)
        maxw = self.W * 0.70
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        txt = headline.upper()
        size = 36
        while size > 22 and tmp.textlength(txt, font=self._f(size)) + self._px(70) > maxw:
            size -= 1
        f = self._f(size)
        tw = int(tmp.textlength(txt, font=f)) + self._px(64)
        cut = self._px(26)
        img = Image.new("RGBA", (self.W, th), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.polygon([(0, 0), (tw + cut, 0), (tw, th), (0, th)], fill=RED + (255,))
        d.text((self._px(self.M), th // 2 + 1), txt, font=f, fill="white", anchor="lm")
        if 1 <= story <= self.n_stories:
            d.text((self.W - self._px(self.M), th // 2 + 1), self.labels["story"].format(i=story, n=self.n_stories),
                   font=self._f(24), fill=ICE, anchor="rm")
        return _bgra(img)

    @lru_cache(maxsize=64)
    def _keytext(self, highlight: str):
        bh = self._px(self.BAR_H) - self._px(6)
        maxw = self.W - self._px(56) - self._px(self.M) - self._px(10)
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        words = highlight.upper().split()
        text = " ".join(words)
        lines, f = None, None
        for size in range(66, 47, -2):                 # one line, big type
            fnt = self._f(size)
            if tmp.textlength(text, font=fnt) <= maxw:
                lines, f = [text], fnt
                break
        if lines is None and len(words) > 1:           # two lines
            for size in range(58, 37, -2):
                fnt = self._f(size)
                a, b = self._best_split(words, fnt, tmp)
                if max(tmp.textlength(a, font=fnt), tmp.textlength(b, font=fnt)) <= maxw:
                    lines, f = [a, b], fnt
                    break
        if lines is None:                              # last resort only
            f = self._f(38)
            a, b = self._best_split(words, f, tmp) if len(words) > 1 else (text, "")
            out = []
            for ln in ([a, b] if b else [a]):
                while ln and tmp.textlength(ln, font=f) > maxw and len(ln) > 8:
                    ln = ln[:-2].rstrip()
                out.append(ln)
            out[-1] = out[-1] + "…"
            lines = out
        img = Image.new("RGBA", (self.W, bh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        line_h = int(f.size * 1.12)
        first = bh // 2 - (len(lines) - 1) * line_h // 2
        for j, ln in enumerate(lines):
            d.text((self._px(56), first + j * line_h), ln, font=f, fill=NAVY_TXT, anchor="lm")
        return _bgra(img)

    def _build_portrait_ticker(self, timeline):
        self.t_h = self._px(self.TICK_H)
        tmp = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        label_size = 30
        while label_size > 18 and tmp.textlength(self.topic, font=self._f(label_size)) > self._px(300):
            label_size -= 1
        self.t_label_font = self._f(label_size)
        self.t_label_w = int(tmp.textlength(self.topic, font=self.t_label_font)) + self._px(48)
        self.t_speed = self._px(150)                    # px / second
        items = [i for i in timeline.get("ticker", []) if i] or [self.studio.upper() + " " + self.topic]
        f = self._f(32)
        gap = self._px(80)
        win_w = self.W - self.t_label_w
        seq = list(items)
        total = lambda s: sum(tmp.textlength(x.upper(), font=f) + gap for x in s)
        while total(seq) < win_w + self._px(200):
            seq += items
        one = int(total(seq))
        strip = Image.new("RGBA", (one * 2, self.t_h), (0, 0, 0, 0))
        d = ImageDraw.Draw(strip)
        x = 0
        for _ in range(2):
            for it in seq:
                txt = it.upper()
                d.text((x, self.t_h // 2 + 1), txt, font=f, fill="white", anchor="lm")
                x += int(tmp.textlength(txt, font=f))
                cx, s = x + gap // 2, self._px(8)
                d.polygon([(cx, self.t_h // 2 - s), (cx + s, self.t_h // 2), (cx, self.t_h // 2 + s),
                           (cx - s, self.t_h // 2)], fill=RED + (255,))
                x += gap
        self.t_strip = _bgra(strip)
        self.t_one = one
        bg = Image.new("RGBA", (self.W, self.t_h), (6, 16, 44, 250))
        bd = ImageDraw.Draw(bg)
        bd.rectangle((0, 0, self.t_label_w, self.t_h), fill=RED + (255,))
        bd.text((self.t_label_w // 2, self.t_h // 2 + 1), self.topic, font=self.t_label_font, fill="white", anchor="mm")
        self.t_bg = _bgra(bg)

    # -------------------------------------------------------------------- frame
    def draw(self, frame: np.ndarray, t: float) -> None:
        M = self._px(self.M)
        a = _ease((t - 0.2) / 0.8)
        top = self._px(self.TOP)
        blend(frame, self.bug, M, top, a)
        pulse = 0.35 + 0.65 * (0.5 + 0.5 * np.cos(2 * np.pi * t / 1.2))          # blinking LIVE dot
        dh = self.dot.shape[0]
        blend(frame, self.dot, M + self._px(34) - dh // 2, top + self._px(38) - dh // 2, a * pulse)
        y2 = top + self.bug.shape[0] + self._px(14)
        blend(frame, self.datepill, M, y2, a)
        if self.ailabel is not None:
            blend(frame, self.ailabel, M, y2 + self.datepill.shape[0] + self._px(10), a * 0.95)
        if self.show_countdown:
            remaining = max(0, int(np.ceil(self.countdown_total - t - 1e-6)))
            badge = self._portrait_countdown(min(remaining, int(self.countdown_total)))
            blend(frame, badge, self.W - M - badge.shape[1], top, a)

        # footer: rises as one block at the start, then stays put
        e = _ease((t - 0.3) / 0.7)
        fy = self._px(self.F_TOP) + int((1 - e) * self._px(420))
        blend(frame, self.footer_bg, 0, fy, 1.0)
        ty = fy + self._px(self.TAB_H) + self._px(self.BAR_H)
        blend(frame, self.t_bg, 0, ty, 1.0)
        off = int(t * self.t_speed) % self.t_one
        win = self.W - self.t_label_w
        blend(frame, self.t_strip[:, off:off + win], self.t_label_w, ty, 1.0)

        i = self._state_at(t)
        st = self.states[i]
        story = st["key"][0]
        # story tab: slides in from the left when the story changes
        j = i
        while j > 0 and self.states[j - 1]["key"][0] == story:
            j -= 1
        s_start = self.states[j]["start"] if j > 0 else 1.0
        e_tab = _ease((t - s_start) / 0.35)
        if e_tab > 0:
            blend(frame, self._tab(st["headline"], story), -int((1 - e_tab) * self._px(500)), fy, e_tab)
        # key point: slides in from the right and fades up each time it changes
        e2 = _ease((t - st["start"]) / 0.3) if i > 0 else _ease((t - 1.0) / 0.5)
        if e2 > 0:
            blend(frame, self._keytext(st["highlight"]), int((1 - e2) * self._px(60)), fy + self._px(self.TAB_H), e2)
