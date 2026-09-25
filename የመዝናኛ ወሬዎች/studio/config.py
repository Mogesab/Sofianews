"""Settings + API-key lookup."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output"
WORK = ROOT / "output" / "_work"

DEFAULTS = {
    "studio_name": "MA Studio",
    "topic_label": "WORLD NEWS",         # short tag shown in the studio bug + ticker + on-screen text
    "gemini_api_key": "",
    "gemini_model": "gemini-2.5-flash",
    "voice": "en-US-AndrewNeural",       # natural American English male voice
    "target_minutes": 1.0,               # 1.0 = a 60-second short; use 5 (etc.) for a long-form broadcast
    "width": 1080,                       # 1080x1920 = vertical 9:16, ready for YouTube Shorts
    "height": 1920,
    "fps": 30,
    "lipsync": "auto",                   # "auto" = words->phonemes->mouth shapes; "audio" = old loudness-only
    "lipsync_intensity": 1.0,            # 0.7 = subtle, 1.0 = natural, 1.3 = very expressive
    "lipsync_lead_ms": 25,               # mouth leads the sound by this much (real speakers do)
    "presenter_image": "journalist.png",
    "presenter_today": "",               # set to a name in assets/presenters/ to force that photo today
    "presenter_rotation": True,          # if presenter_today is empty, rotate daily through assets/presenters/
    "show_ai_label": True,               # small "AI-generated presenter" tag on screen
    "news_focus": "world",               # "world" = major world headlines; "us" = mostly US stories
    "outfit_rotation": True,             # robot anchor: new suit + tie colour/pattern every day
    "outfit_today": "",                  # robot anchor: a number forces one outfit, e.g. "7"
    "camera_moves": True,                # robot anchor: slow push-in + cut to a closer shot between stories
    "show_countdown": True,              # 60->0 countdown clock at the top of the frame
    "countdown_seconds": 0,              # 0 = match target_minutes*60 automatically
    "max_highlight_seconds": 18,         # guidance for the scriptwriter: keep each on-screen highlight under this
    "lead_in_sec": 0.6,                  # silent studio shot before the anchor starts talking
    "tail_sec": 1.2,                     # silent studio shot after the last word (sign-off card time)
    "show_inset": False,                 # over-the-shoulder screen with story footage (off by default for Shorts)
    "inset_fallback": "generated",       # "generated" graphics when no clip is found, or "none"
    "pexels_api_key": "",                # free key: https://www.pexels.com/api/
    "pixabay_api_key": "",               # free key: https://pixabay.com/api/docs/
    "max_age_hours": 48,
    "render_workers": 0,                 # 0 = auto (CPU cores - 1)
    "x264_preset": "veryfast",
    "x264_crf": 17,
    "port": 8765,
}


def load() -> dict:
    cfg = dict(DEFAULTS)
    p = ROOT / "config.json"
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:  # pragma: no cover
            print(f"[config] could not read config.json: {exc}")
    return cfg


def font(cfg: dict, key: str) -> str:
    """Font file for on-screen text: config path (absolute or in assets/fonts), else the first
    installed font that has Ethiopic letters, else Liberation Sans."""
    bold = key == "font_bold"
    want = str(cfg.get(key) or "")
    cands = [want, str(ASSETS / "fonts" / want)] if want else []
    win = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    cands += [str(win / ("ebrimabd.ttf" if bold else "ebrima.ttf")), str(win / "nyala.ttf"),
              "/System/Library/Fonts/Supplemental/Kefa.ttc", "/usr/share/fonts/truetype/noto/NotoSansEthiopic-Bold.ttf",
              str(ASSETS / "fonts" / "AbyssinicaSIL-Regular.ttf"),
              str(ASSETS / "fonts" / ("LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"))]
    for c in cands:
        if c and Path(c).is_file():
            return c
    return cands[-1]


def file_prefix(cfg: dict) -> str:
    """Output-filename prefix derived from the studio name, e.g.
    'World News with AI' -> 'World-News-with-AI'. Used for both writing and
    finding today's video, so it must stay in sync everywhere it's used."""
    import re
    if cfg.get("file_prefix"):
        return re.sub(r"[^A-Za-z0-9_-]+", "-", str(cfg["file_prefix"])).strip("-") or "MA-Studio"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", str(cfg.get("studio_name") or "MA Studio")).strip("-")
    return slug or "MA-Studio"


def gemini_key(cfg: dict) -> str:
    """Order: env var -> config.json -> config/api_keys.json (this folder or a
    sibling Mark-LIV folder, so the key you already use just works)."""
    key = os.environ.get("GEMINI_API_KEY", "").strip() or str(cfg.get("gemini_api_key", "")).strip()
    if key:
        return key
    candidates = [ROOT / "config" / "api_keys.json"]
    for sib in ROOT.parent.glob("Mark-LIV*"):
        candidates.append(sib / "config" / "api_keys.json")
    for c in candidates:
        try:
            k = json.loads(c.read_text(encoding="utf-8")).get("gemini_api_key", "").strip()
            if k:
                return k
        except Exception:
            pass
    return ""
