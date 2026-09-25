"""
B-roll for the over-the-shoulder screen.

Finds a short clip that matches each story, then builds ONE full-length video,
already cropped to the on-screen box, that the render workers read frame by
frame. Building it up front means the workers never have to think about
timing - inset frame N always belongs with presenter frame N.

Where the footage comes from, in order of preference:

  1. assets/broll/*.mp4        - your own clips (filename is matched to the
                                 story keywords, e.g. "chips-factory.mp4")
  2. Pexels Videos API         - free, allowed for commercial use (needs a
                                 free key in config.json -> "pexels_api_key")
  3. Pixabay Videos API        - same idea ("pixabay_api_key")

IMPORTANT: do NOT point this at clips ripped from YouTube, TV news or agency
footage. That is someone else's copyright, and it is the fastest way to get
strikes on your channel. The stock sources above are licensed for exactly this
kind of reuse.
"""
from __future__ import annotations

import json
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from . import media

CACHE_NAME = "broll_cache"
TIMEOUT = 25
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".m4v")


def is_still(p: Path | None) -> bool:
    return p is not None and p.suffix.lower() in IMG_EXTS

# story words -> what actually looks good as footage
VISUAL_HINTS = [
    (r"chip|semiconductor|nvidia|gpu|wafer|silicon", "semiconductor chip manufacturing closeup"),
    (r"robot|humanoid|android|boston dynamics", "robot arm technology"),
    (r"data ?cent|server|cloud|compute|infrastructur", "server room data center"),
    (r"car|driving|waymo|tesla|autonom", "autonomous car city driving"),
    (r"health|medical|doctor|patient|drug|clinic|cancer", "medical technology hospital"),
    (r"regulat|law|court|lawsuit|congress|senate|polic|eu |brussels", "government building official"),
    (r"school|student|teacher|education|classroom|exam", "students classroom computer"),
    (r"job|worker|layoff|hiring|employ|workplace|office", "modern office people working"),
    (r"money|fund|invest|billion|valuation|ipo|stock|market", "stock market financial data"),
    (r"phone|mobile|app|iphone|android app", "person using smartphone"),
    (r"secur|hack|cyber|breach|malware|fraud|scam", "cyber security code screen"),
    (r"energy|power|electric|grid|climate|carbon", "power plant energy grid"),
    (r"image|video|art|music|creat|deepfake", "digital creative screen art"),
    (r"search|browser|google|web", "person typing laptop screen"),
    (r"open ?ai|anthropic|claude|chatgpt|gemini|llm|model|chatbot", "artificial intelligence interface screen"),
]
DEFAULT_QUERY = "artificial intelligence technology abstract"

STOP = set("""a an the and or but if as at by for from in into of on onto over per so than that to up via with
within without after before about is are was were be been has have had will would can could may says said it
its his her their this these those not no new more most now today first says report reports according""".split())


# --------------------------------------------------------------------- queries
def keywords(story: dict) -> list[str]:
    text = " ".join([story.get("headline", "")] + [s.get("highlight", "") for s in story.get("sentences", [])])
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-']+", text)]
    out, seen = [], set()
    for w in words:
        if w in STOP or len(w) < 3 or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out[:12]


def query_for(story: dict) -> str:
    text = " ".join([story.get("headline", "")] + [s.get("say", "") for s in story.get("sentences", [])]).lower()
    for pat, q in VISUAL_HINTS:
        if re.search(pat, text):
            return q
    return DEFAULT_QUERY


# ------------------------------------------------------------------ local pool
def _local_pool(assets: Path) -> list[Path]:
    d = assets / "broll"
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv", ".m4v"))


def _pick_local(pool: list[Path], kws: list[str], used: set) -> Path | None:
    best, score = None, 0
    for p in pool:
        name = re.sub(r"[^a-z]+", " ", p.stem.lower())
        sc = sum(1 for k in kws if k in name) - (2 if p in used else 0)
        if sc > score:
            best, score = p, sc
    if best is None:                                   # nothing matched: round-robin
        free = [p for p in pool if p not in used]
        best = free[0] if free else (pool[0] if pool else None)
    return best


# ---------------------------------------------------------------- stock search
def _get_json(url: str, headers: dict) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def _download(url: str, dest: Path) -> Path | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MA-Studio/1.0"})
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 16)
        if tmp.stat().st_size < 50_000:
            tmp.unlink(missing_ok=True)
            return None
        tmp.replace(dest)
        return dest
    except Exception:
        return None


def _fetch_story_image(story: dict, items: list[dict], cache: Path, used: set) -> Path | None:
    """Download the image the news outlet itself attached to a story used in this segment."""
    for i in story.get("sources", []):
        if not (isinstance(i, int) and 1 <= i <= len(items)):
            continue
        url = items[i - 1].get("image") or ""
        if not url:
            continue
        stem = re.sub(r"[^A-Za-z0-9]+", "_", url)[-80:] or f"story_{i}"
        # keep the outlet's extension when it looks like an image, else default to .jpg
        ext = ".jpg"
        for e in IMG_EXTS:
            if url.lower().split("?", 1)[0].endswith(e):
                ext = e
                break
        dest = cache / f"item_{stem}{ext}"
        if dest in used:
            continue
        if dest.exists() and dest.stat().st_size > 10_000:
            return dest
        got = _download(url, dest)
        if got:
            return got
    return None


def _pexels_photo(key: str, q: str, cache: Path, used: set) -> Path | None:
    url = "https://api.pexels.com/v1/search?per_page=12&orientation=portrait&size=medium&query=" + urllib.parse.quote(q)
    try:
        data = _get_json(url, {"Authorization": key, "User-Agent": "MA-Studio/1.0"})
    except Exception:
        return None
    for photo in data.get("photos", []):
        pid = photo.get("id")
        dest = cache / f"pexels_photo_{pid}.jpg"
        if dest in used:
            continue
        if dest.exists():
            return dest
        src = (photo.get("src") or {})
        pick = src.get("large") or src.get("medium") or src.get("large2x")
        if not pick:
            continue
        got = _download(pick, dest)
        if got:
            return got
    return None


def _pixabay_photo(key: str, q: str, cache: Path, used: set) -> Path | None:
    url = (f"https://pixabay.com/api/?key={urllib.parse.quote(key)}&per_page=12&image_type=photo&safesearch=true&q="
           + urllib.parse.quote(q))
    try:
        data = _get_json(url, {"User-Agent": "MA-Studio/1.0"})
    except Exception:
        return None
    for hit in data.get("hits", []):
        dest = cache / f"pixabay_photo_{hit.get('id')}.jpg"
        if dest in used:
            continue
        if dest.exists():
            return dest
        pick = hit.get("largeImageURL") or hit.get("webformatURL")
        if not pick:
            continue
        got = _download(pick, dest)
        if got:
            return got
    return None


def _pexels(key: str, q: str, cache: Path, used: set) -> Path | None:
    url = ("https://api.pexels.com/videos/search?per_page=12&orientation=landscape&size=medium&query="
           + urllib.parse.quote(q))
    try:
        data = _get_json(url, {"Authorization": key, "User-Agent": "MA-Studio/1.0"})
    except Exception:
        return None
    for vid in data.get("videos", []):
        vid_id = vid.get("id")
        dest = cache / f"pexels_{vid_id}.mp4"
        if dest in used:
            continue
        if dest.exists():
            return dest
        files = [f for f in vid.get("video_files", []) if (f.get("width") or 0) >= 960]
        files.sort(key=lambda f: f.get("width") or 0)
        if not files:
            continue
        got = _download(files[0]["link"], dest)
        if got:
            return got
    return None


def _pixabay(key: str, q: str, cache: Path, used: set) -> Path | None:
    url = (f"https://pixabay.com/api/videos/?key={urllib.parse.quote(key)}&per_page=12&safesearch=true&q="
           + urllib.parse.quote(q))
    try:
        data = _get_json(url, {"User-Agent": "MA-Studio/1.0"})
    except Exception:
        return None
    for hit in data.get("hits", []):
        dest = cache / f"pixabay_{hit.get('id')}.mp4"
        if dest in used:
            continue
        if dest.exists():
            return dest
        v = hit.get("videos", {})
        pick = v.get("medium") or v.get("small") or v.get("large")
        if not pick or not pick.get("url"):
            continue
        got = _download(pick["url"], dest)
        if got:
            return got
    return None


def find_story_images(script: dict, items: list[dict], cfg: dict, assets: Path,
                      log=lambda m: None) -> list[Path | None]:
    """One STILL IMAGE per story. Priority: the outlet's own picture (from RSS), then
    Pexels/Pixabay photo search. Returns None for a story where nothing was found —
    the over-shoulder screen simply won't appear during that story."""
    cache = assets / CACHE_NAME
    cache.mkdir(parents=True, exist_ok=True)
    pex = str(cfg.get("pexels_api_key", "")).strip()
    pix = str(cfg.get("pixabay_api_key", "")).strip()

    out, used = [], set()
    for i, story in enumerate(script["stories"]):
        img = _fetch_story_image(story, items, cache, used)
        if img is None and pex:
            img = _pexels_photo(pex, query_for(story), cache, used)
        if img is None and pix:
            img = _pixabay_photo(pix, query_for(story), cache, used)
        if img is not None:
            used.add(img)
        out.append(img)
        log(f"Story image {i + 1}/{len(script['stories'])}: " + (img.name if img else "none"))
    return out


def find_clips(script: dict, cfg: dict, assets: Path, log=lambda m: None) -> list[Path | None]:
    """One clip per story in script['stories'] (intro and outro included)."""
    cache = assets / CACHE_NAME
    cache.mkdir(parents=True, exist_ok=True)
    pool = _local_pool(assets)
    pex = str(cfg.get("pexels_api_key", "")).strip()
    pix = str(cfg.get("pixabay_api_key", "")).strip()
    if not pool and not pex and not pix:
        log("No b-roll source configured (assets/broll is empty and there is no Pexels/Pixabay key), "
            "so the screen will show generated motion graphics.")
        return [None] * len(script["stories"])

    out, used = [], set()
    for i, story in enumerate(script["stories"]):
        clip = None
        if pool:
            clip = _pick_local(pool, keywords(story), used)
        if clip is None and pex:
            clip = _pexels(pex, query_for(story), cache, used)
        if clip is None and pix:
            clip = _pixabay(pix, query_for(story), cache, used)
        if clip is not None:
            used.add(clip)
        out.append(clip)
        log(f"B-roll {i + 1}/{len(script['stories'])}: " + (clip.name if clip else "none"))
    return out



# ------------------------------------------------------- generated fallback
# When there is no clip for a story we synthesise an abstract "data wall" loop:
# a drifting node network over a dark grid. It is generated here from scratch,
# so there is nothing to license and the screen is never empty.
PALETTES = [
    ((236, 150, 44), (255, 214, 140)),        # BGR: broadcast blue
    ((196, 176, 58), (245, 226, 150)),        # teal
    ((214, 96, 96), (246, 186, 186)),         # indigo
    ((120, 120, 226), (190, 190, 250)),       # red-ish, for alert stories
]


def generate(dest: Path, seconds: float, bw: int, bh: int, fps: int, seed: int = 0) -> Path | None:
    """Render a seamlessly looping abstract graphic straight into `dest`."""
    import subprocess

    import cv2
    import numpy as np

    n = max(2, int(round(seconds * fps)))
    rng = np.random.default_rng(seed)
    dim, bright = PALETTES[seed % len(PALETTES)]
    N = 34
    bx = rng.uniform(-0.1, 1.1, N)
    by = rng.uniform(-0.1, 1.1, N)
    # integer harmonics of the loop -> the last frame joins the first cleanly
    fx = rng.integers(1, 4, N)
    fy = rng.integers(1, 4, N)
    px = rng.uniform(0, 2 * np.pi, N)
    py = rng.uniform(0, 2 * np.pi, N)
    ax = rng.uniform(0.02, 0.09, N)
    ay = rng.uniform(0.02, 0.07, N)
    size = rng.uniform(1.6, 4.2, N)
    pulse = rng.uniform(0, 2 * np.pi, N)

    # static background: vertical gradient + faint grid
    yy = np.linspace(0, 1, bh, dtype=np.float32)[:, None, None]
    base = (np.array([26, 14, 6], np.float32) * (1 - yy) + np.array([70, 38, 14], np.float32) * yy)
    bg = np.repeat(base, bw, axis=1)
    for gx in range(0, bw, max(24, bw // 22)):
        bg[:, gx] += np.array([20, 12, 5], np.float32)
    for gy in range(0, bh, max(24, bh // 12)):
        bg[gy, :] += np.array([20, 12, 5], np.float32)
    bg = np.clip(bg, 0, 255)

    cmd = [media.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{bw}x{bh}", "-r", str(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(dest)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            creationflags=media._creationflags())
    try:
        for i in range(n):
            u = 2 * np.pi * i / n
            X = ((bx + ax * np.sin(fx * u + px)) * bw).astype(np.int32)
            Y = ((by + ay * np.sin(fy * u + py)) * bh).astype(np.int32)
            # draw the network onto its own layer and ADD it, so every element
            # glows over the background instead of punching dark holes in it
            glow = np.zeros((bh, bw, 3), np.float32)
            for a in range(N):
                for b in range(a + 1, N):
                    dx, dy = X[a] - X[b], Y[a] - Y[b]
                    d2 = dx * dx + dy * dy
                    if d2 < (bw * 0.20) ** 2:
                        w = 1.0 - np.sqrt(d2) / (bw * 0.20)
                        col = tuple(float(c) * w * 0.62 for c in dim)
                        cv2.line(glow, (X[a], Y[a]), (X[b], Y[b]), col, 1, cv2.LINE_AA)
            for a in range(N):
                pl = 0.55 + 0.45 * np.sin(u * (1 + a % 3) + pulse[a])
                r = float(size[a]) * (0.8 + 0.4 * pl)
                col = tuple(float(c) * (0.5 + 0.5 * pl) for c in bright)
                cv2.circle(glow, (X[a], Y[a]), max(1, int(r)), col, -1, cv2.LINE_AA)
            # soft halo around the nodes
            glow = glow * 1.25 + cv2.GaussianBlur(glow, (0, 0), bw * 0.012) * 0.9
            fr = bg + glow
            # slow sweep of light across the panel, looping with the animation
            sx = int((i / n) * bw * 1.6 - bw * 0.3)
            band = np.clip(1 - np.abs(np.arange(bw) - sx) / (bw * 0.16), 0, 1) ** 2
            fr += (band[None, :, None] * np.array(dim, np.float32) * 0.18)
            proc.stdin.write(np.clip(fr, 0, 255).astype(np.uint8).tobytes())
        proc.stdin.close()
        rc = proc.wait()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        return None
    return dest if rc == 0 and dest.exists() else None


def generated_clip(cache: Path, seed: int, bw: int, bh: int, fps: int, seconds: float = 8.0) -> Path | None:
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"generated_{seed}_{bw}x{bh}.mp4"
    if dest.exists():
        return dest
    return generate(dest, seconds, bw, bh, fps, seed)


# ---------------------------------------------------------------- build strip
def build_strip(clips: list[Path | None], stories: list[dict], duration: float, work: Path,
                bw: int, bh: int, fps: int, log=lambda m: None, cache: Path | None = None,
                fallback: str = "generated") -> tuple[str | None, list[tuple]]:
    """Render every clip into one continuous video the size of the on-screen box.

    Returns (path, spans) where spans is [(start, end, has_footage, kind)] so the
    renderer knows when to show the box and how to label it.
    """
    # anything we could not source becomes a generated graphic, so the screen
    # stays alive right through the episode
    kinds = ["still" if is_still(c) else ("clip" if c is not None else "generated") for c in clips]
    if fallback == "generated" and cache is not None:
        clips = list(clips)
        for i, c in enumerate(clips):
            if c is None:
                clips[i] = generated_clip(cache, i, bw, bh, fps)
    if not any(c is not None for c in clips):
        return None, []
    segs_dir = work / "broll"
    segs_dir.mkdir(parents=True, exist_ok=True)
    parts, spans, cursor = [], [], 0.0

    def filler(seconds: float, idx: int):
        if seconds <= 0.02:
            return
        p = segs_dir / f"gap_{idx:03d}.mp4"
        media.run(["-f", "lavfi", "-i", f"color=c=black:s={bw}x{bh}:r={fps}", "-t", f"{seconds:.3f}",
                   "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-pix_fmt", "yuv420p", str(p)])
        if p.exists():
            parts.append(p)

    for i, (clip, st, kind) in enumerate(zip(clips, stories, kinds)):
        start, end = float(st["start"]), float(st["end"])
        if start > cursor + 0.02:                               # gap before this story
            filler(start - cursor, i)
            spans.append((cursor, start, False, "none"))
            cursor = start
        seconds = max(0.3, end - start)
        if clip is None:
            filler(seconds, 100 + i)
            spans.append((start, end, False, "none"))
            cursor = end
            continue
        seg = segs_dir / f"seg_{i:03d}.mp4"
        # Ken-Burns pan/zoom for a still image; loop-and-crop for a video clip.
        if is_still(clip):
            zoom_frames = max(2, int(round(seconds * fps)))
            vf = (f"scale=iw*2:ih*2,"
                  f"zoompan=z='min(zoom+0.0009,1.10)':d={zoom_frames}:s={bw}x{bh}:fps={fps},"
                  f"setsar=1,eq=saturation=1.05:contrast=1.02")
            p = media.run(["-loop", "1", "-i", str(clip), "-t", f"{seconds:.3f}",
                           "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                           "-pix_fmt", "yuv420p", str(seg)])
        else:
            # loop the clip to cover the whole story, crop to the box, strip audio
            vf = (f"scale={bw}:{bh}:force_original_aspect_ratio=increase,"
                  f"crop={bw}:{bh},fps={fps},setsar=1,eq=saturation=1.06:contrast=1.03")
            p = media.run(["-stream_loop", "-1", "-i", str(clip), "-t", f"{seconds:.3f}",
                           "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                           "-pix_fmt", "yuv420p", str(seg)])
        if p.returncode != 0 or not seg.exists():
            log(f"Could not prepare b-roll for story {i + 1}; that story runs without the screen.")
            filler(seconds, 200 + i)
            spans.append((start, end, False, "none"))
        else:
            parts.append(seg)
            spans.append((start, end, True, kind))
        cursor = end

    if duration > cursor + 0.02:
        filler(duration - cursor, 999)
        spans.append((cursor, duration, False, "none"))

    listing = segs_dir / "strip.txt"
    listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    out = segs_dir / "inset.mp4"
    p = media.run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(out)])
    if p.returncode != 0 or not out.exists():
        p = media.run(["-f", "concat", "-safe", "0", "-i", str(listing), "-c:v", "libx264",
                       "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)])
    if p.returncode != 0 or not out.exists():
        log("Could not assemble the b-roll strip; rendering without the screen.")
        return None, []
    return str(out), spans
