"""
Checks the over-the-shoulder screen without rendering a whole episode.

    python tools/check_broll.py

Tells you which footage source will be used, tests any API key you have set,
and writes output/broll_preview.png so you can see exactly how the screen will
look in the frame.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from studio import broll, config, presenters
from studio.face import FaceAnimator
from studio.inset import InsetScreen
from studio.overlays import Overlays


def main():
    cfg = config.load()
    W, H, fps = int(cfg["width"]), int(cfg["height"]), int(cfg["fps"])
    x0, y0, bw, bh = InsetScreen.box_size(W, H)
    print(f"screen box      : {bw}x{bh} at ({x0}, {y0})")
    print(f"show_inset      : {cfg.get('show_inset', True)}")
    if not cfg.get("show_inset", True):
        print("\n-> The screen is switched OFF in config.json. Set \"show_inset\": true.")
        return

    pool = broll._local_pool(config.ASSETS)
    pex = str(cfg.get("pexels_api_key", "")).strip()
    pix = str(cfg.get("pixabay_api_key", "")).strip()
    print(f"local clips     : {len(pool)} in assets/broll" + (f" -> {[p.name for p in pool[:6]]}" if pool else ""))
    print(f"pexels key      : {'set' if pex else 'not set'}")
    print(f"pixabay key     : {'set' if pix else 'not set'}")
    print(f"fallback        : {cfg.get('inset_fallback', 'generated')}")

    if pex:
        print("\ntesting the Pexels key…")
        cache = config.ASSETS / broll.CACHE_NAME
        got = broll._pexels(pex, "artificial intelligence technology", cache, set())
        print("  -> " + (f"OK, downloaded {got.name}" if got else
                         "FAILED. Check the key, or your internet/firewall."))

    if not pool and not pex and not pix:
        print("\n-> No real footage source. The screen will still appear, using generated")
        print("   motion graphics. To use real video, either drop clips into assets/broll")
        print("   or get a free key at https://www.pexels.com/api/ and put it in config.json.")

    # build a one-story preview so you can actually look at it
    print("\nbuilding a preview frame…")
    story = dict(headline="PREVIEW · AI NEWS",
                 sentences=[dict(say="A chip maker reports record data centre revenue.",
                                 highlight="Chip maker reports record data centre revenue")])
    script = dict(stories=[story])
    stories = [dict(headline=story["headline"], start=0.0, end=6.0)]
    work = config.OUTPUT / "_brollcheck"
    work.mkdir(parents=True, exist_ok=True)
    clips = broll.find_clips(script, cfg, config.ASSETS, log=lambda m: print("  " + m))
    strip, spans = broll.build_strip(clips, stories, 6.0, work, bw, bh, fps,
                                     log=lambda m: print("  " + m),
                                     cache=config.ASSETS / broll.CACHE_NAME,
                                     fallback=str(cfg.get("inset_fallback", "generated")))
    if not strip:
        print("  -> could not build the strip; the episode would render without the screen.")
        return
    cues = [dict(start=0.0, end=6.0, story=0, headline=story["headline"],
                 highlight=story["sentences"][0]["highlight"], text="")]
    tl = dict(duration=6.0, cues=cues, stories=stories, ticker=["PREVIEW"])
    pres_image, pres_landmarks = presenters.resolve(cfg, config.ASSETS)
    face = FaceAnimator(pres_image, pres_landmarks, W, H)
    ov = Overlays(config.ASSETS / "fonts", W, H, tl, cfg["studio_name"], "PREVIEW", bool(cfg["show_ai_label"]))
    ins = InsetScreen(strip, spans, W, H, config.ASSETS / "fonts", tl, fps)
    t = 3.0
    i = int(t * fps)
    ins.seek(i)
    frame = face.render(dict(mopen=0.35, mspread=0.0, tx=0.4, ty=0.2, rot=0.1, blink=0.0,
                             gx=0.0, gy=0.0, lid=0.0, brow=0.4, breath=0.2,
                             handshift=0.1, pageflip=0.0, t=t))
    ins.draw(frame, i, t)
    ov.draw(frame, t)
    out = config.OUTPUT / "broll_preview.png"
    cv2.imwrite(str(out), frame)
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    print(f"  -> wrote {out}")


if __name__ == "__main__":
    main()
