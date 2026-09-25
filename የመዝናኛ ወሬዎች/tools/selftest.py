"""
Offline self-test: runs the whole animation + render pipeline with a synthetic
"voice" (no internet, no API keys). Produces output/selftest.mp4.

    python tools/selftest.py [seconds]
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from datetime import datetime
from studio import config, lipsync, motion, outfit, presenters, robot_motion, voice, scriptwriter
from studio.media import write_wav
from studio.render import render_video

SR = voice.SR
from fakevoice import FakeEngine          # phoneme-accurate synthetic voice (tools/fakevoice.py)

def main(seconds=25):
    cfg = config.load()
    today = datetime.now()
    intro, outro = scriptwriter.intro_outro(cfg["studio_name"], cfg.get("topic_label", "WORLD NEWS"), today)
    story = dict(headline="FED · RATE CUT", ticker="Federal Reserve cuts interest rates again", sentences=[
        dict(say="The Federal Reserve has cut interest rates for the third time this year.", highlight="FED CUTS RATES AGAIN"),
        dict(say="Officials say inflation is cooling faster than expected.", highlight="INFLATION COOLING FASTER"),
        dict(say="Markets rose on the news, with major indexes closing higher.", highlight="MARKETS CLOSE HIGHER"),
    ])
    script = dict(stories=[intro, story, outro])
    audio, tl = voice.build(script, FakeEngine(), seconds, lead_in=float(cfg.get("lead_in_sec", 0.6)),
                            tail=float(cfg.get("tail_sec", 1.2)))
    fps, W, H = 30, cfg["width"], cfg["height"]
    n = int(np.ceil(tl["duration"] * fps))
    lip = lipsync.analyze(audio, SR, fps, n, cues=tl["cues"], mode=cfg.get("lipsync", "auto"), log=print)
    pres_image, pres_landmarks = presenters.resolve(cfg, config.ASSETS, today.date())
    kind = presenters.kind(pres_landmarks)
    starts = [s["start"] for s in tl["stories"][1:-1]]
    if kind == "robot":
        arrays = robot_motion.build(n, fps, lip, story_starts=starts, cue_starts=[c["start"] for c in tl["cues"]],
                                    cue_ends=[c["end"] for c in tl["cues"]], camera=bool(cfg.get("camera_moves", True)))
    else:
        arrays = {**{k: v for k, v in lip.items() if k.startswith("m")}, **motion.build(n, fps, lip, story_starts=starts)}
    work = config.WORK / "selftest"; work.mkdir(parents=True, exist_ok=True)
    wav = str(work / "n.wav"); write_wav(wav, audio, SR)
    job = dict(W=W, H=H, fps=fps, n_frames=n, workdir=str(work), preset="veryfast", crf=20,
               image=str(pres_image), landmarks=str(pres_landmarks), kind=kind, outfit=outfit.pick(cfg, today.date()),
               fonts=str(config.ASSETS / "fonts"), timeline=tl, studio=cfg["studio_name"],
               topic_label=cfg.get("topic_label", "WORLD NEWS"),
               show_countdown=bool(cfg.get("show_countdown", True)),
               countdown_seconds=float(cfg.get("countdown_seconds") or tl["duration"]),
               date_label=today.strftime("%A · %b %d, %Y").upper(), show_ai=True,
               arrays=arrays)
    out = config.OUTPUT / "selftest.mp4"
    t0 = time.time()
    render_video(job, wav, str(out), int(cfg["render_workers"]), progress=lambda f, m="": print(f"\r{m:60s}", end=""))
    dt = time.time() - t0
    print(f"\nrendered {n} frames in {dt:.1f}s  ({dt / n * 1000:.0f} ms/frame on this machine)  -> {out}")

if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 25)
