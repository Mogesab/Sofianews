"""
Quick lip-sync check with your real voice: renders a short clip of the presenter saying one sentence.

    python tools/lipsync_preview.py "Hello, welcome to MA Studio."
    python tools/lipsync_preview.py "Nvidia spent five billion dollars." --audio        # old loudness-only mouth, to compare
    python tools/lipsync_preview.py "Some text" --fake                                    # offline, robotic test voice

Needs internet for the real voice (Microsoft Edge TTS, the voice from config.json). Output: output/lipsync_preview.mp4
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from studio import config, lipsync, motion, presenters, voice
from studio.media import write_wav
from studio.render import render_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--audio", action="store_true", help="use the old loudness-only lip-sync")
    ap.add_argument("--fake", action="store_true", help="offline robotic voice instead of Edge TTS")
    a = ap.parse_args()
    cfg = config.load()
    if a.fake:
        from fakevoice import FakeEngine
        engine = FakeEngine()
    else:
        engine = voice.EdgeEngine(cfg["voice"])
    story = dict(headline="LIP-SYNC PREVIEW", ticker="Lip-sync preview",
                 sentences=[dict(say=a.text, highlight="LIP-SYNC PREVIEW")])
    audio, tl = voice.build(dict(stories=[story]), engine, 5.0, lead_in=0.8, tail=1.0)
    fps, W, H = int(cfg["fps"]), int(cfg["width"]), int(cfg["height"])
    n = int(np.ceil(tl["duration"] * fps))
    lip = lipsync.analyze(audio, voice.SR, fps, n, cues=tl["cues"], mode="audio" if a.audio else cfg.get("lipsync", "auto"),
                          intensity=float(cfg.get("lipsync_intensity", 1.0)), lead_ms=float(cfg.get("lipsync_lead_ms", 25)), log=print)
    mot = motion.build(n, fps, lip, story_starts=[])
    work = config.WORK / "preview"
    work.mkdir(parents=True, exist_ok=True)
    wav = str(work / "n.wav")
    write_wav(wav, audio, voice.SR)
    today = datetime.now()
    img, lmk = presenters.resolve(cfg, config.ASSETS, today.date())
    job = dict(W=W, H=H, fps=fps, n_frames=n, workdir=str(work), preset="veryfast", crf=20, image=str(img), landmarks=str(lmk),
               fonts=str(config.ASSETS / "fonts"), timeline=tl, studio=cfg["studio_name"], date_label=today.strftime("%A").upper(),
               show_ai=bool(cfg["show_ai_label"]), arrays={**{k: v for k, v in lip.items() if k.startswith("m")}, **mot})
    out = config.OUTPUT / "lipsync_preview.mp4"
    config.OUTPUT.mkdir(exist_ok=True)
    render_video(job, wav, str(out), int(cfg["render_workers"]), progress=lambda f, m="": print(f"\r{m:60s}", end=""))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
