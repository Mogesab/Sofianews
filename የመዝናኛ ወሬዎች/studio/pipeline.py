"""End-to-end: news -> script -> voice -> video (+ YouTube text file)."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime

import numpy as np

from . import broll, config, ethiodate, lipsync, motion, news, outfit, presenters, robot_motion, scriptwriter, voice
from .render import render_video


def _mmss(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60}:{sec % 60:02d}"


def _youtube_text(script: dict, timeline: dict, items: list[dict], cfg: dict, today: datetime) -> str:
    studio = cfg["studio_name"]
    topic = cfg.get("topic_label", "WORLD NEWS").title()
    lines = [script["title"], "", script.get("summary", "").strip() or f"የዛሬው ዋና ዋና ዜናዎች ከ{studio}።", "",
             "ምዕራፎች (CHAPTERS)", f"0:00 {studio}"]
    for st in timeline["stories"][1:-1]:
        lines.append(f"{_mmss(st['start'])} {st['headline']}")
    lines += ["", "ምንጮች (SOURCES)"]
    seen = set()
    for st in script["stories"]:
        for i in st.get("sources", []):
            if 1 <= i <= len(items) and i not in seen:
                seen.add(i)
                lines.append(f"- {items[i - 1]['source']}: {items[i - 1]['title']} {items[i - 1]['link']}")
    lines += ["", "ይህ ዝግጅት በሰው ሰራሽ አስተውሎት (AI) የተዘጋጀ ነው፤ የዜናው ጽሑፍ ከሕዝብ የዜና ምንጮች በAI የተጻፈ ሲሆን አቅራቢው ሮቦትና ድምፁ በAI የተፈጠሩ ናቸው።",
              "This episode was produced with AI: the Amharic news script is AI-written from public news sources, "
              "and the robot presenter and voice are AI-generated.", "",
              "#Shorts #Amharic #AmharicNews #Ethiopia #ዜና #የእለቱ_ዜና #WorldNews"]
    return "\n".join(lines)


def run(cfg: dict, progress=lambda stage, frac, msg="": None) -> dict:
    today = datetime.now()
    W, H, fps = int(cfg["width"]), int(cfg["height"]), int(cfg["fps"])
    target = float(cfg["target_minutes"]) * 60
    stamp = today.strftime("%Y-%m-%d_%H%M")
    out_dir = config.OUTPUT
    out_dir.mkdir(exist_ok=True)
    work = config.WORK / stamp
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    log = lambda m: progress("script", 0.0, m)

    # 1 ── headlines
    progress("news", 0.0, "Collecting today's top world headlines…")
    feeds = cfg.get("news_feeds") or news.FEEDS
    world_feeds = [f for f in feeds if (f[2] if len(f) > 2 else "world") != "am"]
    am_feeds = [f for f in feeds if len(f) > 2 and f[2] == "am"]
    # world news only - nothing about Ethiopia. The Amharic newsrooms are only kept as a
    # backup to read from if Gemini cannot write the script.
    items = [i for i in news.gather(int(cfg["max_age_hours"]), limit=30 if target > 90 else 24,
                                    feeds=world_feeds, us_ratio=0.0, priority_region="world")
             if not news.about_ethiopia(i)]
    if am_feeds:
        items += [dict(i, region="am") for i in news.gather(int(cfg["max_age_hours"]), limit=24, feeds=am_feeds,
                                                            us_ratio=0.0) if not news.about_ethiopia(i)]
    if not items:
        raise RuntimeError("Could not download any news. Check your internet connection and try again.")
    progress("news", 1.0, f"Found {len(items)} fresh stories")

    # 2 ── script
    progress("script", 0.1, "Writing today's script…")
    key = config.gemini_key(cfg)
    script = scriptwriter.write(items, cfg, key, today, target, log)
    if script["mode"] == "offline":
        progress("script", 1.0, "Script ready (no Gemini key/response: reading headlines directly)")
    else:
        progress("script", 1.0, "Script ready")

    # 3 ── narration
    backup = voice.EdgeEngine(cfg.get("backup_voice", "am-ET-AmehaNeural"))
    if str(cfg.get("tts_engine", "gemini")).lower() == "gemini":
        engine = voice.GeminiTTSEngine(key, voice=cfg.get("gemini_voice", "Charon"),
                                       models=cfg.get("tts_models") or None, fallback=backup,
                                       log=lambda m: progress("voice", 0.0, m))
    else:
        engine = backup
    audio, timeline = voice.build(script, engine, target, lead_in=float(cfg.get("lead_in_sec", 0.6)),
                                  tail=float(cfg.get("tail_sec", 1.2)),
                                  progress=lambda f, m="": progress("voice", f, m))
    dur = timeline["duration"]

    # 4 ── animation curves from the audio
    n_frames = int(np.ceil(dur * fps))
    progress("render", 0.0, "Analyzing speech for lip-sync…")
    lip = lipsync.analyze(audio, voice.SR, fps, n_frames, cues=timeline["cues"],
                          mode=str(cfg.get("lipsync", "auto")), intensity=float(cfg.get("lipsync_intensity", 1.0)),
                          lead_ms=float(cfg.get("lipsync_lead_ms", 25)), log=log)
    story_starts = [s["start"] for s in timeline["stories"][1:-1]]  # skip intro/outro
    cue_starts = [c["start"] for c in timeline["cues"]]
    cue_ends = [c["end"] for c in timeline["cues"]]
    pres_image, pres_landmarks = presenters.resolve(cfg, config.ASSETS, today.date())
    pres_kind = presenters.kind(pres_landmarks)
    log(f"Presenter for today: {pres_image.name}")
    todays_outfit = None
    if pres_kind == "robot":
        todays_outfit = outfit.pick(cfg, today.date())
        if todays_outfit:
            log(f"Today's outfit: {todays_outfit['name']}")
        arrays = robot_motion.build(n_frames, fps, lip, story_starts=story_starts, cue_starts=cue_starts,
                                    cue_ends=cue_ends, camera=bool(cfg.get("camera_moves", True)),
                                    lead_in=float(cfg.get("lead_in_sec", 0.6)))
    else:
        mot = motion.build(n_frames, fps, lip, story_starts=story_starts,
                           cue_starts=cue_starts, cue_ends=cue_ends)
        arrays = {k: v for k, v in lip.items() if k.startswith("m")}      # mouth shape curves
        arrays.update(mot)

    from .media import write_wav
    wav = str(work / "narration.wav")
    write_wav(wav, audio, voice.SR)

    # 5 ── over-the-shoulder screen: story-matched footage
    inset_strip, inset_spans = None, []
    if cfg.get("show_inset", True):
        progress("render", 0.0, "Finding footage for the over-the-shoulder screen…")
        try:
            from .inset import InsetScreen
            _, _, bw, bh = InsetScreen.box_size(W, H)
            clips = broll.find_clips(script, cfg, config.ASSETS, log=lambda m: progress("render", 0.0, m))
            inset_strip, inset_spans = broll.build_strip(
                clips, timeline["stories"], dur, work, bw, bh, fps,
                log=lambda m: progress("render", 0.0, m),
                cache=config.ASSETS / broll.CACHE_NAME,
                fallback=str(cfg.get("inset_fallback", "generated")))
        except Exception as exc:
            progress("render", 0.0, f"Over-the-shoulder screen skipped ({exc})")
            inset_strip, inset_spans = None, []

    countdown_seconds = float(cfg.get("countdown_seconds") or dur)
    job = dict(W=W, H=H, fps=fps, n_frames=n_frames, workdir=str(work), preset=cfg["x264_preset"], crf=cfg["x264_crf"],
               image=str(pres_image), landmarks=str(pres_landmarks), kind=pres_kind, outfit=todays_outfit,
               fonts=str(config.ASSETS / "fonts"), timeline=timeline, studio=cfg["studio_name"],
               topic_label=cfg.get("topic_label", "WORLD NEWS"),
               show_countdown=bool(cfg.get("show_countdown", True)), countdown_seconds=countdown_seconds,
               date_label=ethiodate.label(today.date()) if cfg.get("date_style") == "ethiopian"
               else today.strftime("%A · %b %d, %Y").upper().replace(" 0", " "), show_ai=bool(cfg["show_ai_label"]),
               font_bold=config.font(cfg, "font_bold"), font_regular=config.font(cfg, "font_regular"),
               labels=cfg.get("labels") or {},
               arrays=arrays, inset_strip=inset_strip, inset_spans=inset_spans)
    name = f"{config.file_prefix(cfg)}_{today.strftime('%Y-%m-%d')}_{today.strftime('%H%M')}"
    out_mp4 = out_dir / f"{name}.mp4"
    render_video(job, wav, str(out_mp4), int(cfg["render_workers"]),
                 progress=lambda f, m="": progress("render", f, m))

    (out_dir / f"{name}_youtube.txt").write_text(_youtube_text(script, timeline, items, cfg, today), encoding="utf-8")
    (out_dir / f"{name}_script.json").write_text(json.dumps(dict(script=script, timeline=timeline), indent=1, default=str), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    return dict(video=str(out_mp4), name=name, duration=dur, title=script["title"])
