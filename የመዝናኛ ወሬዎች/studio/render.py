"""
Video rendering.

The timeline is cut into ~10 s chunks. Every worker process renders its chunks
frame by frame (face animation + graphics) and pipes them straight into
ffmpeg (H.264). The chunks are then joined and the narration is muxed in as
AAC, giving a 1080p30 MP4 that YouTube accepts as-is.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import media

_ctx: dict = {}


def _init(job: dict) -> None:
    from .face import FaceAnimator
    from .overlays import Overlays, PortraitOverlays
    W, H = job["W"], job["H"]
    _ctx["job"] = job
    if job.get("kind") == "robot":
        from .robot import RobotAnimator
        _ctx["face"] = RobotAnimator(Path(job["image"]), Path(job["landmarks"]), W, H, outfit=job.get("outfit"))
    else:
        _ctx["face"] = FaceAnimator(Path(job["image"]), Path(job["landmarks"]), W, H)
    _ctx["ov"] = (PortraitOverlays if H > W else Overlays)(Path(job["fonts"]), W, H, job["timeline"], job["studio"], job["date_label"], job["show_ai"],
                          topic_label=job.get("topic_label", "WORLD NEWS"),
                          show_countdown=job.get("show_countdown", True),
                          countdown_seconds=job.get("countdown_seconds", 0.0),
                          font_bold=job.get("font_bold"), font_regular=job.get("font_regular"),
                          labels=job.get("labels"))
    _ctx["inset"] = None
    if job.get("inset_strip"):
        from .inset import InsetScreen
        _ctx["inset"] = InsetScreen(job["inset_strip"], job.get("inset_spans") or [], W, H,
                                    Path(job["fonts"]), job["timeline"], job["fps"])


def _params(i: int) -> dict:
    a = _ctx["job"]["arrays"]
    return {k: float(v[i]) for k, v in a.items()}


def _render_chunk(args):
    idx, start, end = args
    job = _ctx["job"]
    W, H, fps = job["W"], job["H"], job["fps"]
    out = str(Path(job["workdir"]) / f"chunk_{idx:04d}.mp4")
    log = open(Path(job["workdir"]) / f"chunk_{idx:04d}.log", "wb")
    cmd = [media.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p",
           "-c:v", "libx264", "-preset", job["preset"], "-crf", str(job["crf"]),
           "-profile:v", "high", "-g", str(fps * 2), "-bf", "2",
           "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
           out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log, creationflags=media._creationflags())
    face, ov, inset = _ctx["face"], _ctx["ov"], _ctx["inset"]
    if inset is not None:
        inset.seek(start)
    for i in range(start, end):
        frame = face.render(_params(i))
        if inset is not None:
            inset.draw(frame, i, i / fps)      # screen sits behind the lower third
        ov.draw(frame, i / fps)
        p.stdin.write(frame.tobytes())
    p.stdin.close()
    rc = p.wait()
    log.close()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed on chunk {idx}: " + (Path(job["workdir"]) / f"chunk_{idx:04d}.log").read_text(errors="ignore")[-300:])
    return idx, end - start, out


def render_video(job: dict, wav_path: str, out_path: str, workers: int = 0, progress=lambda f, m="": None) -> None:
    n, fps = job["n_frames"], job["fps"]
    workdir = Path(job["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    chunk = fps * 10
    ranges = [(i, s, min(n, s + chunk)) for i, s in enumerate(range(0, n, chunk))]
    if not workers:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    workers = min(workers, len(ranges))

    done, results = 0, {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(job,)) as pool:
        for idx, nfr, path in pool.imap_unordered(_render_chunk, ranges):
            results[idx] = path
            done += nfr
            progress(done / n, f"Rendering video… {int(100 * done / n)}%  ({done // fps}s of {n // fps}s)")

    listing = workdir / "chunks.txt"
    listing.write_text("".join(f"file '{Path(results[i]).as_posix()}'\n" for i in sorted(results)), encoding="utf-8")
    progress(1.0, "Finishing MP4 (joining video + audio)…")
    p = media.run(["-f", "concat", "-safe", "0", "-i", str(listing), "-i", wav_path,
                   "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                   "-ar", "48000", "-ac", "2", "-shortest", "-movflags", "+faststart", str(out_path)])
    if p.returncode != 0:
        raise RuntimeError("ffmpeg mux failed: " + p.stderr.decode("utf-8", "ignore")[-400:])
