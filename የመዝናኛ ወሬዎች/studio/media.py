"""ffmpeg helpers (bundled ffmpeg via imageio-ffmpeg, or the one on PATH)."""
from __future__ import annotations

import shutil
import subprocess
import sys

import numpy as np


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
    raise RuntimeError("ffmpeg not found. Run: pip install imageio-ffmpeg")


def _creationflags() -> int:
    return 0x08000000 if sys.platform == "win32" else 0        # CREATE_NO_WINDOW


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args],
                          capture_output=True, creationflags=_creationflags(), **kw)


def decode(path: str, sr: int = 24000) -> np.ndarray:
    """Any audio file -> float32 mono @ sr."""
    p = run(["-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"])
    if p.returncode != 0:
        raise RuntimeError("ffmpeg decode failed: " + p.stderr.decode("utf-8", "ignore")[-400:])
    return np.frombuffer(p.stdout, dtype="<f4").copy()


def write_wav(path: str, audio: np.ndarray, sr: int = 24000) -> None:
    import wave
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
