"""
Narration: one neural-voice clip per sentence, stitched together with natural
pauses. Because every sentence is its own clip we know *exactly* when each one
starts, which is what keeps the on-screen key points in sync with the speech.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import media

SR = 24000


class EdgeEngine:
    """Microsoft Edge neural TTS (free, needs internet, no key)."""

    def __init__(self, voice: str = "en-US-GuyNeural", rate: str = "+0%"):
        self.voice, self.rate = voice, rate

    def synth(self, text: str) -> np.ndarray:
        import edge_tts
        last = None
        for attempt in range(4):
            fd, tmp = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            try:
                async def go():
                    await edge_tts.Communicate(text, self.voice, rate=self.rate).save(tmp)
                asyncio.run(go())
                if os.path.getsize(tmp) < 500:
                    raise RuntimeError("empty audio")
                return media.decode(tmp, SR)
            except Exception as exc:                      # network hiccup -> retry
                last = exc
                time.sleep(1.5 * (attempt + 1))
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        raise RuntimeError(f"Text-to-speech failed (is the internet on?): {last}")


def _trim(x: np.ndarray, thr: float = 0.010) -> np.ndarray:
    idx = np.where(np.abs(x) > thr)[0]
    if len(idx) == 0:
        return x[: int(0.1 * SR)]
    a = max(0, idx[0] - int(0.03 * SR))
    b = min(len(x), idx[-1] + int(0.09 * SR))
    return x[a:b]


class GeminiTTSEngine:
    """Google Gemini text-to-speech (google-genai SDK) - speaks Amharic natively.

    The free plan allows only a few TTS requests per minute/day, so instead of one
    request per sentence the script is recorded in a handful of chunks (several
    sentences each, one per line). Each chunk is then cut back into sentences at
    the pauses the voice naturally leaves between them, so the on-screen key
    points still change exactly when each sentence starts.

    If Gemini TTS is unavailable (no key, quota used up, network) the whole show
    is recorded with the `fallback` engine instead, so the voice never changes
    half-way through a broadcast."""

    STYLE = ("Read the following aloud in Amharic, as a calm, clear and confident TV news anchor, "
             "at a steady broadcast pace, with a short pause after every line:\n")

    def __init__(self, api_key: str, voice: str = "Charon", models=None, fallback=None,
                 max_chars: int = 420, log=lambda m: None):
        self.key, self.voice = api_key, voice
        self.models = list(models or ["gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"])
        self.fallback, self.max_chars, self.log = fallback, max_chars, log

    # ---------------------------------------------------------------- request
    def _request(self, text: str) -> np.ndarray:
        import re
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self.key, http_options=types.HttpOptions(timeout=180_000))
        cfg = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice))))
        err = None
        for model in self.models:
            for attempt in range(4):
                try:
                    r = client.models.generate_content(model=model, contents=self.STYLE + text, config=cfg)
                    part = next(p for p in r.candidates[0].content.parts if getattr(p, "inline_data", None))
                    pcm = np.frombuffer(part.inline_data.data, dtype="<i2").astype(np.float32) / 32768.0
                    rate = re.search(r"rate=(\d+)", part.inline_data.mime_type or "")
                    rate = int(rate.group(1)) if rate else 24000
                    if rate != SR:
                        idx = np.arange(0, len(pcm), rate / SR)
                        pcm = np.interp(idx, np.arange(len(pcm)), pcm).astype(np.float32)
                    if len(pcm) < SR // 2:
                        raise RuntimeError("empty audio")
                    return pcm
                except Exception as exc:
                    err, msg = exc, str(exc)
                    busy = any(c in msg for c in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "empty audio"))
                    if "404" in msg or "NOT_FOUND" in msg or not busy or attempt == 3:
                        break
                    wait = re.search(r"retry(?:Delay)?[\"': ]+(\d+)", msg)
                    delay = min(75.0, float(wait.group(1)) + 2) if wait else 12.0 * (attempt + 1)
                    if "per day" in msg.lower() or "PerDay" in msg:
                        break                                   # daily quota: waiting won't help
                    self.log(f"Gemini voice busy, retrying in {delay:.0f}s…")
                    time.sleep(delay)
        raise RuntimeError(f"Gemini TTS failed: {str(err)[:300]}")

    # ----------------------------------------------------------------- split
    @staticmethod
    def _split(audio: np.ndarray, texts: list[str]) -> list[np.ndarray]:
        """Cut one recording into len(texts) sentences at the pauses between them."""
        n = len(texts)
        if n == 1:
            return [audio]
        hop = int(0.01 * SR)
        frames = len(audio) // hop
        env = np.sqrt(np.convolve((audio[:frames * hop].reshape(frames, hop) ** 2).mean(axis=1), np.ones(3) / 3, "same"))
        db = 20 * np.log10(env + 1e-7)
        quiet = db < (np.percentile(db[db > -90], 90) - 32)
        # candidate pauses: runs of quiet frames >= 90 ms, scored by length
        runs, i = [], 0
        while i < frames:
            if quiet[i]:
                j = i
                while j < frames and quiet[j]:
                    j += 1
                if j - i >= 9 and i > 0 and j < frames:
                    runs.append(((i + j) / 2.0, j - i))
                i = j
            else:
                i += 1
        weights = np.array([max(1, len(t)) for t in texts], np.float32)
        expect = np.cumsum(weights)[:-1] / weights.sum() * frames
        cuts, last = [], 0.0
        for k, e in enumerate(expect):
            cand = [(abs(c - e) / (1.0 + 0.12 * L), c) for c, L in runs if c > last + 30 and c < frames - 30 * (n - 1 - k)]
            if cand:
                c = min(cand)[1]
            else:                                              # no clean pause: quietest point near the estimate
                a, b = int(max(last + 20, e - 60)), int(min(frames - 20, e + 60))
                c = a + int(np.argmin(db[a:b])) if b > a else e
            cuts.append(c)
            last = c
        bounds = [0] + [int(c * hop) for c in cuts] + [len(audio)]
        return [audio[bounds[k]:bounds[k + 1]] for k in range(n)]

    # ----------------------------------------------------------------- batch
    def synth_batch(self, texts: list[str], progress=lambda f, m="": None) -> list[np.ndarray]:
        chunks, cur = [], []
        for i, t in enumerate(texts):
            if cur and sum(len(texts[j]) for j in cur) + len(t) > self.max_chars:
                chunks.append(cur)
                cur = []
            cur.append(i)
        if cur:
            chunks.append(cur)
        out: list[np.ndarray | None] = [None] * len(texts)
        try:
            if not self.key:
                raise RuntimeError("no Gemini API key")
            for k, idx in enumerate(chunks):
                progress(k / len(chunks), f"Recording Amharic voice with Gemini… part {k + 1}/{len(chunks)}")
                audio = self._request("\n".join(texts[i] for i in idx))
                for i, clip in zip(idx, self._split(audio, [texts[i] for i in idx])):
                    out[i] = clip
            progress(1.0, "Voice recorded (Gemini)")
            return out
        except Exception as exc:
            if self.fallback is None:
                raise
            self.log(f"{exc} - recording the whole show with the backup voice instead.")
            res = []
            for i, t in enumerate(texts):
                res.append(self.fallback.synth(t))
                progress((i + 1) / len(texts), f"Recording voice (backup)… {i + 1}/{len(texts)} sentences")
            return res


def _pause_after(text: str) -> float:
    t = text.rstrip()
    if t.endswith(("?", "!", "፧")):
        return 0.42
    if t.endswith(":"):
        return 0.5
    return 0.30


def build(script: dict, engine, target_seconds: float, lead_in: float = 1.8, tail: float = 3.2,
          progress=lambda f, m="": None, workers: int = 4) -> tuple[np.ndarray, dict]:
    """Returns (audio @24 kHz, timeline)."""
    jobs = [(si, ci, s["say"]) for si, st in enumerate(script["stories"]) for ci, s in enumerate(st["sentences"])]
    clips: dict[tuple[int, int], np.ndarray] = {}
    done = 0

    if hasattr(engine, "synth_batch"):                    # engines that record many sentences per request
        for j, clip in zip(jobs, engine.synth_batch([j[2] for j in jobs], progress=progress)):
            clips[(j[0], j[1])] = _trim(clip)
    else:
        def work(j):
            return j, _trim(engine.synth(j[2]))

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for j, clip in ex.map(work, jobs):
                clips[(j[0], j[1])] = clip
                done += 1
                progress(done / len(jobs), f"Recording voice… {done}/{len(jobs)} sentences")

    # ---- stitch speech (no lead-in / tail yet)
    parts, cues, stories = [], [], []
    cursor = 0.0
    for si, st in enumerate(script["stories"]):
        s_start = cursor
        for ci, s in enumerate(st["sentences"]):
            clip = clips[(si, ci)]
            dur = len(clip) / SR
            cues.append(dict(start=cursor, end=cursor + dur, story=si, headline=st["headline"],
                             highlight=s.get("highlight") or st["headline"], text=s["say"]))
            parts.append(clip)
            pz = _pause_after(s["say"])
            if ci == len(st["sentences"]) - 1:
                pz = 0.9 if si < len(script["stories"]) - 1 else 0.4
            parts.append(np.zeros(int(pz * SR), np.float32))
            cursor += dur + pz
        stories.append(dict(headline=st["headline"], start=s_start, end=cursor))
    speech = np.concatenate(parts).astype(np.float32)

    # ---- fit to the target length with a gentle, pitch-preserving tempo change
    want = max(30.0, target_seconds - lead_in - tail)
    f = float(np.clip(len(speech) / SR / want, 0.90, 1.12))
    if abs(f - 1.0) > 0.015:
        progress(1.0, "Fine-tuning pace…")
        d = tempfile.mkdtemp()
        a, b = os.path.join(d, "a.wav"), os.path.join(d, "b.wav")
        media.write_wav(a, speech, SR)
        p = media.run(["-i", a, "-filter:a", f"atempo={f:.5f}", b])
        if p.returncode == 0:
            new = media.decode(b, SR)
            ratio = len(new) / len(speech)
            speech = new
            for c in cues:
                c["start"] *= ratio
                c["end"] *= ratio
            for s in stories:
                s["start"] *= ratio
                s["end"] *= ratio

    # ---- loudness: peak-normalise with headroom
    peak = np.percentile(np.abs(speech), 99.8) + 1e-6
    speech = np.clip(speech * (0.80 / peak), -0.98, 0.98).astype(np.float32)

    audio = np.concatenate([np.zeros(int(lead_in * SR), np.float32), speech, np.zeros(int(tail * SR), np.float32)])
    for c in cues:
        c["start"] += lead_in
        c["end"] += lead_in
    for s in stories:
        s["start"] += lead_in
        s["end"] += lead_in
    timeline = dict(duration=len(audio) / SR, cues=cues, stories=stories,
                    ticker=[st.get("ticker") or st["headline"] for st in script["stories"][1:-1]] or
                           [st["headline"] for st in script["stories"]])
    return audio, timeline
