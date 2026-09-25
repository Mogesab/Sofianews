"""
Narration -> mouth motion.

Two engines live here:

  * analyze_speech()  (default)  text-driven.  We know the words of every sentence, so they are
    converted to phonemes (phonemes.py), pinned to the real audio with a forced alignment
    (align.py) and turned into lip shapes with coarticulation (visemes.py).  Lips close on
    m/b/p, tuck on f/v, round on oo/w/sh, spread on ee ... exactly where those sounds happen.

  * _analyze_audio()   the original loudness-only analysis.  Still used (a) as a safety net if
    the text-driven path fails, and (b) to provide the loud/emph curves that drive head nods
    and eyebrow lifts in motion.py.

The audio-only description follows.  For every video frame we look at ~43 ms of the narration and derive:

  mopen   0..1   how far the jaw drops   (loudness, less for hissy consonants)
  mspread -1..1  lips rounded (oo/oh) .. stretched (ee)   (spectral balance)
  loud    0..1   overall loudness
  emph    0..1   slow "stress" curve, used for head nods and eyebrow lifts

It is language-agnostic and needs no transcript, so it can never drift out of
sync with the voice: the mouth is literally computed from the sound.
"""
from __future__ import annotations

import numpy as np


def _one_pole(x: np.ndarray, a_up: float, a_dn: float) -> np.ndarray:
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += (a_up if v > acc else a_dn) * (v - acc)
        y[i] = acc
    return y


def _onset_dips(openness: np.ndarray, fps: int) -> np.ndarray:
    """A perfectly smooth open/close curve reads as a flapping jaw, not speech -
    real mouths briefly narrow right before a loud syllable bursts open (the lip
    or tongue closure of a consonant). We don't know the phonemes, but we know
    where the loudness jumps, which is where those closures actually fall.
    Returns a 0..1 "how much to dip" curve to subtract from mopen."""
    d = np.diff(openness, prepend=openness[0])
    rise = np.clip(d, 0, None)
    dip = np.zeros_like(openness)
    thresh = max(0.05, float(np.percentile(rise, 90)))
    i, n, lead = 0, len(openness), max(1, int(fps * 0.045))
    while i < n:
        if rise[i] > thresh:
            strength = float(np.clip(rise[i] / (thresh * 2.2), 0.25, 1.0))
            lo = max(0, i - lead)
            width = i - lo + 1
            dip[lo:i + 1] = np.maximum(dip[lo:i + 1], strength * np.linspace(0.15, 1.0, width))
            i += max(1, int(fps * 0.09))          # don't re-trigger inside the same syllable
        else:
            i += 1
    return dip


def _analyze_audio(audio: np.ndarray, sr: int, fps: int, n_frames: int) -> dict:
    hop = sr / fps
    win = 1024
    window = np.hanning(win).astype(np.float32)
    pad = np.concatenate([np.zeros(win // 2, np.float32), audio.astype(np.float32),
                          np.zeros(win + int(hop) * 4, np.float32)])
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    band = lambda lo, hi: (freqs >= lo) & (freqs < hi)
    b_low, b_mid, b_hi, b_all = band(250, 1000), band(1000, 3000), band(4500, 11000), band(100, 11000)

    rms = np.zeros(n_frames, np.float32)
    F = np.zeros(n_frames, np.float32)
    fric = np.zeros(n_frames, np.float32)
    for i in range(n_frames):
        st = int(i * hop)
        seg = pad[st:st + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        rms[i] = float(np.sqrt(np.mean(seg ** 2)) + 1e-9)
        mag = np.abs(np.fft.rfft(seg * window)) ** 2
        e_low, e_mid, e_hi, e_all = mag[b_low].sum(), mag[b_mid].sum(), mag[b_hi].sum(), mag[b_all].sum() + 1e-12
        F[i] = e_mid / (e_low + e_mid + 1e-12)
        fric[i] = e_hi / e_all

    db = 20 * np.log10(rms)
    ref = np.percentile(db[db > -70], 95) if np.any(db > -70) else -20.0
    loud = np.clip((db - (ref - 30.0)) / 30.0, 0, 1)
    loud[db < ref - 33.0] = 0.0

    openness = loud ** 0.8 * (0.66 + 0.34 * (1.0 - F)) * (1.0 - 0.5 * np.clip(fric * 3.0, 0, 1))
    openness = np.clip(openness * 1.05, 0, 1)
    spread = np.clip((F - 0.5) * 2.6 + fric * 0.9, -1, 1) * (loud > 0.05)

    # the lips lead the sound by about a frame in real speech
    openness = np.concatenate([openness[1:], openness[-1:]])
    spread = np.concatenate([spread[1:], spread[-1:]])

    # faster attack AND faster release than before: a flatter, slower-decaying
    # curve is what makes a mouth read as "gulping" instead of articulating -
    # real jaws snap open on a stressed syllable and fall shut just as quickly
    # the instant the loudness drops, they don't ease down.
    mopen = _one_pole(openness, 0.78, 0.66)

    # expand the dynamic range: push near-silence fully shut and let genuine
    # peaks reach fully open, instead of hovering in a narrow "half open" band
    mopen = np.clip((mopen - 0.07) / 0.86, 0, 1) ** 0.88

    mspread = _one_pole(spread, 0.62, 0.46)

    # consonant-like closures right before loud onsets, so the mouth doesn't just
    # balloon open and stay there - it snaps toward closed and bursts, like speech
    rng = np.random.default_rng(0)
    dip = _onset_dips(mopen, fps)
    mopen = np.clip(mopen * (1.0 - 0.62 * dip), 0, 1)

    # tiny fast texture so consecutive frames are never identically smooth -
    # real lips/tongue flutter even within a single vowel
    jitter = _one_pole(rng.standard_normal(n_frames).astype(np.float32), 0.5, 0.5) * 0.05
    mopen = np.clip(mopen + jitter * (mopen > 0.04), 0, 1).astype(np.float32)

    # mouths are never perfectly symmetric while talking - a slow, loudness-gated
    # wander that nudges one corner a little more than the other
    asym_raw = _one_pole(rng.standard_normal(n_frames).astype(np.float32), 0.05, 0.05)
    mcorner = np.clip(asym_raw * 2.2, -1, 1) * np.clip(mopen * 2.2, 0, 1)
    mcorner = mcorner.astype(np.float32)

    sm = _one_pole(loud, 1 - np.exp(-1 / (fps * 0.20)), 1 - np.exp(-1 / (fps * 0.35)))
    slow = _one_pole(loud, 1 - np.exp(-1 / (fps * 2.0)), 1 - np.exp(-1 / (fps * 2.0)))
    emph = np.clip(sm - slow, 0, None)
    p95 = np.percentile(emph, 95) + 1e-6
    emph = np.clip(emph / p95, 0, 1)
    return dict(mopen=mopen.astype(np.float32), mspread=mspread.astype(np.float32),
                mcorner=mcorner, loud=sm.astype(np.float32), emph=emph.astype(np.float32))


# ---------------------------------------------------------------------------------------------
def analyze_speech(audio: np.ndarray, sr: int, fps: int, n_frames: int, cues: list[dict],
                   base: dict, intensity: float = 1.0, lead_ms: float = 25.0, log=lambda m: None) -> dict:
    """Text-driven mouth curves. `base` is the audio-only analysis (used to patch any sentence that fails)."""
    from . import align, phonemes, visemes
    total = len(audio) / sr
    segs, failed = [], []
    pad = 0.15
    for ci, c in enumerate(cues):
        t0, t1 = max(0.0, c["start"] - pad), min(total, c["end"] + pad)
        try:
            items = phonemes.text_to_items(c.get("text", ""))
            clip = audio[int(t0 * sr):int(t1 * sr)]
            sg = align.align(clip, sr, items)
            if not sg:
                raise ValueError("nothing aligned")
        except Exception as exc:                            # never let one odd sentence break the whole video
            log(f"lip-sync: sentence {ci + 1} fell back to audio-only ({exc})")
            failed.append((c["start"], c["end"]))
            continue
        for g in sg:
            g.start += t0
            g.end += t0
        segs.extend(sg)
    if not segs:
        raise RuntimeError("no sentence could be aligned")
    segs.sort(key=lambda g: g.start)
    # drop overlaps produced by the padded windows of neighbouring sentences
    clean, edge = [], 0.0
    for g in segs:
        if g.end <= edge:
            continue
        g.start = max(g.start, edge)
        clean.append(g)
        edge = g.end
    cur = visemes.curves(clean, total, fps, n_frames, lead=lead_ms / 1000.0, intensity=intensity)
    out = dict(mopen=cur["jaw"], mspread=cur["wide"], mround=cur["round"], mpress=cur["press"],
               mtuck=cur["tuck"], mtongue=cur["tongue"], mteeth=cur["teeth"],
               mcorner=np.zeros(n_frames, np.float32), loud=base["loud"], emph=base["emph"])
    for (a, b) in failed:                                   # patch failed sentences with the audio-only mouth
        i0, i1 = max(0, int(a * fps)), min(n_frames, int(np.ceil(b * fps)) + 1)
        for k in ("mopen", "mspread"):
            out[k][i0:i1] = base[k][i0:i1]
        for k in ("mround", "mpress", "mtuck", "mtongue", "mteeth"):
            out[k][i0:i1] = 0.0
    out["_aligned"] = clean
    return out


def analyze(audio: np.ndarray, sr: int, fps: int, n_frames: int, cues: list[dict] | None = None,
            mode: str = "auto", intensity: float = 1.0, lead_ms: float = 25.0, log=lambda m: None) -> dict:
    """mode: "auto" (text-driven, audio-only as fallback) | "audio" (original loudness-only)."""
    base = _analyze_audio(audio, sr, fps, n_frames)
    if mode == "audio" or not cues:
        return base
    try:
        return analyze_speech(audio, sr, fps, n_frames, cues, base, intensity, lead_ms, log)
    except Exception as exc:
        log(f"lip-sync: text-driven engine failed ({exc}); using audio-only lip-sync")
        return base
