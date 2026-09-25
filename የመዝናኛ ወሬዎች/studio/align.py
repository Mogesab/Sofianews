"""
Forced alignment of a known phoneme sequence to the audio it was spoken in.

We know the text of every narrated sentence, so the question is only *when*
each sound happens. This module answers that without any speech-recognition
model: it measures a few cheap acoustic cues per 10 ms frame (loudness,
voicing, high-frequency hiss, low-frequency murmur) and runs a left-to-right
HMM / Viterbi search that finds the most plausible time for every phoneme,
where each phoneme is described by the *kind* of sound it is (vowel, nasal,
hissy fricative, stop closure + burst, silence ...) and how long that kind of
sound typically lasts.

That is enough to place mouth closures on m/b/p, lip-rounding on oo/w, the
teeth-on-lip of f/v and the open jaw of vowels in exactly the right frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .phonemes import VOWELS, Item

HOP_S = 0.010

# ------------------------------------------------------------------ acoustic classes
CLASSES = ["V", "G", "N", "FS", "FW", "FV", "CL", "CV", "BR", "SIL"]
CI = {c: i for i, c in enumerate(CLASSES)}
# mean / std of the four features per class:  loudness (dB rel. to speech level),
# voicing 0..1, high-frequency share 0..1, low-frequency share 0..1
_MU = {
    "V":   (-6.0, 0.85, 0.05, 0.65),
    "G":   (-12.0, 0.85, 0.03, 0.70),
    "N":   (-17.0, 0.88, 0.01, 0.92),
    "FS":  (-11.0, 0.15, 0.62, 0.05),
    "FW":  (-25.0, 0.15, 0.35, 0.20),
    "FV":  (-15.0, 0.55, 0.30, 0.35),
    "CL":  (-46.0, 0.10, 0.20, 0.30),
    "CV":  (-29.0, 0.70, 0.02, 0.95),
    "BR":  (-15.0, 0.20, 0.45, 0.15),
    "SIL": (-58.0, 0.10, 0.25, 0.35),
}
_SD = {
    "V":   (8.0, 0.22, 0.10, 0.28),
    "G":   (8.0, 0.22, 0.09, 0.28),
    "N":   (8.0, 0.22, 0.06, 0.16),
    "FS":  (10.0, 0.28, 0.30, 0.12),
    "FW":  (10.0, 0.28, 0.32, 0.28),
    "FV":  (10.0, 0.32, 0.32, 0.32),
    "CL":  (16.0, 0.32, 0.40, 0.40),
    "CV":  (11.0, 0.30, 0.12, 0.14),
    "BR":  (10.0, 0.30, 0.32, 0.30),
    "SIL": (16.0, 0.32, 0.50, 0.50),
}

# phoneme -> [(class, min_frames, mean_frames)]
_STOP_UNV = {"P", "T", "K"}
_STOP_V = {"B", "D", "G"}
_DIPH = {"AW", "AY", "OY", "EY", "OW"}


def _spec(sym: str, stress: int) -> list[tuple[str, int, float]]:
    if sym in VOWELS:
        if sym in _DIPH:
            return [("V", 5, 13.0 if stress else 10.0)]
        if sym == "ER":
            return [("V", 4, 9.0 if stress else 6.0)]
        return [("V", 3 if stress else 2, 9.5 if stress else 5.0)]
    if sym in _STOP_UNV:
        return [("CL", 2, 5.5), ("BR", 1, 3.0)]
    if sym in _STOP_V:
        return [("CV", 2, 4.5), ("BR", 1, 2.0)]
    if sym in ("M", "N", "NG"):
        return [("N", 3, 6.5)]
    if sym in ("L", "R", "W", "Y"):
        return [("G", 3, 6.0)]
    if sym == "HH":
        return [("FW", 2, 5.0)]
    if sym in ("F", "TH"):
        return [("FW", 3, 8.0)]
    if sym in ("S", "SH"):
        return [("FS", 4, 10.0)]
    if sym in ("V", "DH"):
        return [("FV", 2, 5.0)]
    if sym in ("Z", "ZH"):
        return [("FV", 3, 8.0)]
    if sym == "CH":
        return [("CL", 2, 3.5), ("FS", 3, 6.5)]
    if sym == "JH":
        return [("CV", 2, 3.5), ("FV", 3, 6.5)]
    return [("V", 2, 6.0)]


# ------------------------------------------------------------------ features
def features(audio: np.ndarray, sr: int) -> dict:
    """Per-10 ms frame cues. Returns arrays of length n_frames."""
    hop = int(round(sr * HOP_S))
    n = max(1, len(audio) // hop)
    win = 1024
    w = np.hanning(win).astype(np.float32)
    pad = np.concatenate([np.zeros(win // 2, np.float32), audio.astype(np.float32), np.zeros(win, np.float32)])
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    frames = pad[idx]
    # spectrum from a 512-sample core (~21 ms), voicing from the full 1024 (~43 ms)
    core = frames[:, win // 2 - 256:win // 2 + 256] * np.hanning(512).astype(np.float32)
    P = np.abs(np.fft.rfft(core, axis=1)) ** 2
    freqs = np.fft.rfftfreq(512, 1.0 / sr)
    tot = P.sum(axis=1) + 1e-12
    db = 10 * np.log10(tot / 512.0 + 1e-10)
    hf = P[:, freqs >= 4000].sum(axis=1) / tot
    low = P[:, freqs < 1000].sum(axis=1) / tot

    fw = frames * w
    spec = np.fft.rfft(fw, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)
    win_ac = np.fft.irfft(np.abs(np.fft.rfft(w)) ** 2)
    lo_lag, hi_lag = int(sr / 400), min(int(sr / 70), win // 2 - 1)
    norm = ac[:, 0:1] + 1e-9
    r = (ac[:, lo_lag:hi_lag] / norm) / (win_ac[lo_lag:hi_lag] / win_ac[0] + 1e-6)[None, :]
    voicing = np.clip(r.max(axis=1), 0, 1)
    voicing[db < db.max() - 62] = 0.0                    # nothing to be voiced about in silence

    active = db > (np.percentile(db, 5) + 18)
    ref = float(np.percentile(db[active], 90)) if active.sum() > 5 else float(db.max())
    er = np.clip(db - ref, -75, 12)

    # spectral change: phoneme boundaries inside runs of vowels/nasals/liquids show up as
    # formant movement rather than as a change in loudness
    edges = np.geomspace(150, min(8000, sr / 2 - 1), 21)
    bands = np.stack([P[:, (freqs >= a) & (freqs < b)].sum(axis=1) for a, b in zip(edges[:-1], edges[1:])], axis=1)
    L = np.log10(bands + 1e-9 * tot[:, None] + 1e-12)
    L = L - L.mean(axis=1, keepdims=True)                  # shape only (ignore overall level)
    Lp = np.pad(L, ((2, 2), (0, 0)), mode="edge")
    flux = np.sqrt(((Lp[4:] - Lp[:-4]) ** 2).mean(axis=1))
    loud_enough = np.clip((er + 40) / 20.0, 0, 1)          # ignore "change" in near-silence
    flux = flux * (0.4 + 0.6 * loud_enough)
    flux = np.clip(flux / (np.percentile(flux, 75) + 1e-6), 0, 3.0)
    return dict(er=er.astype(np.float32), vo=voicing.astype(np.float32), hf=hf.astype(np.float32),
                low=low.astype(np.float32), flux=flux.astype(np.float32), n=n, ref=ref)


def _emission(feat: dict) -> np.ndarray:
    """(T, n_classes) log-likelihoods"""
    X = np.stack([feat["er"], feat["vo"], feat["hf"], feat["low"]], axis=1)
    out = np.empty((X.shape[0], len(CLASSES)), np.float32)
    for c in CLASSES:
        mu, sd = np.array(_MU[c], np.float32), np.array(_SD[c], np.float32)
        z = (X - mu) / sd
        out[:, CI[c]] = (-0.5 * z ** 2 - np.log(sd)).sum(axis=1)
    return out


# ------------------------------------------------------------------ HMM construction
@dataclass
class Seg:
    sym: str
    stress: int
    start: float
    end: float
    word: int
    level: float = 0.0       # loudness of the segment relative to the sentence, dB (0 = typical)
    kind: str = "ph"         # "ph" | "pause"


def _build(items: list[Item], T: int, rate: float, force_single: bool = False):
    """Expand the item list into HMM states. Returns arrays + per-state unit index."""
    cls, stay, leave, pred1, pc1, pred2, pc2, unit_of = [], [], [], [], [], [], [], []
    units: list[dict] = []
    NEG = -1e9

    def add_chain(subs, unit_idx):
        """subs = [(class, min, mean)] -> (first_state, last_state).
        Each sound is an Erlang chain of n sub-states: durations cluster around the mean
        (a plain geometric loop would let a vowel shrink to nothing next to a long 'r')."""
        first = last = None
        for (c, mn, mean) in subs:
            n = 1 if force_single else int(np.clip(round(mean / 2.2), 1, 4))
            p_stay = float(np.clip(1.0 - n / max(mean, n + 0.01), 0.0, 0.92))
            for k in range(n):
                s = len(cls)
                cls.append(CI[c])
                stay.append(float(np.log(p_stay)) if p_stay > 1e-4 else NEG)
                leave.append(float(np.log(max(1 - p_stay, 1e-4))))
                pred1.append(last if last is not None else -1); pc1.append(0.0)
                pred2.append(-1); pc2.append(0.0); unit_of.append(unit_idx)
                first = s if first is None else first
                last = s
        return first, last

    prev_last = None
    prev_prev_last = None                 # end of the unit before an optional pause
    pending_skip_from = None
    pending_skip_cost = 0.0
    first_of_next_needs = None

    def connect(first_state, from_last, cost=0.0):
        if pred1[first_state] == -1:
            pred1[first_state] = from_last; pc1[first_state] = cost
        else:
            pred2[first_state] = from_last; pc2[first_state] = cost

    def add_unit(kind, sym, stress, word, subs, sp_optional=False):
        ui = len(units)
        units.append(dict(kind=kind, sym=sym, stress=stress, word=word))
        f, l = add_chain(subs, ui)
        return ui, f, l

    # lead-in silence
    ui, f, l = add_unit("pause", "", 0, -1, [("SIL", 1, 6.0)])
    prev_last = l
    skip_src = None                       # when an optional pause precedes: (state to skip from, cost)
    n_items = len(items)
    i = 0
    while i < n_items:
        it = items[i]
        if it.kind in ("wb", "pause"):
            p_through = 0.10 if it.kind == "wb" else (0.85 if it.pause == "short" else 0.92)
            # optional silence unit
            pause_mean = 4.0 if it.kind == "wb" else (16.0 if it.pause == "short" else 24.0)
            ui, f, l = add_unit("pause", "", 0, -1, [("SIL", 1, pause_mean * rate)])
            connect(f, prev_last, float(np.log(p_through)))
            skip_src = (prev_last, float(np.log(1 - p_through)))
            prev_last = l
            i += 1
            continue
        subs = [(c, mn, mean * rate) for (c, mn, mean) in _spec(it.sym, it.stress)]
        ui, f, l = add_unit("ph", it.sym, it.stress, it.word, subs)
        connect(f, prev_last, 0.0)
        if skip_src is not None:
            connect(f, skip_src[0], skip_src[1])
            skip_src = None
        prev_last = l
        i += 1
    # trailing silence (also reachable directly from the last phoneme)
    ui, f, l = add_unit("pause", "", 0, -1, [("SIL", 1, 10.0)])
    connect(f, prev_last, 0.0)
    if skip_src is not None:
        connect(f, skip_src[0], skip_src[1])
    S = len(cls)
    arr = dict(cls=np.array(cls, np.int32), stay=np.array(stay, np.float32), leave=np.array(leave, np.float32),
               pred1=np.array(pred1, np.int32), pc1=np.array(pc1, np.float32),
               pred2=np.array(pred2, np.int32), pc2=np.array(pc2, np.float32), unit_of=np.array(unit_of, np.int32))
    return arr, units, S


BOUNDARY_W = 1.6


def _viterbi(emit: np.ndarray, g: dict, S: int, flux: np.ndarray | None = None) -> np.ndarray:
    T = emit.shape[0]
    NEG = np.float32(-1e9)
    cls, stay, leave = g["cls"], g["stay"], g["leave"]
    p1, c1, p2, c2 = g["pred1"], g["pc1"], g["pred2"], g["pc2"]
    has1, has2 = p1 >= 0, p2 >= 0
    p1s, p2s = np.where(has1, p1, 0), np.where(has2, p2, 0)
    add1 = np.where(has1, leave[p1s] + c1, NEG).astype(np.float32)
    add2 = np.where(has2, leave[p2s] + c2, NEG).astype(np.float32)
    unit = g["unit_of"]
    isb1 = (has1 & (unit[p1s] != unit)).astype(np.float32)      # transitions that cross a phoneme boundary
    isb2 = (has2 & (unit[p2s] != unit)).astype(np.float32)
    if flux is None:
        bonus = np.zeros(T, np.float32)
    else:
        bonus = (BOUNDARY_W * (np.clip(flux, 0, 2.5) - 0.9)).astype(np.float32)
    delta = np.full(S, NEG, np.float32)
    delta[0] = emit[0, cls[0]]
    bp = np.zeros((T, S), np.int8)
    for t in range(1, T):
        e = emit[t, cls]
        a = delta + stay
        b = delta[p1s] + add1 + bonus[t] * isb1
        c = delta[p2s] + add2 + bonus[t] * isb2
        best = a
        choice = np.zeros(S, np.int8)
        m = b > best
        best = np.where(m, b, best); choice[m] = 1
        m = c > best
        best = np.where(m, c, best); choice[m] = 2
        delta = np.maximum(best + e, NEG)
        bp[t] = choice
    s = S - 1
    if delta[s] <= NEG / 2:                               # end state unreachable: take best anywhere near the end
        s = int(np.argmax(delta))
    path = np.empty(T, np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = s
        ch = bp[t, s]
        if ch == 1:
            s = int(p1s[s])
        elif ch == 2:
            s = int(p2s[s])
    return path


def align(audio: np.ndarray, sr: int, items: list[Item]) -> list[Seg]:
    """Returns one Seg per phoneme (and per non-trivial pause) with times in seconds, relative to `audio`."""
    ph_items = [it for it in items if it.kind == "ph"]
    if not ph_items or len(audio) < sr * 0.05:
        return []
    feat = features(audio, sr)
    T = feat["n"]
    emit = _emission(feat)

    # speaking-rate estimate: active span vs the sum of typical durations
    active = np.where(feat["er"] > -30)[0]
    span = (active[-1] - active[0] + 1) if len(active) else T
    typical = sum(sum(m for (_, _, m) in _spec(it.sym, it.stress)) for it in ph_items)
    n_short = sum(1 for it in items if it.kind == "pause" and it.pause == "short")
    n_long = sum(1 for it in items if it.kind == "pause" and it.pause == "long")
    typical += n_short * 16 * 0.6
    rate = float(np.clip(span / max(typical, 1.0), 0.55, 1.7))
    g, units, S = _build(items, T, rate)
    if S > T * 0.97:                                     # clip too short for the full chains: 1 sub-state per sound
        g, units, S = _build(items, T, rate, force_single=True)
    path = _viterbi(emit, g, S, feat["flux"])
    unit_path = g["unit_of"][path]

    er = feat["er"]
    segs: list[Seg] = []
    t = 0
    while t < T:
        u = int(unit_path[t])
        t0 = t
        while t < T and unit_path[t] == u:
            t += 1
        info = units[u]
        if info["kind"] == "ph":
            lvl = float(np.mean(er[t0:t]))
            segs.append(Seg(info["sym"], info["stress"], t0 * HOP_S, t * HOP_S, info["word"], lvl))
        elif (t - t0) >= 2:
            segs.append(Seg("", 0, t0 * HOP_S, t * HOP_S, -1, float(np.mean(er[t0:t])), "pause"))
    return segs
