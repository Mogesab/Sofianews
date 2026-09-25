"""
A tiny offline "voice" for tests: turns text into a robotic-but-phonetically-honest
signal (formant vowels, noisy fricatives, silent-closure + burst stops, nasal
murmur) so the lip-sync can be exercised with no internet and no TTS engine.

It also returns the true start/end time of every phoneme, which the tests use to
measure how accurately studio.align recovers the timing from the audio alone.
Not used in the real show.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio import phonemes                                     # noqa: E402
from studio.phonemes import VOWELS                              # noqa: E402

SR = 24000
# F1, F2, F3 (Hz)
_FORM = {
    "AA": (730, 1090, 2440), "AE": (660, 1720, 2410), "AH": (640, 1190, 2390), "AO": (570, 840, 2410),
    "AW": (700, 1200, 2400), "AY": (700, 1300, 2400), "EH": (530, 1840, 2480), "ER": (490, 1350, 1690),
    "EY": (480, 1900, 2500), "IH": (390, 1990, 2550), "IY": (270, 2290, 3010), "OW": (500, 900, 2400),
    "OY": (550, 1000, 2400), "UH": (440, 1020, 2240), "UW": (300, 870, 2240),
    "L": (360, 1300, 2800), "R": (420, 1300, 1600), "W": (300, 700, 2200), "Y": (300, 2200, 3000),
    "M": (250, 1100, 2500), "N": (250, 1500, 2500), "NG": (250, 1900, 2500),
}
_DUR = {"V": 0.095, "DIPH": 0.15, "STOP": 0.085, "NAS": 0.07, "LIQ": 0.065, "S": 0.11, "F": 0.09, "VF": 0.06, "AFF": 0.11}


def _db(x):
    return 10 ** (x / 20.0)


def _noise_band(n, lo, hi, rng):
    if n < 8:
        return np.zeros(n, np.float32)
    x = rng.standard_normal(n).astype(np.float32)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    shape = np.clip((f - lo) / max(lo * 0.25, 200), 0, 1) * np.clip((hi - f) / max(hi * 0.15, 300), 0, 1)
    y = np.fft.irfft(X * shape, n).astype(np.float32)
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-9)


def _env(n, a=0.008, r=0.010):
    e = np.ones(n, np.float32)
    na, nr = min(n // 2, int(a * SR)), min(n // 2, int(r * SR))
    if na:
        e[:na] = np.linspace(0, 1, na)
    if nr:
        e[-nr:] = np.linspace(1, 0, nr)
    return e


def synth(items, rate: float = 1.0, rng=None, f0: float = 112.0, noise_db: float = -64.0, jitter: float = 0.25):
    """items: phonemes.Item list. Returns (audio float32 @24k, truth [(sym, t0, t1)])."""
    rng = rng or np.random.default_rng(0)
    # ---- 1. durations
    plan = []          # (sym, stress, dur, kind)
    for it in items:
        if it.kind == "ph":
            s = it.sym
            if s in VOWELS:
                d = _DUR["DIPH"] if s in ("AW", "AY", "OY", "EY", "OW") else _DUR["V"] * (1.25 if it.stress else 0.75)
            elif s in ("P", "T", "K", "B", "D", "G"):
                d = _DUR["STOP"]
            elif s in ("M", "N", "NG"):
                d = _DUR["NAS"]
            elif s in ("L", "R", "W", "Y"):
                d = _DUR["LIQ"]
            elif s in ("S", "SH"):
                d = _DUR["S"]
            elif s in ("F", "TH", "HH"):
                d = _DUR["F"]
            elif s in ("CH", "JH"):
                d = _DUR["AFF"]
            else:
                d = _DUR["VF"] if s in ("V", "DH") else 0.085
            d *= float(np.exp(rng.normal(0, jitter))) / rate
            plan.append((s, it.stress, max(d, 0.03), "ph"))
        elif it.kind == "pause":
            plan.append(("", 0, (0.22 if it.pause == "short" else 0.34) / rate * float(np.exp(rng.normal(0, 0.2))), "pause"))
        elif it.kind == "wb" and rng.random() < 0.10:
            plan.append(("", 0, 0.05 / rate, "pause"))
    lead, tail = 0.05, 0.10
    total = lead + tail + sum(p[2] for p in plan)
    N = int(total * SR) + 1
    out = np.zeros(N, np.float32)
    voiced_f0 = np.zeros(N, np.float32)
    F = np.zeros((3, N), np.float32)
    amp_v = np.zeros(N, np.float32)            # voiced-source amplitude
    truth = []
    t = lead
    prev_form = (500.0, 1500.0, 2500.0)
    n_ph = sum(1 for p in plan if p[3] == "ph")
    k_ph = 0
    for (s, stress, d, kind) in plan:
        a, b = int(t * SR), min(N, int((t + d) * SR))
        n = b - a
        if kind == "pause" or n <= 0:
            t += d
            continue
        k_ph += 1
        truth.append((s, t, t + d))
        prog = k_ph / max(1, n_ph)
        f0_here = f0 * (1.12 - 0.22 * prog) * (1.08 if stress else 1.0)
        voiced_f0[a:b] = f0_here
        if s in _FORM:
            fm = _FORM[s]
            if s in ("AW", "AY", "OY", "EY", "OW"):
                tgt = {"AW": (700, 1200, 2400), "AY": (700, 1300, 2400), "OY": (550, 1000, 2400),
                       "EY": (480, 1900, 2500), "OW": (500, 900, 2400)}[s]
                end = {"AW": (330, 900, 2200), "AY": (400, 2000, 2500), "OY": (400, 2000, 2500),
                       "EY": (300, 2200, 2900), "OW": (330, 850, 2200)}[s]
                for k in range(3):
                    F[k, a:b] = np.linspace(tgt[k], end[k], n)
            else:
                for k in range(3):
                    F[k, a:b] = fm[k]
            if s in VOWELS:
                lvl = 0.0 if stress else -4.0
                amp_v[a:b] = _db(lvl) * _env(n, 0.012, 0.012)
            elif s in ("M", "N", "NG"):
                amp_v[a:b] = _db(-13.0) * _env(n, 0.012, 0.012)
                for k in range(3):
                    F[k, a:b] = (250, 1000, 2500)[k]
            else:
                amp_v[a:b] = _db(-7.0) * _env(n, 0.012, 0.012)
        elif s in ("P", "T", "K", "B", "D", "G"):
            voiced = s in ("B", "D", "G")
            clos = int(n * 0.70)
            band = {"P": (500, 4500), "B": (500, 4000), "T": (3000, 9000), "D": (3000, 8000),
                    "K": (1500, 4500), "G": (1500, 4000)}[s]
            if voiced:
                voiced_f0[a:a + clos] = f0_here
                for k in range(3):
                    F[k, a:a + clos] = (250, 1000, 2400)[k]
                amp_v[a:a + clos] = _db(-29.0)
            burst_n = n - clos
            nz = _noise_band(burst_n, *band, rng)
            g = _db(-12.0 if not voiced else -16.0)
            asp = np.linspace(1.0, 0.25, burst_n) if not voiced else np.linspace(1.0, 0.1, burst_n)
            out[a + clos:b] += nz * g * asp
        elif s in ("S", "SH", "F", "TH", "HH", "Z", "ZH", "V", "DH", "CH", "JH"):
            lo, hi, g = {"S": (4500, 11000, -8.0), "SH": (2200, 7000, -8.0), "F": (1500, 11000, -25.0),
                         "TH": (1500, 11000, -26.0), "HH": (500, 6000, -22.0), "Z": (4500, 11000, -14.0),
                         "ZH": (2200, 7000, -14.0), "V": (1500, 11000, -27.0), "DH": (1500, 11000, -28.0),
                         "CH": (2200, 7000, -9.0), "JH": (2200, 7000, -14.0)}[s]
            start = a
            if s in ("CH", "JH"):
                clos = int(n * 0.35)
                if s == "JH":
                    voiced_f0[a:a + clos] = f0_here
                    for k in range(3):
                        F[k, a:a + clos] = (250, 1000, 2400)[k]
                    amp_v[a:a + clos] = _db(-29.0)
                start = a + clos
            nz = _noise_band(b - start, lo, hi, rng)
            out[start:b] += nz * _db(g) * _env(b - start, 0.010, 0.012)
            if s in ("Z", "ZH", "V", "DH", "JH"):
                voiced_f0[start:b] = f0_here
                for k in range(3):
                    F[k, start:b] = (300, 1400, 2500)[k]
                amp_v[start:b] = _db(-15.0 if s in ("Z", "ZH", "JH") else -22.0)
            if s == "HH":                      # aspiration takes the colour of what follows -> just leave noise
                pass
        t += d

    # ---- 2. voiced source: harmonics shaped by (smoothed) formant tracks
    kern = np.hanning(int(0.018 * SR)); kern /= kern.sum()
    for k in range(3):
        F[k] = np.convolve(F[k], kern, mode="same")
    amp_v = np.convolve(amp_v, kern, mode="same")
    f0c = np.where(voiced_f0 > 0, voiced_f0, f0)
    vib = 1 + 0.012 * np.sin(2 * np.pi * 4.7 * np.arange(N) / SR) + 0.004 * rng.standard_normal(N)
    phase = 2 * np.pi * np.cumsum(f0c * vib) / SR
    voiced = np.zeros(N, np.float32)
    bw = (90.0, 140.0, 220.0)
    for h in range(1, 45):
        fh = h * f0c
        a_h = np.zeros(N, np.float32)
        for k, gk in zip(range(3), (1.0, 0.7, 0.35)):
            a_h += gk * np.exp(-((fh - F[k]) / bw[k]) ** 2)
        a_h += 0.05 / h
        voiced += (a_h * np.sin(h * phase)).astype(np.float32)
    voiced *= amp_v * 0.06
    out += voiced
    out += rng.standard_normal(N).astype(np.float32) * _db(noise_db)
    peak = np.percentile(np.abs(out), 99.9) + 1e-9
    out = (out * (0.7 / peak)).astype(np.float32)
    return out, truth


def speak(text: str, rate: float = 1.0, seed: int = 0):
    items = phonemes.text_to_items(text)
    return synth(items, rate=rate, rng=np.random.default_rng(seed)), items


class FakeEngine:
    """Drop-in for studio.voice.EdgeEngine, driven by the actual words."""
    def synth(self, text: str) -> np.ndarray:
        (a, _), _ = speak(text, rate=1.0, seed=abs(hash(text)) % (2 ** 31))
        return a
