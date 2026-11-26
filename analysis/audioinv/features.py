"""Per-file audio analysis: levels, envelope class, pitch (YIN), spectral centroid, stereo checks.

Only the first few seconds and the last 100 ms of each file are decoded, which keeps a full
inventory of tens of thousands of samples fast.
"""
from __future__ import annotations

import math

import numpy as np
import soundfile as sf

FMIN = 27.5
FMAX = 4200.0
WINDOW_S = 4.0


def yin(seg, sr, fmin=FMIN, fmax=FMAX, thr=0.15):
    """YIN pitch estimate for one segment. Returns (f0_hz, confidence) or None.

    The segment must be at least (integration window + sr/fmin) samples long.
    """
    tau_max = int(sr / fmin) + 1
    tau_min = max(2, int(sr / fmax))
    length = len(seg)
    w = length - tau_max
    if w < 256:
        return None
    x = np.asarray(seg, dtype=np.float64)
    n = 1 << int(length + w).bit_length()
    fa = np.fft.rfft(x[:w], n)
    fb = np.fft.rfft(x, n)
    r = np.fft.irfft(np.conj(fa) * fb, n)[: tau_max + 1]
    cs = np.concatenate(([0.0], np.cumsum(x * x)))
    t = np.arange(tau_max + 1)
    d = (cs[w] - cs[0]) + (cs[t + w] - cs[t]) - 2.0 * r
    d[0] = 0.0
    cum = np.cumsum(d[1:])
    dn = np.ones(tau_max + 1)
    dn[1:] = d[1:] * t[1:] / np.maximum(cum, 1e-12)

    below = np.where(dn[tau_min:] < thr)[0]
    if below.size:
        cand = tau_min + int(below[0])
        while cand + 1 <= tau_max and dn[cand + 1] < dn[cand]:
            cand += 1
    else:
        cand = tau_min + int(np.argmin(dn[tau_min:]))
    tau = float(cand)
    if 1 <= cand < tau_max:
        a, b, c = dn[cand - 1], dn[cand], dn[cand + 1]
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-12:
            tau = cand + max(-1.0, min(1.0, 0.5 * (a - c) / denom))
    if tau <= 0:
        return None
    return sr / tau, float(1.0 - dn[cand])


def detect_pitch(mono, sr, start):
    """Estimate pitch from up to three windows after the attack. Returns a small dict."""
    tau_max = int(sr / FMIN) + 1
    w = max(2048, int(1.5 * tau_max))
    length = w + tau_max
    starts = []
    for off in (0.03, 0.12, 0.30):
        s = max(min(start + int(off * sr), len(mono) - length), 0)
        if s not in starts:
            starts.append(s)
    valid = []
    for s in starts:
        seg = mono[s:s + length]
        if len(seg) < length:
            seg = np.pad(seg, (0, length - len(seg)))
        if float(np.sqrt(np.mean(seg * seg))) < 1e-4:
            continue
        res = yin(seg, sr)
        if res and res[0] > 0:
            valid.append((69.0 + 12.0 * math.log2(res[0] / 440.0), res[1]))
    if not valid:
        return {"pitched": False}
    good = [m for m, c in valid if c >= 0.85]
    conf = float(np.median([c for _, c in valid]))
    out = {"pitched": False, "pitch_conf": round(conf, 2)}
    if len(good) >= min(2, len(starts)) and (max(good) - min(good)) <= 0.6:
        out["pitched"] = True
        out["pitch_midi"] = round(float(np.median(good)), 2)
    return out


def _db(value):
    return 20.0 * math.log10(max(float(value), 1e-9))


def _true_tail(f, sr, tail_n, cap_s=600):
    """Last tail_n frames of an open SoundFile, found by decoding from the start.

    Capped so a hostile file claiming a huge length cannot keep the worker busy for long.
    """
    f.seek(0)
    keep = None
    done = 0
    while done < cap_s * sr:
        chunk = f.read(sr, dtype="float32", always_2d=True)
        if chunk.shape[0] == 0:
            break
        done += chunk.shape[0]
        keep = chunk if keep is None else np.concatenate([keep, chunk])[-tail_n:]
    return keep


def _finite(record):
    """Replace NaN or infinite numbers with None so the output is always valid JSON."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in record.items()}


def analyze_file(path, window_s=WINDOW_S):
    """Decode part of one audio file and return a feature dict (or {'error': ...})."""
    try:
        with sf.SoundFile(path) as f:
            sr = int(f.samplerate)
            ch = int(f.channels)
            frames = int(f.frames)
            info = {"format": f.format, "subtype": f.subtype, "sr": sr, "channels": ch, "frames": frames}
            if frames <= 0:
                return {**info, "error": "empty file"}
            n = min(frames, int(window_s * sr))
            x = f.read(n, dtype="float32", always_2d=True)
            tail_n = max(int(0.1 * sr), 1)
            if frames > n + tail_n:
                f.seek(frames - tail_n)
                tail = f.read(tail_n, dtype="float32", always_2d=True)
                if tail.shape[0] == 0:
                    # Some MP3 headers overstate the length, so the seek lands past the real
                    # end. Decode forward to find the true last 100 ms instead.
                    tail = _true_tail(f, sr, tail_n)
            else:
                tail = x[-tail_n:]
            if tail is None or tail.shape[0] == 0:
                tail = x[-tail_n:]
    except Exception as exc:  # untrusted input: never let one bad file stop the run
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}

    if x.shape[0] == 0:
        return {**info, "error": "no samples decoded"}
    out = dict(info)
    out["dur_s"] = round(frames / sr, 3)
    mono = x.mean(axis=1)
    peak = float(np.max(np.abs(x)))
    if peak < 1e-9:
        out["silent"] = True
        return out
    out["peak_db"] = round(_db(peak), 1)

    # Leading silence and attack.
    thr = max(peak * 0.01, 1e-4)
    above = np.nonzero(np.abs(mono) > thr)[0]
    onset = int(above[0]) if above.size else 0
    out["lead_s"] = round(onset / sr, 3)
    seg300 = mono[onset:onset + int(0.3 * sr)]
    out["level_db"] = round(_db(np.sqrt(np.mean(seg300 * seg300))), 1)

    # Envelope in 10 ms frames.
    hop = max(sr // 100, 1)
    nfr = len(mono) // hop
    if nfr >= 2:
        env = np.sqrt(np.mean(mono[: nfr * hop].reshape(nfr, hop) ** 2, axis=1))
    else:
        env = np.array([float(np.sqrt(np.mean(mono * mono)))])
    pk = int(np.argmax(env))
    pk_db = _db(env[pk])
    rel = 20.0 * np.log10(np.maximum(env, 1e-9)) - pk_db
    out["attack_s"] = round(max(0.0, (pk * hop - onset) / sr), 3)

    def fall(db):
        idx = np.nonzero(rel[pk:] < -db)[0]
        return round(float(idx[0]) * hop / sr, 3) if idx.size else None

    t20 = fall(20)
    out["t20_s"] = t20
    out["t40_s"] = fall(40)
    tail_rms = float(np.sqrt(np.mean(tail.mean(axis=1) ** 2)))
    out["tail_db"] = round(_db(tail_rms) - pk_db, 1)
    # True when the file stops while still audible (more than -40 dB below its peak), so a
    # fade-out or loop is needed to avoid a click. It is a to-do marker, not a defect.
    out["ends_cut"] = bool(out["tail_db"] > -40.0)
    if t20 is None:
        # No 20 dB fall inside the analysed window. It is only truly sustained if it is still
        # audible at the end of the file; otherwise it is a slow decay longer than the window.
        out["env"] = "sustained" if out["ends_cut"] else "long-decay"
    elif t20 <= 0.35:
        out["env"] = "percussive"
    elif t20 <= 2.5:
        out["env"] = "decaying"
    else:
        out["env"] = "long-decay"

    # Spectral centroid around the loudest point.
    p0 = pk * hop
    seg = mono[p0:p0 + 8192]
    if len(seg) < 8192:
        seg = np.pad(seg, (0, 8192 - len(seg)))
    mag = np.abs(np.fft.rfft(seg * np.hanning(8192)))
    if mag.sum() > 0:
        freqs = np.fft.rfftfreq(8192, 1.0 / sr)
        out["centroid_hz"] = int(round(float((freqs * mag).sum() / mag.sum())))

    if ch >= 2:
        l, r = x[:, 0], x[:, 1]
        denom = float(np.sqrt(np.mean(l * l))) + 1e-12
        out["dual_mono"] = bool(float(np.sqrt(np.mean((l - r) ** 2))) / denom < 1e-3)
    out["clipped"] = int(np.sum(np.abs(x) >= 0.9999))

    out.update(detect_pitch(mono, sr, p0))
    return _finite(out)
