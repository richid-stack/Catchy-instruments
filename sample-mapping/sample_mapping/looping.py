"""Find loop points for sustained tones so held notes do not end when the recording does."""
from __future__ import annotations

import math

import numpy as np


def _upward_zero_crossings(x, lo, hi):
    lo = max(lo, 1)
    hi = min(hi, len(x) - 1)
    if hi <= lo:
        return np.zeros(0, dtype=np.int64)
    seg = x[lo - 1:hi + 1]
    idx = np.nonzero((seg[:-1] <= 0.0) & (seg[1:] > 0.0))[0] + lo
    return idx.astype(np.int64)


def _corr(a, b):
    na = float(np.sqrt(np.dot(a, a)))
    nb = float(np.sqrt(np.dot(b, b)))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _best_for_length(mono, sr, f0_hz, length_s, tail_guard_s, max_level_step_db):
    n = len(mono)
    period = sr / f0_hz
    xf = int(min(max(0.03 * sr, 2 * period), 0.1 * sr))
    end_limit = n - int(tail_guard_s * sr)
    k = max(1, int(round(length_s * f0_hz)))
    target = k * period
    start_min = max(int(0.15 * sr), xf)
    ends = _upward_zero_crossings(mono, end_limit - int(2 * period), end_limit)
    best = None
    for e in ends[::-1][:6]:
        s0 = e - target
        starts = _upward_zero_crossings(mono, int(s0 - period), int(s0 + period))
        starts = starts[starts >= start_min]
        w = min(xf, int(e))
        tail_e = mono[e - w:e]
        rms_e = float(np.sqrt(np.mean(tail_e * tail_e))) + 1e-12
        for s in starts:
            if s - w < 0 or s >= e - 8:
                continue
            head = mono[s - w:s]
            score = _corr(head, tail_e)
            ratio = (float(np.sqrt(np.mean(head * head))) + 1e-12) / rms_e
            if abs(20.0 * math.log10(ratio)) > max_level_step_db:
                continue
            total = score - 0.5 * abs(math.log(ratio))
            if best is None or total > best[0]:
                best = (total, int(s), int(e), score)
    if best is None:
        return None
    return {"start": best[1], "end": best[2], "xfade": xf, "score": round(best[3], 3)}


def find_loop(mono, sr, f0_hz, lengths_s=(1.0, 0.7, 0.5, 0.35, 0.25), tail_guard_s=0.05,
              min_score=0.85, max_level_step_db=2.5):
    """Pick loop start/end frames near the end of a held tone.

    The loop spans a whole number of pitch periods. Both points sit on upward zero crossings,
    and the waveform leading into the end point must resemble the waveform leading into the
    start point, so the wrap does not click. The level at the two points must also agree to
    within max_level_step_db, otherwise a decaying tone would pulse once per loop. Longer loops
    are tried first and shorter ones are used when the tone decays too fast for a long loop.
    Returns {'start', 'end', 'xfade', 'score'} in frames, or None when no good loop exists.
    """
    n = len(mono)
    if not f0_hz or f0_hz <= 0 or n < int(0.6 * sr):
        return None
    for length_s in lengths_s:
        if length_s > 0.45 * n / sr:
            continue
        found = _best_for_length(mono, sr, f0_hz, length_s, tail_guard_s, max_level_step_db)
        if found and found["score"] >= min_score:
            return found
    return None
