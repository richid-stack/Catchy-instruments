"""Small DSP helpers for the reference sampler: interpolation, loop baking, fades, tone filters."""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def cubic_read(x, pos):
    """Read x (frames, channels) at fractional frame positions with Catmull-Rom interpolation.

    Positions are clamped to the signal, so callers should keep them inside it.
    """
    n = x.shape[0]
    i = np.floor(pos).astype(np.int64)
    f = (pos - i).astype(np.float32)[:, None]
    p0 = x[np.clip(i - 1, 0, n - 1)]
    p1 = x[np.clip(i, 0, n - 1)]
    p2 = x[np.clip(i + 1, 0, n - 1)]
    p3 = x[np.clip(i + 2, 0, n - 1)]
    return p1 + 0.5 * f * (p2 - p0 + f * (2 * p0 - 5 * p1 + 4 * p2 - p3 + f * (3 * (p1 - p2) + p3 - p0)))


def bake_loop(x, loop_start, loop_end, xfade, need):
    """Return x extended by looping [loop_start, loop_end) until it holds at least `need` frames.

    The end of each pass is cross-faded into the audio just before the loop start, so the wrap
    from the last frame to the first is continuous. Needs loop_start >= xfade.
    """
    n = x.shape[0]
    loop_end = min(int(loop_end), n)
    loop_start = int(loop_start)
    if loop_end - loop_start < 8 or need <= loop_end:
        return x
    xf = int(min(xfade, loop_start, (loop_end - loop_start) // 2))
    body = x[loop_start:loop_end].copy()
    if xf > 0:
        t = np.linspace(0.0, 1.0, xf, endpoint=False, dtype=np.float32)[:, None]
        body[-xf:] = x[loop_end - xf:loop_end] * (1.0 - t) + x[loop_start - xf:loop_start] * t
    reps = int(np.ceil((need - loop_start) / body.shape[0])) + 1
    return np.concatenate([x[:loop_start]] + [body] * reps, axis=0)


def one_pole(x, cutoff_hz, sr, kind):
    """One-pole low-pass ('lp') or high-pass ('hp') over a (frames, channels) array."""
    a = float(np.exp(-2.0 * np.pi * cutoff_hz / sr))
    lp = lfilter([1.0 - a], [1.0, -a], x, axis=0).astype(np.float32)
    return lp if kind == "lp" else (x - lp).astype(np.float32)


def release_curve(n):
    """Raised-cosine fade from 1 to 0 over n frames."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    return (0.5 * (1.0 + np.cos(np.pi * np.arange(n) / n))).astype(np.float32)
