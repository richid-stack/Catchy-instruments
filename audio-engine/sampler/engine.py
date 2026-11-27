"""Offline renderer: plays notes through an instrument patch and returns stereo audio."""
from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from .dsp import bake_loop, cubic_read, one_pole, release_curve


class Engine:
    def __init__(self, instrument, datasets_root, sr=44100, cache_mb=1500):
        self.inst = instrument
        self.root = os.path.realpath(datasets_root)
        self.sr = int(sr)
        self.cache = {}
        self.cache_bytes = 0
        self.cache_limit = cache_mb * 1_000_000

    def load_sample(self, rel):
        """Decode a sample (cached). Returns (float32 frames x channels, source sample rate)."""
        hit = self.cache.get(rel)
        if hit is not None:
            return hit
        path = os.path.realpath(os.path.join(self.root, rel))
        if path != self.root and not path.startswith(self.root + os.sep):
            raise ValueError(f"sample path leaves the datasets folder: {rel}")
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if data.shape[1] > 2:
            data = data[:, :2]
        if self.cache_bytes + data.nbytes > self.cache_limit:
            self.cache.clear()
            self.cache_bytes = 0
        self.cache[rel] = (data, int(sr))
        self.cache_bytes += data.nbytes
        return self.cache[rel]

    def _voice(self, patch, zone, note):
        v = patch.voice
        data, sr_src = self.load_sample(zone["s"])
        sr = self.sr
        ratio = 2.0 ** ((note.key - zone["root"]) / 12.0) * sr_src / sr
        start_src = zone.get("st", 0.0) * sr_src
        rel_n = max(int(v["release_s"] * sr), 1)
        hold_n = max(int(note.dur * sr), 1)
        total = hold_n + rel_n
        loop = zone.get("loop")
        looped = False
        if loop:
            ls, le, xf = (int(loop[0] * sr_src), int(loop[1] * sr_src), int(loop[2] * sr_src))
            need = int(start_src + total * ratio) + 8
            if le - ls >= 8 and need > le:
                data = bake_loop(data, ls, le, xf, need)
                looped = True
        avail = int((data.shape[0] - 4 - start_src) / ratio)
        n = min(total, avail)
        if n <= 8:
            return None
        pos = start_src + np.arange(n, dtype=np.float64) * ratio
        buf = cubic_read(data, pos)
        if buf.shape[1] == 1:
            buf = np.repeat(buf, 2, axis=1)

        env = np.ones(n, dtype=np.float32)
        a = min(int(v["attack_s"] * sr), n)
        if a > 1:
            env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
        if n > hold_n:
            env[hold_n:] = release_curve(rel_n)[: n - hold_n]
        if not looped and avail < total:  # the recording ended before the note did
            f = min(int(0.02 * sr), n // 4)
            env[n - f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
        if looped and v.get("loop_decay_s"):
            beyond = np.maximum(pos - le, 0.0) / sr_src
            env *= np.exp(-beyond / v["loop_decay_s"]).astype(np.float32)

        # Loudness follows the patch's velocity curve: 0 dB at velocity 127, falling by
        # dynamic_range_db towards velocity 1. The layer's own recorded level is taken out first.
        curve = -v["dynamic_range_db"] * (1.0 - note.vel / 127.0) ** v["velocity_curve"]
        db = v["gain_db"] + zone.get("g", 0.0) + curve - zone.get("rl", 0.0)
        db = max(-60.0, min(24.0, db))
        return buf * (env[:, None] * np.float32(10.0 ** (db / 20.0)))

    def render(self, notes, patch=None, tail_s=1.0, peak_limit=0.95):
        """Render notes (objects with time, dur, key, vel) to a (frames, 2) float32 array.

        Returns (audio, stats). If the mix would exceed peak_limit it is scaled down as a whole
        (never clipped); stats['scale'] records the factor.
        """
        p = self.inst.patches[patch or self.inst.default_patch]
        p.reset_round_robin()
        sr = self.sr
        end = max((n.time + n.dur for n in notes), default=0.0) + p.voice["release_s"] + tail_s
        out = np.zeros((int(end * sr) + 1, 2), dtype=np.float32)
        played = skipped = 0
        for note in sorted(notes, key=lambda n: (n.time, n.key)):
            zones = p.pick(note.key, note.vel)
            if not zones:
                skipped += 1
                continue
            start = int(round(note.time * sr))
            for zone in zones:
                buf = self._voice(p, zone, note)
                if buf is None:
                    continue
                stop = min(start + buf.shape[0], out.shape[0])
                out[start:stop] += buf[: stop - start]
            played += 1
        v = p.voice
        if v.get("highpass_hz"):
            out = one_pole(out, v["highpass_hz"], sr, "hp")
        if v.get("lowpass_hz"):
            out = one_pole(out, v["lowpass_hz"], sr, "lp")
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        scale = 1.0
        if peak > peak_limit:
            scale = peak_limit / peak
            out *= np.float32(scale)
        return out, {"patch": p.id, "notes_played": played, "notes_without_sample": skipped,
                     "peak": round(peak * scale, 4), "scale": round(scale, 4), "seconds": round(out.shape[0] / sr, 2)}


def write_wav(path, audio, sr, subtype="PCM_16"):
    sf.write(path, audio, sr, subtype=subtype)
