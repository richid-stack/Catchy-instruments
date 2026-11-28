"""Offline renderer: plays notes through an instrument patch and returns stereo audio."""
from __future__ import annotations

import os

import numpy as np
import soundfile as sf

from .dsp import bake_loop, cubic_read, one_pole, release_curve
from .midi import Note


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

    def _voice(self, patch, zone, note, release_s=None):
        v = patch.voice
        data, sr_src = self.load_sample(zone["s"])
        sr = self.sr
        ratio = 2.0 ** ((note.key - zone["root"]) / 12.0) * sr_src / sr
        start_src = zone.get("st", 0.0) * sr_src
        oneshot = bool(zone.get("os"))
        rel_n = max(int((v["release_s"] if release_s is None else release_s) * sr), 1)
        hold_n = max(int(note.dur * sr), 1)
        if oneshot:   # plays to the end of the recording whatever the key does
            hold_n = max(int((data.shape[0] - 4 - start_src) / ratio), 1)
            rel_n = 1
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
        if v.get("decay_s"):   # a faster decay than the recording's, as a palm-muted string has
            env *= np.exp(-np.arange(n, dtype=np.float32) / (sr * float(v["decay_s"])))
        if oneshot:
            f = min(int(0.01 * sr), n // 4)
            if f > 1:
                env[n - f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
        elif not looped and avail < total:  # the recording ended before the note did
            f = min(int(0.02 * sr), n // 4)
            env[n - f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
        if looped and v.get("loop_decay_s"):
            beyond = np.maximum(pos - le, 0.0) / sr_src
            env *= np.exp(-beyond / v["loop_decay_s"]).astype(np.float32)

        # Loudness follows the patch's velocity curve: 0 dB at velocity 127, falling by
        # dynamic_range_db towards velocity 1. The layer's own recorded level is taken out first.
        curve = -v["dynamic_range_db"] * (1.0 - note.vel / 127.0) ** v["velocity_curve"]
        db = v["gain_db"] + zone.get("g", 0.0) + curve - zone.get("rl", 0.0)
        db = max(-80.0, min(40.0, db))
        return buf * (env[:, None] * np.float32(10.0 ** (db / 20.0)))

    def _mono_cuts(self, notes, cut_s):
        """Monophonic patches: a later note ends the sounding one. Returns [(Note, release_s or None)]."""
        ordered = sorted(notes, key=lambda n: (n.time, n.key))
        times = sorted({n.time for n in ordered})
        out, cuts = [], 0
        for n in ordered:
            later = [t for t in times if t > n.time + 1e-6]
            if later and later[0] < n.time + n.dur:
                out.append((Note(n.time, later[0] - n.time, n.key, n.vel, getattr(n, "channel", 0)), cut_s))
                cuts += 1
            else:
                out.append((n, None))
        return out, cuts

    @staticmethod
    def _apply_chokes(voices, choke_s, sr):
        """A voice whose zone has off_by=g is cut when a later voice of group g starts."""
        choked = 0
        live = []
        for v in voices:   # voices are in start order
            g = v["grp"]
            if g is not None:
                for o in live:
                    if o["off_by"] == g and o["end"] > v["start"] and o["start"] < v["start"]:
                        cut = v["start"] - o["start"]
                        fade = min(int(choke_s * sr), o["buf"].shape[0] - cut)
                        o["buf"] = o["buf"][: cut + fade].copy()
                        if fade > 1:
                            o["buf"][cut:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
                        o["end"] = o["start"] + o["buf"].shape[0]
                        choked += 1
            live = [o for o in live if o["end"] > v["start"]]
            live.append(v)
        return choked

    def _render_stack(self, p, notes, tail_s, peak_limit):
        """A stack patch plays the same notes through several other patches, each transposed and gained."""
        sr, v = self.sr, p.voice
        parts, played, skipped = [], 0, 0
        for layer in p.stack:
            shift = int(layer.get("transpose", 0))
            moved = [Note(n.time, n.dur, n.key + shift, n.vel, getattr(n, "channel", 0))
                     for n in notes if 0 <= n.key + shift <= 127]
            audio, st = self.render(moved, patch=layer["patch"], tail_s=tail_s, peak_limit=1e9)
            parts.append(audio * np.float32(10.0 ** (float(layer.get("gain_db", 0.0)) / 20.0)))
            played += st["notes_played"]
            skipped += st["notes_without_sample"]
        out = np.zeros((max(a.shape[0] for a in parts), 2), dtype=np.float32)
        for a in parts:
            out[: a.shape[0]] += a
        out *= np.float32(10.0 ** (float(v.get("gain_db", 0.0)) / 20.0))
        if v.get("highpass_hz"):
            out = one_pole(out, v["highpass_hz"], sr, "hp")
        if v.get("lowpass_hz"):
            out = one_pole(out, v["lowpass_hz"], sr, "lp")
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        scale = 1.0
        if peak > peak_limit:
            scale = peak_limit / peak
            out *= np.float32(scale)
        return out, {"patch": p.id, "notes_played": played, "notes_without_sample": skipped, "mono_cuts": 0,
                     "choked": 0, "peak": round(peak * scale, 4), "scale": round(scale, 4),
                     "seconds": round(out.shape[0] / sr, 2), "layers": len(parts)}

    def render(self, notes, patch=None, tail_s=1.0, peak_limit=0.95):
        """Render notes (objects with time, dur, key, vel) to a (frames, 2) float32 array.

        Returns (audio, stats). If the mix would exceed peak_limit it is scaled down as a whole
        (never clipped); stats['scale'] records the factor.

        Patch voice options used here: mono (a new note cuts the last one over mono_release_s),
        choke_s (fade of a voice cut by a choke group). One-shot zones ignore note length.
        """
        p = self.inst.patches[patch or self.inst.default_patch]
        if p.stack:
            return self._render_stack(p, notes, tail_s, peak_limit)
        p.reset_round_robin()
        sr = self.sr
        v = p.voice
        if v.get("mono"):
            plan, mono_cuts = self._mono_cuts(notes, float(v.get("mono_release_s", 0.04)))
        else:
            plan, mono_cuts = [(n, None) for n in sorted(notes, key=lambda n: (n.time, n.key))], 0
        voices, played, skipped = [], 0, 0
        for note, rel in plan:
            zones = p.pick(note.key, note.vel)
            if not zones:
                skipped += 1
                continue
            start = int(round(note.time * sr))
            for zone in zones:
                buf = self._voice(p, zone, note, release_s=rel)
                if buf is None:
                    continue
                voices.append({"start": start, "buf": buf, "end": start + buf.shape[0],
                               "grp": zone.get("grp"), "off_by": zone.get("off_by")})
            played += 1
        voices.sort(key=lambda o: o["start"])
        choked = self._apply_chokes(voices, float(v.get("choke_s", 0.02)), sr) if any(
            o["grp"] is not None for o in voices) else 0
        nominal = max((n.time + n.dur for n in notes), default=0.0) + v["release_s"] + tail_s
        frames = max([int(nominal * sr) + 1] + [o["end"] + int(0.05 * sr) for o in voices])
        out = np.zeros((frames, 2), dtype=np.float32)
        for o in voices:
            out[o["start"]:o["end"]] += o["buf"]
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
                     "mono_cuts": mono_cuts, "choked": choked,
                     "peak": round(peak * scale, 4), "scale": round(scale, 4), "seconds": round(out.shape[0] / sr, 2)}


def write_wav(path, audio, sr, subtype="PCM_16"):
    sf.write(path, audio, sr, subtype=subtype)
