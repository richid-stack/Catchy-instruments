"""Cut single notes out of loop stems so they can be mapped like a note-per-file instrument.

Kontakt-style libraries are recorded one note at a time. Producer loop packs rarely ship those,
but a dry guitar stem is a long run of real, played notes. This module finds the attacks in a
stem, keeps the ones that are a single clear pitch, cuts each note out with a fade where the next
note starts, tunes it to equal temperament, and writes it as a file that the pitched automapper
understands (`<family>_<Note><octave>_v<layer>_rr<n>.wav`).

Three checks keep chords, noise and misread pitches out:
  * YIN pitch must be stable across the note's first 150 ms;
  * a harmonic-sum detector, working on the spectrum, must agree with YIN to a quarter tone;
  * most of the spectrum's energy must sit on the harmonics of that pitch (a strummed chord does not).
Nothing here judges how good a note sounds; it only guarantees the note is one pitch, on pitch,
and cleanly cut. The ears decide the rest.
"""
from __future__ import annotations

import collections
import json
import math
import os
import re
import sys
from fractions import Fraction

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as ss
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "analysis"))

from audioinv.features import yin  # noqa: E402

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

DEFAULTS = {
    "fmin": 70.0,               # YIN search range (guitar low E is 82 Hz)
    "fmax": 1500.0,
    "onset_k": 24.0,            # onset threshold above the local median, in MADs of the spectral flux
    "min_gap_s": 0.10,          # a note needs this long before the next attack to be judged
    "midi_range": [38, 90],     # notes outside this range (D2 to F#6 for a guitar) are not trusted
    "max_len_s": 4.0,           # longest note kept
    "min_level_db": -60.0,      # quieter attacks (dBFS, before normalising) are ignored
    "yin_conf": 0.8,
    "pitch_tol_st": 0.5,        # YIN windows must agree to this many semitones
    "hss_tol_st": 0.25,         # and the spectral detector must agree to this
    "min_harmonic_ratio": 0.55,
    "target_attack_db": -22.0,  # every stem's median note is brought to this level
    "layers": 3,
    "max_rr": 6,
    "tune": True,               # correct each note to exact equal temperament
    "fade_in_s": 0.002,
    "fade_cut_s": 0.04,         # fade where the next note cuts this one off
    "fade_end_s": 0.04,         # fade after a note that died away by itself
}


def midi_to_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


# ------------------------------------------------------------------ onsets

def find_onsets(mono, sr, k=4.0):
    """Attack start (sample index) of every note-like event, refined to the start of the rise."""
    _, t, z = ss.stft(mono, sr, nperseg=2048, noverlap=2048 - 256, boundary=None, padded=False)
    mag = np.log1p(np.abs(z) * 50.0)
    ref = ndi.maximum_filter1d(mag, size=3, axis=0)          # SuperFlux: ignore vibrato-sized bin moves
    lag = 2
    flux = np.zeros(mag.shape[1])
    flux[lag:] = np.maximum(mag[:, lag:] - ref[:, :-lag], 0.0).sum(axis=0)
    mad = float(np.median(np.abs(flux - np.median(flux)))) + 1e-9
    thr = ndi.median_filter(flux, size=129, mode="nearest") + k * mad
    # on a very clean stem the MAD is tiny and fluctuations inside a ringing note would pass; an attack
    # must also reach a tenth of the strongest attack nearby (and of a quarter of the typical peak overall)
    near_peak = ndi.maximum_filter1d(flux, size=int(3.0 * sr / 256), mode="nearest")
    thr = np.maximum(thr, 0.1 * np.maximum(near_peak, 0.25 * float(np.percentile(flux, 99.5))))
    peaks, _ = ss.find_peaks(flux, height=thr, distance=8)
    env = ndi.uniform_filter1d(np.abs(mono), size=max(int(0.0015 * sr), 1))
    starts = []
    for p in peaks:
        c = int(t[p] * sr)
        lo, hi = max(c - int(0.035 * sr), 0), min(c + int(0.03 * sr), len(mono))
        if hi - lo < 16:
            continue
        pk = lo + int(np.argmax(env[lo:hi]))
        floor = 0.12 * env[pk]
        i = pk
        while i > lo and env[i] > floor:
            i -= 1
        starts.append(i)
    out = []
    for s in sorted(set(starts)):
        if not out or s - out[-1] > int(0.04 * sr):
            out.append(s)
    return out


# ------------------------------------------------------------------ pitch

def _hss_score(db, freqs, f0, floor):
    """Mean height (dB over the spectrum's median) at the first ten harmonics of f0."""
    vals = []
    for k in range(1, 11):
        f = f0 * k
        if f > freqs[-1] * 0.9:
            break
        lo, hi = np.searchsorted(freqs, [f * 0.985, f * 1.015])
        vals.append(np.clip(db[lo:max(hi, lo + 1)].max() - floor, 0.0, 40.0))
    return float(np.mean(vals)) if vals else 0.0


def analyse_note(mono, sr, start, end, o):
    """Pitch verdict for the note between two attacks. Returns a dict, or {'reject': reason}."""
    gap = end - start
    win = int(2 * sr / o["fmin"] + 1 * sr / o["fmin"])           # YIN needs integration window + max lag
    wins = [int(off * sr) for off in (0.02, 0.035, 0.07, 0.12, 0.18)]
    wins = [w for w in wins if w + win <= gap - int(0.008 * sr)]
    if len(wins) < 2:
        return {"reject": "too short to judge"}
    ests = []
    for w in wins:
        res = yin(mono[start + w:start + w + win], sr, o["fmin"], o["fmax"], 0.15)
        if not res or res[1] < o["yin_conf"]:
            return {"reject": "unstable pitch"}
        ests.append(69.0 + 12.0 * math.log2(res[0] / 440.0))
    if max(ests) - min(ests) > o["pitch_tol_st"]:
        return {"reject": "unstable pitch"}
    midi = float(np.median(ests))
    f_yin = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)

    n = 1 << int(math.log2(max(min(gap - int(0.04 * sr) - int(0.008 * sr), 16384), 2048)))
    seg = mono[start + int(0.03 * sr):start + int(0.03 * sr) + n]
    if len(seg) < n:
        return {"reject": "too short to judge"}
    spec = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    db = 10.0 * np.log10(spec + 1e-18)
    band = (freqs > 60) & (freqs < 6000)
    floor = float(np.median(db[band]))
    cands = {name: f_yin * m for name, m in (("low", 0.5), ("yin", 1.0), ("high", 2.0))}
    scores = {name: _hss_score(db, freqs, f, floor) for name, f in cands.items()}
    best = max(scores, key=scores.get)
    # a wrong octave leaves the right candidate with about twice the score; a sound with few harmonics ties
    if best != "yin" and scores[best] > 1.25 * scores["yin"] + 1.0:
        return {"reject": "octave disagreement"}
    # refine on the spectrum: the strongest bin near the fundamental's harmonics gives the exact pitch
    est = []
    for k in range(1, 7):
        f = f_yin * k
        if f > 5000:
            break
        lo, hi = np.searchsorted(freqs, [f * 0.97, f * 1.03])
        j = lo + int(np.argmax(spec[lo:hi])) if hi > lo else None
        if j is not None and 0 < j < len(spec) - 1:
            a, b, c = np.log(spec[j - 1] + 1e-30), np.log(spec[j] + 1e-30), np.log(spec[j + 1] + 1e-30)
            d = a - 2 * b + c
            off = 0.5 * (a - c) / d if abs(d) > 1e-12 else 0.0
            est.append(((j + max(-1.0, min(1.0, off))) * sr / n) / k)
    f_refined = float(np.median(est)) if est else f_yin
    midi_ref = 69.0 + 12.0 * math.log2(f_refined / 440.0)
    if abs(midi_ref - midi) > o["hss_tol_st"]:
        return {"reject": "detectors disagree"}
    if not o["midi_range"][0] <= midi_ref <= o["midi_range"][1]:
        return {"reject": "out of range"}
    # share of energy on harmonics (a chord puts a lot of it elsewhere)
    mask = np.zeros(len(freqs), bool)
    for k in range(1, 40):
        f = f_refined * k
        if f > 6000:
            break
        mask |= np.abs(freqs - f) <= max(f * 0.025, 2.0 * sr / n)
    ratio = float(spec[band & mask].sum() / max(spec[band].sum(), 1e-18))
    if ratio < o["min_harmonic_ratio"]:
        return {"reject": "not a single pitch"}
    return {"midi": midi_ref, "harmonic_ratio": ratio}


# ------------------------------------------------------------------ one stem

def _attack_db(mono, sr, start, end):
    a, b = start + int(0.01 * sr), min(start + int(0.21 * sr), end)
    seg = mono[a:b]
    return 20.0 * math.log10(max(float(np.sqrt(np.mean(seg * seg))) if len(seg) else 0.0, 1e-9))


def scan_stem(path, options=None):
    """Candidate notes of one stem: metadata only (no audio kept). Returns (notes, rejections)."""
    o = {**DEFAULTS, **(options or {})}
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = x.mean(axis=1)
    onsets = find_onsets(mono, sr, o["onset_k"])
    notes, rej = [], collections.Counter()
    for i, s in enumerate(onsets):
        nxt = onsets[i + 1] if i + 1 < len(onsets) else len(mono)
        cut_by_next = i + 1 < len(onsets)
        gap = nxt - s
        if gap < int(o["min_gap_s"] * sr):
            rej["next note too soon"] += 1
            continue
        lvl = _attack_db(mono, sr, s, nxt)
        if lvl < o["min_level_db"]:
            rej["too quiet"] += 1
            continue
        verdict = analyse_note(mono, sr, s, nxt, o)
        if "reject" in verdict:
            rej[verdict["reject"]] += 1
            continue
        end = min(nxt - int(0.006 * sr), s + int(o["max_len_s"] * sr))
        # a note that dies away by itself ends where it falls 50 dB below its attack
        frame = int(0.01 * sr)
        rms = np.array([np.sqrt(np.mean(mono[j:j + frame] ** 2)) for j in range(s, end - frame, frame)] or [0.0])
        alive = np.where(rms > 10.0 ** ((lvl - 50.0) / 20.0))[0]
        natural = s + (int(alive[-1]) + 1) * frame + int(0.05 * sr) if alive.size else end
        dies = natural < end
        end = min(end, natural)
        notes.append({"stem": path, "start": int(s), "end": int(end), "dies": bool(dies), "sr": int(sr),
                      "cut_by_next": bool(cut_by_next and not dies), "midi": verdict["midi"],
                      "harmonic_ratio": round(verdict["harmonic_ratio"], 3), "attack_db": lvl})
    return notes, dict(rej)


# ------------------------------------------------------------------ grouping and export

def _retune(x, cents):
    """Resample so that a note that is `cents` sharp ends up on pitch (also lengthens it by the same ratio)."""
    ratio = Fraction(2.0 ** (cents / 1200.0)).limit_denominator(1500)
    return ss.resample_poly(x, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)


def plan_groups(notes, options=None):
    """Assign each note a velocity layer and round-robin slot. Mutates and returns the kept notes."""
    o = {**DEFAULTS, **(options or {})}
    by_stem = collections.defaultdict(list)
    for n in notes:
        by_stem[n["stem"]].append(n)
    for ns in by_stem.values():                       # bring every stem to the same median attack level
        med = float(np.median([n["attack_db"] for n in ns]))
        for n in ns:
            n["norm_db"] = n["attack_db"] + (o["target_attack_db"] - med)
    for n in notes:
        n["key"] = int(round(n["midi"]))
        n["cents"] = (n["midi"] - n["key"]) * 100.0
    levels = np.array([n["norm_db"] for n in notes])
    cuts = [float(np.quantile(levels, q / o["layers"])) for q in range(1, o["layers"])] if len(notes) else []
    for n in notes:
        n["layer"] = 1 + sum(n["norm_db"] > c for c in cuts)
    groups = collections.defaultdict(list)
    for n in notes:
        groups[(n["key"], n["layer"])].append(n)
    kept = []
    for members in groups.values():
        members.sort(key=lambda n: (-n["harmonic_ratio"], n["stem"], n["start"]))
        members = members[:o["max_rr"]]
        members.sort(key=lambda n: (n["stem"], n["start"]))
        med = float(np.median([n["norm_db"] for n in members]))
        for rr, n in enumerate(members, start=1):
            n["rr"] = rr
            n["match_db"] = max(-6.0, min(6.0, med - n["norm_db"]))     # round robins of a layer play at one level
            kept.append(n)
    return kept


def _exported_pitch_ok(seg, sr, key, lead, o):
    """Re-measure the finished note the way the pack inventory will: it must be on its key to within 35 cents."""
    mono = seg.mean(axis=1)
    win = int(3 * sr / o["fmin"])
    for off in (0.02, 0.05):
        a = lead + int(off * sr)
        if a + win > len(mono):
            continue
        res = yin(mono[a:a + win], sr, o["fmin"], o["fmax"], 0.15)
        if not res or abs(69.0 + 12.0 * math.log2(res[0] / 440.0) - key) > 0.35:
            return False
    return True


def export_notes(notes, out_dir, family, options=None):
    """Write the planned notes as WAV files into out_dir. Returns the list of written file names."""
    o = {**DEFAULTS, **(options or {})}
    os.makedirs(out_dir, exist_ok=True)
    by_stem = collections.defaultdict(list)
    for n in notes:
        by_stem[n["stem"]].append(n)
    written = []
    for stem, ns in by_stem.items():
        x, sr = sf.read(stem, dtype="float32", always_2d=True)
        for n in ns:
            seg = x[max(n["start"] - int(0.002 * sr), 0):n["end"]].copy()
            lead = n["start"] - max(n["start"] - int(0.002 * sr), 0)
            seg *= 10.0 ** (((n["norm_db"] - n["attack_db"]) + n["match_db"]) / 20.0)   # stem level, then round-robin match
            if o["tune"] and abs(n["cents"]) >= 4.0:
                seg = _retune(seg, n["cents"])
            fi = max(int(o["fade_in_s"] * sr), 1)
            seg[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)[:, None]
            fo = min(int((o["fade_end_s"] if n["dies"] else o["fade_cut_s"]) * sr), len(seg) // 4)
            seg[-fo:] *= (0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, fo, dtype=np.float32)))[:, None]
            if not _exported_pitch_ok(seg, sr, n["key"], lead, o):
                n["dropped"] = True
                continue
            peak = float(np.max(np.abs(seg)))
            if peak > 0.99:
                seg *= 0.99 / peak
            name = f"{family}_{midi_to_name(n['key'])}_v{n['layer']}_rr{n['rr']}.wav"
            sf.write(os.path.join(out_dir, name), seg, sr, subtype="PCM_24")
            n["file"] = name
            n["lead_s"] = round(lead / sr, 4)
            written.append(name)
    return written


def harvest(stems, out_dir, family, options=None):
    """Scan the stems, plan, export. Returns a report dict (also written to out_dir/harvest-report.json)."""
    notes, rejected = [], collections.Counter()
    per_stem = {}
    for path in stems:
        ns, rej = scan_stem(path, options)
        notes += ns
        rejected.update(rej)
        per_stem[os.path.basename(path)] = {"notes": len(ns), "rejected": rej}
    kept = plan_groups(notes, options)
    written = export_notes(kept, out_dir, family, options)
    dropped = sum(1 for n in kept if n.get("dropped"))
    kept = [n for n in kept if not n.get("dropped")]
    keys = sorted({n["key"] for n in kept})
    report = {
        "family": family, "stems": len(stems), "candidates": len(notes), "files_written": len(written),
        "rejected": dict(rejected), "failed_export_check": dropped, "pitches": len(keys),
        "range": [midi_to_name(keys[0]), midi_to_name(keys[-1])] if keys else None,
        "layers_per_pitch": dict(collections.Counter(len({n['layer'] for n in kept if n['key'] == k}) for k in keys)),
        "notes": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in n.items() if k not in ("sr",)} for n in kept],
        "per_stem": per_stem,
    }
    with open(os.path.join(out_dir, "harvest-report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    return report
