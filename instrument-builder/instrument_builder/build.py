"""Turn an instrument spec into a playable instrument manifest.

A spec lists patches. Each patch names a sample pack and one of its SFZ programs, plus how the
voice should behave (attack, release, tone). The builder maps the pack into zones, matches the
loudness of the patches to one target level, and records credits for every pack used.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "sample-mapping"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "audio-engine"))

from sample_mapping.zones import build_zones  # noqa: E402
from sampler.engine import Engine  # noqa: E402
from sampler.instrument import Instrument  # noqa: E402
from sampler.midi import Note  # noqa: E402

FORMAT = "catchy-instrument/1"
VOICE_DEFAULTS = {"attack_s": 0.002, "release_s": 0.3, "gain_db": 0.0, "dynamic_range_db": 30.0,
                  "velocity_curve": 1.0, "lowpass_hz": None, "highpass_hz": None, "loop_decay_s": None}
TIERS = ("clean", "research-private")


class BuildError(Exception):
    pass


def read_pack_table(datasets_root):
    """Read datasets/manifest.psv into {pack_id: {license, attribution, tier}}."""
    table = {}
    path = os.path.join(datasets_root, "manifest.psv")
    if not os.path.exists(path):
        return table
    with open(path, encoding="utf-8") as fh:
        for line in fh.read().splitlines()[1:]:
            parts = line.split("|")
            if len(parts) >= 5:
                table[parts[0]] = {"tier": parts[2], "license": parts[3], "attribution": parts[4]}
    return table


def measure_loudness_db(manifest, patch_id, datasets_root, keys=range(48, 73, 4), vel=100, dur=0.9):
    """Render mid-keyboard notes at forte and return their mean-square level in dBFS.

    This is a rendered measurement, not an estimate from file levels, so it includes the layer
    chosen for the velocity, pitch shifting, loops and the patch's own filters.
    """
    engine = Engine(Instrument(manifest), datasets_root, sr=44100)
    notes = [Note(time=i * 2.2, dur=dur, key=k, vel=vel) for i, k in enumerate(keys)]
    audio, _ = engine.render(notes, patch=patch_id, tail_s=0.5, peak_limit=1e9)
    mono = audio.mean(axis=1)
    sq = np.concatenate([mono[int(n.time * 44100):int((n.time + 1.5) * 44100)] ** 2 for n in notes])
    return float(10.0 * np.log10(max(float(np.mean(sq)), 1e-12)))


def _key_levels_db(manifest, patch_id, datasets_root, keys, vel, window_s, repeats):
    """Rendered level (dBFS, mean-square over the first window_s) of each key; round robins are averaged."""
    sr = 44100
    inst = Instrument(manifest)
    notes, slots = [], []
    for k in keys:
        for _ in range(repeats):
            slots.append((k, len(notes) * 2.2))
            notes.append(Note(time=len(notes) * 2.2, dur=1.0, key=k, vel=vel))
    audio, _ = Engine(inst, datasets_root, sr=sr).render(notes, patch=patch_id, tail_s=0.3, peak_limit=1e9)
    mono = audio.mean(axis=1)
    power = {}
    for k, t in slots:
        seg = mono[int((t + 0.02) * sr):int((t + 0.02 + window_s) * sr)]
        power.setdefault(k, []).append(float(np.mean(seg ** 2)))
    return {k: float(10.0 * np.log10(max(float(np.mean(v)), 1e-12))) for k, v in power.items()}


def balance_keys(manifest, patch, datasets_root, keys, max_db=9.0, strength=0.85, vel=100, window_s=0.5):
    """Even out recording-level differences between keys by adjusting zone gains.

    Each key is rendered at the same velocity and its level is compared with the median level of
    the range; every zone is moved towards that median by `strength` of the difference, at most
    max_db. Returns {'before_spread_db', 'after_spread_db', 'max_correction_db'} measured from
    renders, so the effect is verified rather than assumed.
    """
    ks = list(keys)
    repeats = max([z["rr"][1] for z in patch["zones"] if "rr" in z] or [1])
    before = _key_levels_db(manifest, patch["id"], datasets_root, ks, vel, window_s, repeats)
    ref = float(np.median(list(before.values())))
    corr_of = {k: max(-max_db, min(max_db, strength * (ref - lvl))) for k, lvl in before.items()}
    lo, hi = min(ks), max(ks)
    applied = 0.0
    for z in patch["zones"]:
        k = int(round(max(lo, min(hi, z["root"]))))
        c = corr_of.get(k, 0.0)
        z["g"] = round(z.get("g", 0.0) + c, 2)
        applied = max(applied, abs(c))
    after = _key_levels_db(manifest, patch["id"], datasets_root, ks, vel, window_s, repeats)
    spread = lambda d: round(max(d.values()) - min(d.values()), 1)  # noqa: E731
    return {"before_spread_db": spread(before), "after_spread_db": spread(after), "max_correction_db": round(applied, 1)}


def build_instrument(spec, datasets_root, inventory_dir, allow_private=False):
    """Return (manifest, report). Raises BuildError for a bad spec or a blocked tier."""
    patches_in = spec.get("patches") or []
    if not patches_in:
        raise BuildError("spec has no patches")
    packs = read_pack_table(datasets_root)
    built, report, used = [], {"patches": {}}, {}
    for p in patches_in:
        src = p.get("source") or {}
        tier = src.get("tier", "clean")
        if tier not in TIERS:
            raise BuildError(f"patch {p.get('id')}: unknown tier {tier!r}")
        if tier == "research-private" and not allow_private:
            raise BuildError(f"patch {p.get('id')} uses research-private samples; pass --allow-private "
                             "for a private build that stays out of git")
        if not all(k in src for k in ("pack", "sfz")):
            raise BuildError(f"patch {p.get('id')}: source needs 'pack' and 'sfz'")
        zones, rep = build_zones(datasets_root, tier, src["pack"], src["sfz"], inventory_dir, src.get("options"))
        if not zones:
            raise BuildError(f"patch {p.get('id')}: no playable zones from {src['pack']}")
        used[src["pack"]] = {"pack": src["pack"], "tier": tier, **{k: v for k, v in packs.get(src["pack"], {}).items() if k != "tier"}}
        voice = {**VOICE_DEFAULTS, **(p.get("voice") or {})}
        built.append({"id": p["id"], "name": p.get("name", p["id"]), "note": p.get("note", ""),
                      "voice": voice, "zones": zones})
        report["patches"][p["id"]] = rep

    trial = {"format": FORMAT, "name": spec["name"], "patches": built}
    by_id = {p["id"]: p for p in patches_in}
    target = spec.get("target_rms_db", -24.0)
    for b in built:
        trim = 0.0
        bal = {**(spec.get("balance") or {}), **(by_id[b["id"]].get("balance") or {})}
        if bal.get("keys"):
            lo, hi = bal["keys"][0], bal["keys"][1]
            report["patches"][b["id"]]["key_balance"] = balance_keys(
                trial, b, datasets_root, range(int(lo), int(hi) + 1), max_db=bal.get("max_db", 9.0),
                strength=bal.get("strength", 0.85))
        if spec.get("match_loudness", True):
            m = {**(spec.get("measure") or {}), **(by_id[b["id"]].get("measure") or {})}
            kw = {}
            if "keys" in m:
                lo, hi, step = (list(m["keys"]) + [4])[:3] if len(m["keys"]) == 2 else m["keys"]
                kw["keys"] = range(int(lo), int(hi) + 1, int(step))
            for name in ("vel", "dur"):
                if name in m:
                    kw[name] = m[name]
            measured = measure_loudness_db(trial, b["id"], datasets_root, **kw)
            trim = max(-12.0, min(12.0, target - measured))
            report["patches"][b["id"]]["measured_rms_db_before_trim"] = round(measured, 2)
        b["voice"]["gain_db"] = round(b["voice"]["gain_db"] + trim, 2)
        report["patches"][b["id"]]["auto_gain_db"] = round(trim, 2)

    manifest = {
        "format": FORMAT, "name": spec["name"], "genre": spec.get("genre"), "type": spec.get("type"),
        "description": spec.get("description", ""), "default_patch": spec.get("default_patch", built[0]["id"]),
        "distributable": all(u["tier"] == "clean" for u in used.values()),
        "credits": sorted(used.values(), key=lambda u: u["pack"]), "patches": built,
    }
    return manifest, report
