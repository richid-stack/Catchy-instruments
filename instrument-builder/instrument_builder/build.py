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


def measure_loudness_db(manifest, patch_id, datasets_root, keys=range(48, 73, 4), vel=100):
    """Render mid-keyboard notes at forte and return their mean-square level in dBFS.

    This is a rendered measurement, not an estimate from file levels, so it includes the layer
    chosen for the velocity, pitch shifting, loops and the patch's own filters.
    """
    engine = Engine(Instrument(manifest), datasets_root, sr=44100)
    notes = [Note(time=i * 2.2, dur=0.9, key=k, vel=vel) for i, k in enumerate(keys)]
    audio, _ = engine.render(notes, patch=patch_id, tail_s=0.5, peak_limit=1e9)
    mono = audio.mean(axis=1)
    sq = np.concatenate([mono[int(n.time * 44100):int((n.time + 1.5) * 44100)] ** 2 for n in notes])
    return float(10.0 * np.log10(max(float(np.mean(sq)), 1e-12)))


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
    target = spec.get("target_rms_db", -24.0)
    for b in built:
        trim = 0.0
        if spec.get("match_loudness", True):
            measured = measure_loudness_db(trial, b["id"], datasets_root)
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
