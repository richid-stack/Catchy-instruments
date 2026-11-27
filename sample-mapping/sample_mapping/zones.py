"""Build playable zones (key range, velocity range, root pitch, gain, loop) for one sample pack.

Zones start from the pack's own SFZ mapping, which carries the author's key spreads, velocity
splits and round robins. The audio analysis then checks and improves them: it compares the
declared root pitch against the detected pitch (and uses the detected pitch when the two are
within 50 cents, which puts naturally tuned recordings in equal temperament), trims leading
silence, and finds loops for sustained samples that stop while still audible.

Level matching between keys is deliberately left to the pack authors: a 300 ms RMS reading is
too noisy on decaying instruments (it flagged 10 dB "outliers" in Salamander that are just fast
decays), so correcting from it would add errors rather than remove them.
"""
from __future__ import annotations

import collections
import gzip
import json
import os
import statistics
import sys

import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "analysis"))

from audioinv.link import AudioIndex  # noqa: E402
from audioinv.sfzparse import parse_sfz, read_bank_defines  # noqa: E402

from .looping import find_loop  # noqa: E402

DEFAULTS = {
    "sw_last": None,          # keep only regions for this keyswitch (plus regions with none)
    "refine_tuning": True,    # use the detected pitch as root when it is within 50 cents of the declared one
    "auto_loop": True,        # find loops for samples that end while still audible
    "trim_lead": True,        # skip leading silence
}


def load_records(inventory_dir, pack_id):
    """Load analysis/inventory/<pack>.jsonl.gz into {path: record}."""
    path = os.path.join(inventory_dir, f"{pack_id}.jsonl.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no inventory for {pack_id}; run analysis/inventory.py first")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {r["path"]: r for r in map(json.loads, fh)}


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def _auto_loop(pack_dir, rel, record, f0_pitch):
    """Decode a sample and look for loop points. Returns seconds or None."""
    try:
        with sf.SoundFile(os.path.join(pack_dir, rel)) as f:
            sr = int(f.samplerate)
            data = f.read(min(int(f.frames), 30 * sr), dtype="float32", always_2d=True)
    except Exception:
        return None
    mono = data.mean(axis=1)
    f0 = 440.0 * 2.0 ** ((f0_pitch - 69.0) / 12.0)
    found = find_loop(mono, sr, f0)
    if not found:
        return None
    return [round(found["start"] / sr, 4), round(found["end"] / sr, 4), round(found["xfade"] / sr, 4)]


def build_zones(datasets_root, tier, pack_id, sfz_rel, inventory_dir, options=None):
    """Build zones for one SFZ program of a pack. Returns (zones, report)."""
    opts = {**DEFAULTS, **(options or {})}
    pack_dir = os.path.join(datasets_root, tier, pack_id)
    records = load_records(inventory_dir, pack_id)
    index = AudioIndex(list(records))
    regions, warnings = parse_sfz(os.path.join(pack_dir, sfz_rel), pack_dir, defines=read_bank_defines(pack_dir))

    skipped = collections.Counter()
    zones = []
    tuning = []
    loops = collections.Counter()
    loop_cache = {}
    for r in regions:
        if str(r.get("trigger", "attack")).lower().startswith("release"):
            skipped["release trigger"] += 1
            continue
        if opts["sw_last"] is not None and r.get("sw_last") not in (None, opts["sw_last"]):
            skipped["other keyswitch"] += 1
            continue
        if any(v is not None and v >= 1 for k, v in (r.get("cc") or {}).items() if k.startswith("locc")):
            skipped["controller-gated"] += 1
            continue
        if r["root"] is None:
            skipped["no root key"] += 1
            continue
        rel = index.resolve(r["sample"])
        rec = records.get(rel) if rel else None
        if rec is None or "error" in rec:
            skipped["sample missing or unreadable"] += 1
            continue

        root_pitch = r["root"] - _num(r.get("transpose")) - _num(r.get("tune")) / 100.0
        det = rec.get("pitch_midi") if rec.get("pitched") else None
        if det is not None:
            delta = det - root_pitch
            near_octave = abs(delta - 12 * round(delta / 12))
            tuning.append((delta * 100.0, 0 if abs(delta) <= 0.6 else (1 if near_octave <= 0.6 else 2)))
            if opts["refine_tuning"] and abs(delta) <= 0.5:
                root_pitch = det

        zone = {"s": f"{tier}/{pack_id}/{rel}", "_lvl": rec.get("level_db"), "lk": r["lokey"], "hk": r["hikey"], "lv": max(r["lovel"], 1),
                "hv": min(r["hivel"], 127), "root": round(root_pitch, 3), "g": 0.0, "st": 0.0,
                "_seq": (r["seq_position"], r["seq_length"]), "_rand": (r["lorand"], r["hirand"])}
        if opts["trim_lead"] and rec.get("lead_s", 0.0) > 0.004:
            zone["st"] = round(rec["lead_s"] - 0.002, 4)

        sr = rec["sr"]
        lm = str(r.get("loop_mode", "")).lower()
        if lm in ("loop_continuous", "loop_sustain") and r.get("loop_end") not in (None, ""):
            zone["loop"] = [round(_num(r.get("loop_start")) / sr, 4), round(_num(r.get("loop_end")) / sr, 4), 0.02]
            loops["from sfz"] += 1
        elif opts["auto_loop"] and rec.get("env") == "sustained":
            if rel not in loop_cache:
                loop_cache[rel] = _auto_loop(pack_dir, rel, rec, det if det is not None else root_pitch)
            if loop_cache[rel]:
                zone["loop"] = loop_cache[rel]
                loops["auto found"] += 1
            else:
                loops["auto failed"] += 1
        zones.append(zone)

    # Round robins: zones that share key and velocity range but have a sequence position or random range.
    groups = collections.defaultdict(list)
    for z in zones:
        groups[(z["lk"], z["hk"], z["lv"], z["hv"])].append(z)
    rr_groups = layered = 0
    for members in groups.values():
        if len(members) == 1:
            continue
        seqs = {z["_seq"] for z in members}
        rands = {z["_rand"] for z in members}
        if any(s[1] > 1 for s in seqs) and len(seqs) == len(members):
            for z in members:
                z["rr"] = [z["_seq"][0], z["_seq"][1]]
            rr_groups += 1
        elif len(rands) == len(members):
            for i, z in enumerate(sorted(members, key=lambda z: z["_rand"])):
                z["rr"] = [i + 1, len(members)]
            rr_groups += 1
        else:
            layered += 1
    for z in zones:
        z.pop("_seq", None)
        z.pop("_rand", None)

    # Reference level of each velocity layer (median over its samples, in dB below the loudest layer).
    # The engine uses it to make loudness follow the patch's velocity curve whatever level the
    # layers were recorded at, while the timbre still comes from the layer that is chosen.
    by_layer = collections.defaultdict(list)
    for z in zones:
        if z["_lvl"] is not None:
            by_layer[(z["lv"], z["hv"])].append(z["_lvl"])
    layer_level = {k: statistics.median(v) for k, v in by_layer.items()}
    top = max(layer_level.values()) if layer_level else 0.0
    for z in zones:
        lvl = layer_level.get((z["lv"], z["hv"]))
        z["rl"] = round(lvl - top, 2) if lvl is not None else 0.0
        z.pop("_lvl", None)

    zones.sort(key=lambda z: (z["lv"], z["lk"], z.get("rr", [1])[0]))
    covered = {k for z in zones for k in range(z["lk"], z["hk"] + 1)}
    report = {
        "pack": pack_id, "sfz": sfz_rel, "regions_in_sfz": len(regions), "zones": len(zones),
        "skipped": dict(skipped), "sfz_warnings": warnings[:5],
        "keys": [min(covered), max(covered)] if covered else None,
        "velocity_layers": len({(z["lv"], z["hv"]) for z in zones}),
        "round_robin_groups": rr_groups, "layered_groups_left_stacked": layered,
        "layer_levels_db_below_loudest": {f"{lv}-{hv}": round(v - top, 1) for (lv, hv), v in sorted(layer_level.items())},
        "loops": dict(loops),
    }
    if tuning:
        cents = [c for c, _ in tuning]
        close = [abs(c) for c, kind in tuning if kind == 0]
        report["declared_vs_detected_pitch"] = {
            "compared": len(tuning), "within_60_cents": len(close),
            "octave_off": sum(1 for _, k in tuning if k == 1), "other": sum(1 for _, k in tuning if k == 2),
            "median_abs_cents": round(statistics.median(close), 1) if close else None,
            "p95_abs_cents": round(_percentile(close, 0.95), 1) if close else None,
            "mean_cents": round(statistics.fmean(cents), 1)}
    return zones, report
