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
import re
import statistics
import sys

import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "analysis"))

from audioinv.link import AudioIndex  # noqa: E402
from audioinv.sfzparse import cc_active, parse_sfz, read_bank_defines  # noqa: E402

from .looping import find_loop  # noqa: E402

DEFAULTS = {
    "sw_last": None,          # keep only regions for this keyswitch (plus regions with none)
    "cc": None,               # controller values to play the SFZ with, {cc_number: value}; the pack's set_cc defaults otherwise
    "include": None,          # regex: keep only samples whose pack-relative path matches
    "exclude": None,          # regex: drop samples whose pack-relative path matches
    "pitch_convention": "sounding",   # "sounding": a key plays its own pitch; "pack": keep the author's octave
    "refine_tuning": True,    # use the detected pitch as root when it is within 50 cents of the declared one
    "auto_loop": True,        # find loops for samples that end while still audible
    "trim_lead": True,        # skip leading silence
    "sfz_volume": True,       # apply the SFZ author's per-region volume (dB) as zone gain
    "level_scope": "patch",   # "patch": compare velocity layers across the whole patch; "key": per key range
    "key_range": None,        # [low, high] playable keys (after the pitch convention); zones outside are dropped
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


def auto_loop(pack_dir, rel, f0_pitch, sustained_region=False):
    """Decode a sample and look for loop points. Returns [start, end, xfade] in seconds or None."""
    try:
        with sf.SoundFile(os.path.join(pack_dir, rel)) as f:
            sr = int(f.samplerate)
            data = f.read(min(int(f.frames), 30 * sr), dtype="float32", always_2d=True)
    except Exception:
        return None
    mono = data.mean(axis=1)
    f0 = 440.0 * 2.0 ** ((f0_pitch - 69.0) / 12.0)
    found = find_loop(mono, sr, f0, stable_region=sustained_region)
    if not found:
        return None
    return [round(found["start"] / sr, 4), round(found["end"] / sr, 4), round(found["xfade"] / sr, 4)]


def assign_round_robins(zones):
    """Give zones that share key and velocity range a round-robin position. Returns (rr_groups, layered_groups).

    Positions come from seq_position/seq_length or from distinct random ranges (zones carry them
    as _seq and _rand). Zones that are neither, such as microphone layers, stay stacked.
    """
    groups = collections.defaultdict(list)
    for z in zones:
        groups[(z["lk"], z["hk"], z["lv"], z["hv"])].append(z)
    rr_groups = layered = 0
    for members in groups.values():
        if len(members) == 1:
            continue
        seqs = {z["_seq"] for z in members}
        rands = {z["_rand"] for z in members}
        if any(sq[1] > 1 for sq in seqs) and len(seqs) == len(members):
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
    return rr_groups, layered


def set_layer_levels(zones, scope="patch"):
    """Set each zone's reference level 'rl': its velocity layer's level in dB below the loudest layer.

    The engine uses it to make loudness follow the patch's velocity curve whatever level the
    layers were recorded at, while the timbre still comes from the layer that is chosen. scope
    'patch' compares layers across the whole patch (a pitched instrument); 'key' compares them
    within each key (a drum kit, where every key is a different instrument).
    Returns {layer label: dB below loudest} for the report (patch scope only).
    """
    def scope_key(z):
        return (z["lk"], z["hk"]) if scope == "key" else None

    by_layer = collections.defaultdict(list)
    for z in zones:
        if z.get("_lvl") is not None:
            by_layer[(scope_key(z), z["lv"], z["hv"])].append(z["_lvl"])
    level = {k: statistics.median(v) for k, v in by_layer.items()}
    top = {}
    for (sk, _, _), v in level.items():
        top[sk] = max(top.get(sk, -1e9), v)
    for z in zones:
        lvl = level.get((scope_key(z), z["lv"], z["hv"]))
        z["rl"] = round(lvl - top[scope_key(z)], 2) if lvl is not None else 0.0
        z.pop("_lvl", None)
    return {f"{lv}-{hv}": round(v - top[sk], 1) for (sk, lv, hv), v in sorted(level.items(), key=lambda kv: (str(kv[0][0]), kv[0][1]))}


def tuning_report(pairs):
    """pairs: (residual cents, octave offset, kind). Summarize declared-vs-detected pitch."""
    close = [abs(c) for c, _, kind in pairs if kind == 0]
    return {"compared": len(pairs), "within_60_cents": len(close),
            "octave_shifted": sum(1 for _, o, k in pairs if k == 0 and o != 0),
            "other": sum(1 for _, _, k in pairs if k != 0),
            "median_abs_cents": round(statistics.median(close), 1) if close else None,
            "p95_abs_cents": round(_percentile(close, 0.95), 1) if close else None,
            "mean_cents": round(statistics.fmean(c for c, _, k in pairs if k == 0), 1) if close else None}


def build_zones(datasets_root, tier, pack_id, sfz_rel, inventory_dir, options=None):
    """Build zones for one SFZ program of a pack. Returns (zones, report)."""
    opts = {**DEFAULTS, **(options or {})}
    cc_over = {int(k): int(v) for k, v in (opts["cc"] or {}).items()}
    inc = re.compile(opts["include"]) if opts["include"] else None
    exc = re.compile(opts["exclude"]) if opts["exclude"] else None
    pack_dir = os.path.join(datasets_root, tier, pack_id)
    records = load_records(inventory_dir, pack_id)
    index = AudioIndex(list(records))
    regions, warnings = parse_sfz(os.path.join(pack_dir, sfz_rel), pack_dir, defines=read_bank_defines(pack_dir))

    skipped = collections.Counter()
    cands = []
    for r in regions:
        if str(r.get("trigger", "attack")).lower().startswith("release"):
            skipped["release trigger"] += 1
            continue
        if opts["sw_last"] is not None and r.get("sw_last") not in (None, opts["sw_last"]):
            skipped["other keyswitch"] += 1
            continue
        if not cc_active(r, cc_over):
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
        if (inc and not inc.search(rel)) or (exc and exc.search(rel)):
            skipped["filtered out by include/exclude"] += 1
            continue
        declared = r["root"] - _num(r.get("transpose")) - _num(r.get("tune")) / 100.0
        det = rec.get("pitch_midi") if rec.get("pitched") else None
        delta = octave = residual = None
        if det is not None:
            delta = det - declared
            octave = int(round(delta / 12.0))
            residual = delta - 12 * octave
        cands.append((r, rec, rel, declared, det, octave, residual))

    # The octave the pack's key map is written in: most common offset between the sound and the declared root.
    # Only trusted when most pitched samples agree and the offset is small: drum and noise samples
    # "detect" as random pitches, which must not move a key map.
    octs = collections.Counter(c[5] for c in cands if c[5] is not None and abs(c[6]) <= 0.6)
    pitched = sum(1 for c in cands if c[5] is not None)
    pack_octave = 0
    if octs:
        top, count = octs.most_common(1)[0]
        if abs(top) <= 2 and count >= 0.6 * pitched:
            pack_octave = top

    zones, tuning, loops, loop_cache = [], [], collections.Counter(), {}
    for r, rec, rel, declared, det, octave, residual in cands:
        if det is not None:
            tuning.append((residual * 100.0, octave, 0 if abs(residual) <= 0.6 else 2))
        # "sounding": a key plays the pitch it is named after. Packs written an octave off (bass
        # guitar is notated an octave above where it sounds) have their whole key map moved so the
        # sample that really sounds E1 sits on key E1. "pack" keeps the author's key map.
        shift = 12 * pack_octave if opts["pitch_convention"] == "sounding" else 0
        sounding = declared + 12 * pack_octave        # where the sample really sits if undetected
        root = declared + shift
        if det is not None and opts["refine_tuning"] and abs(residual) <= 0.5:
            root = det - 12 * octave + shift
        lo_key, hi_key = max(r["lokey"] + shift, 0), min(r["hikey"] + shift, 127)
        if lo_key > hi_key:
            skipped["moved outside the keyboard"] += 1
            continue
        pitch_for_loop = det if det is not None else sounding

        zone = {"s": f"{tier}/{pack_id}/{rel}", "_lvl": rec.get("level_db"), "lk": lo_key, "hk": hi_key,
                "lv": max(r["lovel"], 1), "hv": min(r["hivel"], 127), "root": round(root, 3), "g": 0.0, "st": 0.0,
                "_seq": (r["seq_position"], r["seq_length"]), "_rand": (r["lorand"], r["hirand"])}
        if opts["sfz_volume"] and "volume" in r:
            zone["g"] = round(_num(r["volume"]), 2)
        if opts["trim_lead"] and rec.get("lead_s", 0.0) > 0.004:
            zone["st"] = round(rec["lead_s"] - 0.002, 4)
        if str(r.get("loop_mode", "")).lower() == "one_shot":
            zone["os"] = 1
        if r.get("group") not in (None, ""):
            zone["grp"] = int(_num(r["group"]))
        if r.get("off_by") not in (None, ""):
            zone["off_by"] = int(_num(r["off_by"]))

        sr = rec["sr"]
        lm = str(r.get("loop_mode", "")).lower()
        if lm in ("loop_continuous", "loop_sustain") and r.get("loop_end") not in (None, ""):
            zone["loop"] = [round(_num(r.get("loop_start")) / sr, 4), round(_num(r.get("loop_end")) / sr, 4), 0.02]
            loops["from sfz"] += 1
        elif opts["auto_loop"] and rec.get("env") == "sustained" and "os" not in zone:
            if rel not in loop_cache:
                loop_cache[rel] = auto_loop(pack_dir, rel, pitch_for_loop)
            if loop_cache[rel]:
                zone["loop"] = loop_cache[rel]
                loops["auto found"] += 1
            else:
                loops["auto failed"] += 1
        zones.append(zone)

    if opts["key_range"]:
        lo, hi = int(opts["key_range"][0]), int(opts["key_range"][1])
        kept = []
        for z in zones:
            if z["hk"] < lo or z["lk"] > hi:
                skipped["outside key_range"] += 1
                continue
            z["lk"], z["hk"] = max(z["lk"], lo), min(z["hk"], hi)
            kept.append(z)
        zones = kept
    rr_groups, layered = assign_round_robins(zones)
    layer_levels = set_layer_levels(zones, opts["level_scope"])
    zones.sort(key=lambda z: (z["lv"], z["lk"], z.get("rr", [1])[0]))
    covered = {k for z in zones for k in range(z["lk"], z["hk"] + 1)}
    report = {
        "pack": pack_id, "sfz": sfz_rel, "regions_in_sfz": len(regions), "zones": len(zones),
        "skipped": dict(skipped), "sfz_warnings": warnings[:5],
        "keys": [min(covered), max(covered)] if covered else None,
        "velocity_layers": len({(z["lv"], z["hv"]) for z in zones}),
        "round_robin_groups": rr_groups, "layered_groups_left_stacked": layered,
        "layer_levels_db_below_loudest": layer_levels,
        "loops": dict(loops), "pack_octave": pack_octave,
        "key_shift_semitones": 12 * pack_octave if opts["pitch_convention"] == "sounding" else 0,
    }
    if tuning:
        report["declared_vs_detected_pitch"] = tuning_report(tuning)
    return zones, report
