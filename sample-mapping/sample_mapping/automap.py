"""Map packs that have no SFZ file, from their file names and the analysis.

Pitched mode is for instruments recorded one note at a time (VSCO brass, say). File names such as
'Trumpet_sus_A#3_v2_rr1.wav' carry the note, the velocity layer and the round-robin number. The
sounding pitch comes from the analysis (the detected pitch, when it agrees with the name's pitch
class), so a pack that labels notes in another octave still plays each key at its own pitch.

Keys are given to samples by their sounding pitch; each sampled note covers the keys halfway to its
neighbours (and a few keys past the ends); velocity ids become equal velocity bands; repeated
samples of one note and velocity become round robins; leading silence is trimmed; sustained
notes can be looped in their steadiest stretch.
"""
from __future__ import annotations

import collections
import os
import re

from .zones import _num, assign_round_robins, auto_loop, load_records, set_layer_levels

NOTE_RE = (r"(?P<note>[A-G][#b]?)(?P<oct>-?\d+)_v(?P<vel>\d+)(?:_(?:rr)?(?P<rr>\d+))?")
_PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

DEFAULTS = {
    "include": None,        # regex on the pack-relative path (required): which samples form this patch
    "exclude": None,
    "name_regex": NOTE_RE,  # must define groups note, oct, vel and optionally rr
    "loop": "none",         # "stable": loop the steadiest stretch of each sample; "auto": loop at the end of the file; "none"
    "one_shot": False,      # play each sample to its end whatever the key does (staccato stabs)
    "key_span": 3,          # keys covered beyond the lowest and highest sampled notes
    "trim_lead": True,
    "level_scope": "key",
}


def note_to_midi(note: str, octave: int) -> int:
    pc = _PITCH_CLASS[note[0]] + (1 if note.endswith("#") else -1 if note.endswith("b") else 0)
    return 12 * (octave + 1) + pc


def build_pitched_zones(datasets_root, tier, pack_id, inventory_dir, options=None):
    """Zones for one patch of a note-per-file pack. Returns (zones, report)."""
    opts = {**DEFAULTS, **(options or {})}
    if not opts["include"]:
        raise ValueError("automap needs an 'include' pattern that selects the samples of this patch")
    inc = re.compile(opts["include"])
    exc = re.compile(opts["exclude"]) if opts["exclude"] else None
    name_re = re.compile(opts["name_regex"])
    pack_dir = os.path.join(datasets_root, tier, pack_id)
    records = load_records(inventory_dir, pack_id)

    skipped = collections.Counter()
    items = []
    for path, rec in sorted(records.items()):
        if not inc.search(path) or (exc and exc.search(path)):
            continue
        if "error" in rec:
            skipped["unreadable"] += 1
            continue
        m = name_re.search(os.path.splitext(path)[0])
        if not m:
            skipped["name did not match"] += 1
            continue
        named = note_to_midi(m.group("note"), int(m.group("oct")))
        det = rec.get("pitch_midi") if rec.get("pitched") else None
        octave = residual = None
        if det is not None:
            delta = det - named
            octave = int(round(delta / 12.0))
            residual = delta - 12 * octave
            if abs(residual) > 0.6:      # the detector and the name disagree about the note: distrust the detector
                det = octave = residual = None
        items.append({"path": path, "rec": rec, "named": named, "det": det, "octave": octave, "residual": residual,
                      "vel": int(m.group("vel")), "rr": int(m.group("rr")) if m.group("rr") else None})
    if not items:
        raise ValueError(f"no samples of {pack_id} match include={opts['include']!r}")

    agree = collections.Counter(i["octave"] for i in items if i["octave"] is not None)
    pack_octave = 0
    if agree:
        top, count = agree.most_common(1)[0]
        if abs(top) <= 2 and count >= 0.6 * len(items):
            pack_octave = top
    from_name = 0
    for i in items:
        if i["det"] is not None:
            i["root"] = i["det"]
        else:
            i["root"] = float(i["named"] + 12 * pack_octave)
            from_name += 1
        i["key"] = int(round(i["root"]))

    # velocity ids -> equal bands
    ids = sorted({i["vel"] for i in items})
    n = len(ids)
    band = {vid: (1 if idx == 0 else round(127 * idx / n) + 1, round(127 * (idx + 1) / n)) for idx, vid in enumerate(ids)}

    keys = sorted({i["key"] for i in items})
    span = int(opts["key_span"])
    lo = {k: (k - span if idx == 0 else (keys[idx - 1] + k) // 2 + 1) for idx, k in enumerate(keys)}
    hi = {k: (k + span if idx == len(keys) - 1 else (k + keys[idx + 1]) // 2) for idx, k in enumerate(keys)}

    by_cell = collections.defaultdict(list)
    for i in items:
        by_cell[(i["key"], i["vel"])].append(i)

    loops, loop_cache, zones = collections.Counter(), {}, []
    for (key, vid), members in sorted(by_cell.items()):
        members.sort(key=lambda i: (i["rr"] or 0, i["path"]))
        for pos, i in enumerate(members, start=1):
            rec = i["rec"]
            zone = {"s": f"{tier}/{pack_id}/{i['path']}", "_lvl": rec.get("level_db"), "lk": max(lo[key], 0),
                    "hk": min(hi[key], 127), "lv": band[vid][0], "hv": band[vid][1], "root": round(i["root"], 3),
                    "g": 0.0, "st": 0.0, "_seq": (pos, len(members)), "_rand": (0.0, 1.0)}
            if opts["trim_lead"] and rec.get("lead_s", 0.0) > 0.004:
                zone["st"] = round(rec["lead_s"] - 0.002, 4)
            if opts["one_shot"]:
                zone["os"] = 1
            elif opts["loop"] != "none":
                if i["path"] not in loop_cache:
                    loop_cache[i["path"]] = auto_loop(pack_dir, i["path"], i["root"], sustained_region=opts["loop"] == "stable")
                if loop_cache[i["path"]]:
                    zone["loop"] = loop_cache[i["path"]]
                    loops["found"] += 1
                else:
                    loops["failed"] += 1
            zones.append(zone)

    rr_groups, layered = assign_round_robins(zones)
    layer_levels = set_layer_levels(zones, opts["level_scope"])
    zones.sort(key=lambda z: (z["lv"], z["lk"], z.get("rr", [1])[0]))
    report = {
        "pack": pack_id, "mode": "pitched", "include": opts["include"], "samples_matched": len(items), "zones": len(zones),
        "skipped": dict(skipped), "keys": [zones[0]["lk"] if zones else None, max(z["hk"] for z in zones)],
        "sampled_notes": len(keys), "velocity_layers": n, "round_robin_groups": rr_groups,
        "layered_groups_left_stacked": layered, "roots_from_name": from_name, "pack_octave": pack_octave,
        "loops": dict(loops), "layer_levels_db_below_loudest": layer_levels,
    }
    report["keys"] = [min(z["lk"] for z in zones), max(z["hk"] for z in zones)]
    return zones, report
