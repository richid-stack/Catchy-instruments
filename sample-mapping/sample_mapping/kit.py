"""Map drum hits into a kit: one key per drum, round robins from similar hits, one-shot playback.

A drum pack is a folder of single hits with no mapping. The spec lists slots (key, what to look for)
and this module picks the hits that fit each slot, brings them to a common peak level, and writes
one-shot zones. A slot with several hits becomes round robins; to keep the robins sounding like one
drum it takes the hits closest to the typical one (by brightness and decay), not the first few by name.
Choke groups let a closed hat cut an open hat.

Slot options: key, name, pack (default: the patch's pack), include/exclude (regex on the pack-relative
path; add (?i) for any case), name_include/name_exclude (the same, on the file name alone), dur / t20 / centroid (min, max ranges from the inventory), pitched
(true/false), rr (how many hits, default 4), peak_db (level the slot's typical hit peaks at, default -3),
gain_db (extra), grp / off_by (choke groups: a voice whose zone has off_by=g is cut when group g plays).
"""
from __future__ import annotations

import math
import os
import re

import numpy as np
import soundfile as sf

from .zones import load_records

DEFAULT_SLOT = {"rr": 4, "peak_db": -3.0, "gain_db": 0.0}


def _in_range(value, rng):
    return value is not None and rng[0] <= value <= rng[1]


def _fits(rec, slot):
    path = rec["path"]
    name = path.rsplit("/", 1)[-1]
    if slot.get("include") and not re.search(slot["include"], path):
        return False
    if slot.get("exclude") and re.search(slot["exclude"], path):
        return False
    if slot.get("name_include") and not re.search(slot["name_include"], name):
        return False
    if slot.get("name_exclude") and re.search(slot["name_exclude"], name):
        return False
    for field, key in (("dur", "dur_s"), ("t20", "t20_s"), ("centroid", "centroid_hz")):
        if field in slot and not _in_range(rec.get(key), slot[field]):
            return False
    if "pitched" in slot and bool(rec.get("pitched")) != bool(slot["pitched"]):
        return False
    return True


def _typical(cands, n):
    """The n hits nearest the median in (log brightness, log decay), so round robins belong together."""
    if len(cands) <= n:
        return sorted(cands, key=lambda r: r["path"])
    feats = np.array([[math.log10(max(r.get("centroid_hz") or 1.0, 1.0)), math.log10(max(r.get("t20_s") or 0.01, 0.005))]
                      for r in cands])
    spread = np.maximum(feats.std(axis=0), 0.05)       # a feature that barely varies must not turn rounding noise into distance
    dist = np.sqrt((((feats - np.median(feats, axis=0)) / spread) ** 2).sum(axis=1))
    order = np.argsort(dist, kind="stable")[:n]
    return sorted((cands[i] for i in order), key=lambda r: r["path"])


def _attack_db(datasets_root, tier, pack_id, rec, window_s=0.12):
    """Level (dBFS RMS) of the first 120 ms after the hit starts, measured from the file."""
    path = os.path.join(datasets_root, tier, pack_id, *rec["path"].split("/"))
    sr = int(rec.get("sr") or 44100)
    start = int((rec.get("lead_s") or 0.0) * sr)
    x, _ = sf.read(path, start=start, frames=int(window_s * sr), dtype="float32", always_2d=True)
    mono = x.mean(axis=1) if len(x) else np.zeros(1)
    return 10.0 * math.log10(max(float(np.mean(mono ** 2)), 1e-12))


def _gains(datasets_root, tier, pid, chosen, slot):
    """Per-hit gain: round robins are matched on attack loudness (at most 8 dB), then the slot's typical peak lands on peak_db."""
    att = [_attack_db(datasets_root, tier, pid, r) for r in chosen]
    med = float(np.median(att))
    match = [max(-8.0, min(8.0, med - a)) for a in att]
    peaks = [float(r.get("peak_db") or 0.0) + m for r, m in zip(chosen, match)]
    offset = float(slot["peak_db"]) - float(np.median(peaks)) + float(slot["gain_db"])
    # a hit that is mostly one sharp spike must not end up louder than 2 dB over the slot's peak level
    return [max(-24.0, min(24.0, min(m + offset, float(slot["peak_db"]) + float(slot["gain_db"]) + 2.0 - float(r.get("peak_db") or 0.0))))
            for r, m in zip(chosen, match)]


def build_kit_zones(datasets_root, tier, pack_id, inventory_dir, options=None):
    """Return (zones, report). Slots that match nothing are reported, not silently dropped."""
    opts = options or {}
    slots = opts.get("slots") or []
    if not slots:
        raise ValueError("a kit needs 'slots'")
    cache, zones, report, packs = {}, [], {}, []
    for slot in slots:
        s = {**DEFAULT_SLOT, **slot}
        if "key" not in s:
            raise ValueError(f"slot {s.get('name')!r} has no key")
        pid = s.get("pack", pack_id)
        if pid is None:
            raise ValueError(f"slot {s.get('name')!r} names no pack")
        if pid not in cache:
            cache[pid] = load_records(inventory_dir, pid)
        if pid not in packs:
            packs.append(pid)
        cands = [r for r in cache[pid].values() if _fits(r, s)]
        label = s.get("name") or str(s["key"])
        if not cands:
            report[label] = {"key": s["key"], "pack": pid, "matched": 0, "chosen": []}
            continue
        chosen = _typical(cands, int(s["rr"]))
        gains = _gains(datasets_root, tier, pid, chosen, s)
        for pos, (r, gain) in enumerate(zip(chosen, gains), start=1):
            z = {"s": f"{tier}/{pid}/{r['path']}", "lk": int(s["key"]), "hk": int(s["key"]), "lv": 1, "hv": 127,
                 "root": int(s["key"]), "g": round(gain, 2), "st": 0.0, "os": 1, "rl": 0.0}
            if (r.get("lead_s") or 0.0) > 0.004:
                z["st"] = round(r["lead_s"] - 0.002, 4)
            if len(chosen) > 1:
                z["rr"] = [pos, len(chosen)]
            if "grp" in s:
                z["grp"] = int(s["grp"])
            if "off_by" in s:
                z["off_by"] = int(s["off_by"])
            zones.append(z)
        report[label] = {"key": s["key"], "pack": pid, "matched": len(cands),
                         "chosen": [r["path"].split("/")[-1] for r in chosen]}
    zones.sort(key=lambda z: (z["lk"], z.get("rr", [1])[0]))
    empty = [k for k, v in report.items() if not v["matched"]]
    return zones, {"mode": "kit", "packs": packs, "zones": len(zones), "slots": report, "empty_slots": empty,
                   "keys": [zones[0]["lk"], zones[-1]["hk"]] if zones else None}
