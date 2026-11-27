"""Load an instrument manifest and choose the zone that plays for a given key and velocity."""
from __future__ import annotations

import collections
import json
import os

FORMAT = "catchy-instrument/1"
MAX_MANIFEST_BYTES = 40_000_000


class InstrumentError(Exception):
    pass


def _check_zone(z, where):
    for k in ("s", "lk", "hk", "lv", "hv", "root"):
        if k not in z:
            raise InstrumentError(f"{where}: zone missing {k!r}")
    path = z["s"]
    if not isinstance(path, str) or path.startswith(("/", "\\")) or ".." in path.replace("\\", "/").split("/"):
        raise InstrumentError(f"{where}: unsafe sample path {path!r}")
    if not (0 <= z["lk"] <= z["hk"] <= 127 and 0 <= z["lv"] <= z["hv"] <= 127):
        raise InstrumentError(f"{where}: key or velocity range out of bounds")
    if not isinstance(z["root"], (int, float)):
        raise InstrumentError(f"{where}: root must be a number")


class Patch:
    def __init__(self, raw):
        self.id = raw["id"]
        self.name = raw.get("name", self.id)
        self.voice = dict(raw["voice"])
        self.zones = raw["zones"]
        self.by_key = collections.defaultdict(list)
        self.extend_semitones = int(self.voice.get("extend_semitones", 12))
        for i, z in enumerate(self.zones):
            _check_zone(z, f"patch {self.id} zone {i}")
            for key in range(z["lk"], z["hk"] + 1):
                self.by_key[key].append(z)
        self._counters = collections.Counter()

    def reset_round_robin(self):
        self._counters.clear()

    def pick(self, key, vel):
        """Zones to play for (key, velocity).

        A key outside the sampled range borrows the nearest sampled key (pitch-shifted by the
        engine) up to extend_semitones away; further out it is empty.
        """
        cands = self.by_key.get(key)
        if not cands:
            if not self.by_key:
                return []
            near = min(self.by_key, key=lambda k: abs(k - key))
            if abs(near - key) > self.extend_semitones:
                return []
            cands = self.by_key[near]
        hit = [z for z in cands if z["lv"] <= vel <= z["hv"]]
        if not hit:  # velocity gap: use the layer nearest to this velocity
            def gap(z):
                return z["lv"] - vel if vel < z["lv"] else vel - z["hv"]
            best = min(gap(z) for z in cands)
            hit = [z for z in cands if gap(z) == best]
        groups = collections.defaultdict(list)
        for z in hit:
            groups[(z["lk"], z["hk"], z["lv"], z["hv"])].append(z)
        gkey = min(groups, key=lambda g: (abs(key - groups[g][0]["root"]), g[3] - g[2]))
        members = groups[gkey]
        rr = sorted((z for z in members if "rr" in z), key=lambda z: z["rr"][0])
        if rr:
            n = len(rr)
            idx = self._counters[(self.id, gkey)] % n
            self._counters[(self.id, gkey)] += 1
            return [rr[idx]]
        return members


class Instrument:
    def __init__(self, manifest):
        if manifest.get("format") != FORMAT:
            raise InstrumentError(f"unsupported format {manifest.get('format')!r}")
        raw_patches = manifest.get("patches") or []
        if not raw_patches:
            raise InstrumentError("instrument has no patches")
        self.manifest = manifest
        self.name = manifest.get("name", "")
        self.patches = {}
        for raw in raw_patches:
            if "id" not in raw or "zones" not in raw or "voice" not in raw:
                raise InstrumentError("patch needs id, voice and zones")
            self.patches[raw["id"]] = Patch(raw)
        self.default_patch = manifest.get("default_patch") or raw_patches[0]["id"]
        if self.default_patch not in self.patches:
            raise InstrumentError("default_patch is not one of the patches")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            raw = fh.read(MAX_MANIFEST_BYTES + 1)
        if len(raw) > MAX_MANIFEST_BYTES:
            raise InstrumentError("manifest too large")
        try:
            return cls(json.loads(raw.decode("utf-8")))
        except (ValueError, KeyError, TypeError) as exc:
            raise InstrumentError(f"bad manifest: {exc}") from exc
