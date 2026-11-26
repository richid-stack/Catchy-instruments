"""Link SFZ regions to the audio files on disk, and count velocity layers and round robins."""
from __future__ import annotations

import collections
import posixpath


class AudioIndex:
    """Find the audio file an SFZ sample path points at.

    Matching is by exact path first (ignoring case), then ignoring the file extension, and last
    by the trailing path components. The last step covers packs whose SFZ files point at a
    folder variable that the host normally fills in. It only accepts a unique match.
    """

    def __init__(self, audio_paths):
        self.exact = {p.lower(): p for p in audio_paths}
        self.stem_of = {p: posixpath.splitext(p.lower())[0] for p in audio_paths}
        self.by_stem = collections.defaultdict(list)
        self.by_base = collections.defaultdict(list)
        for p, stem in self.stem_of.items():
            self.by_stem[stem].append(p)
            self.by_base[posixpath.basename(stem)].append(p)
        self.fuzzy_links = 0

    def resolve(self, sample: str, fuzzy: bool = True):
        """Return the audio path for an SFZ sample path, or None. fuzzy=False allows exact matches only."""
        s = posixpath.normpath(sample.replace("\\", "/").lower())
        if s in self.exact:
            return self.exact[s]
        stem = posixpath.splitext(s)[0]
        hits = self.by_stem.get(stem, [])
        if len(hits) == 1:
            return hits[0]
        if not fuzzy:
            return None
        parts = [p for p in stem.split("/") if p not in ("", ".", "..") and not p.startswith("$")]
        if not parts:
            return None
        cands = self.by_base.get(parts[-1], [])
        for n in range(len(parts), 0, -1):
            tail = "/".join(parts[-n:])
            found = [c for c in cands if self.stem_of[c] == tail or self.stem_of[c].endswith("/" + tail)]
            if len(found) == 1:
                self.fuzzy_links += 1
                return found[0]
            if found:
                return None  # ambiguous even with the longest matching tail
        return None


def _plays_normally(region) -> bool:
    """False for release triggers (key-up noise, pedal and string resonance samples)."""
    return not str(region.get("trigger", "attack")).lower().startswith("release")


def layer_stats(regions) -> dict:
    """Velocity layers and round robins per key, as the maximum over keys and SFZ programs.

    Velocity layers are the distinct (lovel, hivel) ranges on one key. Round robins are the
    distinct seq_position values, or the distinct random ranges, inside one velocity layer.
    Release-triggered samples and full-range add-ons on a velocity-split key are left out.
    Controller-gated layers (a microphone blend, say) are counted like any other.
    """
    buckets = collections.defaultdict(lambda: collections.defaultdict(lambda: ([set(), set()])))
    for r in regions:
        if not _plays_normally(r):
            continue
        for key in range(max(r["lokey"], 0), min(r["hikey"], 127) + 1):
            seqs, rands = buckets[(r["sfz"], key)][(r["lovel"], r["hivel"])]
            if r["seq_length"] > 1:
                seqs.add(r["seq_position"])
            if (r["lorand"], r["hirand"]) != (0.0, 1.0):
                rands.add((r["lorand"], r["hirand"]))
    if not buckets:
        return {}
    # A full-range region next to velocity-split ones is an add-on (noise, ambience), not a layer.
    for b in buckets.values():
        if len(b) > 1 and (0, 127) in b:
            del b[(0, 127)]
    layers = sorted(len(b) for b in buckets.values())
    rr = max(max(len(seqs), len(rands), 1) for b in buckets.values() for seqs, rands in b.values())
    return {"velocity_layers_max": layers[-1], "velocity_layers_median": layers[len(layers) // 2],
            "round_robins_max": rr, "keys_with_regions": len({k for _, k in buckets})}
