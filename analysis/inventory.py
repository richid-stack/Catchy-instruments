#!/usr/bin/env python3
"""Inventory every sample pack under datasets/.

For each audio file it records format, levels, envelope class, detected pitch and the
hints in the file name; SFZ mappings are attached when a pack ships them. Files are then
grouped by folder (a folder is normally one instrument or articulation) and summarized.

The audio is untrusted input, so run this in isolated mode:

    python3 -I analysis/inventory.py --datasets datasets \
        --out analysis/inventory --reports analysis/reports

Packs already inventoried are skipped unless --force is given.
"""
import argparse
import collections
import gzip
import json
import os
import posixpath
import statistics
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audioinv.features import analyze_file  # noqa: E402
from audioinv.nameparse import midi_to_name, parse_name  # noqa: E402
from audioinv.link import AudioIndex, layer_stats  # noqa: E402
from audioinv.sfzparse import parse_sfz, read_bank_defines  # noqa: E402

AUDIO_EXT = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".mp3"}
TIERS = ("clean", "research-private")
_ROOT = None


def walk_pack(pack_dir):
    audio, sfz = [], []
    for dirpath, dirnames, filenames in os.walk(pack_dir, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d != "__MACOSX")
        for fn in sorted(filenames):
            if fn.startswith("._"):
                continue
            full = os.path.join(dirpath, fn)
            if os.path.islink(full):
                continue
            ext = os.path.splitext(fn)[1].lower()
            rel = os.path.relpath(full, pack_dir).replace(os.sep, "/")
            if ext in AUDIO_EXT:
                audio.append(rel)
            elif ext == ".sfz":
                sfz.append(rel)
    return audio, sfz


def _init(root):
    global _ROOT
    _ROOT = root


def _reconcile(rec):
    """Combine the detected pitch with the pitch hinted by the file name."""
    detected = rec.get("pitch_midi")
    named = rec.get("name_midi")
    if "release" in rec.get("artic", []) and named is not None:
        # Key-release noise has no reliable pitch, so trust the name and skip the comparison.
        rec["note_midi"] = named
        rec["note_name"] = midi_to_name(named)
        rec["note_source"] = "name"
        return
    if detected is not None:
        nearest = int(round(detected))
        rec["note_midi"] = nearest
        rec["note_name"] = midi_to_name(nearest)
        rec["cents"] = int(round((detected - nearest) * 100))
        rec["note_source"] = "detected"
        if named is not None:
            diff = detected - named
            k = round(diff / 12.0)
            rec["name_agrees"] = abs(diff) <= 0.6
            rec["name_octave_shift"] = int(k) if abs(diff - 12 * k) <= 0.6 else None
    elif named is not None:
        rec["note_midi"] = named
        rec["note_name"] = midi_to_name(named)
        rec["note_source"] = "name"


def _work(rel):
    rec = {"path": rel}
    rec.update(parse_name(rel))
    rec.update(analyze_file(os.path.join(_ROOT, rel)))
    _reconcile(rec)
    return rec


def load_sfz(pack_dir, sfz_files, audio):
    """Parse the pack's SFZ files and link every region to an audio file.

    Files that other SFZ files include are fragments (a mapping for one microphone, say), not
    instruments, so only the root files are counted. Returns (by_sample, regions, stats) where
    by_sample maps an audio path to the regions that play it.
    """
    defines = read_bank_defines(pack_dir)
    parsed, included = {}, set()
    warnings, errors = [], []
    for rel in sfz_files:
        inc = set()
        try:
            regions, warns = parse_sfz(os.path.join(pack_dir, rel), pack_dir, defines=defines, included=inc)
        except Exception as exc:  # untrusted input
            errors.append(f"{rel}: {type(exc).__name__}: {exc}"[:200])
            continue
        parsed[rel] = regions
        included |= inc
        warnings.extend(f"{rel}: {w}" for w in warns)
    roots = [rel for rel in parsed if rel not in included] or list(parsed)

    index = AudioIndex(audio)
    # An included-by-nobody file whose sample paths do not resolve on their own is an orphan
    # fragment (an unused tuning variant, say), not an instrument. If that would drop every
    # root, keep them all and rely on the looser matching.
    solid = []
    for rel in roots:
        regs = parsed[rel]
        exact = sum(1 for r in regs if index.resolve(r["sample"], fuzzy=False) is not None)
        if regs and exact * 2 >= len(regs):
            solid.append(rel)
    skipped = len(roots) - len(solid) if solid else 0
    roots = solid or roots
    by_sample = collections.defaultdict(list)
    all_regions, unresolved = [], set()
    for rel in roots:
        for region in parsed[rel]:
            target = index.resolve(region["sample"])
            if target is None:
                unresolved.add(region["sample"])
                continue
            region["sample"] = target
            by_sample[target.lower()].append(region)
            all_regions.append(region)
    stats = {"root_files": len(roots), "orphan_fragments_skipped": skipped, "regions": len(all_regions), "unresolved": sorted(unresolved),
             "fuzzy_links": index.fuzzy_links, "warnings": warnings[:20], "errors": errors[:20]}
    return by_sample, all_regions, stats


def _compact(region):
    out = {"f": region["sfz"], "lo": region["lokey"], "hi": region["hikey"], "root": region["root"],
           "lv": region["lovel"], "hv": region["hivel"], "seq": region["seq_position"]}
    if region["lorand"] != 0.0 or region["hirand"] != 1.0:
        out["rand"] = [region["lorand"], region["hirand"]]
    return out


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 3) if values else None


def summarize_group(recs):
    ok = [r for r in recs if "error" not in r]
    g = {"files": len(recs), "errors": len(recs) - len(ok)}
    if not ok:
        return g
    g["minutes"] = round(sum(r["frames"] / r["sr"] for r in ok) / 60.0, 2)
    g["formats"] = sorted({r["format"] for r in ok})
    g["subtypes"] = sorted({r["subtype"] for r in ok})
    g["sample_rates"] = sorted({r["sr"] for r in ok})
    g["channels"] = sorted({r["channels"] for r in ok})
    g["pitched_ratio"] = round(sum(1 for r in ok if r.get("pitched")) / len(ok), 2)

    notes = [r["note_midi"] for r in ok if "note_midi" in r]
    distinct = sorted(set(notes))
    if distinct:
        g["pitch_range"] = [midi_to_name(distinct[0]), midi_to_name(distinct[-1])]
        g["distinct_pitches"] = len(distinct)
        gaps = [[midi_to_name(a), midi_to_name(b), b - a] for a, b in zip(distinct, distinct[1:]) if b - a > 4]
        if gaps:
            g["gaps_over_4_semitones"] = gaps[:6]
        per_pitch = collections.Counter(notes)
        g["files_per_pitch_median"] = statistics.median(per_pitch.values())

    vel = {r["name_vel"] for r in ok if "name_vel" in r}
    if vel:
        g["name_velocity_layers"] = len(vel)
    dyn = sorted({r["name_dyn"] for r in ok if "name_dyn" in r})
    if dyn:
        g["name_dynamics"] = dyn
    rr = {r["name_rr"] for r in ok if "name_rr" in r}
    if rr:
        g["name_round_robins"] = len(rr)
    artic = collections.Counter(a for r in ok for a in r.get("artic", []))
    if artic:
        g["articulations"] = dict(artic.most_common(8))

    diffs = [r["sfz_root_diff"] for r in ok if "sfz_root_diff" in r]
    if diffs:
        close = sum(1 for d in diffs if abs(d) <= 0.6)
        octave_off = sum(1 for d in diffs if abs(d) > 0.6 and abs(d - 12 * round(d / 12)) <= 0.6)
        # by_octave shows the pack's mapping convention: "-1" means the SFZ root is one octave
        # above the sounding pitch (typical for basses written an octave above they sound).
        by_octave = collections.Counter(str(int(round(d / 12))) for d in diffs
                                        if abs(d - 12 * round(d / 12)) <= 0.6)
        g["sfz_root_vs_pitch"] = {"compared": len(diffs), "within_60_cents": close,
                                  "octave_off": octave_off, "other": len(diffs) - close - octave_off,
                                  "by_octave": dict(sorted(by_octave.items()))}

    sfz_refs = [ref for r in ok for ref in r.get("sfz", [])]
    if sfz_refs:
        g["sfz_regions_seen"] = sum(r.get("sfz_n", 0) for r in ok)

    g["envelope"] = dict(collections.Counter(r["env"] for r in ok if "env" in r))
    for key, label in (("centroid_hz", "centroid_hz_median"), ("level_db", "level_db_median"),
                       ("lead_s", "lead_silence_s_median"), ("attack_s", "attack_s_median"),
                       ("dur_s", "duration_s_median")):
        g[label] = _median([r.get(key) for r in ok])

    shifts = collections.Counter(r["name_octave_shift"] for r in ok
                                 if "name_agrees" in r and r.get("name_octave_shift") is not None)
    compared = sum(1 for r in ok if "name_agrees" in r)
    if compared:
        g["name_vs_pitch"] = {
            "compared": compared,
            "agree": sum(1 for r in ok if r.get("name_agrees")),
            "octave_shift_mode": shifts.most_common(1)[0][0] if shifts else None,
            "mismatch": sum(1 for r in ok if "name_agrees" in r and r.get("name_octave_shift") is None),
        }
    flags = {
        "dual_mono_files": sum(1 for r in ok if r.get("dual_mono")),
        "cut_off_ending": sum(1 for r in ok if r.get("ends_cut")),
        "clipped_files": sum(1 for r in ok if r.get("clipped", 0) > 3),
        "silent_files": sum(1 for r in ok if r.get("silent")),
    }
    g["flags"] = {k: v for k, v in flags.items() if v}
    return g


def process_pack(tier, pack_id, pack_dir, out_dir, rep_dir, workers, force):
    out_path = os.path.join(out_dir, f"{pack_id}.jsonl.gz")
    if os.path.exists(out_path) and not force:
        return f"{pack_id}: skipped (already inventoried)"
    start = time.time()
    audio, sfz_files = walk_pack(pack_dir)
    by_sample, linked_regions, sfz_stats = load_sfz(pack_dir, sfz_files, audio)

    with Pool(workers, initializer=_init, initargs=(pack_dir,)) as pool:
        records = list(pool.imap_unordered(_work, audio, chunksize=16))
    records.sort(key=lambda r: r["path"])

    referenced = set()
    for rec in records:
        refs = by_sample.get(rec["path"].lower())
        if refs:
            referenced.add(rec["path"].lower())
            rec["sfz_n"] = len(refs)
            rec["sfz"] = [_compact(r) for r in refs[:6]]
            root = refs[0].get("root")
            if root is not None and rec.get("note_source") == "detected":
                rec["sfz_root_diff"] = round(rec["pitch_midi"] - root, 2)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)
    tmp = out_path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    os.replace(tmp, out_path)

    groups = collections.defaultdict(list)
    for rec in records:
        groups[posixpath.dirname(rec["path"]) or "."].append(rec)
    summary = summarize_group(records)
    summary["sfz_files"] = len(sfz_files)
    if sfz_files:
        summary["sfz_root_files"] = sfz_stats["root_files"]
        if sfz_stats["orphan_fragments_skipped"]:
            summary["sfz_orphan_fragments_skipped"] = sfz_stats["orphan_fragments_skipped"]
        summary["sfz_regions"] = sfz_stats["regions"]
        summary["sfz_layers"] = layer_stats(linked_regions)
        summary["sfz_missing_samples"] = len(sfz_stats["unresolved"])
        if sfz_stats["unresolved"]:
            summary["sfz_missing_examples"] = sfz_stats["unresolved"][:5]
        if sfz_stats["fuzzy_links"]:
            summary["sfz_fuzzy_links"] = sfz_stats["fuzzy_links"]
        summary["audio_not_in_any_sfz"] = len(records) - len(referenced)
    if sfz_stats["warnings"]:
        summary["sfz_warnings"] = sfz_stats["warnings"]
    if sfz_stats["errors"]:
        summary["sfz_errors"] = sfz_stats["errors"]
    bad = [{"path": r["path"], "error": r["error"]} for r in records if "error" in r]
    if bad:
        summary["unreadable_examples"] = bad[:10]

    report = {"pack": pack_id, "tier": tier, "summary": summary,
              "groups": {k: summarize_group(v) for k, v in sorted(groups.items())}}
    with open(os.path.join(rep_dir, f"{pack_id}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    return (f"{pack_id}: {len(records)} audio files, {len(sfz_files)} sfz, {len(groups)} folders, "
            f"{summary.get('errors', 0)} unreadable, {time.time() - start:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--out", default="analysis/inventory")
    ap.add_argument("--reports", default="analysis/reports")
    ap.add_argument("--pack", action="append", help="only this pack id (repeatable)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for tier in TIERS:
        tier_dir = os.path.join(args.datasets, tier)
        if not os.path.isdir(tier_dir):
            continue
        for pack_id in sorted(os.listdir(tier_dir)):
            pack_dir = os.path.join(tier_dir, pack_id)
            if not os.path.isdir(pack_dir) or (args.pack and pack_id not in args.pack):
                continue
            print(process_pack(tier, pack_id, pack_dir, args.out, args.reports, args.workers, args.force),
                  flush=True)


if __name__ == "__main__":
    main()
