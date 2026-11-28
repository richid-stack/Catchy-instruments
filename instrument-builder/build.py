#!/usr/bin/env python3
"""Build an instrument manifest from a spec.

    python3 -I instrument-builder/build.py --spec generated-instruments/afrobeat/piano/spec.json \
        --datasets datasets --inventory analysis/inventory --out generated-instruments/afrobeat/piano

Writes instrument.json (zones, one per line) and build-report.json next to each other. A spec that
uses research-private samples needs --allow-private and is written as instrument.private.json,
which .gitignore keeps out of the repository.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instrument_builder.build import BuildError, build_instrument  # noqa: E402


def dump_manifest(manifest, path):
    """Write the manifest with one zone per line so diffs stay readable."""
    head = {k: v for k, v in manifest.items() if k != "patches"}
    patches = manifest["patches"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{\n")
        for k, v in head.items():
            fh.write(f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)},\n")
        fh.write(' "patches": [\n')
        for i, p in enumerate(patches):
            meta = ", ".join(f"{json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in p.items() if k != "zones")
            fh.write("  {" + meta + ',\n   "zones": [\n')
            fh.write(",\n".join("    " + json.dumps(z, separators=(",", ":")) for z in p["zones"]))
            fh.write("\n   ]}" + ("," if i < len(patches) - 1 else "") + "\n")
        fh.write(" ]\n}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--inventory", default="analysis/inventory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-private", action="store_true")
    args = ap.parse_args()

    with open(args.spec, encoding="utf-8") as fh:
        spec = json.load(fh)
    try:
        manifest, report = build_instrument(spec, args.datasets, args.inventory, allow_private=args.allow_private)
    except BuildError as exc:
        sys.exit(f"build failed: {exc}")
    os.makedirs(args.out, exist_ok=True)
    name = "instrument.json" if manifest["distributable"] else "instrument.private.json"
    dump_manifest(manifest, os.path.join(args.out, name))
    with open(os.path.join(args.out, "build-report.json" if manifest["distributable"] else "build-report.private.json"),
              "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    for p in manifest["patches"]:
        r = report["patches"][p["id"]]
        if "stack" in r:
            print(f"{p['id']}: stack of {', '.join(l['patch'] for l in r['stack'])}, "
                  f"measured {r.get('measured_rms_db_before_trim')} dB, trim {r['auto_gain_db']} dB")
            continue
        if r.get("mode") == "kit":
            print(f"{p['id']}: kit of {len(r['slots']) - len(r['empty_slots'])} drums ({r['zones']} zones), keys {r['keys']}, "
                  f"measured {r.get('measured_rms_db_before_trim')} dB, trim {r['auto_gain_db']} dB"
                  + (f", EMPTY SLOTS: {', '.join(r['empty_slots'])}" if r["empty_slots"] else ""))
            continue
        print(f"{p['id']}: {r['zones']} zones, {r['velocity_layers']} velocity layers, keys {r['keys']}, "
              f"measured {r.get('measured_rms_db_before_trim')} dB, trim {r['auto_gain_db']} dB, loops {r['loops']}"
              + (f", key balance {r['key_balance']['before_spread_db']} -> {r['key_balance']['after_spread_db']} dB spread"
                 if "key_balance" in r else ""))
    print(f"wrote {os.path.join(args.out, name)}")


if __name__ == "__main__":
    main()
