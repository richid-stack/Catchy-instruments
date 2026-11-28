#!/usr/bin/env python3
"""Cut single notes out of loop stems into datasets/research-private/harvest-guitar-loops/<family>/.

    python3 -I sample-mapping/harvest_loops.py --family inq --packs 'inqboi-afrolistic-guitar-pack' \
        --include '/Dry/' --exclude 'Scream'

--packs, --include and --exclude are regular expressions: packs match whole folder names under
datasets/research-private, include/exclude match the stem's path. Run analysis/inventory.py afterwards
(delete the pack's old inventory first) so the pitched automapper can see the new files.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sample_mapping import harvest  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, help="name of the output folder and file prefix")
    ap.add_argument("--packs", required=True, help="regex of pack folder names to read")
    ap.add_argument("--include", default=".", help="regex a stem's path must match")
    ap.add_argument("--exclude", default=None, help="regex that removes stems")
    ap.add_argument("--datasets", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasets"))
    ap.add_argument("--out-pack", default="harvest-guitar-loops")
    ap.add_argument("--option", action="append", default=[], metavar="KEY=VALUE", help="override a harvest option (JSON value)")
    args = ap.parse_args()
    root = os.path.join(args.datasets, "research-private")
    stems = sorted(p for d in os.listdir(root) if re.fullmatch(args.packs, d)
                   for p in glob.glob(os.path.join(root, d, "**", "*.wav"), recursive=True)
                   if re.search(args.include, p) and not (args.exclude and re.search(args.exclude, p)))
    if not stems:
        sys.exit("no stems matched")
    options = {k: json.loads(v) for k, v in (o.split("=", 1) for o in args.option)}
    out = os.path.join(root, args.out_pack, args.family)
    shutil.rmtree(out, ignore_errors=True)
    rep = harvest.harvest(stems, out, args.family, options)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("notes", "per_stem")}, indent=1))


if __name__ == "__main__":
    main()
