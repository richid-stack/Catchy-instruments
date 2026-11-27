#!/usr/bin/env python3
"""Render a MIDI file through a built instrument.

    python3 -I audio-engine/render.py --instrument generated-instruments/afrobeat/piano/instrument.json \
        --datasets datasets --patch grand --midi song.mid --out song.wav
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sampler.engine import Engine, write_wav  # noqa: E402
from sampler.instrument import Instrument  # noqa: E402
from sampler.midi import notes_from_midi, read_midi  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--patch")
    ap.add_argument("--midi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--no-pedal", action="store_true")
    args = ap.parse_args()

    inst = Instrument.load(args.instrument)
    if args.patch and args.patch not in inst.patches:
        sys.exit(f"unknown patch {args.patch!r}; choose from {', '.join(inst.patches)}")
    notes = notes_from_midi(read_midi(args.midi), pedal=not args.no_pedal)
    audio, stats = Engine(inst, args.datasets, sr=args.sr).render(notes, patch=args.patch)
    write_wav(args.out, audio, args.sr)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
