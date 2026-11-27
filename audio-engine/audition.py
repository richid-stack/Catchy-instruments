#!/usr/bin/env python3
"""Play one MIDI part through every patch of an instrument, one after another, into a single WAV.

    python3 -I audio-engine/audition.py --instrument generated-instruments/afrobeat/bass/instrument.json \
        --datasets datasets --midi generated-instruments/afrobeat/bass/demo/afrobeat-bass-108bpm.mid \
        --seconds 9 --out bass-audition.wav

Prints the start time of each patch so a listener knows which is which.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sampler.engine import Engine, write_wav  # noqa: E402
from sampler.instrument import Instrument  # noqa: E402
from sampler.midi import Note, notes_from_midi, read_midi  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instrument", required=True)
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--midi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=0.0, help="play only the first N seconds of the part (0 = all)")
    ap.add_argument("--gap", type=float, default=0.8, help="silence between patches, seconds")
    ap.add_argument("--sr", type=int, default=44100)
    args = ap.parse_args()

    inst = Instrument.load(args.instrument)
    notes = notes_from_midi(read_midi(args.midi))
    if args.seconds > 0:
        notes = [Note(n.time, min(n.dur, args.seconds - n.time), n.key, n.vel, n.channel)
                 for n in notes if n.time < args.seconds]
    engine = Engine(inst, args.datasets, sr=args.sr)
    pieces, marks, cursor = [], {}, 0.0
    gap = np.zeros((int(args.gap * args.sr), 2), dtype=np.float32)
    for pid in inst.patches:
        audio, _ = engine.render(notes, patch=pid, tail_s=0.6)
        marks[pid] = round(cursor, 1)
        pieces += [audio, gap]
        cursor += (audio.shape[0] + gap.shape[0]) / args.sr
    write_wav(args.out, np.concatenate(pieces), args.sr)
    print(json.dumps({"seconds": round(cursor, 1), "patch_starts_s": marks}))


if __name__ == "__main__":
    main()
