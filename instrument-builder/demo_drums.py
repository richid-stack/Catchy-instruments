#!/usr/bin/env python3
"""Write a four-bar drum groove (kick, snare with clap, closed and open hats, a hat roll) as MIDI.

    python3 -I instrument-builder/demo_drums.py --out generated-instruments/trap/drums/demo/trap-drums-100bpm.mid

Uses only keys that the Trap Drums kits share (kick 36, rim 37, snare 38, clap 39, closed hat 42, open hat 46).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audio-engine"))

from sampler.midi import write_midi  # noqa: E402

BPM = 100.0


def groove():
    notes = []
    for bar in range(4):
        b = bar * 4.0
        for beat, vel in ((0.0, 118), (2.5, 104), (3.75, 96) if bar % 2 else (1.75, 92)):
            notes.append((b + beat, 0.2, 36, vel))
        notes.append((b + 1.0, 0.2, 38, 112)); notes.append((b + 1.0, 0.2, 39, 80))
        notes.append((b + 3.0, 0.2, 38, 116)); notes.append((b + 3.0, 0.2, 39, 84))
        if bar == 3:
            notes.append((b + 3.5, 0.2, 38, 52)); notes.append((b + 3.75, 0.2, 38, 70))     # ghost notes into the loop
        for i in range(8):                                                                   # eighth-note hats, accents on the beat
            if bar == 3 and i == 7:
                notes.append((b + 3.5, 0.2, 46, 96))                                          # open hat at the end of the bar
                continue
            notes.append((b + i * 0.5, 0.2, 42, 98 if i % 2 == 0 else 66))
        if bar == 1:                                                                          # a sixteenth roll before beat 3
            for i, v in enumerate((60, 72, 84, 96)):
                notes.append((b + 2.5 + i * 0.125, 0.1, 42, v))
        notes.append((b + 1.5, 0.2, 37, 58))
    return notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    notes = groove()
    write_midi(args.out, notes, bpm=BPM)
    print(f"wrote {args.out} ({len(notes)} notes, {BPM:g} bpm)")


if __name__ == "__main__":
    main()
