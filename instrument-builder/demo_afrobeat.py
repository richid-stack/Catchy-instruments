#!/usr/bin/env python3
"""Write a short Afrobeat keys demo as a MIDI file (108 BPM, A minor, eight bars).

    python3 -I instrument-builder/demo_afrobeat.py generated-instruments/afrobeat/piano/demo/afrobeat-comp-108bpm.mid

Syncopated chord stabs with ghost notes, a bass line in the left hand, and a pentatonic lick in
the second half. Velocities are humanized with a fixed seed so the file is reproducible.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audio-engine"))

from sampler.midi import write_midi  # noqa: E402

STEP = 0.25  # one sixteenth, in beats
BPM = 108.0
CHORDS = {
    "Am9": {"stab": [57, 60, 64, 67, 71], "bass": [33, 45, 40]},
    "D9": {"stab": [54, 57, 60, 64, 69], "bass": [38, 50, 45]},
}
BARS = ["Am9", "Am9", "D9", "Am9", "Am9", "Am9", "D9", "Am9"]
STAB_STEPS = [(0, 2, 96), (3, 2, 78), (6, 2, 102), (8, 2, 84), (11, 2, 74), (14, 1, 92)]
GHOST_STEPS = [(1, 1, 46), (5, 1, 52), (9, 1, 44), (13, 1, 50)]
BASS_STEPS = [(0, 3, 94, 0), (6, 2, 88, 1), (10, 2, 92, 2), (13, 2, 86, 1)]
LICK = [(2, 69), (4, 72), (7, 74), (9, 76), (12, 79), (14, 76)]  # A minor pentatonic, bars 5-8


def lcg(seed):
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (2 ** 31)
        yield state / 2 ** 31


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    rnd = lcg(7)
    notes = []

    def add(bar, step, dur_steps, key, vel):
        jitter = (next(rnd) - 0.5) * 0.02          # up to 10 ms early or late at this tempo
        vel = max(1, min(127, int(vel + (next(rnd) - 0.5) * 10)))
        start = max(bar * 4 + step * STEP + jitter, 0.0)
        notes.append((start, dur_steps * STEP * 0.9, key, vel))

    for bar, name in enumerate(BARS):
        chord = CHORDS[name]
        for step, dur, vel in STAB_STEPS:
            for key in chord["stab"]:
                add(bar, step, dur, key, vel)
        for step, dur, vel in GHOST_STEPS:
            for key in chord["stab"][1:4]:
                add(bar, step, dur, key, vel)
        for step, dur, vel, which in BASS_STEPS:
            add(bar, step, dur, chord["bass"][which], vel)
        if bar >= 4:
            for step, key in LICK:
                add(bar, step, 2, key + 12 * (bar == 7), 100)
    write_midi(sys.argv[1], notes, bpm=BPM)
    print(f"wrote {sys.argv[1]}: {len(notes)} notes, {len(BARS)} bars at {BPM:g} BPM")


if __name__ == "__main__":
    main()
