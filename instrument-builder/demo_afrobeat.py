#!/usr/bin/env python3
"""Write short Afrobeat demo parts as MIDI files (108 BPM, A minor, eight bars: Am9 Am9 D9 Am9, twice).

    python3 -I instrument-builder/demo_afrobeat.py [keys|bass] OUT.mid

keys (the default): syncopated chord stabs with ghost notes, a bass line in the left hand and a
pentatonic lick in the second half. bass: a syncopated root/octave/fifth line with passing notes,
some of them overlapping so monophonic patches show their note cutting. Velocities are humanized
with a fixed seed so the files are reproducible.
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


BASS_NOTES = {   # sounding MIDI keys: root, octave, fifth, third, seventh, fourth/passing
    "Am9": {"r": 33, "o": 45, "f": 40, "t": 36, "s": 43, "p": 38},
    "D9": {"r": 38, "o": 50, "f": 45, "t": 42, "s": 48, "p": 40},
}
# (step, length in steps, note name, velocity); two bar shapes that alternate.
BASS_BAR_A = [(0, 3, "r", 100), (3, 1, "o", 70), (6, 2, "f", 92), (8, 2, "r", 98), (10, 1, "t", 80), (11, 3, "o", 94), (14, 2, "s", 86)]
BASS_BAR_B = [(0, 2, "r", 104), (2, 1, "r", 66), (4, 3, "f", 90), (7, 1, "p", 72), (8, 3, "r", 98), (11, 2, "o", 88), (13, 1, "s", 76), (14, 2, "f", 90)]


def bass_part(rnd, bars):
    notes = []
    for bar, name in enumerate(bars):
        shape = BASS_BAR_A if bar % 2 == 0 else BASS_BAR_B
        for step, length, which, vel in shape:
            jitter = (next(rnd) - 0.5) * 0.02
            vel = max(1, min(127, int(vel + (next(rnd) - 0.5) * 8)))
            # Some notes run a little into the next one, as a bassist plays legato.
            overlap = 1.15 if (step + bar) % 5 == 0 else 0.92
            notes.append((max(bar * 4 + step * STEP + jitter, 0.0), length * STEP * overlap, BASS_NOTES[name][which], vel))
    return notes


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    kind = sys.argv[1] if len(sys.argv) == 3 else "keys"
    out = sys.argv[-1]
    rnd = lcg(7)
    if kind == "bass":
        notes = bass_part(rnd, BARS)
        write_midi(out, notes, bpm=BPM)
        print(f"wrote {out}: {len(notes)} bass notes, {len(BARS)} bars at {BPM:g} BPM")
        return
    if kind != "keys":
        sys.exit(f"unknown part {kind!r}\n{__doc__}")
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
    write_midi(out, notes, bpm=BPM)
    print(f"wrote {out}: {len(notes)} notes, {len(BARS)} bars at {BPM:g} BPM")


if __name__ == "__main__":
    main()
