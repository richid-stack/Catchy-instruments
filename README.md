# AI-Instrument-Architect

Builds playable sampled and synthesized instruments (standalone and VST3) from real instrument recordings, tuned for Afrobeat, R&B, trap, drill and amapiano.

## Dataset tiers

- `datasets/clean/` holds CC0, CC-BY and public-domain sources. Samples from here may ship inside instruments.
- `datasets/research-private/` holds everything else: non-commercial packs, unverified licenses, and anything added by hand. It is used for analysis and training only. Release builds must never bundle samples from this tier.

`datasets/manifest.psv` lists every source with its license. `datasets/PULLED.psv` records the exact commit pulled.

## Setup

Run `scripts/fetch_datasets.sh`. It is safe to re-run and skips datasets already present.

## Adding your own downloads

Put each pack in its own folder under `datasets/research-private/<pack-name>/`, with a `SOURCE.txt` giving the URL it came from and its license if known. After a license is verified, move the pack to `datasets/clean/` and update the manifest.

## Building and playing an instrument

```
python3 -I analysis/inventory.py                      # analyze the packs (once; results are in analysis/inventory)
python3 -I instrument-builder/build.py --spec generated-instruments/afrobeat/piano/spec.json \
        --out generated-instruments/afrobeat/piano    # map the packs into zones, write instrument.json
python3 -I audio-engine/render.py --instrument generated-instruments/afrobeat/piano/instrument.json \
        --patch wurlitzer --midi generated-instruments/afrobeat/piano/demo/afrobeat-comp-108bpm.mid --out demo.wav
python3 -I audio-engine/audition.py --instrument generated-instruments/afrobeat/bass/instrument.json \
        --midi generated-instruments/afrobeat/bass/demo/afrobeat-bass-108bpm.mid --seconds 9 --out bass-audition.wav
python3 -I -m unittest discover -s tests
```

`audition.py` plays one MIDI part through every patch of an instrument in turn and prints where each patch starts.

An instrument is a set of patches. Each patch is one sample pack program mapped into zones (key range,
velocity range, root pitch, loop). Zones start from the pack's own SFZ mapping. The analysis then
sets root pitch from the detected pitch when it is within 50 cents of the declared one (this puts
naturally tuned recordings in equal temperament), trims leading silence, loops sustained samples
that end while still audible, and records each velocity layer's level. The builder matches patch
loudness by rendering test notes. Loudness follows a per-patch velocity curve whatever level the
layers were recorded at.

Spec options worth knowing (see `generated-instruments/afrobeat/bass/spec.json`):
`pitch_convention` ("sounding" moves a pack written an octave off, such as bass guitar, so each key plays
the pitch it is named after), `cc` (controller values to play an SFZ with, e.g. to pick acoustic or electric),
`include`/`exclude` (regex on sample paths), `key_range`, `level_scope` ("key" keeps velocity layers
smooth key by key), `balance` (evens out recording-level differences between keys, verified by rendering)
and the voice option `mono` (a new note cuts the last, for bass lines). One-shot samples and choke groups
(open and closed hi-hat) come from the SFZ.

Known limits of the reference engine: no key-release or pedal-noise samples yet, no effects, and loops
are checked by measurement (seam continuity) but not by ear.

A manifest that uses `research-private` samples is only built with `--allow-private`, is written as
`instrument.private.json`, and is kept out of git.

## Layout

- `analysis/`: inventory of every sample (pitch, envelope, velocity layers, round robins, SFZ links)
- `sample-mapping/`: turns an analyzed pack into zones (loop finding, tuning, velocity layers)
- `instrument-builder/`: builds instrument manifests from specs; demo MIDI generators
- `audio-engine/`: reference sampler engine (MIDI to audio); the VST3 comes later
- `generated-instruments/`: specs, built manifests and demo MIDI per genre
- `tests/`: unit tests and render tests
- `licenses/`: copies of the original license texts
