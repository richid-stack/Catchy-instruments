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

## Layout

- `analysis/`: pitch, velocity-layer, round-robin and loop-point detection
- `sample-mapping/`: SFZ generation
- `instrument-builder/`: instrument definitions per genre
- `audio-engine/`: standalone engine and VST3
- `generated-instruments/`: build output
- `tests/`: MIDI render tests
- `licenses/`: copies of the original license texts
