#!/bin/sh
# Rebuild the harvested notes this instrument is made from (needs the loop packs in datasets/research-private),
# then re-inventory them and build:
#   rm -f analysis/inventory/harvest-guitar-loops.jsonl.gz analysis/reports/harvest-guitar-loops.json
#   python3 -I analysis/inventory.py --datasets datasets --out analysis/inventory --reports analysis/reports
#   python3 -I instrument-builder/build.py --spec generated-instruments/afrobeat/guitar-loops/spec.json \
#       --datasets datasets --inventory analysis/inventory --out generated-instruments/afrobeat/guitar-loops --allow-private
set -e
H="python3 -I sample-mapping/harvest_loops.py"
FX='(?i)(tape|lofi|echo|filter|stack|fx|reverse|ambient|resampled|buzzy|wah|chops|cuts|layer|muffled|chords|amped)'
$H --family inq  --packs 'inqboi-afrolistic-guitar-pack' --include '/Dry/' --exclude 'Scream'
$H --family elec --packs '(91vocals-.*|pure-water-.*)' --include '/Electric_Guitar/' --exclude "$FX"
$H --family acou --packs '(91vocals-.*|pure-water-.*)' --include '/Acoustic_Guitar/' --exclude "$FX"
