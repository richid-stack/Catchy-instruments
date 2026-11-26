"""Pull musical hints (note, velocity, round robin, articulation) out of sample paths.

These are hints only. The audio analysis is the source of truth for pitch, and the
inventory reports where the two disagree.
"""
from __future__ import annotations

import re

_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACC = {None: 0, "#": 1, "♯": 1, "b": -1, "♭": -1}
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Scientific pitch notation, C4 = MIDI 60. The letter must not follow another letter,
# so "SD2" or "Bass3" are not read as notes.
NOTE_RE = re.compile(r"(?<![A-Za-z])([A-G])(#|♯|b|♭)?(-?\d)(?![0-9])")
VEL_RE = re.compile(r"(?<![A-Za-z])(?:velocity|vel|vl|layer|lyr|v)[_\- ]?(\d{1,3})(?![0-9])", re.I)
RR_RE = re.compile(r"(?<![A-Za-z])(?:rr|rand|random|take|variation|var|alt)[_\- ]?(\d{1,2})(?![0-9])", re.I)
# Dynamics are matched case-sensitively in lowercase so "F" (a note) is not read as forte.
DYN_RE = re.compile(r"(?<![A-Za-z0-9])(ppp|pp|mp|mf|fff|ff|p|f)(?![A-Za-z0-9])")

_DYN_WORDS = {
    "soft": "soft", "quiet": "soft", "light": "soft",
    "medium": "medium", "med": "medium", "mid": "medium",
    "hard": "hard", "loud": "hard", "heavy": "hard",
}

_ARTIC = {
    "pizz": "pizzicato", "pizzicato": "pizzicato",
    "stac": "staccato", "staccato": "staccato", "spic": "spiccato", "spiccato": "spiccato",
    "legato": "legato", "sus": "sustain", "sustain": "sustain", "sustained": "sustain",
    "trem": "tremolo", "tremolo": "tremolo",
    "vib": "vibrato", "vibrato": "vibrato", "nonvib": "non-vibrato", "novib": "non-vibrato",
    "marc": "marcato", "marcato": "marcato", "flutter": "flutter",
    "mute": "muted", "muted": "muted", "dead": "muted", "palm": "palm-muted",
    "harm": "harmonic", "harmonic": "harmonic", "harmonics": "harmonic",
    "slap": "slap", "pop": "pop", "ghost": "ghost", "rim": "rim", "rimshot": "rimshot",
    "open": "open", "closed": "closed", "roll": "roll", "release": "release", "rel": "release",
    "arco": "arco", "bend": "bend", "slide": "slide", "gliss": "glissando", "glissando": "glissando",
    "trill": "trill", "shake": "shake", "swell": "swell", "cresc": "crescendo",
    "crescendo": "crescendo", "fall": "fall", "doit": "doit", "bow": "arco",
    "damped": "muted", "choke": "choke", "bell": "bell",
}


def note_to_midi(letter: str, accidental: str | None, octave: int) -> int:
    return 12 * (octave + 1) + _BASE[letter] + _ACC[accidental]


def midi_to_name(midi: int) -> str:
    midi = int(midi)
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


# Some packs name files by note alone ("db4mp", "c2ff", "057_A3"). Accept a lowercase or
# uppercase note only at the very start of the name, optionally after a number prefix, so that
# words such as "bd1" or "Bass3" are still rejected.
LEAD_NOTE_RE = re.compile(r"^(?:\d{1,3}[_\-. ]+)?([a-gA-G])(#|b)?(-?\d)(?![0-9])")


def _find_note(text: str, lead: bool = False):
    m = NOTE_RE.search(text)
    if not m and lead:
        m = LEAD_NOTE_RE.match(text)
    if not m:
        return None
    letter = m.group(1).upper()
    return note_to_midi(letter, m.group(2), int(m.group(3)))


def parse_name(relpath: str) -> dict:
    """Return hints parsed from a path relative to the pack root. Empty fields are omitted."""
    parts = relpath.replace("\\", "/").split("/")
    stem = parts[-1].rsplit(".", 1)[0]
    out: dict = {}

    midi = _find_note(stem, lead=True)
    src = "file"
    if midi is None and len(parts) > 1:
        midi = _find_note(parts[-2])
        src = "folder"
    if midi is not None:
        out["name_midi"] = midi
        out["name_note"] = midi_to_name(midi)
        out["name_note_src"] = src

    m = VEL_RE.search(stem)
    if m:
        out["name_vel"] = int(m.group(1))
    m = DYN_RE.search(stem)
    if m:
        out["name_dyn"] = m.group(1)
    else:
        for tok in re.findall(r"[A-Za-z]+", stem.lower()):
            if tok in _DYN_WORDS:
                out["name_dyn"] = _DYN_WORDS[tok]
                break
    m = RR_RE.search(stem)
    if m:
        out["name_rr"] = int(m.group(1))

    # Articulations can live in folder names, so look at every path part.
    artics = set()
    for part in parts:
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", part.rsplit(".", 1)[0] if part is parts[-1] else part)
        for tok in re.findall(r"[A-Za-z]+", spaced.lower()):
            if tok in _ARTIC:
                artics.add(_ARTIC[tok])
    if artics:
        out["artic"] = sorted(artics)
    return out
