"""Minimal Standard MIDI File reader and writer, and conversion of events to played notes.

MIDI files are untrusted input: sizes and event counts are capped and malformed data raises
MidiError instead of being guessed at.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

MAX_BYTES = 8_000_000
MAX_EVENTS = 500_000


class MidiError(Exception):
    pass


@dataclass
class Note:
    time: float      # seconds from the start
    dur: float       # seconds held (pedal already applied)
    key: int
    vel: int
    channel: int = 0


def _vlq(data, pos):
    value = 0
    for _ in range(4):
        if pos >= len(data):
            raise MidiError("truncated variable-length value")
        b = data[pos]
        pos += 1
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, pos
    raise MidiError("variable-length value too long")


def read_midi(source):
    """Parse an SMF (path or bytes). Returns {'ppq', 'tempos': [(tick, usec)], 'events': [...]}.

    Each event is (tick, kind, channel, a, b) with kind 'on', 'off', 'cc' or 'program'.
    """
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
    else:
        with open(source, "rb") as fh:
            data = fh.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise MidiError("midi file too large")
    if data[:4] != b"MThd" or len(data) < 14:
        raise MidiError("not a MIDI file")
    hlen, _fmt, ntrks, division = struct.unpack(">IHHH", data[4:14])
    if division & 0x8000:
        raise MidiError("SMPTE time division is not supported")
    if division == 0:
        raise MidiError("zero ticks per quarter note")
    pos = 8 + hlen
    events, tempos = [], []
    for _ in range(ntrks):
        if data[pos:pos + 4] != b"MTrk" or pos + 8 > len(data):
            raise MidiError("bad track header")
        length = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        pos += 8
        end = pos + length
        if end > len(data):
            raise MidiError("track runs past end of file")
        tick, status = 0, 0
        while pos < end:
            delta, pos = _vlq(data, pos)
            tick += delta
            if pos >= end:
                raise MidiError("truncated event")
            b = data[pos]
            if b == 0xFF:
                if pos + 2 > end:
                    raise MidiError("truncated meta event")
                mtype = data[pos + 1]
                mlen, pos = _vlq(data, pos + 2)
                if pos + mlen > end:
                    raise MidiError("meta event runs past track")
                if mtype == 0x51 and mlen == 3:
                    tempos.append((tick, int.from_bytes(data[pos:pos + 3], "big")))
                pos += mlen
                if mtype == 0x2F:
                    break
                continue
            if b in (0xF0, 0xF7):
                slen, pos = _vlq(data, pos + 1)
                pos += slen
                continue
            if b & 0x80:
                status = b
                pos += 1
            elif not status:
                raise MidiError("data byte without status")
            kind = status & 0xF0
            ch = status & 0x0F
            need = 1 if kind in (0xC0, 0xD0) else 2
            if pos + need > end:
                raise MidiError("truncated channel event")
            a = data[pos]
            bb = data[pos + 1] if need == 2 else 0
            pos += need
            if kind == 0x90:
                events.append((tick, "on" if bb > 0 else "off", ch, a, bb))
            elif kind == 0x80:
                events.append((tick, "off", ch, a, bb))
            elif kind == 0xB0:
                events.append((tick, "cc", ch, a, bb))
            elif kind == 0xC0:
                events.append((tick, "program", ch, a, 0))
            if len(events) > MAX_EVENTS:
                raise MidiError("too many events")
        pos = end
    tempos.sort()
    return {"ppq": division, "tempos": tempos or [(0, 500_000)], "events": sorted(events, key=lambda e: e[0])}


def _tick_to_seconds(ppq, tempos):
    """Return a function mapping ticks to seconds using the tempo map."""
    marks = []  # (tick, seconds_at_tick, usec_per_quarter)
    sec, last_tick, last_us = 0.0, 0, 500_000
    for tick, us in tempos:
        sec += (tick - last_tick) * last_us / 1e6 / ppq
        marks.append((tick, sec, us))
        last_tick, last_us = tick, us
    if not marks or marks[0][0] > 0:
        marks.insert(0, (0, 0.0, 500_000))

    def convert(tick):
        base = marks[0]
        for m in marks:
            if m[0] <= tick:
                base = m
            else:
                break
        return base[1] + (tick - base[0]) * base[2] / 1e6 / ppq

    return convert


def notes_from_midi(midi, pedal=True, channels=None):
    """Turn parsed events into Note objects, holding notes while the sustain pedal (CC 64) is down."""
    to_s = _tick_to_seconds(midi["ppq"], midi["tempos"])
    pedal_down = {}   # channel -> time pedal went down
    pedal_ups = {}    # channel -> sorted list of (down_time, up_time)
    held = {}         # (channel, key) -> list of (start, vel)
    raw = []          # (start, end, key, vel, channel)
    last = 0.0
    for tick, kind, ch, a, b in midi["events"]:
        if channels is not None and ch not in channels:
            continue
        t = to_s(tick)
        last = max(last, t)
        if kind == "on":
            held.setdefault((ch, a), []).append((t, b))
        elif kind == "off":
            stack = held.get((ch, a))
            if stack:
                start, vel = stack.pop(0)
                raw.append((start, t, a, vel, ch))
        elif kind == "cc" and a == 64 and pedal:
            if b >= 64 and ch not in pedal_down:
                pedal_down[ch] = t
            elif b < 64 and ch in pedal_down:
                pedal_ups.setdefault(ch, []).append((pedal_down.pop(ch), t))
    for ch, down in pedal_down.items():
        pedal_ups.setdefault(ch, []).append((down, last + 2.0))
    for (ch, key), stack in held.items():
        for start, vel in stack:
            raw.append((start, last + 0.25, key, vel, ch))
    notes = []
    for start, end, key, vel, ch in raw:
        for down, up in pedal_ups.get(ch, []):
            if down <= end < up:
                end = up
                break
        notes.append(Note(time=start, dur=max(end - start, 0.01), key=key, vel=vel, channel=ch))
    notes.sort(key=lambda n: (n.time, n.key))
    return notes


def write_midi(path, notes, bpm=100.0, ppq=480, pedal_events=()):
    """Write a format-0 SMF. notes: (start_beat, dur_beats, key, vel); pedal_events: (beat, 0 or 127)."""
    ev = []
    for start, dur, key, vel in notes:
        t0 = int(round(start * ppq))
        ev.append((t0, 1, bytes([0x90, key, int(vel)])))
        ev.append((max(int(round((start + dur) * ppq)), t0 + 1), 0, bytes([0x80, key, 0])))
    for beat, value in pedal_events:
        ev.append((int(round(beat * ppq)), 0, bytes([0xB0, 64, int(value)])))
    ev.sort(key=lambda e: (e[0], e[1]))

    def vlq(n):
        out = [n & 0x7F]
        n >>= 7
        while n:
            out.append((n & 0x7F) | 0x80)
            n >>= 7
        return bytes(reversed(out))

    track = bytearray()
    track += vlq(0) + b"\xFF\x51\x03" + int(60_000_000 / bpm).to_bytes(3, "big")
    last = 0
    for tick, _, raw in ev:
        track += vlq(tick - last) + raw
        last = tick
    track += vlq(0) + b"\xFF\x2F\x00"
    with open(path, "wb") as fh:
        fh.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq))
        fh.write(b"MTrk" + struct.pack(">I", len(track)) + bytes(track))
