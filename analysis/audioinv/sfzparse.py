"""A small, defensive SFZ reader that turns .sfz files into flat region dicts.

SFZ files are untrusted input: sizes are capped, includes must stay inside the pack, and
sample paths that would escape the pack are dropped.
"""
from __future__ import annotations

import os
import posixpath
import re

MAX_BYTES = 8_000_000
MAX_INCLUDE_DEPTH = 5
MAX_INCLUDES = 2000
MAX_EXPANDED_BYTES = 64_000_000

_NOTE = re.compile(r"^([a-gA-G])(#|b)?(-?\d)$")
_BASE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_HEADER = re.compile(r"(<[A-Za-z_]+>)")
_OPCODE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=")


class SfzError(Exception):
    pass


def key_to_midi(value):
    """Convert an SFZ key value (60, c4, c#4, db3) to a MIDI number; C4 = 60."""
    v = str(value).strip()
    try:
        return int(float(v))
    except (ValueError, OverflowError):
        pass
    m = _NOTE.match(v)
    if not m:
        return None
    acc = {"#": 1, "b": -1, None: 0}[m.group(2)]
    return 12 * (int(m.group(3)) + 1) + _BASE[m.group(1).lower()] + acc


def _int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _inside(root: str, real: str) -> bool:
    return real == root or real.startswith(root + os.sep)


def _read_text(path: str) -> str:
    with open(path, "rb") as fh:
        raw = fh.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise SfzError("sfz file too large")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


class _Defs:
    """#define table. Names are replaced at the point of use, longest name first."""

    def __init__(self, initial=None):
        self.table: dict = {}
        self._rx = None
        for name, value in (initial or {}).items():
            self.set(name, value)

    def set(self, name: str, value: str) -> None:
        if name not in self.table:
            self._rx = None
        self.table[name] = value

    def apply(self, text: str) -> str:
        if "$" not in text or not self.table:
            return text
        if self._rx is None:
            names = sorted(self.table, key=len, reverse=True)
            self._rx = re.compile("(?:" + "|".join(re.escape(n) for n in names) + r")(?![A-Za-z0-9_])")
        for _ in range(3):  # a define may refer to another define
            new = self._rx.sub(lambda m: self.table[m.group(0)], text)
            if new == text:
                break
            text = new
        return text


def read_bank_defines(pack_root: str) -> dict:
    """Read <Define name="$x" value="..."/> entries from a top-level sforzando *.bank.xml.

    Some packs keep their sample folder variable there instead of in the .sfz files.
    """
    out: dict = {}
    try:
        names = sorted(n for n in os.listdir(pack_root) if n.lower().endswith(".bank.xml"))
    except OSError:
        return out
    for name in names[:3]:
        try:
            with open(os.path.join(pack_root, name), "rb") as fh:
                raw = fh.read(1_000_000)
        except OSError:
            continue
        text = raw.decode("utf-8", errors="replace")
        for m in re.finditer(r'<Define\s+name="(\$[A-Za-z0-9_]+)"\s+value="([^"]*)"', text):
            out[m.group(1)] = m.group(2)
    return out


def _find_ci(base: str, rel: str, root: str):
    """Find base/rel, ignoring letter case, without leaving the pack. Returns a path or None."""
    direct = os.path.join(base, rel)
    if os.path.exists(direct):
        return direct
    cur = base
    for part in rel.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            cur = os.path.dirname(cur)
            continue
        if not _inside(root, os.path.realpath(cur)):
            return None
        try:
            names = os.listdir(cur)
        except OSError:
            return None
        hit = next((n for n in names if n.lower() == part.lower()), None)
        if hit is None:
            return None
        cur = os.path.join(cur, hit)
    return cur if os.path.exists(cur) else None


_INCLUDE = re.compile(r'(#include\s+"[^"]*")')


def _expand(path, root, depth, state, defs, warnings):
    if depth > MAX_INCLUDE_DEPTH:
        warnings.append("include nesting too deep; skipped")
        return []
    real = os.path.realpath(path)
    if not _inside(root, real):
        if depth == 0:
            raise SfzError("sfz file is outside the pack")
        warnings.append("include outside pack ignored")
        return []
    text = _read_text(real)
    state["bytes"] += len(text)
    if state["bytes"] > MAX_EXPANDED_BYTES:
        warnings.append("expanded size limit reached; rest ignored")
        return []
    if depth > 0:
        state["included"].add(os.path.relpath(real, root).replace(os.sep, "/"))
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    base_dir = os.path.dirname(real)
    out = []
    for line in text.splitlines():
        line = line.split("//", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#define"):
            parts = line.split(None, 2)
            if len(parts) >= 2:
                defs.set(parts[1], parts[2].strip() if len(parts) == 3 else "")
            continue
        # Includes may sit in the middle of a line ("<group> #include ... lovel=1 #include ...").
        for piece in _INCLUDE.split(line):
            if piece.startswith("#include"):
                m = re.search(r'"([^"]+)"', piece)
                if not m:
                    continue
                state["includes"] += 1
                if state["includes"] > MAX_INCLUDES:
                    if state["includes"] == MAX_INCLUDES + 1:
                        warnings.append("too many includes; rest ignored")
                    continue
                inc = defs.apply(m.group(1)).replace("\\", "/")
                cand = None
                for base in (base_dir, state["main_dir"], root):
                    cand = _find_ci(base, inc, root)
                    if cand:
                        break
                if cand:
                    out.extend(_expand(cand, root, depth + 1, state, defs, warnings))
            elif piece.strip():
                out.append(defs.apply(piece))
    return out


def _opcodes(chunk: str, target: dict) -> None:
    matches = list(_OPCODE.finditer(chunk))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chunk)
        target[m.group(1).lower()] = chunk[m.end():end].strip()


def _emit(regions, merged, ctrl, sfz_rel, main_dir_rel, warnings):
    sample = merged.get("sample")
    if not sample or sample.startswith("*"):
        return  # no file (generator or empty)
    offset = _int(ctrl.get("note_offset"), 0) + 12 * _int(ctrl.get("octave_offset"), 0)
    default_path = ctrl.get("default_path", "").replace("\\", "/")
    rel = posixpath.normpath(posixpath.join(main_dir_rel, default_path, sample.replace("\\", "/")))
    if rel.startswith("..") or posixpath.isabs(rel):
        if len(warnings) < 20:
            warnings.append(f"sample path outside pack dropped: {sample[:80]}")
        return

    def key(name):
        v = key_to_midi(merged[name]) if name in merged else None
        return v + offset if v is not None else None

    k = key("key")
    lokey = key("lokey")
    hikey = key("hikey")
    root = key("pitch_keycenter")
    if k is not None:
        lokey = k if lokey is None else lokey
        hikey = k if hikey is None else hikey
        root = k if root is None else root
        if "lokey" not in merged and "hikey" not in merged:
            lokey = hikey = k
    region = {
        "sfz": sfz_rel,
        "sample": rel,
        "lokey": 0 if lokey is None else lokey,
        "hikey": 127 if hikey is None else hikey,
        "root": root,
        "lovel": _int(merged.get("lovel"), 0),
        "hivel": _int(merged.get("hivel"), 127),
        "lorand": _float(merged.get("lorand"), 0.0),
        "hirand": _float(merged.get("hirand"), 1.0),
        "seq_position": _int(merged.get("seq_position"), 1),
        "seq_length": _int(merged.get("seq_length"), 1),
    }
    for name in ("loop_mode", "trigger", "group", "off_by", "transpose", "tune", "offset", "end",
                 "loop_start", "loop_end", "sw_last", "sw_lokey", "sw_hikey", "sw_previous",
                 "volume", "ampeg_release"):
        if name in merged:
            region[name] = merged[name]
    cc = {n: _int(v, None) for n, v in merged.items() if n.startswith(("locc", "hicc"))}
    if cc:
        region["cc"] = cc
        region["cc_set"] = {int(n[6:]): _int(v, 0) for n, v in ctrl.items()
                            if n.startswith("set_cc") and n[6:].isdigit()}
    regions.append(region)


def parse_sfz(sfz_path: str, pack_root: str, defines=None, included=None):
    """Parse one SFZ file. Returns (regions, warnings). Raises SfzError for unusable files.

    defines: extra #define values supplied by the host (for example from a *.bank.xml).
    included: optional set that receives the pack-relative path of every file this one includes,
    so callers can tell root instruments from fragments that only exist to be included.
    """
    root = os.path.realpath(pack_root)
    main_real = os.path.realpath(sfz_path)
    warnings: list = []
    state = {"includes": 0, "bytes": 0, "included": set(), "main_dir": os.path.dirname(main_real)}
    lines = _expand(sfz_path, root, 0, state, _Defs(defines), warnings)
    if included is not None:
        included.update(state["included"])
    text = "\n".join(lines)

    sfz_rel = os.path.relpath(main_real, root).replace(os.sep, "/")
    main_dir_rel = posixpath.dirname(sfz_rel)

    ctrl: dict = {}
    glob: dict = {}
    mast: dict = {}
    grp: dict = {}
    reg = None
    cur = None
    regions: list = []

    def flush():
        if reg is not None:
            merged = {**glob, **mast, **grp, **reg}
            _emit(regions, merged, ctrl, sfz_rel, main_dir_rel, warnings)

    for tok in _HEADER.split(text):
        stripped = tok.strip()
        if not stripped:
            continue
        if _HEADER.fullmatch(stripped):
            name = stripped[1:-1].lower()
            flush()
            reg = None
            if name == "region":
                reg = {}
                cur = reg
            elif name == "group":
                grp = {}
                cur = grp
            elif name == "master":
                mast, grp = {}, {}
                cur = mast
            elif name == "global":
                glob, mast, grp = {}, {}, {}
                cur = glob
            elif name == "control":
                cur = ctrl
            else:
                cur = {}
        elif cur is not None:
            _opcodes(tok, cur)
    flush()
    return regions, warnings[:20]


def cc_active(region, overrides=None) -> bool:
    """True if the region's controller ranges (locc/hicc) admit the controller values in force.

    Values come from the pack's own set_cc defaults (zero when unset), then from overrides
    ({cc_number: value}). This is how a pack's blend, voice-count and mic switches choose which
    regions play.
    """
    cc = region.get("cc")
    if not cc:
        return True
    values = dict(region.get("cc_set") or {})
    values.update(overrides or {})
    for name, bound in cc.items():
        if bound is None or not name[4:].isdigit():
            continue
        now = values.get(int(name[4:]), 0)
        if name.startswith("locc") and now < bound:
            return False
        if name.startswith("hicc") and now > bound:
            return False
    return True
