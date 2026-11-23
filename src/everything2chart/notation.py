"""Notation output: MusicXML + MIDI (music21) and a readable jianpu sketch.

M0 deliberately does not depend on MuseScore/LilyPond — those are optional
renderers plugged in later for PDF.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .types import Arrangement, NoteEvent

log = logging.getLogger(__name__)

MAJOR_SHARPS = ["C", "G", "D", "A", "E", "B", "F#"]
MAJOR_FLATS = ["C", "F", "Bb", "Eb", "Ab", "Db", "Gb"]
MINOR_SHARPS = ["Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m"]
MINOR_FLATS = ["Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm", "Abm"]


def _quantised_events(events: list[NoteEvent], bpm: float, grid: float = 0.125):
    """Seconds -> (offset_q, quarter_length) on a grid expressed in quarter notes."""
    spq = 60.0 / bpm
    out = []
    for e in events:
        off = round(e.start / spq / grid) * grid
        dur = max(round(e.dur / spq / grid) * grid, grid)
        out.append((off, dur, e))
    return out


def _clean_pieces(ql: float, off: float = 0.0, bar: float = 4.0) -> list[float]:
    """Split a quarter length into pieces that print with at most one dot.

    music21 writes 3.75 beats as a triple-dotted half note: arithmetically
    right, unreadable at sight, and 670 of the 1502 left-hand notes came out
    that way.  Values needing two dots (1.75, 3.5, 3.75, 7...) are cut greedily
    into values needing at most one (4, 3, 2, 1.5, 1, 0.75, 0.5, 0.375, 0.25,
    0.125) and tied back together, so the page reads as a chain a player can
    count while the sounding length -- and the MIDI, which merges ties -- is
    exactly what it was.

    The cut has to respect bar lines as well as note values.  music21 splits a
    note that runs past a bar line by itself, and its remainder is whatever is
    left: a 3.75 starting on beat 3 became 1.75 + 2.0, and 1.75 is a
    double-dotted quarter -- the same unreadable page by a different route.
    """
    from music21 import duration as m21duration

    def dots(v: float) -> int:
        try:
            return m21duration.Duration(quarterLength=v).dots
        except Exception:
            return 0

    clean = (4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125)
    pieces: list[float] = []
    left = ql
    at = off
    bar = bar if bar > 0 else 4.0
    while left > 1e-9:
        to_bar = (int((at + 1e-9) / bar) + 1) * bar - at
        cap = min(left, to_bar if to_bar > 1e-9 else bar)
        take = next((c for c in clean if c <= cap + 1e-9), clean[-1])
        # a remainder too small to be a clean value is folded into the last piece
        if take > cap:
            take = cap
        pieces.append(take)
        left -= take
        at += take
    if len(pieces) > 1 and any(dots(v) > 1 for v in pieces):
        return [ql]
    return pieces or [ql]


def _attach_finger(note_obj, finger: int) -> None:
    """Print a finger number above/next to one note, if this note has one.

    music21's ``Fingering`` is the standard representation and Verovio renders
    it as a small digit; nothing about the pitch, duration or beaming changes.
    """
    if not finger or not 1 <= int(finger) <= 5:
        return
    from music21 import articulations

    fi = articulations.Fingering(int(finger))
    fi.placement = "above"
    note_obj.articulations.append(fi)


def _attach_fingers(note_obj, fingers: list[int]) -> None:
    """Print one digit per chord note, in the chord's own note order.

    ``music21`` writes the n-th ``Fingering`` in a Chord's ``articulations``
    as the digit of the n-th chord note and ignores fingerings hung on the
    members, so the whole column has to be attached at the chord level.  A
    partly fingered chord cannot be represented -- the digits would land on
    the wrong notes -- so it prints no digit at all.
    """
    if not fingers or not all(1 <= int(f) <= 5 for f in fingers):
        return
    from music21 import articulations

    for f in fingers:
        fi = articulations.Fingering(int(f))
        fi.placement = "above"
        note_obj.articulations.append(fi)


def _insert_group(part, group: list[NoteEvent], off: float, dur: float,
                  bar: float = 4.0) -> None:
    """Insert one strike, split into tied pieces that need at most one dot."""
    from music21 import chord as m21chord
    from music21 import note, tie

    pieces = _clean_pieces(dur, off, bar)
    at = off
    for i, piece in enumerate(pieces):
        if len(pieces) == 1:
            tp = None
        elif i == 0:
            tp = tie.Tie("start")
        elif i == len(pieces) - 1:
            tp = tie.Tie("stop")
        else:
            tp = tie.Tie("continue")
        if len(group) == 1:
            # note.Note(<int>) keeps a pitch.Accidental('natural') on white keys
            # in music21 10.5, and the writer then prints a natural on the first
            # appearance of every step in each measure -- 1897 of them in the
            # easy tier, where the left hand has no black keys at all.  Setting
            # the midi value on a blank note spells the same pitch without an
            # accidental to print.
            n = note.Note()
            n.pitch.midi = group[0].pitch
            n.quarterLength = piece
            if group[0].velocity:
                n.volume.velocity = int(np.clip(group[0].velocity, 20, 127))
            if tp is not None:
                n.tie = tp
            _attach_finger(n, group[0].finger if i == 0 else 0)
            part.insert(at, n)
        else:
            c = m21chord.Chord()
            for e in sorted(group, key=lambda x: x.pitch):
                q = note.Note()
                q.pitch.midi = e.pitch
                c.add(q)
            # music21 drops a Fingering hung on a chord *member* note and keeps
            # only the n-th Fingering on the Chord itself as the n-th printed
            # digit, matched to chord notes in sorted-pitch order.  Attaching to
            # the members produced a MusicXML file with zero <fingering>.
            _attach_fingers(
                c,
                [e.finger for e in sorted(group, key=lambda x: x.pitch)]
                if i == 0
                else [],
            )
            c.quarterLength = piece
            if tp is not None:
                c.tie = tp
            part.insert(at, c)
        at += piece


def to_music21(arr: Arrangement):
    from music21 import chord as m21chord
    from music21 import clef, key, meter, stream, tempo

    score = stream.Score()
    score.insert(0, tempo.MetronomeMark(number=round(arr.info.bpm)))

    for name, events, cl, rng in (
        ("RH", arr.right, clef.TrebleClef(), (60, 84)),
        ("LH", arr.left, clef.BassClef(), (36, 60)),
    ):
        part = stream.Part(id=name)
        part.insert(0, cl)
        part.insert(0, meter.TimeSignature(f"{arr.info.beats_per_bar}/4"))
        try:
            k = key.Key(arr.key_signature)
            part.insert(0, k)
        except Exception:
            pass
        by_offset: dict[float, list[NoteEvent]] = {}
        for off, dur, e in _quantised_events(events, arr.info.bpm):
            by_offset.setdefault(off, []).append(e)
        for off in sorted(by_offset):
            group = by_offset[off]
            dur = max(round(max(e.dur for e in group) / (60.0 / arr.info.bpm) * 8) / 8, 0.125)
            _insert_group(part, group, off, dur, float(arr.info.beats_per_bar))
        score.insert(0, part)
    return score


def _renumber_voices(xml: Path) -> None:
    """Pin part ids and (if present) shift music21's 0-based voice to 1.

    music21 emits ``<voice>0</voice>`` for single-voice parts; Verovio wants
    1-based layers.  This score is one voice per part, so the current output
    carries no ``<voice>`` element at all and that branch is a no-op here -- it
    stays for the multi-voice case.  The shift is written as "0 -> 1" rather
    than "n -> n+1" so running twice cannot push a voice to 2.
    """
    import re

    text = xml.read_text(encoding="utf-8")
    # music21 names every part with a fresh uuid, so two runs of the same score
    # produced different files and "did my change alter the output" could only be
    # answered by massaging the diff.  Rename the parts in score order: the ids
    # are internal references (this is the only place that defines and uses
    # them), and P1/P2 is what any hand-written MusicXML would say.
    ids: dict[str, str] = {}

    def _stable(m: re.Match[str]) -> str:
        old = m.group(1)
        return f'id="{ids.setdefault(old, f"P{len(ids) + 1}")}"'

    before = text
    text = re.sub(r'\bid="(P[0-9a-f]{32})"', _stable, text)
    text = re.sub(r"<voice>0</voice>", "<voice>1</voice>", text)
    # Write when *either* rewrite changed something.  Gating the write on the
    # voice shift alone silently dropped the id rewrite on any score whose
    # voices were already 1-based.
    if text != before:
        xml.write_text(text, encoding="utf-8")


def _pin_encoding_date(xml: Path) -> None:
    """Make ``<encoding-date>`` reproducible when the caller asks for it.

    music21 writes ``datetime.date.today()`` into ``<encoding-date>`` (see
    ``m21ToXml.ScoreExporter.setEncoding``) with no knob to override it, so two
    runs of an unchanged score always differed by that one line and byte-level
    reproducibility could not be checked without massaging the diff.  Honour
    ``SOURCE_DATE_EPOCH`` (the reproducible-builds convention): set it and the
    date is derived from that instant, unset it and nothing changes.
    """
    import os
    import re
    from datetime import datetime, timezone

    raw = os.environ.get("SOURCE_DATE_EPOCH", "").strip()
    if not raw:
        return
    try:
        stamp = datetime.fromtimestamp(int(raw), tz=timezone.utc).date().isoformat()
    except (ValueError, OSError):
        return
    text = xml.read_text(encoding="utf-8")
    text = re.sub(r"<encoding-date>[^<]*</encoding-date>",
                  f"<encoding-date>{stamp}</encoding-date>", text)
    xml.write_text(text, encoding="utf-8")


def write_score(arr: Arrangement, out_dir: str | Path, stem: str = "score") -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score = to_music21(arr)
    paths = {}
    xml = out_dir / f"{stem}-{arr.preset}.musicxml"
    score.write("musicxml", fp=str(xml))
    _renumber_voices(xml)
    _pin_encoding_date(xml)
    paths["musicxml"] = xml
    midi = out_dir / f"{stem}-{arr.preset}.mid"
    score.write("midi", fp=str(midi))
    paths["midi"] = midi
    jp = out_dir / f"{stem}-{arr.preset}.jianpu.txt"
    jp.write_text(render_jianpu(arr), encoding="utf-8")
    paths["jianpu"] = jp
    return paths


# --------------------------------------------------------------------------- jianpu

_PC_TO_SOLFA = {0: "1", 2: "2", 4: "3", 5: "4", 7: "5", 9: "6", 11: "7"}
# jianpu calls the minor tonic "6" of its relative major, so a flat key needs b-degrees
FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb",
             "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm", "Abm", "Dbm"}
_IS_MINOR = {"m", "min"}


_PC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def _key_parts(key_sig: str) -> tuple[int, bool]:
    """Key signature -> (pitch class of the jianpu "1", is minor).

    A flat key has to be looked up in the flat spelling table: ``Ebm`` is
    ``Eb``, not ``C``.  The previous single sharp-spelled table silently
    resolved every flat tonic to ``C``, which put ``Ebm`` a fourth off and
    printed the wrong ``1=`` line above a correct score.
    """
    minor = key_sig.endswith("m") and len(key_sig) > 1
    name = key_sig[:-1] if minor else key_sig
    for i, spelled in enumerate(zip(_PC_SHARP, _PC_FLAT)):
        if name in spelled:
            tonic = i + (3 if minor else 0)  # relative major: Am -> C, Dm -> F
            return tonic % 12, minor
    # A name neither table knows used to fall through as pc 0 and print the page
    # under "1=C".  Refusing is the only answer that cannot silently transpose.
    raise ValueError(f"unknown key signature {key_sig!r}")


def _solfa(pitch: int, tonic: int, flat: bool = False) -> str:
    pc = (pitch - tonic) % 12
    base = _PC_TO_SOLFA.get(pc)
    if base is None:            # chromatic note: spell it the way the key signature leans
        if flat:
            upper = _PC_TO_SOLFA.get((pc + 1) % 12)
            base = ("b" + upper) if upper else ("#" + _PC_TO_SOLFA.get((pc - 1) % 12, "?"))
        else:
            base = "#" + _PC_TO_SOLFA.get((pc - 1) % 12, "?")
    # The plain octave is the one that starts at the tonic, so a note below the
    # tonic is in the octave below even when its pitch class is high (a fifth
    # below the tonic is "5," and not "5").
    octave = (pitch - tonic - 60) // 12
    if octave > 0:
        return base + "'" * octave
    if octave < 0:
        return base + "," * (-octave)
    return base


def render_jianpu(arr: Arrangement, beats_per_bar: int | None = None,
                  sub: int = 4) -> str:
    """Plain-text numbered notation (简谱) on a ``sub``-slot-per-beat grid.

    The header legend is the contract: an empty ``_`` suffix means a full beat,
    one underscore halves it, two quarter it, ``-`` holds the previous note for
    one more beat, ``[ ]`` groups notes struck together and ``( )`` is the left
    hand.  ``sub=4`` engraves down to sixteenth notes, which is the shortest
    value any of the three tiers needs to stay readable; the staff and MIDI stay
    exact.
    """
    bpb = beats_per_bar or arr.info.beats_per_bar
    spq = 60.0 / arr.info.bpm
    tonic, minor = _key_parts(arr.key_signature)
    flat = arr.key_signature in FLAT_KEYS
    tonic_name = (_PC_FLAT if flat else _PC_SHARP)[tonic]
    lines = [
        f"# 简谱 {arr.preset}  1={tonic_name}  key={arr.key_signature}  bpm={round(arr.info.bpm)}  "
        f"{bpb}/4",
        "# 数字=级数  '=高八度  ,=低八度  b=降 #=升  _=减半(一条=八分,两条=十六分)  "
        ".=附点(加一半)  - = 延长一拍  0 = 休止  [+]=同时弹  ( )=左手",
        "",
    ]

    total_beats = max(1, int(round(arr.info.duration / spq)))

    # A text line per hand can only carry one duration per strike, so "how long"
    # is decided here and the reader (tools/jpround.py) has to agree: a note that
    # is still sounding under a later strike cannot be written, and is cut short
    # rather than silently extended to its neighbour's length.
    strikes: dict[str, dict[int, dict[int, list[str]]]] = {"R": {}, "L": {}}
    lengths: dict[str, dict[int, list[float]]] = {"R": {}, "L": {}}
    for hand, events in (("R", arr.right), ("L", arr.left)):
        for e in events:
            beat_pos = e.start / spq
            beat = int(beat_pos + 1e-6)
            slot = min(sub - 1, int((beat_pos - beat) * sub + 1e-6))
            beats = e.dur / spq
            whole, rest = divmod(max(1, int(round(beats * sub))), sub)
            dot, under = "", 0
            if rest:
                for k in range(4):
                    if rest * (2 ** k) == sub:
                        under = k
                        break
                    if rest * (2 ** k) * 2 == sub * 3:
                        under, dot = k, "."
                        break
                else:
                    under = max(0, min(2, int(round(np.log2(sub / rest)))))
            token = _solfa(e.pitch, tonic, flat) + dot + "_" * under
            bucket = strikes[hand].setdefault(beat, {}).setdefault(slot, [])
            if token not in bucket:             # the same pitch twice is a tie
                bucket.append(token)
                lengths[hand].setdefault(beat, []).append(beats)

    def beat_text(hand: str, beat: int, hold_until: int) -> tuple[str, int]:
        slots = strikes[hand].get(beat)
        if slots:
            columns = ["[%s]" % "+".join(strikes[hand][beat][k]) if len(strikes[hand][beat][k]) > 1
                       else strikes[hand][beat][k][0] for k in sorted(strikes[hand][beat])]
            # The token only carries the fractional part of the length; every
            # "-" adds exactly one beat, so the hold window is the strike's
            # whole-beat count.  Taking the shortest note of the strike keeps a
            # "-" from stretching a short note into its neighbour's length.
            # A token with no dot and no underscore already means one whole
            # beat, so such a strike needs one dash fewer than its beat count.
            holes = min(max(0, int(round(bl * sub)) // sub
                            - (1 if int(round(bl * sub)) % sub == 0 else 0))
                        for bl in lengths[hand][beat])
            return " ".join(columns), beat + holes
        if beat <= hold_until:
            return "-", hold_until
        return "", beat - 1

    hold_until = {"R": -1, "L": -1}
    for bar in range(0, total_beats, bpb):
        cells = []
        for b in range(bar, min(bar + bpb, total_beats)):
            rh, hold_until["R"] = beat_text("R", b, hold_until["R"])
            lh, hold_until["L"] = beat_text("L", b, hold_until["L"])
            if not rh and not lh:
                cells.append("0")
            elif rh and lh:
                cells.append(rh + "(" + lh + ")")
            else:
                cells.append(rh or "(" + lh + ")")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- staff render

def _cairo_lib() -> str | None:
    """Locate a Cairo dylib without relying on ctypes.util.find_library.

    On macOS, find_library() only searches dyld's default paths and misses a
    Homebrew install, so cairocffi raises "no library called libcairo-2".
    """
    import glob

    for pat in ("/opt/homebrew/lib/libcairo.[0-9]*.dylib",
                "/usr/local/lib/libcairo.[0-9]*.dylib"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def _png_from_svg(svg: str, png: Path, scale: float) -> None:
    """Rasterise one SVG page; import-time Cairo lookup is patched if needed."""
    import ctypes.util

    lib = _cairo_lib()
    if lib:
        orig = ctypes.util.find_library
        ctypes.util.find_library = lambda n: lib if "cairo" in n else orig(n)
        try:
            import cairosvg
        finally:
            ctypes.util.find_library = orig
    else:
        import cairosvg
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png), scale=scale)


def render_staff(musicxml: Path, out_dir: Path, stem: str, fmt: str = "png",
                 page_width: int = 2100, scale: float = 2.0) -> list[Path]:
    """Engrave MusicXML to page images with Verovio (BSD, fully offline).

    Pages are written and released one at a time: a whole-book run in a single
    process peaks at several hundred MB and has been OOM-killed on this host.
    """
    import gc
    import os

    import verovio
    os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")

    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": page_width, "pageHeight": 2970, "scale": 42,
        "adjustPageHeight": True, "footer": "none", "header": "none",
        "spacingSystem": 12, "spacingStaff": 6,
    })
    if not tk.loadFile(str(musicxml)):
        raise RuntimeError(f"verovio could not load {musicxml}")
    pages: list[Path] = []
    svg_dir = out_dir / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, tk.getPageCount() + 1):
        svg = tk.renderToSVG(i)
        if not svg:
            raise RuntimeError(f"verovio returned an empty page {i} for {musicxml}")
        svg_path = svg_dir / f"{stem}-p{i:02d}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        if fmt == "svg":
            pages.append(svg_path)
        else:
            png = out_dir / f"{stem}-p{i:02d}.png"
            _png_from_svg(svg, png, scale)
            pages.append(png)
        del svg
        gc.collect()
    return pages
