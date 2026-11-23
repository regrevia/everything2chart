"""Core data types shared by the pipeline.

Everything downstream of separation/transcription speaks in these objects, so
backends (demucs / transkun / basic-pitch / librosa) stay swappable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

StemName = Literal["vocals", "drums", "bass", "guitar", "piano", "other"]

# Demucs htdemucs_6s stem order.
SIX_STEMS: tuple[str, ...] = ("drums", "bass", "other", "vocals", "guitar", "piano")
FOUR_STEMS: tuple[str, ...] = ("drums", "bass", "other", "vocals")


@dataclass(slots=True)
class NoteEvent:
    """One note, seconds-based, before any quantisation."""

    pitch: int           # MIDI pitch number
    start: float         # seconds
    dur: float           # seconds
    velocity: int = 80   # 1..127
    stem: str = "other"  # which separated stem it came from
    confidence: float = 1.0
    finger: int = 0      # 1..5 for a printed fingering, 0 for none

    @property
    def end(self) -> float:
        return self.start + self.dur

    def transpose(self, semitones: int) -> "NoteEvent":
        return NoteEvent(
            pitch=self.pitch + semitones,
            start=self.start,
            dur=self.dur,
            velocity=self.velocity,
            stem=self.stem,
            confidence=self.confidence,
            finger=self.finger,
        )


@dataclass(slots=True)
class ChordEvent:
    """A harmonic label active over [start, end)."""

    start: float
    end: float
    root: int            # MIDI pitch class 0..11
    quality: str         # "maj" | "min" | "dim" | "aug" | "maj7" | "min7" | "sus4" | "5"
    confidence: float = 1.0

    @property
    def intervals(self) -> tuple[int, ...]:
        return {
            "maj": (0, 4, 7),
            "min": (0, 3, 7),
            "dim": (0, 3, 6),
            "aug": (0, 4, 8),
            "maj7": (0, 4, 7, 11),
            "min7": (0, 3, 7, 10),
            "7": (0, 4, 7, 10),
            "sus4": (0, 5, 7),
            "5": (0, 7),
        }[self.quality]

    def pitch_classes(self) -> list[int]:
        return [(self.root + i) % 12 for i in self.intervals]


@dataclass(slots=True)
class SongInfo:
    path: str
    duration: float
    bpm: float
    beats: list[float] = field(default_factory=list)  # beat onset times, seconds
    beats_per_bar: int = 4

    def quantise(self, t: float, grid: float = 0.125) -> float:
        """Snap a time in seconds to a beat-subdivision grid."""
        if not self.beats:
            step = 60.0 / self.bpm
            return round(t / (step * grid)) * step * grid
        step = 60.0 / self.bpm * grid
        return round(t / step) * step


@dataclass(slots=True)
class Arrangement:
    """What the arranger outputs: two hands' worth of notes plus metadata."""

    right: list[NoteEvent]
    left: list[NoteEvent]
    info: SongInfo
    preset: str
    key_signature: str = "C"
    meta: dict = field(default_factory=dict)

    def stats(self) -> dict:
        return {
            "right_notes": len(self.right),
            "left_notes": len(self.left),
            "right_range": [min((n.pitch for n in self.right), default=0),
                            max((n.pitch for n in self.right), default=0)],
            "left_range": [min((n.pitch for n in self.left), default=0),
                           max((n.pitch for n in self.left), default=0)],
            "duration": round(self.info.duration, 2),
            "key_signature": self.key_signature,
            "transpose": int(self.meta.get("transpose", 0)),
            **{k: v for k, v in self.meta.items() if k != "transpose"},
        }
