"""Monophonic/multi-instrument transcription with lightweight, dependency-free
backends.

M0 backend = librosa: pYIN pitch tracking per stem (bass / melody) and chroma
template matching for harmony.  It is deliberately swappable: `transcribe.py`
exposes plain functions returning `NoteEvent` lists, so Transkun / basic-pitch
can be dropped in later without touching the arranger.
"""

from __future__ import annotations

import logging

import librosa
import numpy as np

from .types import ChordEvent, NoteEvent, SongInfo

log = logging.getLogger(__name__)

QUALITIES = ("maj", "min", "5", "dim", "sus4", "maj7", "min7")


def _grid_contrast(onset_env: np.ndarray, sr: int, hop: int,
                   tempo: float, duration: float) -> float:
    """How much stronger the onsets on a grid line are than between them.

    At the true tempo every attack sits on a line and the halfway points are
    quiet; at half tempo the halfway points land on real beats, so the ratio
    collapses.  Unlike a coverage count this cannot be fooled by a grid that
    simply has more lines in it.
    """
    spb = 60.0 / tempo
    lines = np.arange(0.0, duration - spb, spb)
    if len(lines) < 4:
        return -1e9
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop)

    def energy(ts):
        i = np.clip(np.searchsorted(times, np.asarray(ts, dtype=float)), 1,
                    len(onset_env) - 2)
        return np.maximum(np.maximum(onset_env[i - 1], onset_env[i]), onset_env[i + 1])

    on = energy(lines).mean()
    off = energy(lines + 0.5 * spb).mean()
    return float(on / (off + 1e-9))


def track_beats(y: np.ndarray, sr: int) -> tuple[float, list[float]]:
    mono = librosa.to_mono(y.T) if y.ndim > 1 else y
    tempo, beats = librosa.beat.beat_track(y=mono, sr=sr, units="time", trim=False)
    tempo = float(np.atleast_1d(tempo)[0])

    # librosa's default start_bpm=120 prior locks onto half the beat for fast
    # music (167 became 83.35).  Only ever correct upward: doubling an already
    # correct tempo would be the worse mistake, and a slow reading is the one
    # this prior actually produces.
    if 0 < tempo < 100:
        hop = 512
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr, hop_length=hop)
        duration = len(mono) / float(sr)
        cand = tempo * 2.0
        if cand <= 240:
            here = _grid_contrast(onset_env, sr, hop, tempo, duration)
            there = _grid_contrast(onset_env, sr, hop, cand, duration)
            if there > here * 4.0:
                doubled, nbeats = librosa.beat.beat_track(
                    y=mono, sr=sr, units="time", trim=False, bpm=cand)
                # keep the onsets' own timing; only accept if it re-finds the grid
                if len(nbeats) >= 0.75 * len(beats) * 1.8:
                    tempo = float(np.atleast_1d(doubled)[0])
                    beats = nbeats
    return tempo, [float(b) for b in beats]


def pyin_notes(
    y: np.ndarray,
    sr: int,
    fmin: float = librosa.note_to_hz("C2"),
    fmax: float = librosa.note_to_hz("C7"),
    stem: str = "other",
    min_dur: float = 0.06,
    frame_length: int = 2048,
) -> list[NoteEvent]:
    """Track one dominant pitch line and turn it into note events."""
    mono = librosa.to_mono(y.T) if y.ndim > 1 else y
    if mono.size < frame_length:
        return []
    f0, voiced, prob = librosa.pyin(
        mono, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=256, fill_na=np.nan,
    )
    f0 = _fix_octaves(f0, voiced)
    times = librosa.times_like(f0, sr=sr, hop_length=256)
    rms = librosa.feature.rms(y=mono, frame_length=frame_length, hop_length=256)[0]

    notes: list[NoteEvent] = []
    cur: dict | None = None
    for i, (t, f, v) in enumerate(zip(times, f0, voiced)):
        if not v or np.isnan(f):
            if cur is not None:
                notes.append(_close(cur, t, min_dur, stem))
                cur = None
            continue
        midi = int(round(librosa.hz_to_midi(f)))
        if cur is None or abs(midi - cur["pitch"]) > 1:
            if cur is not None:
                notes.append(_close(cur, t, min_dur, stem))
            cur = {"pitch": midi, "start": float(t), "vel": float(rms[min(i, len(rms) - 1)]),
                   "conf": float(prob[i]) if prob is not None else 1.0}
        else:
            cur["vel"] = max(cur["vel"], float(rms[min(i, len(rms) - 1)]))
            if prob is not None:
                cur["conf"] = max(cur["conf"], float(prob[i]))
    if cur is not None:
        notes.append(_close(cur, float(times[-1]), min_dur, stem))
    return [n for n in notes if n is not None]


def _fix_octaves(f0: np.ndarray, voiced: np.ndarray, window: int = 24) -> np.ndarray:
    """pYIN on a dense mix loves octave jumps; drag outliers back to the local median."""
    midi = np.where(voiced, np.log2(np.maximum(f0, 1e-6) / 440.0) * 12 + 69, np.nan)
    out = midi.copy()
    for i in range(len(midi)):
        if np.isnan(midi[i]):
            continue
        lo, hi = max(0, i - window), min(len(midi), i + window)
        ref_vals = midi[lo:hi]
        ref_vals = ref_vals[~np.isnan(ref_vals)]
        if ref_vals.size < 4:
            continue
        ref = float(np.median(ref_vals))
        m = midi[i]
        while m - ref > 7:
            m -= 12
        while ref - m > 7:
            m += 12
        out[i] = m
    hz = np.where(np.isnan(out), np.nan, 440.0 * 2 ** ((out - 69) / 12))
    return hz


def _close(cur: dict, end_t: float, min_dur: float, stem: str) -> NoteEvent | None:
    dur = end_t - cur["start"]
    if dur < min_dur:
        return None
    vel = int(np.clip(40 + 400 * cur["vel"], 30, 120))
    return NoteEvent(pitch=cur["pitch"], start=cur["start"], dur=dur,
                     velocity=vel, stem=stem, confidence=cur["conf"])


def estimate_chords(
    y: np.ndarray,
    sr: int,
    beats: list[float],
    hop: int = 512,
    min_dur: float = 0.25,
) -> list[ChordEvent]:
    """Per-beat chord estimation by matching chroma against chord templates."""
    mono = librosa.to_mono(y.T) if y.ndim > 1 else y
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=hop)
    times = librosa.times_like(chroma[0], sr=sr, hop_length=hop)

    templates: list[tuple[int, str, np.ndarray]] = []
    for root in range(12):
        for q in QUALITIES:
            iv = ChordEvent(0, 0, root, q).intervals
            v = np.zeros(12)
            v[[(root + i) % 12 for i in iv]] = 1.0 / len(iv)
            templates.append((root, q, v / np.linalg.norm(v)))

    bounds = beats if beats and len(beats) > 1 else [times[0], times[-1]]
    chords: list[ChordEvent] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 0.05:
            continue
        mask = (times >= a) & (times < b)
        if not mask.any():
            continue
        vec = chroma[:, mask].mean(axis=1)
        n = np.linalg.norm(vec)
        if n < 1e-6:
            continue
        vec = vec / n
        best = max(templates, key=lambda t: float(np.dot(t[2], vec)))
        score = float(np.dot(best[2], vec))
        chords.append(ChordEvent(start=a, end=b, root=best[0], quality=best[1],
                                 confidence=score))

    # merge identical consecutive labels, drop very short ones
    merged: list[ChordEvent] = []
    for c in chords:
        if merged and merged[-1].root == c.root and merged[-1].quality == c.quality:
            merged[-1].end = c.end
            merged[-1].confidence = max(merged[-1].confidence, c.confidence)
        else:
            merged.append(c)
    return [c for c in merged if c.end - c.start >= min_dur] or merged


def analyse(y: np.ndarray, sr: int) -> SongInfo:
    bpm, beats = track_beats(y, sr)
    if not beats:
        step = 60.0 / bpm if bpm else 0.5
        beats = list(np.arange(0, y.shape[0] / sr, step))
    return SongInfo(path="", duration=y.shape[0] / sr, bpm=bpm, beats=beats)


# --------------------------------------------------------------------- melody (skyline)

def _fix_midi_octaves(midi: np.ndarray, window: int = 16) -> np.ndarray:
    """Pull octave jumps back toward the local median (pitch trackers love them)."""
    out = midi.copy()
    for i in range(len(midi)):
        if np.isnan(midi[i]):
            continue
        lo, hi = max(0, i - window), min(len(midi), i + window)
        ref_win = midi[lo:hi]
        ref_win = ref_win[~np.isnan(ref_win)]
        if ref_win.size < 3:
            continue
        ref = float(np.median(ref_win))
        m = float(midi[i])
        while m - ref > 7:
            m -= 12
        while ref - m > 7:
            m += 12
        out[i] = m
    return out


def cqt_melody(y, sr, fmin_hz=130.81, fmax_hz=2093.0, stem="other",
               hop=512, bins_per_octave=36, min_dur=0.09,
               continuity=1.0, gamma=20.0):
    """Melody line by CQT harmonic salience + skyline tracking.

    pYIN on a dense orchestral mix reports voice only ~30% of the time, so the
    line comes out dotted. Here every frame keeps a candidate set and a simple
    continuity prior picks the top line, which yields a singable contour.
    """
    import librosa

    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    C = np.abs(librosa.cqt(y, sr=sr, fmin=fmin_hz, hop_length=hop,
                           n_bins=int(np.log2(fmax_hz / fmin_hz) * bins_per_octave) + 1,
                           bins_per_octave=bins_per_octave))
    bpo = bins_per_octave
    n_bins, n_frames = C.shape
    # harmonic sum: a real note lights up f, 2f, 3f ...
    sal = C.copy()
    for h, w in ((2, 0.6), (3, 0.4), (4, 0.25)):
        shift = int(round(np.log2(h) * bpo))
        if shift < n_bins:
            sal[: n_bins - shift] += w * C[shift:]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)
    freqs = librosa.cqt_frequencies(n_bins, fmin=fmin_hz, bins_per_octave=bpo)
    midi = np.round(69 + 12 * np.log2(freqs / 440.0)).astype(int)

    # per-frame candidate peaks (local maxima, robust threshold)
    thr = np.percentile(sal, 60, axis=0) * 0.9
    peaks = []
    for t in range(n_frames):
        col = sal[:, t]
        cand = []
        if col.max() > 1e-6:
            for i in range(1, n_bins - 1):
                if col[i] > thr[t] and col[i] >= col[i - 1] and col[i] >= col[i + 1] \
                        and 24 <= midi[i] <= 96:
                    cand.append((float(col[i]), int(midi[i])))
        cand.sort(reverse=True)
        peaks.append(cand[:12])

    # Viterbi over candidate pitches: continuity beats loudness for a singable line
    cands = [sorted({m for _, m in c}) for c in peaks]
    emit = []
    for t, c in enumerate(peaks):
        d = {m: float(np.log1p(a * 120.0)) for a, m in c}
        emit.append(d)
    NEG = -1e9
    prev_cost = {m: emit[0].get(m, 0.0) for m in cands[0]} if cands[0] else {}
    costs = [dict(prev_cost)]          # per-frame state costs, for the backtrace
    back = []
    for t in range(1, n_frames):
        cur, bp = {}, {}
        if not cands[t]:
            back.append({})
            prev_cost = {}
            costs.append({})
            continue
        for m in cands[t]:
            base = emit[t].get(m, 0.0)
            if not prev_cost:
                cur[m] = base
                bp[m] = None
                continue
            best, best_pm = NEG, None
            for pm, pc in prev_cost.items():
                step = abs(m - pm)
                cost = pc + base - continuity * step - (6.0 if step > 12 else 0.0) \
                       - (2.0 if step == 12 else 0.0)
                if cost > best:
                    best, best_pm = cost, pm
            cur[m], bp[m] = best, best_pm
        back.append(bp)
        costs.append(dict(cur))
        prev_cost = cur
    # Walk back from the last frame that actually has a state.  A track can end in
    # silence (the tail of a mix, a gap after the final chord); starting from the
    # final frame then throws away the entire take instead of its last frames.
    chosen = np.full(n_frames, np.nan)
    t = next((i for i in range(n_frames - 1, -1, -1) if costs[i]), None)
    if t is not None:
        state = max(costs[t], key=costs[t].get)
        while t >= 0:
            chosen[t] = state
            pm = back[t - 1].get(state) if t > 0 and t - 1 < len(back) else None
            if pm is None:
                # segment start (first frame, or a hop over unvoiced frames)
                nxt = next((i for i in range(t - 1, -1, -1) if costs[i]), None)
                if nxt is None:
                    break
                t = nxt
                state = max(costs[t], key=costs[t].get)
                continue
            t -= 1
            state = pm
    # octave repair + median smoothing, then merge into notes
    chosen = _fix_midi_octaves(chosen)
    ok = ~np.isnan(chosen)
    idx = np.arange(n_frames)
    if ok.sum() > 5:
        sm = np.full(n_frames, np.nan)
        for i in idx:
            win = chosen[max(0, i - 3): i + 4]
            win = win[~np.isnan(win)]
            if win.size:
                sm[i] = np.median(win)
        chosen = np.round(sm)
    notes = []
    cur_p, cur_s = None, None
    for t in range(n_frames + 1):
        # an unvoiced frame stays NaN; NaN is not None, and int(round(nan)) raises
        p = chosen[t] if t < n_frames else np.nan
        p = None if p is None or not np.isfinite(p) else int(round(p))
        if cur_p is None:
            if p is not None:
                cur_p, cur_s = p, times[t]
            continue
        if p != cur_p:
            t_end = times[t] if t < n_frames else times[-1] + hop / sr
            dur = t_end - cur_s
            if dur >= min_dur:
                notes.append(NoteEvent(pitch=cur_p, start=float(cur_s), dur=float(dur),
                                       velocity=80, stem=stem,
                                       confidence=float(min(1.0, dur / 0.4))))
            cur_p = p
            cur_s = times[t] if p is not None else None
    return notes
