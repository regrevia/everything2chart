"""End-to-end CLI: audio file -> two-hand piano score (MusicXML/MIDI/jianpu).

    python -m everything2chart.cli --input work/astr2.m4a --preset easy --out out

Stems are cached to .npz so separation runs exactly once per track.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from . import arrange, audio, eval as peval, notation, transcribe
from . import fingering

log = logging.getLogger("everything2chart")


def _cache_path(cache_dir: Path, src: Path, model: str, seconds: float | None) -> Path:
    tag = f"{src.stem}-{model}-{int(seconds or 0)}s.npz"
    return cache_dir / tag


def get_stems(src: Path, model: str, cache_dir: Path,
              seconds: float | None, device: str | None,
              no_separate: bool) -> dict[str, np.ndarray]:
    cache = _cache_path(cache_dir, src, model, seconds)
    if cache.exists():
        log.info("using cached stems %s", cache)
        data = np.load(cache)
        return {k: data[k] for k in data.files}
    y, sr = audio.load_mono_stereo(src)
    if seconds:
        y = y[: int(seconds * sr)]
    if no_separate:
        mono = y.mean(axis=1, keepdims=True)
        return {"mix": mono, "other": mono, "bass": mono}
    from .separate import separate
    stems = separate(y, model_name=model, device=device)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **stems)
    log.info("stems cached -> %s", cache)
    return stems


def piano_roll(arr: arrange.Arrangement, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=130)
    for events, color, label in ((arr.left, "#2b6cb0", "LH"), (arr.right, "#c53030", "RH")):
        for n in events:
            ax.add_patch(plt.Rectangle((n.start, n.pitch - 0.4), n.dur, 0.8,
                                       color=color, alpha=0.85))
    ax.set_xlim(0, arr.info.duration)
    ax.set_ylim(30, 90)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("MIDI pitch")
    ax.set_title(f"{arr.preset} | key={arr.key_signature} | bpm={round(arr.info.bpm)}"
                 f" | RH {len(arr.right)} notes / LH {len(arr.left)} notes")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def _report_value(value, out_dir: Path):
    """Paths stay machine-readable in report.json: names relative to out_dir."""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_report_value(v, out_dir) for v in value]
    p = Path(value)
    return str(p.relative_to(out_dir)) if p.is_relative_to(out_dir) else str(p)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="everything2chart")
    p.add_argument("--input", required=True)
    p.add_argument("--out", default="out")
    p.add_argument("--preset", default="easy", choices=["easy", "medium", "hard", "all"])
    p.add_argument("--seconds", type=float, default=None, help="analyse only the first N seconds")
    p.add_argument("--model", default="htdemucs")
    p.add_argument("--demucs-repo", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--no-separate", action="store_true", help="skip demucs (single-file quick test)")
    p.add_argument("--transpose", type=int, default=0)
    p.add_argument("--stem-name", default=None)
    p.add_argument("--no-staff", action="store_true", help="skip staff rendering (fast)")
    p.add_argument("--fingering", default="key", choices=["key", "all", "none"],
                   help="finger numbers in the staff score: key = only where the hand "
                        "changes position (default), all = every note, none = off")
    p.add_argument("--notes-cache", default=None,
                   help="reuse a pickled analysis ({mel,bass,chords,info,mix}) instead of "
                        "re-running transcription; stems are still not separated")
    p.add_argument("--notes-out", default=None,
                   help="write the analysis this run actually used to this path, in the "
                        "--notes-cache format; makes a freshly separated chart verifiable "
                        "by tools/jpcheck.py, which can only check a chart against the "
                        "exact notes behind it")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.demucs_repo:
        from . import separate as sep_mod
        sep_mod.DEFAULT_REPO = Path(args.demucs_repo)

    src = Path(args.input)
    if args.notes_cache:
        log.info("reusing analysis %s", args.notes_cache)
        cached = pickle.loads(Path(args.notes_cache).read_bytes())
        melody, bass_notes = cached["mel"], cached["bass"]
        chords, info, mix = cached["chords"], cached["info"], cached["mix"]
    else:
        stems = get_stems(src, args.model, Path(args.cache_dir), args.seconds,
                          args.device, args.no_separate)

        # reference mix for tempo / harmony: everything except drums
        mix = sum(stems[k] for k in stems if k != "drums")
        if mix.ndim > 1:
            mix = mix.mean(axis=1)
        info = transcribe.analyse(mix, 44100)

        melody_stem = "other" if "other" in stems else list(stems)[0]
        melody = transcribe.cqt_melody(stems[melody_stem], 44100, stem=melody_stem,
                                       fmin_hz=transcribe.librosa.note_to_hz("C3"),
                                       fmax_hz=transcribe.librosa.note_to_hz("C7"))
        bass_notes = []
        if "bass" in stems:
            bass_notes = transcribe.pyin_notes(stems["bass"], 44100, stem="bass",
                                               fmin=transcribe.librosa.note_to_hz("E1"),
                                               fmax=transcribe.librosa.note_to_hz("C4"))
        chords = transcribe.estimate_chords(mix, 44100, info.beats)

    if args.notes_out:
        # Save the analysis *this run* used, before any arranger simplification.
        # The chart on disk is a rearrangement of exactly these notes, so this is
        # the only thing tools/jpcheck.py can legitimately hold it against; with
        # a stale --notes-cache the judge compares two different separations and
        # its verdict means nothing.
        Path(args.notes_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.notes_out).write_bytes(pickle.dumps(
            {"mel": melody, "bass": bass_notes, "chords": chords,
             "info": info, "mix": mix}))

    presets = arrange.load_presets()
    names = list(presets) if args.preset == "all" else [args.preset]
    out_dir = Path(args.out)
    report = {}
    arrangements = {}
    for name in names:
        arr = arrange.build_arrangement(melody, bass_notes, chords, info, name,
                                        presets=presets, transpose=args.transpose,
                                        ref_audio=(mix, 44100))
        if args.fingering != "none":
            arr.meta["fingering"] = fingering.annotate(arr, args.fingering)
        arrangements[name] = arr
        stem = args.stem_name or src.stem
        paths = notation.write_score(arr, out_dir, stem=stem)
        paths["roll"] = piano_roll(arr, out_dir / f"{stem}-{name}-roll.png")
        if not args.no_staff:
            # a shorter re-run must not leave the previous run's tail pages behind
            for old in out_dir.glob(f"{stem}-{name}-p*.png"):
                old.unlink()
            for old in out_dir.glob(f"{stem}-{name}-p*.svg"):
                old.unlink()
            try:
                paths["pages"] = notation.render_staff(paths["musicxml"], out_dir,
                                                       f"{stem}-{name}")
            except Exception as exc:  # engraver is optional
                log.warning("staff rendering skipped: %s", exc)
        entry = {k: _report_value(v, out_dir) for k, v in paths.items()}
        entry["stats"] = arr.stats()
        if "fingering" in arr.meta:
            entry["fingering"] = arr.meta["fingering"]
            try:
                entry["fingering_check"] = fingering.verify_export(
                    arr, Path(paths["musicxml"]).read_text(encoding="utf-8"),
                    arr.meta["fingering"].get("mode", args.fingering))
            except Exception as exc:  # never fail a run over the check itself
                log.warning("fingering check skipped: %s", exc)
        try:
            import librosa
            ref = librosa.resample(mix, orig_sr=44100, target_sr=peval.SR)
            entry["eval"] = {**peval.compare(arr, ref, peval.SR),
                             **peval.melody_contour(arr)}
        except Exception as exc:
            log.warning("eval skipped: %s", exc)
        report[name] = entry
        (out_dir / f"{stem}-{name}.arr.pkl").write_bytes(pickle.dumps(arr))

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v["stats"] for k, v in report.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
