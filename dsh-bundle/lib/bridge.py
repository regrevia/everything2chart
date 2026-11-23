#!/usr/bin/env python3
"""One link in, one score out.

Resolves a source into a local audio file, runs the documented
``everything2chart.cli`` unchanged, then reports the run as one JSON object on
stdout. The CLI stays the only entry point into the pipeline so this bridge
cannot drift from the documented behaviour.

Source resolution has three branches and they are not interchangeable:

``local``
    A path on this machine. Used as-is.
``direct-media``
    An http(s) URL that *is* the audio/video file (``.mp3``, ``.m4a``, ``.wav``,
    ``.flac``, ``.ogg``, ``.opus``, ``.mp4``, ``.webm``, or a ``Content-Type`` of
    ``audio/*`` / ``video/*``). Fetched with ``curl``. Envoy is deliberately not
    consulted: its static reader rejects a bare media URL with
    ``unsupported_source``, so routing these through Envoy only costs a turn.
``page``
    An http(s) URL that is a page hosting the audio. Tried with ``yt-dlp``,
    which reads the page's own player metadata. If that fails the run stops
    with ``needs_media_url`` and a hint, rather than guessing.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import time

MEDIA_EXTENSIONS = (
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
    ".oga",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".mp4",
    ".m4v",
    ".webm",
    ".mkv",
    ".mov",
)

PIPELINE_TIMEOUT_SECONDS = 3600
DOWNLOAD_TIMEOUT_SECONDS = 900
DOWNLOAD_ATTEMPTS = 3


def log(message: str) -> None:
    """Progress goes to stderr so stdout stays one parseable JSON object."""
    print(message, file=sys.stderr, flush=True)


def which(*names: str) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def run(cmd: list[str], timeout: int, cwd: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        errors="replace",
    )


def sniff_is_media(url: str) -> bool:
    """Decide by extension first, then by the response's Content-Type."""
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    if path.endswith(MEDIA_EXTENSIONS):
        return True

    curl = which("curl")
    if not curl:
        return False
    # Header only, and follow redirects: a CDN link rarely ends in an extension.
    probe = run([curl, "-sS", "-I", "-L", "--max-time", "30", "-A", "Mozilla/5.0", url], 45)
    content_type = ""
    for line in probe.stdout.splitlines():
        if line.lower().startswith("content-type:"):
            content_type = line.split(":", 1)[1].strip().lower()
    return content_type.startswith("audio/") or content_type.startswith("video/")


def fetch_direct(url: str, work_dir: str) -> str:
    curl = which("curl")
    if not curl:
        raise RuntimeError("curl is not installed")

    suffix = os.path.splitext(url.split("?", 1)[0])[1].lower()
    if suffix not in MEDIA_EXTENSIONS:
        guessed = mimetypes.guess_extension("audio/mpeg") or ".bin"
        suffix = guessed
    target = os.path.join(work_dir, f"source{suffix}")

    log(f"downloading {url}")
    # A long track over a flaky link drops mid-transfer: curl exits 18 with a
    # partial file. `-C -` resumes where the previous attempt stopped, so three
    # bounded attempts finish what one would abandon.
    last_error = ""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        result = run(
            [curl, "-sS", "-L", "--fail", "--max-time", str(DOWNLOAD_TIMEOUT_SECONDS),
             "-C", "-", "-A", "Mozilla/5.0", "-o", target, url],
            DOWNLOAD_TIMEOUT_SECONDS + 60,
        )
        if result.returncode == 0:
            break
        last_error = result.stderr.strip()[:400] or f"exit {result.returncode}"
        # 33 = server cannot resume; drop the partial file and start over.
        if result.returncode == 33 and os.path.exists(target):
            os.remove(target)
            last_error += " (server refused to resume; restarting)"
        log(f"download attempt {attempt} failed: {last_error}")
    else:
        if os.path.exists(target) and os.path.getsize(target) > 0:
            log("using the partial file from the last attempt")
        else:
            raise RuntimeError(
                f"download failed after {DOWNLOAD_ATTEMPTS} attempts: {last_error}"
            )
    if not os.path.exists(target) or os.path.getsize(target) == 0:
        raise RuntimeError("download produced an empty file")
    return target


class PageNeedsHelp(RuntimeError):
    """A page URL could not be turned into audio here; the caller needs to
    fetch the page itself and hand over the media URL it exposes."""


def fetch_page_media(url: str, work_dir: str) -> str:
    yt_dlp = which("yt-dlp")
    if not yt_dlp:
        raise PageNeedsHelp("yt-dlp is not installed, so a page URL cannot be resolved here")

    template = os.path.join(work_dir, "source.%(ext)s")
    log("resolving the page with yt-dlp")
    result = run(
        [yt_dlp, "-f", "bestaudio/best", "--no-playlist", "--no-progress",
         "-o", template, url],
        DOWNLOAD_TIMEOUT_SECONDS + 60,
    )
    if result.returncode != 0:
        raise PageNeedsHelp((result.stderr or "").strip().splitlines()[-1][:300] or "yt-dlp failed")

    for name in sorted(os.listdir(work_dir)):
        if name.startswith("source.") and not name.endswith(".part"):
            return os.path.join(work_dir, name)
    raise PageNeedsHelp("yt-dlp reported success but wrote no file")


def resolve_source(source: str, work_dir: str) -> tuple[str, str]:
    if not source:
        raise RuntimeError("no source given")
    if os.path.exists(source):
        return os.path.abspath(source), "local"
    if not source.lower().startswith(("http://", "https://", "file://")):
        raise RuntimeError(f"not a path and not an http(s) URL: {source}")

    if source.lower().startswith("file://"):
        path = source[7:]
        if not os.path.exists(path):
            raise RuntimeError(f"no such file: {path}")
        return path, "local"

    if sniff_is_media(source):
        return fetch_direct(source, work_dir), "direct-media"
    return fetch_page_media(source, work_dir), "page"


def duration_of(path: str) -> float | None:
    ffprobe = which("ffprobe")
    if not ffprobe:
        return None
    result = run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                  "-of", "csv=p=0", path], 60)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def pipeline_command(repo: str, audio: str, tier: str, out_dir: str,
                     seconds: float | None, transpose: int | None,
                     fingering: str | None = None) -> list[str]:
    cmd = [
        os.environ.get("PIANOCHART_PYTHON") or sys.executable,
        "-m", "everything2chart.cli",
        "--input", audio,
        "--preset", tier,
        "--out", out_dir,
    ]
    if seconds:
        cmd += ["--seconds", str(seconds)]
    if transpose:
        cmd += ["--transpose", str(transpose)]
    if fingering:
        cmd += ["--fingering", str(fingering)]
    return cmd


def absolute(path: str, out_dir: str) -> str:
    """Report paths are written relative to the output directory; the caller
    needs paths that work from any working directory."""
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(out_dir, path)


def collect(report: dict, tier: str, out_dir: str) -> dict:
    block = report.get(tier) or next(iter(report.values()), {})
    pages = block.get("pages") or []
    if isinstance(pages, str):
        pages = [pages]
    pages = [absolute(page, out_dir) for page in pages]
    jianpu_path = absolute(block.get("jianpu") or "", out_dir)
    jianpu_text = ""
    if jianpu_path and os.path.exists(jianpu_path):
        with open(jianpu_path, "r", encoding="utf-8") as handle:
            jianpu_text = handle.read()

    return {
        "tier": tier,
        "key_signature": (block.get("stats") or {}).get("key_signature"),
        "transpose": (block.get("stats") or {}).get("transpose"),
        "duration": (block.get("stats") or {}).get("duration"),
        "right_notes": (block.get("stats") or {}).get("right_notes"),
        "left_notes": (block.get("stats") or {}).get("left_notes"),
        "similarity": block.get("eval") or {},
        "fingering": block.get("fingering") or {},
        "fingering_check": block.get("fingering_check") or {},
        "staff_pages": pages,
        "jianpu_path": jianpu_path,
        "jianpu": jianpu_text,
        "musicxml": absolute(block.get("musicxml") or "", out_dir),
        "midi": absolute(block.get("midi") or "", out_dir),
        "roll": absolute(block.get("roll") or "", out_dir),
        "out_dir": out_dir,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="everything2chart bridge")
    parser.add_argument("--source", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tier", default="easy", choices=("easy", "medium", "hard"))
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--transpose", type=int, default=None)
    parser.add_argument("--fingering", default="key", choices=("key", "all", "none"))
    parser.add_argument("--jianpu-limit", type=int, default=4000)
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.out, exist_ok=True)
    work_dir = tempfile.mkdtemp(prefix="e2c-", dir=args.out)

    payload: dict = {"ok": False, "tier": args.tier, "out_dir": args.out}
    try:
        audio, branch = resolve_source(args.source, work_dir)
        payload["source"] = args.source
        payload["source_branch"] = branch
        payload["audio_path"] = audio
        payload["audio_seconds"] = duration_of(audio)

        log(f"transcribing {audio} at tier {args.tier}")
        result = subprocess.run(
            pipeline_command(args.repo, audio, args.tier, args.out, args.seconds, args.transpose,
                                args.fingering),
            cwd=args.repo,
            env={**os.environ, "PYTHONPATH": os.path.join(args.repo, "src"),
                 "PYTHONDONTWRITEBYTECODE": "1",
                 "DYLD_FALLBACK_LIBRARY_PATH": os.environ.get(
                     "DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")},
            timeout=PIPELINE_TIMEOUT_SECONDS,
            capture_output=True,
            text=True,
            errors="replace",
        )
        if result.returncode != 0:
            payload["returncode"] = result.returncode
            stderr_tail = (result.stderr or "").strip().splitlines()[-12:]
            stdout_tail = (result.stdout or "").strip().splitlines()[-12:]
            if result.returncode < 0:
                head = f"pipeline killed by signal {-result.returncode}"
                payload["hint"] = (
                    "The transcriber was killed by the operating system, usually for running "
                    "out of memory. Close other memory-heavy apps and try again, or transcribe "
                    "a shorter span with seconds."
                )
            else:
                head = f"pipeline failed (exit {result.returncode})"
            detail = stderr_tail or stdout_tail
            payload["stderr"] = "\n".join(stderr_tail) or "<no stderr>"
            payload["error"] = head + ":\n" + ("\n".join(detail) or "<no output>")
            payload["seconds"] = round(time.time() - started, 1)
            print(json.dumps(payload, ensure_ascii=False))
            return 1

        report_path = os.path.join(args.out, "report.json")
        if not os.path.exists(report_path):
            payload["error"] = f"pipeline finished but wrote no report at {report_path}"
            payload["seconds"] = round(time.time() - started, 1)
            print(json.dumps(payload, ensure_ascii=False))
            return 1

        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)

        payload.update(collect(report, args.tier, args.out))
        if len(payload.get("jianpu") or "") > args.jianpu_limit:
            payload["jianpu_truncated"] = True
            payload["jianpu"] = payload["jianpu"][: args.jianpu_limit]
        payload["ok"] = True
        payload["seconds"] = round(time.time() - started, 1)
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    except subprocess.TimeoutExpired:
        payload["error"] = f"timed out after {PIPELINE_TIMEOUT_SECONDS}s"
    except PageNeedsHelp as error:
        payload["error"] = str(error)
        payload["hint"] = (
            "This link is a page, not an audio file, and the page could not be read here. "
            "Read the page through Envoy (envoy_read) and pass the audio URL or saved file it "
            "exposes as the source."
        )
    except Exception as error:  # noqa: BLE001 - the tool reports, never raises
        payload["error"] = str(error)[:600]
    payload["seconds"] = round(time.time() - started, 1)
    print(json.dumps(payload, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    sys.exit(main())
