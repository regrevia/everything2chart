"""Give it a link, get a score.

    python -m everything2chart.serve --port 8765

Opens one page with one box: paste an audio or video link, pick a tier, get the
staff pages, the numbered notation, MusicXML and MIDI back as download links.

Runs jobs one at a time on purpose: separation is the expensive stage and two
of them at once only makes both slower.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

DOWNLOAD_TIMEOUT = 900
JOB_TIMEOUT = 3600
MAX_JOBS = 40

# what the page shows while it waits
STAGES = {"queued": 0, "fetching": 1, "transcribing": 2, "ready": 3}
FINGERING_MODES = {"key", "all", "none"}

# one heavy job at a time
_lane = threading.Semaphore(1)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------- fetching

class SourceError(RuntimeError):
    """The link could not be turned into a local audio file."""


# curl exit codes worth naming: the page shows these, so they say what to do
_CURL = {
    6: "域名解析不了，链接可能写错了",
    7: "连不上这个地址",
    18: "只下了一半，链接可能不支持续传，再试一次",
    22: "这个链接取不到音频（服务器拒绝了请求）",
    23: "本地落盘失败，磁盘可能满了",
    28: "超时了，链接可能很慢或文件很大",
    33: "服务器不支持续传",
    35: "TLS 握手失败",
    56: "传输中途断了，再试一次",
    60: "对方证书验证不通过",
    404: "这个链接返回 404，文件可能已经没了",
}


def explain(code: int, stderr: str) -> str:
    """Name the failure in words the person can act on."""
    if code in _CURL:
        return _CURL[code]
    m = re.search(r"\b(\d{3})\b", stderr)
    if m and int(m.group(1)) in _CURL:
        return _CURL[int(m.group(1))]
    if "404" in stderr:
        return _CURL[404]
    return f"取音频失败（curl {code}）"


_AUDIO_EXT = re.compile(r"\.(mp3|m4a|aac|wav|flac|ogg|opus|webm|mp4|mkv)(\?|$)", re.I)


def _curl(url: str, dest: Path, tries: int = 3) -> Path:
    """Download with resume; partial transfers are the common failure on big audio."""
    for attempt in range(1, tries + 1):
        proc = subprocess.run(
            ["curl", "-L", "--fail", "--silent", "--show-error",
             "-C", "-", "--max-time", str(DOWNLOAD_TIMEOUT),
             "-A", "Mozilla/5.0 (everything2chart)", "-o", str(dest), url],
            capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT + 30,
        )
        if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 8192:
            return dest
        # 33 = range not supported; start over without -C
        if proc.returncode == 33:
            dest.unlink(missing_ok=True)
            continue
        if attempt == tries:
            raise SourceError(explain(proc.returncode, proc.stderr or ""))
        time.sleep(2)
    raise SourceError("download kept failing")


def fetch_source(source: str, workdir: Path) -> Path:
    """Turn a link into a local file, or explain what it is instead."""
    if not re.match(r"^https?://", source, re.I):
        # A path pasted from a file manager can arrive as file:///a/b.m4a.
        bare = re.sub(r"^file://", "", source, flags=re.I)
        local = Path(unquote(bare)).expanduser()
        if not local.exists():
            raise SourceError(f"no such file: {local}")
        return local

    name = Path(unquote(urlparse(source).path)).name or "audio"
    if not _AUDIO_EXT.search(name):
        name += ".bin" if "." not in name else ""

    dest = workdir / name
    if shutil.which("yt-dlp") and _AUDIO_EXT.search(source) is None:
        proc = subprocess.run(
            ["yt-dlp", "-x", "--audio-format", "m4a", "--no-playlist",
             "--max-filesize", "300M", "-o", str(workdir / "audio.%(ext)s"), source],
            capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT,
        )
        got = sorted(workdir.glob("audio.*"))
        if proc.returncode == 0 and got:
            return got[0]
    return _curl(source, dest)


# ---------------------------------------------------------------- jobs

def _run_job(job_id: str, source: str, tier: str, seconds: float | None,
             fingering: str = "key") -> None:
    job = _jobs[job_id]
    work = Path(job["work"])
    try:
        with _lane:
            job["status"] = "fetching"
            src = fetch_source(source, work)
            job["status"] = "transcribing"
            out = work / "out"
            cmd = [sys.executable, "-m", "everything2chart.cli",
                   "--input", str(src), "--out", str(out), "--preset", tier,
                   "--cache-dir", str(work / "cache")]
            if seconds:
                cmd += ["--seconds", str(seconds)]
            cmd += ["--fingering", fingering]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                                  text=True, timeout=JOB_TIMEOUT, env=env)
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout or "pipeline failed").strip()[-600:])
            job["result"] = _collect(out, tier, fingering)
            job["status"] = "ready"
    except Exception as exc:  # surfaced to the page, not swallowed
        job["status"] = "error"
        job["error"] = str(exc)[:600]


def _collect(out: Path, tier: str, fingering: str = "key") -> dict:
    files = []
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".png", ".musicxml", ".mid", ".txt", ".svg"}:
            files.append({"name": p.name, "path": str(p.relative_to(out)),
                          "bytes": p.stat().st_size})
    jianpu = ""
    for f in files:
        if f["name"].endswith(".jianpu.txt"):
            try:
                jianpu = (out / f["path"]).read_text()[:1800]
            except Exception:
                pass
            break
    fing = {}
    checks = {}
    try:
        rep = json.loads((out / "report.json").read_text())
        for name, entry in rep.items():
            if isinstance(entry, dict) and entry.get("fingering"):
                fing[name] = entry["fingering"]
            if isinstance(entry, dict) and entry.get("fingering_check"):
                checks[name] = entry["fingering_check"]
    except Exception:
        pass
    # The counts alone say how many numbers were written; the check says whether
    # the staff file actually carries them and whether they are in order.
    return {"tier": tier, "files": files, "jianpu": jianpu,
            "fingering": fing, "fingering_check": checks}


# ---------------------------------------------------------------- page

PAGE_PATH = HERE / "web" / "index.html"
ALLOW_ORIGIN = os.environ.get("PIANOCHART_ALLOW_ORIGIN", "*")


class Handler(BaseHTTPRequestHandler):
    server_version = "everything2chart"

    def log_message(self, fmt, *a):  # keep the terminal readable
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        # the published page lives on another host and calls this one by ?api=
        self.send_header("access-control-allow-origin", ALLOW_ORIGIN)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("access-control-allow-origin", ALLOW_ORIGIN)
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type")
        self.send_header("access-control-max-age", "86400")
        self.send_header("content-length", "0")
        self.end_headers()

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        if path.startswith("/api/job/"):
            job_id = path.rsplit("/", 1)[-1]
            job = _jobs.get(job_id)
            if not job:
                self._json(404, {"error": "no such job"})
                return
            out = {"id": job_id, "status": job["status"],
                   "stage": STAGES.get(job["status"], 0)}
            if job.get("error"):
                out["error"] = job["error"]
            if job.get("result"):
                out.update(job["result"])
                out["id"] = job_id
            self._json(200, out)
            return
        if path.startswith("/files/"):
            parts = path.split("/", 3)
            if len(parts) < 4:
                self._send(404, b"no", "text/plain")
                return
            job = _jobs.get(parts[2])
            if not job:
                self._send(404, b"no such job", "text/plain")
                return
            target = Path(job["work"]) / "out" / unquote(parts[3])
            root = (Path(job["work"]) / "out").resolve()
            if not target.resolve().is_file() or not str(target.resolve()).startswith(str(root)):
                self._send(404, b"no such file", "text/plain")
                return
            ctype = {".png": "image/png", ".svg": "image/svg+xml",
                     ".musicxml": "application/xml", ".mid": "audio/midi",
                     ".txt": "text/plain; charset=utf-8"}.get(target.suffix.lower(), "application/octet-stream")
            self._send(200, target.read_bytes(), ctype)
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/score":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("content-length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            source = (payload.get("source") or "").strip()
            if not source:
                self._json(400, {"error": "给个链接"})
                return
            tier = payload.get("tier") or "easy"
            if tier not in {"easy", "medium", "hard", "all"}:
                self._json(400, {"error": f"unknown tier: {tier}"})
                return
            seconds = payload.get("seconds")
            fingering = payload.get("fingering") or "key"
            if fingering not in FINGERING_MODES:
                self._json(400, {"error": f"unknown fingering: {fingering}"})
                return
        except Exception as exc:
            self._json(400, {"error": f"bad request: {exc}"})
            return

        with _lock:
            if len(_jobs) > MAX_JOBS:
                for k in [k for k, v in _jobs.items() if v["status"] in ("ready", "error")][:10]:
                    shutil.rmtree(_jobs[k]["work"], ignore_errors=True)
                    _jobs.pop(k, None)
            job_id = uuid.uuid4().hex[:12]
            _jobs[job_id] = {"status": "queued", "work": tempfile.mkdtemp(prefix=f"pc-{job_id}-"),
                             "created": time.time()}
        threading.Thread(target=_run_job,
                             args=(job_id, source, tier, seconds, fingering),
                             daemon=True).start()
        self._json(200, {"id": job_id})


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="everything2chart.serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"everything2chart serve -> http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
