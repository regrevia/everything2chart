#!/usr/bin/env python3
"""Minimal stdio MCP server: one audio link in, one piano score out.

Speaks JSON-RPC 2.0 over stdin/stdout with one tool, `piano_score`, which is a
thin wrapper around `bridge.py`. The bridge does the work (source
resolution, pipeline run, report collection); this file only translates MCP
frames into a bridge invocation and the bridge's JSON answer back into MCP
content.

Why a subprocess rather than an import: the bridge is also the CLI the bundle's
own tool shells out to, so keeping one entry point means the MCP surface and the
Host tool can never drift. The bridge prints one JSON object as its last line
(its earlier lines are progress log), which is what gets parsed here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _bundle_version():
    """Read the version from the bundle package.json so it never drifts again."""
    try:
        with open(os.path.join(os.path.dirname(HERE), "package.json"), encoding="utf-8") as fh:
            return json.load(fh).get("version") or "0.0.0"
    except Exception:
        return "0.0.0"


VERSION = _bundle_version()

BRIDGE = os.path.join(HERE, "bridge.py")
PROTOCOL_VERSION = "2024-11-05"
# The pipeline runs on CPU and the separation stage alone costs about a tenth of
# the audio length, so a five-minute song needs well over five minutes here.
CALL_TIMEOUT_SECONDS = 1800

TOOL = {
    "name": "piano_score",
    "description": (
        "Turn one audio link into a piano score: five-line staff images, numbered (jianpu) "
        "notation, MusicXML and MIDI. Only background music is transcribed - vocals are "
        "separated out and dropped, and non-melodic instruments are folded into the left hand. "
        "All three tiers are one arrangement in three keys - the same notes transposed, so easy "
        "has the fewest black keys and is the tier to pick when the point is to play it. This runs "
        "a local CPU pipeline and takes minutes: pass seconds for a quick preview. Files are "
        "written to disk; the paths are in the result."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": (
                    "Audio to transcribe: an http(s) link to the audio file, a page hosting it, "
                    "or a local path. A bare player page is fetched with yt-dlp; if that fails, "
                    "read the page first and pass the direct audio URL it exposes."
                ),
            },
            "tier": {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
                "description": (
                    "Difficulty. Same rhythm, density and register in all three; they differ in "
                    "key signature, so easy has the fewest black keys. Default easy."
                ),
            },
            "seconds": {
                "type": "number",
                "description": (
                    "Only transcribe the first N seconds. Use it to preview a long track "
                    "cheaply: the separation stage costs about a tenth of the audio length."
                ),
            },
            "fingering": {
                "type": "string",
                "enum": ["key", "all", "none"],
                "description": (
                    "Which notes carry a fingering number. key marks only the few places "
                    "a player has to be told how to get there (the default, and the one to "
                    "use); all numbers every note; none leaves them out."
                ),
            },
            "transpose": {
                "type": "integer",
                "description": "Shift the whole arrangement by this many semitones.",
            },
            "out": {
                "type": "string",
                "description": "Directory for the score files. Default a new directory under "
                "the pipeline checkout.",
            },
            "jianpuChars": {
                "type": "integer",
                "description": "How much numbered notation to return inline. Default 1800 "
                "characters; the file holds the rest.",
            },
        },
        "required": ["source"],
        "additionalProperties": False,
    },
}


def log(message: str) -> None:
    """Progress goes to stderr; stdout carries JSON-RPC frames only."""
    print(f"[e2c-mcp] {message}", file=sys.stderr, flush=True)


def _is_repo(path: str) -> bool:
    return bool(path) and os.path.isfile(os.path.join(path, "src", "everything2chart", "cli.py"))


def find_repo() -> str:
    """Locate the everything2chart checkout.

    Order: E2C_REPO (PIANOCHART_REPO still accepted), then walk up from this
    file (the bundle normally lives inside the checkout as <repo>/dsh-bundle),
    then home guesses.
    """
    env = os.environ.get("E2C_REPO") or os.environ.get("PIANOCHART_REPO")
    if _is_repo(env):
        return env

    for start in (HERE, os.getcwd()):
        current = os.path.abspath(start)
        while True:
            if _is_repo(current):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

    for name in ("piano-rescue", "piano"):
        guess = os.path.join(os.path.expanduser("~"), name)
        if _is_repo(guess):
            return guess

    searched = env or "E2C_REPO unset"
    raise RuntimeError(
        "everything2chart checkout not found; set E2C_REPO to the directory "
        f"containing src/everything2chart/cli.py (searched from {HERE}, {searched})")


def find_python(repo: str) -> str:
    """E2C_PYTHON wins, then the repo venv, then whatever runs this file."""
    override = os.environ.get("E2C_PYTHON") or os.environ.get("PIANOCHART_PYTHON")
    if override:
        return override
    venv = os.path.join(repo, ".venv", "bin", "python")
    if os.path.exists(venv):
        return venv
    return sys.executable or shutil.which("python3") or "python3"


def run_bridge(arguments: dict) -> dict:
    repo = find_repo()
    out = arguments.get("out")
    if not out:
        import time

        out = os.path.join(repo, "out", f"mcp-{int(time.time())}")
    os.makedirs(out, exist_ok=True)

    argv = [
        find_python(repo),
        BRIDGE,
        "--source", str(arguments["source"]),
        "--repo", repo,
        "--out", out,
        "--tier", str(arguments.get("tier") or "easy"),
        "--jianpu-limit", str(int(arguments.get("jianpuChars") or 1800)),
    ]
    if arguments.get("seconds") is not None:
        argv += ["--seconds", str(arguments["seconds"])]
    if arguments.get("fingering"):
        argv += ["--fingering", str(arguments["fingering"])]
    if arguments.get("transpose") is not None:
        argv += ["--transpose", str(int(arguments["transpose"]))]

    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=CALL_TIMEOUT_SECONDS,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    # The bridge logs progress first and prints its JSON answer last.
    payload = None
    for line in reversed((proc.stdout or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if payload is None:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        return {
            "ok": False,
            "error": "the transcriber produced no JSON answer:\n" + ("\n".join(tail) or "<empty>"),
        }
    return payload


def render(payload: dict) -> str:
    if not payload.get("ok"):
        text = payload.get("error") or "the transcriber failed"
        if payload.get("hint"):
            text += "\n\n" + payload["hint"]
        return text

    shift = payload.get("transpose") or 0
    lines = [
        f"key {payload.get('key_signature')} (written {shift:+d} semitones), tier {payload.get('tier')}",
        f"right hand {payload.get('right_notes')} notes, left hand {payload.get('left_notes')} notes",
    ]
    similarity = payload.get("similarity") or {}
    if similarity:
        lines.append(
            f"similarity to the record: mean {similarity.get('chroma_cos_mean')}, "
            f"tenth percentile {similarity.get('chroma_cos_p10')}"
        )
    for label in ("musicxml", "midi", "jianpu_path"):
        if payload.get(label):
            lines.append(str(payload[label]))
    for page in payload.get("staff_pages") or []:
        lines.append(str(page))
    if payload.get("jianpu"):
        lines.append("")
        lines.append(payload["jianpu"])
    if payload.get("jianpu_truncated"):
        lines.append("(numbered notation truncated; the full text is in the .jianpu.txt file)")
    return "\n".join(lines)


def handle(request: dict):
    method = request.get("method")
    params = request.get("params") or {}

    if method == "initialize":
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "everything2chart", "version": VERSION},
        }
    if method == "tools/list":
        return {"tools": [TOOL]}
    if method == "tools/call":
        name = params.get("name")
        if name != TOOL["name"]:
            raise ValueError(f"unknown tool: {name}")
        arguments = params.get("arguments") or {}
        if not arguments.get("source"):
            raise ValueError("source is required")
        payload = run_bridge(arguments)
        return {
            "content": [{"type": "text", "text": render(payload)}],
            "isError": not payload.get("ok", False),
        }
    if method == "ping":
        return {}
    raise ValueError(f"unknown method: {method}")


def main() -> int:
    log(f"serving from {HERE}")
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "id" not in request:
            # Notification (notifications/initialized and friends): no answer.
            continue
        response = {"jsonrpc": "2.0", "id": request["id"]}
        try:
            result = handle(request)
            response["result"] = result
        except Exception as error:  # noqa: BLE001 - a frame must always get an answer
            response["error"] = {"code": -32603, "message": str(error)[:600]}
        print(json.dumps(response, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
