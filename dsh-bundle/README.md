# dsh-pianochart

The DeepSeek Harness plugin for [everything2chart](https://github.com/regrevia/everything2chart).

One audio link in, one piano score out: five-line staff images, numbered
(jianpu) notation, MusicXML and MIDI.

This bundle is a **Host plugin** that adds one tool, `piano_score`, and mounts
an **MCP server** so the same tool is also reachable as
`mcp__dsh-pianochart__piano_score`. It does not reimplement the transcription: it
resolves the link to a local audio file, runs the documented `everything2chart.cli` in
a child process, and reads the `report.json` that CLI writes. Demucs needs
minutes of CPU and several gigabytes, so it must not run inside the Harness Node
runtime.

## The two entry points

The bundle patch (`cordis.patch.yml`) inserts two rows:

| Row | Module | What it provides |
| --- | --- | --- |
| `dsh-pianochart` | `dsh-pianochart` | Host tool `piano_score`, plus the `pianochart` skill |
| `dsh-pianochart-mcp` | `@deepseek-ai/dsh-mcp-client` | stdio MCP server, tools named `mcp__dsh-pianochart__piano_score` |

The MCP row runs `lib/mcp_server.py`, which is a stdio JSON-RPC server over
the same bridge (`lib/bridge.py`) the Host tool uses. Both cost the
same CPU; pick one. The MCP name is prefixed by the client, so it does not
collide with the Host tool.

Because `@deepseek-ai/dsh-mcp-client` is named by the patch, it must be declared
in this bundle's `dependencies` — the bundle loader rejects a patch that
references a package the bundle does not depend on.

Two settings matter for a real run:

- `toolCallTimeoutMs: 1800000` — the client default is 60 s and a score needs
  minutes, so a default-budget call is cut off mid-pipeline.
- `failOnStartupError: false` — a broken interpreter path leaves the Host
  running instead of failing the whole profile.

The mounted process is owned by the plugin, so the MCP Manager budgets and the
Settings page cannot edit this entry. A change to the patch takes effect when
the Host recomposes — disabling and re-enabling the bundle is enough.

## Install

```
plugin_manager install_bundle <absolute path to this directory>
```

Or copy the directory into `~/.dsh/plugins/` and enable it from **Plugins**.

The pipeline itself must already be on this machine — a checkout containing
`src/everything2chart/cli.py` and a Python environment with its dependencies. This
bundle finds it in this order:

1. `PIANOCHART_REPO`
2. `/Users/apple/Desktop/work/piano`
3. `~/piano`

and the interpreter in this order:

1. `PIANOCHART_PYTHON`
2. `<repo>/.venv/bin/python`
3. `<repo>/.venv/bin/python3`

If neither is found the tool says so instead of failing deep inside the run.

## What the tool does with a link

Three branches, and they are not interchangeable:

| Source | Route |
| --- | --- |
| A local path or `file://` | Used as-is. |
| An http(s) URL that *is* the audio (`Content-Type: audio/*` or `video/*`, or a known media extension) | `curl`, then the pipeline. |
| An http(s) URL that is a page hosting the audio | `yt-dlp`; if that fails the result says so and asks for the media URL. |

Envoy is deliberately not used for direct media URLs: its static reader rejects
a bare audio URL with `unsupported_source`. For a page behind a login, read the
page with Envoy first and pass the audio URL (or the saved file) as `source`.

## Arguments

| Name | Meaning |
| --- | --- |
| `source` | Required. Link or path. |
| `tier` | `easy` (default), `medium`, `hard`. One arrangement in three keys — the same notes, rhythm and left-hand voicing, transposed so `easy` has the fewest black keys and `hard` the most. A tier therefore sits a few semitones above or below the others; pick `easy` when the point is to play it. |
| `seconds` | Transcribe only the first N seconds. Separation costs roughly a tenth of the audio length on CPU, so this is how to preview cheaply. |
| `fingering` | `key` (default), `all`, `none`. `key` marks only the notes where the hand has to be told how to get there; `all` numbers every note; `none` leaves the staff unmarked. The numbers come from a cost model adapted from [performer](https://github.com/michaeljclark/performer) (MIT) — a suggestion, not a teacher's marking. |
| `transpose` | Shift the whole arrangement by semitones. |
| `out` | Output directory. Defaults to `out/link-<timestamp>` under the pipeline checkout. |
| `jianpuChars` | How much numbered notation to return inline (default 1800). The file holds the whole thing. |

## Output

Bounded by design: a few summary lines, the paths to every artifact, and the
first `jianpuChars` characters of the numbered notation. The score directory
holds the rest, including one PNG per staff page and a piano-roll image.

`report.json` carries a `fingering_check` block: how many finger numbers the
arrangement holds, how many reached the staff file, and whether every marked
chord is monotone in its hand's direction. If those disagree the run says so
rather than shipping a score whose digits do not match its notes.

Similarity is a chroma cosine against the source audio: `chroma_cos_mean` over
the whole track and `chroma_cos_p10` for the worst tenth. It measures tonal
content, not whether the part is playable; the tier is what controls difficulty.

## Limits

- Vocals are separated out and dropped on purpose — only the background music is
  transcribed.
- A short clip (seconds, not minutes) gives a readable score but the key
  estimate and the harmony have little to work from; do not read a low
  similarity on a 6-second sample as a failure of the pipeline.
- Only CPU is used. A full track takes minutes.

## Licence

MIT. The pipeline it drives has its own licences; see that project's README.
