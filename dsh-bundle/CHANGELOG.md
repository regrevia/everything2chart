## 0.3.0

Numbered notation could print a beat that was short. A beat holding two notes
half a beat apart came out as two adjacent sixteenths, so the printed beat
covered 0.75 of a beat and the bar after it had drifted; a beat with no
right-hand note printed the left hand alone with no rest in front of it, and a
beat with no left-hand note dropped the right-hand note entirely. All three are
now spelled out.

| case | before | after |
| --- | --- | --- |
| interior hole in a beat | `3'__ 2'__` (0.75 beat) | `3'__ 0__ 2'__` |
| right hand silent, left hand plays | `| ([1,,]) |`, the `2'` vanished | `| 0([1,,]) |` |
| three-four, short final bar | bar summed to 0 | bar sums to the metre |

On the two sample recordings the defect was latent rather than audible: the same
arrangements rendered through the old and the new spelling drop 0 of 87 and 0 of
139 notes. Where it does bite it changes 6 short cells per tier on clip1 and none
on clip2, and the staff output, MIDI and `report.json` stats are byte-identical
either way.

- `scripts/selftest.py` is now a release gate: 11 pure-stdlib cases over the
  numbered notation, run with `PYTHONPATH=src python scripts/selftest.py`. The
  previous spelling fails 5 of them. There is no pytest in this project, so this
  script is what a release has to pass.
- The MCP server now honours `E2C_REPO` and `E2C_PYTHON`. The launcher already
  documented those names, but `lib/mcp_server.py` only read the legacy
  `PIANOCHART_*` pair, so the missing-checkout error told you to set a variable
  that the server itself never read. Legacy names still work.
- `--no-separate` no longer reads a stem cache left by an earlier demucs run. The
  flag means "transcribe the mix", but the cache check came first, so passing it
  could still transcribe the separated accompaniment: on the same 21.0 s clip,
  `--seconds 21` (cached) reported the key as `G` while `--seconds 22` (not
  cached) reported `C`. Both now report `C`. The default path, which does run
  separation, still reuses the cache.

# Changelog

## 0.2.3

The tier the skill calls easiest was, on both sample recordings, the hardest to
play. Tiers were ordered by key signature, on the theory that a key with no
accidentals is the easiest to read. That is a proxy, and it inverts whenever the
piece is written around a key other than C: notating it in C turns each of the
piece's own accidentals into a black key. Measured on the black keys actually
under the hand:

| sample | tier | before | after |
| --- | --- | --- | --- |
| clip1 | easy / medium / hard | 27 / 3 / 9 | 2 / 3 / 6 |
| clip2 | easy / medium / hard | 70 / 16 / 24 | 0 / 16 / 24 |

Tiers now rank on measured black keys, with the signature breaking ties, so
`easy` is the one with the fewest. The notes and rhythm are untouched, so the
tier-isomorphism guarantee still holds and note-F1 did not move.

Also fixes bars that summed to the wrong length — the user complaint "4/4 的曲子
有时候多一拍或者少半拍". A token that began late inside a beat kept its unclamped
duration (bars came out at 4.25 beats), leading rests were emitted for events
that started in an earlier beat, and holds were derived from note length rather
than the note's end. Scanning every bar for its total duration, all tiers:

| sample | jianpu before | after | staff |
| --- | --- | --- | --- |
| clip1 (96 bpm) | 4/8 bars at 4 beats | 8/8 | 16/16 unchanged |
| clip2 (72 bpm) | 11/12 | 12/12 | 24/24 unchanged |

Notes are no longer carried past the last bar line either (they were invisible,
both renderers clipped them, but they made counts disagree: clip1 92 vs 87).
The skill no longer says the same thing about the key twice.


## 0.2.2

Fixes the checkout probe in `index.js`, which `0.2.0` shipped still looking for
`src/pianochart/cli.py`. After the package moved to `src/everything2chart/`,
that path never exists, so the tool could not find the pipeline and answered
"not on this machine" on a machine where it plainly was. Also:

- reads `E2C_REPO` / `E2C_PYTHON` first, with `PIANOCHART_*` kept as a fallback,
  matching what `cordis.patch.yml` and the README already documented;
- looks in `~/everything2chart` as well as `~/piano-rescue` and `~/piano`;
- reports its version from `package.json` in the MCP `initialize` handshake, so
  the handshake can no longer claim a version the package is not.

Anyone on `0.2.0` should move to this one.


## 0.2.1

Functionally fine — it carries every fix in `0.2.2` — but the MCP `initialize`
handshake still hardcoded `0.2.0`, so a client asking who it was talking to got
the wrong answer. Use `0.2.2`.


## 0.2.0

The project is called `everything2chart` from here on. One audio link in, one
playable score out, and the piano is where it starts rather than what it is:
other instruments are next, and the pipeline is portable to any host that can
speak MCP. The GitHub repository is
[`regrevia/everything2chart`](https://github.com/regrevia/everything2chart)
and the Python package is `everything2chart`, so `python -m everything2chart.cli`
is the documented entry point.

Two names did not move, on purpose:

- The npm package stays `pianochart-dsh`. An npm name is fixed per package, so
  "renaming" it means publishing a second package and deprecating the first;
  that is a migration, not a rename, and it is not worth a broken install for
  anyone on `0.1.x`.
- This plugin keeps its catalog name `dsh-pianochart`, which is the only place
  `pianochart` survives: a plugin catalog wants a name that says which product
  it plugs into, and the skill it registers is still invoked as `pianochart`.

What moves inside the bundle: `lib/pianochart_bridge.py` -> `lib/bridge.py`,
`lib/pianochart_mcp.py` -> `lib/mcp_server.py`. The env overrides are now
`E2C_REPO` / `E2C_PYTHON` / `E2C_MCP`, and the old `PIANOCHART_*` spellings are
still read as a fallback so an existing setup keeps working.


## 0.1.8

- The docs stop claiming the tiers share a register. They share the notes,
  rhythm and left-hand voicing, and each is written in its own key, so a tier
  sits a few semitones above or below the others; saying "same register" was
  a promise the transposed score cannot keep. The preset `range` is now
  described for what it is: the working window the arrangement is built in,
  before the final shift.


## 0.1.7

- The three tiers are now the same notes, which is what the docs have been
  promising. The arrangement was being written in the tier's own notated key,
  so snapping the left hand to that key's scale could absorb one source note
  up, down, or not at all, and how many notes the tier had followed from
  that: on one separation pass the three scores differed by 19-23 notes each
  way once each tier's transpose was removed. The arrangement is now built in
  the source key and shifted once at the end, so `easy`, `medium` and `hard`
  are the same 2187 pitches in 690 right-hand and 1497 left-hand notes, and
  only the key signature differs.
- The fingering check in the repository pairs its staff file by name.
  `--preset all` writes three staff files a second apart, so picking the
  newest attached `easy` and `medium` to the `hard` score and reported a
  disagreement that was not there.


## 0.1.6

- The result says whether the numbers actually reached the score. `fingering`
  has always reported how many places were chosen, which cannot tell a page
  full of fingerings from one where the export silently dropped them. A new
  `fingering_check` carries the independent count read back out of the
  written MusicXML, so `in_staff_file` equal to `in_arrangement` with
  `non_monotone_strikes` at zero is a fact rather than an assumption.
- A local file path is accepted as it arrives from a file manager. A leading
  `file://` is stripped before the path is resolved instead of being taken
  as part of the filename.


## 0.1.5

- The plugin layer passes `fingering` through. 0.1.4 accepted the argument in
  the MCP bridge and in the tool schema but the bundled tool never forwarded
  it, so every call from the plugin was scored with the default.
- The result reports the fingerings, and the rendered summary says how many
  were marked out of how many notes.
- The cost model charges a hand for playing away from its own end of the
  keyboard, so the same motif fingers differently low and high.
- `report.json` carries `fingering_check`: the number of finger numbers in the
  arrangement, the number in the staff file, and whether every marked chord is
  monotone in its hand's direction. A run that disagrees says so.


## 0.1.4

- Fingering. The score carries finger numbers, `key` by default: the few notes
  where the hand has to be told how to get there. `all` numbers every note;
  `none` leaves the staff unmarked. New `fingering` argument on the tool.
- `serve.py`, a small local HTTP runner (`POST /api/score`) so a web page can
  hand a link to this machine. It takes the same `fingering` argument.
- Fingering counts are recorded in `report.json` and returned by the tool.


## 0.1.2

- Peer range for `@deepseek-ai/dsh-mcp-client` covers both prerelease lines
  (`>=0.1.0-rc.1 <0.2.0-0 || >=0.2.0-rc.1 <0.3.0-0`), so installing on a 0.1
  harness build no longer resolves to nothing.


## 0.1.1

- `@deepseek-ai/dsh-mcp-client` moved from `dependencies` to `peerDependencies`:
  the host owns the client, the bundle must not pin a second copy.
- Icon redrawn: one note on a faint staff, replacing the previous mark that
  read as a grid at small sizes.


## 0.1.0

First release.

- Host tool `piano_score`: one audio source in, score out.
- Skill `pianochart`: which tier to pick, previewing long audio with `seconds`,
  handing a page link to Envoy when the tool reports it is not an audio file.
- MCP server `pianochart` on stdio, so other MCP clients can call the same
  pipeline.
- Outputs per tier: MusicXML, MIDI, numbered notation text, page PNGs and a
  piano-roll image.

