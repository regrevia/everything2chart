# Changelog

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
