# Piano score from one link

`piano_score` turns an audio link into a playable score. This skill only says
how to drive it; the tool description and its arguments are already in view.

## Pick the tier before calling

| What the person wants | `tier` |
| --- | --- |
| To sit down and play it | `easy` |
| The same piece with more of the record's colour | `medium` |
| As close to the record as possible | `hard` |

The three tiers are one arrangement: the same notes, the same rhythm, the same
left-hand voicing. What changes is the key they are written in — `easy` uses the
fewest black keys, `hard` the most — so a tier sits a few semitones above or
below the others, and playback restores the original pitch so the record's key
still sounds. Name that key when you hand over the images; the result states it.
Do not promise an easier rhythm or fewer notes for `easy`; that is not what differs.

## Fingering

The score carries finger numbers, `key` by default: only the few places where the
hand has to be told how to get there — the first note, after a rest, a leap, a
crossing, a change of grip. Leave it alone. Reach for `all` only if the person
asks for a number on every note (a beginner reading the score note by note), and
`none` if they want the staff clean. The numbers are a suggestion from a cost
model, not a teacher's marking: say so if the person is learning the piece, and
let them change one rather than defend it.

`fingering` counts where a number was chosen; `fingering_check` is read back out
of the written MusicXML afterwards. Quote the second when someone asks whether
the score really has them: `in_staff_file` equal to `in_arrangement` and
`non_monotone_strikes` at 0 means every chosen number survived the export and
each hand's numbers run the right way. If the two counts differ, the export
lost numbers and that is the thing to report.

## Order of operations

1. Long audio: call once with `seconds` (60–120) first. Separation costs about a
   tenth of the audio length even on CPU, so the preview tells you the key, the
   tempo and whether the stem is usable before spending minutes.
2. Then call again without `seconds` for the real score, and say how long it will
   take instead of waiting in silence.
3. Hand over the page PNGs with absolute paths. The `jianpu` text in the result
   is truncated: if the person wants the notation in chat, read the `.txt` file
   rather than assuming the tail is missing from the score.

Same link twice in one conversation: reuse the earlier paths, do not re-run.

## When the link is a page, not a file

The result says so in `error`/`hint` and exits without artifacts. Then read the
page itself (Envoy, or a plain fetch if it is public) and pass the audio URL it
exposes — or the downloaded file — as `source`. Do not retry the same page URL
and do not switch tiers hoping it behaves differently.

## Reading the result

- `similarity.chroma_cos_mean` is tonal agreement with the source over the whole
  track; `chroma_cos_p10` is the worst tenth. They measure pitch content, not
  playability. Never present them as a quality score out of 1 for the person's
  ear.
- A short clip scores lower than a full track for lack of material. Do not call a
  low number on a fifteen-second sample a failure.
- `right_notes`/`left_notes` are the two hands of the arrangement; the left hand
  carries bass and harmony, so it is normally the denser one. Both hands are
  written to be playable: at most 3 keys under one hand at a time, never wider
  than an octave.

## Limits worth stating out loud

- Vocals are dropped on purpose — only the background music is transcribed.
- A drum kit does not become playable piano; percussion can only show up as
  rhythmic weight, not as notes.
- If the person asks for a specific instrument's part that was mixed into the
  background, that part is not separable here; say so instead of inventing it.
