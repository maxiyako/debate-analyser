# Spec: recover both speakers when speech overlaps

## Problem

When two people talk at once, the transcript keeps one of them and loses the
other. Two things cause it:

1. `diarize()` in `src/transcription.py` reads pyannote's
   `exclusive_speaker_diarization` first. That output forces one speaker per
   frame, so every crossing is deleted before ASR runs. Every cached
   `data/transcripts/*.diarization.json` has 0.0 s of overlap between turns.
2. `transcribe()` slices the audio per diarization turn and runs
   `kinit/whisper-large-v3-sk` on each slice. Whisper hears one mixed waveform
   and follows one voice. Feeding it overlap-aware turns as-is would not help:
   both speakers' slices would hold the same mixed audio, and Whisper would
   transcribe the mix twice under two labels.

As a result the `(cez seba)` marker, which `_assign_speaker` sets when a
second speaker overlaps a segment, never fires on current transcripts: with
exclusive turns, every slice overlaps exactly one speaker. The interrupter's
words are either absent or attributed to the dominant speaker.

## Goals

1. Overlap regions are detected and kept as two (or more) active speakers.
2. In regions with two active speakers, each speaker gets their own line with
   their own words, transcribed by the same Slovak Whisper model.
3. Recovered overlap lines stay marked `(cez seba)`, so quote validation and
   fact-checking treat them as low-trust, as they do today.
4. Clean speech outside overlap regions is transcribed exactly as today.
5. The gain is measured on a hand-made gold set before the feature is enabled.

## Non-goals

- Replacing `kinit/whisper-large-v3-sk` or using a multi-speaker ASR model
  (DiCoW, cloud APIs).
- Enrollment-conditioned target-speaker extraction (WeSep, SpEx+). The
  separator sits behind an interface so it can be added later and scored on
  the same gold set.
- Separating regions with three or more simultaneous speakers. They fall back
  to today's behaviour.
- Re-transcribing episodes that already have transcripts. Old caches,
  transcripts and reports stay as they are.
- Changing how recovered lines are used downstream. They stay low-trust.

## Approach

Blind two-speaker separation on overlap regions only, then assignment of each
separated stream to a diarized speaker by voice similarity.

Alternatives considered:

- **Target-speaker extraction conditioned on an enrollment clip** (WeSep,
  REAL-TSE BSRNN). Scales to N speakers, but public weights are research
  previews trained on English LibriMix, a poor match for Slovak TV audio.
- **PixIT** (`pyannote/speech-separation-ami-1.0`). Joint diarization and
  separation over the whole episode, but it pins pyannote.audio 3.3.2 (we run
  4.0.7), is trained on AMI meetings only, has known output-scale problems, and
  would replace the diarizer.

ClearerVoice `MossFormer2_SS_16K` is the separator: 16 kHz, trained on large
open and private data, language-agnostic, runs on CPU or MPS.

## Design

### Diarization

`diarization_model_id` defaults to `pyannote/speaker-diarization-community-1`
(better error rates than 3.1, native to pyannote 4, VBx clustering). This
change is independent of the feature flag.

`diarize()` returns both outputs of `DiarizeOutput`:

- `turns`: `speaker_diarization`, overlap-aware.
- `exclusive_turns`: `exclusive_speaker_diarization`.

Both are cached in `{episode_id}.diarization.v2.json` as
`{"model": ..., "turns": [...], "exclusive_turns": [...]}`. The old
`{episode_id}.diarization.json` (a bare list of exclusive turns) is never read
as overlap-aware. If only the old cache exists, diarization reruns.

With the flag off, the pipeline uses `exclusive_turns` exactly as today.

### Flow with `overlap_separation_enabled`

1. **Overlap regions.** From `turns`, find spans where two or more speakers are
   active. Merge spans less than 0.3 s apart. Drop spans shorter than
   `overlap_min_seconds`. Each kept region records its start, end and the set
   of active speakers. The separator receives the region padded by
   `overlap_pad_seconds` on both sides; transcript timestamps use the unpadded
   region.
2. **Enrollment.** For each speaker, collect their turns from `turns` that do
   not intersect any overlap region (before padding is applied), longest
   first, until `enroll_max_seconds` is reached. Embed the audio with the
   community-1 embedding model (`Model.from_pretrained(
   "pyannote/speaker-diarization-community-1", subfolder="embedding")`, whole
   window) and average the vectors. A speaker with less than
   `enroll_min_seconds` of clean speech gets no enrollment. The pipeline's
   cluster centroids are not used, because VBx centroids may sit in a
   different space from raw embeddings.
3. **Separation.** For each region with exactly two active speakers, both
   enrolled, run the separator on the padded mixed audio. It returns two
   streams. Peak-normalize each stream.
4. **Assignment.** Embed each stream with the same embedding model. Compute
   cosine similarity to the two active speakers' enrollments. Of the two
   possible pairings (stream 1 to speaker A and stream 2 to speaker B, or the
   reverse), pick the one with the higher summed similarity. A stream whose
   similarity to its paired speaker is below `overlap_assign_min_sim` is
   dropped.
5. **Transcription.**
   - Outside regions: today's slicing over `exclusive_turns`, with each slice
     trimmed at region edges so no mixed audio is transcribed.
   - Inside a separated region: Whisper on each assigned stream, cropped back
     to the unpadded region. Each produces one segment with that speaker and
     `crosstalk=True`.
   - Inside a region that falls back (see below): the `exclusive_turns` slice
     for that span, with `crosstalk=True`.
6. **Merge.** ASR segments carry their speaker, so the merge no longer infers
   it from time overlap. A segment with `crosstalk=True` never merges with a
   neighbour that has `crosstalk=False`, and two cross-talk segments of the same
   speaker merge only within one region. Two lines with the same start time
   are ordered with the speaker who was already talking before the region
   first.

### Fallback

A region falls back to the exclusive winner's slice, marked `(cez seba)`, when:

- three or more speakers are active;
- either active speaker has no enrollment;
- both streams are dropped at assignment;
- `clearvoice` cannot be imported or the separator raises.

If only one stream survives assignment, that speaker gets the separated line
and the other active speaker gets nothing for the region. An import failure
disables separation for the whole episode with one warning. Per episode, one
summary log line reports regions found, separated and fallen back by reason.

### Transcript shape

The line format stays `Label [MM:SS]: text`:

```
Speaker A [12:03]: ...pred prerušením
Speaker A [12:41]: (cez seba) prekryté slová A
Speaker B [12:41]: (cez seba) slová B cez A
Speaker A [12:44]: ...pokračovanie
```

`src/transcript_lines.py`, `src/validation.py` (excludes `(cez seba)` lines
from reliable quotes) and `reel_generator/transcript.py` (already handles a
short overlap sharing the long turn's timestamp) need no change.

### Units

New module `src/overlap.py`:

- `Region` dataclass: `start`, `end`, `speakers: frozenset[str]`.
- `find_overlap_regions(turns, min_seconds, merge_gap) -> list[Region]`. Pure.
- `select_enrollment(turns, regions, max_seconds, min_seconds)
  -> dict[str, list[tuple[float, float]]]`. Pure; picks the clean spans.
- `Embedder`: wraps the community-1 embedding model; `embed(audio) -> vector`.
- `Separator` protocol: `separate(audio) -> list[np.ndarray]`.
  `MossFormerSeparator` implements it and imports `clearvoice` lazily.
- `assign_streams(stream_vecs, enroll_vecs, min_sim) -> dict[str, int]`. Pure;
  maps speaker to stream index.
- `trim_spans(spans, regions) -> list[tuple[float, float]]`. Pure; removes
  region time from exclusive slices.

`src/transcription.py` keeps the Whisper loop and gains a way to transcribe
in-memory chunks (separated streams) alongside slices of the episode audio.
`Segment` gets its speaker from the slice it came from.

### Configuration

In `config.py`:

| Setting | Default |
| --- | --- |
| `diarization_model_id` | `pyannote/speaker-diarization-community-1` |
| `overlap_separation_enabled` | `False` |
| `overlap_min_seconds` | `0.5` |
| `overlap_pad_seconds` | `0.25` |
| `overlap_assign_min_sim` | `0.3` |
| `enroll_max_seconds` | `30.0` |
| `enroll_min_seconds` | `3.0` |

`clearvoice` is an optional dependency, listed in `requirements.txt` under a
comment, and only imported when the flag is on. Installing it must not change
the installed torch, torchaudio or transformers versions; the implementation
plan verifies this before anything else.

## Evaluation

Two commands join the `src/eval.py` click group.

**`overlap-candidates <episode_id>`** diarizes with community-1 (or reads the
v2 cache) and writes, for the ~10 longest overlap regions of the episode:

- a WAV clip per region (unpadded region plus 2 s of context each side) under
  `data/eval/overlap_clips/<episode_id>/`;
- an entry appended to `data/eval/overlap_gold.json` with episode id, region
  times, active diarization labels, the lines of the existing transcript that
  touch the region, and an empty `gold: {label: ""}` to fill in.

The user listens and writes what each active speaker said, for about 20
regions across 2 episodes.

**`overlap`** scores the filled gold regions on three rows:

1. the existing transcript (3.1 with exclusive turns);
2. community-1 with exclusive turns (flag off);
3. community-1 with separation (flag on).

Rows 2 and 3 transcribe only the gold regions plus context, not whole
episodes. Per row:

- **word recall** per speaker role, where the dominant speaker is the one
  talking before the region and the interrupter is the other: the fraction of
  gold words found, by token alignment, in lines of that speaker that touch the
  region;
- **misattributed words**: gold words of one speaker found in another
  speaker's line.

Gold labels are diarization labels of the community-1 run; row 1 is mapped to
them by maximum time overlap.

Results go to `data/eval/overlap_result.json` and a printed table.

**Gate to enable the flag.** Compared with row 2, row 3 must:

- raise interrupter word recall by at least 0.25 absolute;
- lower dominant-speaker word recall by at most 0.05;
- not increase misattributed words.

## Testing

Unit tests in `tests/test_overlap.py` and `tests/test_transcription.py`, with
the separator, embedder and Whisper stubbed:

- region finding: overlap detection, merging close spans, dropping short
  spans, three-speaker regions;
- enrollment selection: excludes region time, longest first, caps at the
  maximum, no enrollment under the minimum;
- assignment: both pairings, the threshold drop, one surviving stream;
- slice trimming at region edges;
- merge: cross-talk segments never merge into clean neighbours; same-start
  ordering;
- every fallback reason, including a failing `clearvoice` import;
- v2 cache round-trip and the old cache being ignored;
- flag off reproduces today's output from the same exclusive turns.

The gold eval covers the models themselves.

## Rollout

1. Ship with `overlap_separation_enabled=False`; the diarization model switch
   applies to new episodes.
2. Build the gold set and run `overlap`.
3. Enable the flag only if the gate passes. If it fails, the separator
   interface is where an enrollment-conditioned extractor would be tried next.

Old episodes keep their transcripts and reports.
