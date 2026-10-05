# Spec: multi-source media ingest (STVR + ta3 podcast)

## Goal

Accept debate URLs from more than one broadcaster. First new source: **ta3.com
podcast articles** that embed a Transistor.fm player (e.g. *V politike*). Same
CLI `--url` auto-routes by host; after download, existing ffmpeg → ASR → agents
pipeline stays unchanged.

## Decisions (locked)

| Topic | Choice |
| --- | --- |
| Scope | ta3 **podcast** articles only (Transistor iframe → mp3). No Livebox video. |
| Episode id | Prefixed: `ta3-<articleId>` (e.g. `ta3-1074913`) |
| CLI | Auto-detect source from URL host; no required `--source` flag |
| Architecture | **Source adapter** pattern (`MediaSource` + registry) for future sources |
| Guests / moderator / debate date | Still CLI-only (no page scrape) |
| HTTP client | `requests` (already in deps) |

## Out of scope

- Livebox / Kaltura video on ta3
- Scraping title, guests, or air date from the article
- Custom yt-dlp extractors for ta3 / Transistor
- Generic “any page with audio” crawler
- Renaming the project / package away from “stvr”

## Architecture

```
--url
  → resolve(url)           # first MediaSource where matches(url)
  → source.download(url)   → Path under data/raw/
  → extract_audio(media)   → data/audio/<id>.wav   # shared
  → run_transcription + agents                     # unchanged
```

### `MediaSource` ABC (`src/sources/base.py`)

- `name: str`
- `matches(url: str) -> bool`
- `episode_id(url: str) -> str`
- `download(url: str, settings: Settings) -> Path`

### Registry (`src/sources/__init__.py`)

Ordered list of source instances. `resolve(url)` returns the first match.
Unknown host → clear error (Click `UsageError` / `ValueError` with supported hosts).

### Ship now

1. **`StvrSource`** — current yt-dlp STVR download + `/archiv/...` id parse
2. **`Ta3PodcastSource`** — HTML scrape → Transistor embed → mp3 download

### Later sources

Add `src/sources/<name>.py`, register in `SOURCES`. No changes to ASR or agents.

## Data flow — ta3 podcast

Proven path for
`https://www.ta3.com/clanok/1074913/v-politike-tomas-taraba-vs-michal-simecka`:

1. GET article HTML.
2. Find iframe `src` matching `share.transistor.fm/e/<slug>` (example: `5bc6ce9e`).
3. GET embed page; extract `trackable_media_url` /
   `https://media.transistor.fm/<slug>/<hash>.mp3`.
4. Download mp3 → `data/raw/ta3-<articleId>.mp3` (optional GCS upload, same as STVR).
5. `extract_audio` → `data/audio/ta3-<articleId>.wav`.

**Article id:** capture group from `/clanok/(\d+)/` in the URL.

### Errors (hard fail, clear message)

- URL host not in registry
- ta3 URL but no Transistor iframe (e.g. Livebox-only article)
- Embed page has no mp3 / media URL
- HTTP failure downloading article, embed, or mp3

## File layout

```
src/sources/
  __init__.py      # SOURCES registry + resolve(url)
  base.py          # MediaSource ABC
  stvr.py          # move yt-dlp download + episode_id_from_url from ingestion.py
  ta3.py           # Transistor scrape + mp3 download
src/ingestion.py   # thin: resolve → download → extract_audio
main.py            # --url help lists supported hosts
README.md          # document ta3 podcast URL usage
tests/test_sources.py
```

`episode_id_from_url` (STVR) moves into `StvrSource` (or a thin re-export from
`ingestion` if anything still imports it — prefer updating call sites).

## Wiring

`ingest(url)` becomes:

1. `source = resolve(url)`
2. `ep_id = source.episode_id(url)`
3. `media = source.download(url, settings)`
4. `audio = extract_audio(media, settings)`
5. return `(ep_id, media, audio)`

STVR behavior and output paths must stay equivalent for existing archive URLs.

## Tests

Unit tests only; no live network:

- `matches` / `episode_id` for representative STVR and ta3 URLs
- ta3 happy path: fixture article HTML (Transistor iframe) + fixture embed HTML
  (mp3 URL) → resolved media URL / id (mock `requests.get`)
- ta3 article without iframe → raises
- `resolve` unknown host → raises
- STVR id regex regression (`/archiv/14036/<id>` and `/archiv/<id>`)

## Usage (after implementation)

```bash
python main.py --url "https://www.stvr.sk/televizia/archiv/14036/<id>"
python main.py --url "https://www.ta3.com/clanok/1074913/v-politike-tomas-taraba-vs-michal-simecka" \
  --guests "Tomáš Taraba;Michal Šimečka" --moderator "Braňo Král"
```

Outputs use `ta3-1074913` as the episode id stem under `data/`.

## Success criteria

1. Same `--url` works for STVR and ta3 podcast articles without extra flags.
2. ta3 run produces `data/raw/ta3-<id>.mp3` (or equivalent) and WAV, then full pipeline.
3. Adding a third source = new module + registry entry only.
4. Existing STVR ingest tests / behavior unchanged.
5. Missing Transistor embed fails fast with an actionable error (not silent wrong media).
