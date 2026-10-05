# STVR Debate Analyzer

Pipeline that downloads an STVR episode (e.g. *O 5 minút 12*), transcribes it with Slovak Whisper + Pyannote diarization, and analyzes it with a CrewAI multi-agent system on Vertex AI Gemini.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# system deps: ffmpeg
brew install ffmpeg  # macOS
cp .env.example .env
```

### GCP auth

Set `GCP_PROJECT_ID` in `.env`, then authenticate (local ADC):

```bash
gcloud auth application-default login --project=<your-project-id>
gcloud config set project <your-project-id>
```

For a Cloud Run job / CI, set `GOOGLE_APPLICATION_CREDENTIALS` to a service-account key instead.

Enable APIs: Vertex AI, and optionally Cloud Storage.

Accept [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) terms and set `HUGGINGFACE_TOKEN`.

## Usage

```bash
python main.py --url "https://www.stvr.sk/televizia/archiv/14036/<episode-id>"
python main.py --url "https://www.ta3.com/clanok/1074913/v-politike-tomas-taraba-vs-michal-simecka" \
  --guests "Tomáš Taraba;Michal Šimečka" --moderator "Braňo Král"
# skip agents (transcription only)
python main.py --url "..." --skip-agents
```

ta3 articles must embed a Transistor.fm podcast player (Livebox-only video pages are not supported yet). Outputs for ta3 use ids like `ta3-1074913`.

Outputs:
- `data/transcripts/<id>.txt` — speaker-tagged transcript
- `data/reports/<id>.json` — structured multi-agent report

## Architecture

1. **Ingestion** — source adapters (`stvr` via yt-dlp, `ta3` via Transistor mp3 scrape) → ffmpeg 16 kHz mono WAV
2. **Transcription** — Pyannote diarization + `kinit/whisper-large-v3-sk` → merged lines
3. **Speaker map** — diarization labels (`Speaker A`) are resolved to real people in code (turn-taking heuristics + one small validated LLM call); pass the roster with `--guests "Erik Tomáš;Marián Viskupič"` and `--moderator "<name>"`. An uncertain map degrades scoring instead of guessing names.
4. **Transcript correction** — the LLM only proposes small edits (spelling, word boundaries, known names, punctuation, turn splits); code applies them only when numbers, negation, and content words stay invariant, and logs every decision.
5. **Agents** (Vertex Gemini) — Behavioral Analyst + Moderator Bias Auditor + Fact Extractor → grounded checker (Vertex Google Search grounding) + per-category specialists (ŠÚSR DATAcube, verified Slovak sources via ddgs) → Chief Critic
6. **Deterministic layer** — question audit (which moderator questions got a real answer), claim funnel (extracted → selected → checked), quote/source validation, accusation guard, fair-play scoring → Facebook post
7. **Fact sources** — Vertex Gemini grounding + FCRI-allowlisted Slovak/international sources; no extra API keys

Designed to run locally now; Cloud Run / Vertex Pipeline later via the included Dockerfile.
