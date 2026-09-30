# Spec: fact-check quality — political context, deep research, auditable evidence

## Problem statement

The fact-checking phase produces verdicts that are frequently wrong in both
directions, cites weak sources, and spends its budget on claims that do not
matter. The root cause is not the reasoning model; it is that the pipeline
deliberately withholds political context and verifies claims from search
snippets in a single batched pass.

Evidence from the stored reports and two full run logs:

| Symptom | Evidence |
| --- | --- |
| No political background anywhere in the pipeline | Extraction sees only the transcript and the debate date. `_CHECK_RULES` forbids background knowledge outright: "NEVER add names, people, dates or specifics that are not present in the claim/quote/context", and "the verdict MUST follow the search results, NOT your prior knowledge" |
| Fact-check slots wasted on trivia | 591624 checked "the band name Juden Mord translates to murder of Jews" and "Hungary is second-to-last in the EU by living standards", while the substantive dispute over EU funds conditionality got one query |
| Salience measures rhetoric, not consequence | `ExtractedClaim.salience` is rated by rhetorical role inside the transcript (attack, deflection, repetition) with no notion of real-world stakes |
| Retrieval silently discards grounded answers | `vertex_grounded_search` drops every citation outside a ~30-domain allowlist and returns "No grounded results from allowlisted sources" — 16 times in the 591624 run alone |
| Findable facts come back Unverified | 591624 left Hungarian turnout at 11:00 (published by valasztas.hu), the Uhrík invoice (contract register), and a 2019 Fico quote unverified after a single failed query each |
| One query per claim, thirty claims per response | The run log shows exactly one `vertex_grounded_search` call per claim, and all verdicts arrive in one `ClaimCheckList` |
| Verdicts rest on snippets, never on page content | `sk_source_search` returns ~200-character DuckDuckGo bodies; `fetch_page` exists only in the manager pass and returns the first 1800 characters of regex-stripped HTML |
| Sources carry no evidence | `sources` is a bare URL list. Nothing records what the source says, when it was published, or whether it supports or refutes |
| Numeric calibration is off | "Hungary contributes over 2 billion euros annually" against an actual 1.9 billion was graded Misleading/trivial; a human checker calls a hedged figure within 5% true |

Current cost is $3.80–8.50 per debate across 100–290 LLM requests.

### What is already good and must survive

Provenance enforcement via `seen_urls`, debate-date anachronism handling,
transcript quote grounding, dead-link stripping, and the code-side downgrade of
unsupported False verdicts. These are guards the reference experience (asking
Gemini directly) does not have, and they are the reason the pipeline's failures
are quiet rather than catastrophic.

## Goals

1. Give the pipeline the political context a desk journalist would have before
   watching the debate: who these people are, what they are fighting about, and
   what happened in the weeks before.
2. Select claims by political consequence rather than by rhetorical flourish.
3. Research each selected claim properly — multiple queries, multiple languages,
   primary sources, pages actually read — instead of one snippet-level query.
4. Make every verdict auditable down to a dated, quoted passage.
5. Measure verdict quality so changes can be shown to help.

## Non-goals

- Replacing the behavioral, moderator, or scoring phases.
- Predicting who won the debate (covered by `docs/verdict-improvement-spec.md`).
- Judging whether a policy is good. The tool checks factual accuracy only.

## Relationship to `docs/verdict-improvement-spec.md`

That spec covers scoring, reproducibility, and bias. Two points overlap and are
reconciled here:

- **Evaluation.** That spec builds a human-labelled gold set measuring verdict
  agreement. This spec adds an LLM judge measuring evidence quality per claim.
  They are complementary — the gold set says whether the verdict is right, the
  judge says whether the evidence justifies it — and both report through the
  same `src.eval` entry point.
- **Claim selection.** That spec proposes per-speaker extraction quotas so no
  speaker enters scoring with a single checked claim. This spec proposes an
  adaptive consequence threshold. They combine: the threshold applies per
  speaker, with a per-speaker floor, so denominators stay comparable.

One interaction to watch: that spec's K=3 repeated extraction multiplies phase-A
cost, and this spec multiplies phase-B cost. Both dials belong in config with
their cost documented.

---

## Architecture

```
Phase 0   Briefing (new)        political desk research      → DebateBriefing
Phase A   Extract               behavioral + moderator + claims, briefing-aware
          Select                adaptive consequence threshold
Phase B   Research (new)        per-claim native genai loop  → ClaimDossier
Phase B2  Adjudicate (new)      verdict from dossier only, no web access
Phase B3  Judge (new)           evidence-quality review, downgrades only
Phase C   Critic + scoring      unchanged
```

The fact-check manager crew and its rounds are removed; the research loop and
the judge subsume its work.

### Phase 0 — Briefing

A `Political Desk Researcher` runs once per debate, before analysis, over the
corrected transcript and the debate date, using grounded search. Every item it
produces carries a source URL and a date, and nothing dated after the debate is
admitted.

`DebateBriefing` fields:

- `participants` — name, party, office held at the debate date, coalition or
  opposition, prior roles, the policy dossiers they own, with sources.
- `entity_index` — role-to-person mapping valid at the debate date ("prezident
  Poľska", "minister financií"). This is what allows entity resolution without
  guessing from model memory.
- `live_disputes` — the three to six substantive conflicts the debate turns on,
  each with the factual question underneath it.
- `timeline` — dated events in roughly the 90 days before the debate touching
  those disputes.
- `glossary` — dossier terms (konsolidácia, plán obnovy, Benešove dekréty) with
  a neutral one-line definition and status at the debate date.
- `stakes_map` — for each dispute, what a claim about it would mean for a
  voter's judgement. This is the input to consequence scoring.

The briefing agent states facts descriptively and never evaluates who is right.

**Consumption rules.** Extraction receives the whole briefing. The research loop
receives `entity_index`, `glossary`, and the relevant `live_disputes` entry for
each claim, and uses them for query construction and for writing the claim's
`interpretation`. The adjudicator receives the interpretation but **not** the
briefing's factual assertions, so background can never become evidence. The
judge and the Facebook publisher receive the briefing for framing only.

### Phase A — Extraction and selection

Changes to extraction:

- `consequence` (1–5) replaces `salience` as the primary ranking, scored against
  `stakes_map`: does the claim bear on policy, money, responsibility, or record
  in a way that changes how a voter judges the speaker? Rhetorical role survives
  as a secondary multiplier.
- `checkability` classifies each claim as empirical, opinion, prediction, or
  definitional. Only empirical claims enter the research loop; the rest are
  labelled, reported, and excluded from scoring.
- Near-duplicate claims are merged, with repetition recorded as a consequence
  multiplier rather than as separate claims.
- Claim text is written in Slovak, matching the rest of the report.

Selection becomes adaptive: every empirical claim at or above the consequence
threshold (config, default 3) is researched. The threshold is applied per
speaker with a per-speaker floor (the top two claims of any speaker are always
researched), so no speaker reaches scoring on a single checked claim.
`max_claims` survives only as a cost fuse (default 40); anything it cuts is
reported in `critic_notes`.

### Phase B — Per-claim research loop

Implemented directly on `google-genai` function calling, not CrewAI: the loop
needs per-claim tool budgets, per-claim retries, structured output, and
concurrency. CrewAI remains for the behavioral, moderator, extraction, critic,
and publisher crews.

Concurrency is bounded (4–6 workers) and the existing rate-limit backoff wraps
the native client. Tool budget scales with consequence: 12 calls at consequence
5, 8 at 4, 6 at 3.

The loop is forced through a protocol:

1. **Plan** — restate the claim as one precise verifiable question, decompose it
   into atomic subfacts, and name what evidence would settle each. Resolve vague
   references through `entity_index`, recording each resolution.
2. **Search wide** — at least two distinct query formulations, Slovak plus
   English, plus the subject's own language for foreign topics. If press search
   returns nothing usable, a second path against the underlying register or
   dataset is mandatory before concluding anything.
3. **Read** — open the best candidates and extract the passage that addresses
   the claim.
4. **Record** — one `Evidence` object per usable passage.
5. **Stop** — when at least one whole-claim source of adequate tier and date
   exists, or the budget is exhausted. Remaining gaps are recorded explicitly.

Tools:

| Tool | Purpose |
| --- | --- |
| `web_search` | Vertex grounding with Google Search; citations resolved and tiered rather than dropped |
| `open_page` | Real extraction (trafilatura, PDF via pypdf), publication-date parsing, returns the passage window around claim keywords |
| `susr_data` | ŠÚSR DATAcube with explicit metric, period, and territory selection |
| `register_lookup` | Targeted queries against nrsr.sk votes, slov-lex, crz.gov.sk, rozpocet.sk |
| `calc` | Deterministic arithmetic, unit, and percentage conversion |

### Evidence model

```python
class Evidence(BaseModel):
    url: str
    tier: SourceTier      # official_primary | factcheck | wire_quality | other_media | unknown
    title: str
    published: str        # ISO date, "" when unknown
    passage: str          # verbatim excerpt, <= 400 chars
    stance: Stance        # supports | refutes | partial | context_only
    covers: Coverage      # whole_claim | subfact | different_metric
    subfact_id: int | None
    note: str
```

`passage` must occur in the text actually fetched from `url`, checked with the
existing `validation.quote_grounded` machinery. A fabricated or drifted citation
therefore dies in code rather than relying on a prompt rule. When the fetch was
inconclusive (bot-blocked, JavaScript-rendered), the check degrades to a warning
instead of a hard drop, mirroring the existing three-state `url_reachable`
semantics.

The loop emits a `ClaimDossier`: claim id, interpretation, resolved entities,
subfacts, evidence, gaps, and a search log.

### Phase B2 — Adjudication

A separate model call with no tools and no web access. Input: the claim, its
transcript quote, the interpretation, the subfacts, and the evidence list.
Output: verdict, per-subfact resolution, confidence, severity, and a rationale
that cites evidence by index.

Separating evidence gathering from judging is the primary precision control: the
researcher cannot talk itself into a narrative, and the adjudicator can only use
what was actually collected.

### Verdict taxonomy

The five existing verdicts (True, False, Misleading, Unverified, Contested) are
kept so `scoring.py` error weights continue to work. Three additions:

- `confidence` (0–1) on every verdict. Below 0.6 the fact is published but
  carries no scoring penalty.
- `checkability` on the claim (empirical, opinion, prediction, definitional).
- `unverified_reason` (not_found, unverifiable_in_principle, conflicting), so
  "we failed to find it" stops reading like "the speaker was evasive".

Numeric calibration, enforced in the adjudicator prompt and by `calc`:

- A figure hedged with "vyše", "asi", "okolo", "približne" within 10% of the
  true value is True, with a precision note.
- An unhedged figure within 2% is True.
- Beyond that, Misleading with severity set by materiality to the argument.

### Phase B3 — Judge

One call per claim, receiving the claim, interpretation, dossier, verdict, and
rationale — but not the researcher's reasoning trace. It scores five axes:

1. **Support** — does the cited evidence entail the verdict?
2. **Metric fidelity** — same metric, period, and territory as the claim?
3. **Date fit** — no anachronism; fresh enough for the claim's time frame?
4. **Coverage** — is the claim addressed as a whole, or only in part?
5. **Independence** — for False, are there two genuinely independent sources?

It returns issues, a quality score, and may force a **downgrade only**; it can
never upgrade a verdict to False. Code applies the downgrades.

The judge doubles as the evaluation harness. Per-claim quality scores aggregate
to a run-level number stored with the report, and a `--rejudge` path scores the
eleven existing reports to establish a baseline before any change lands.

### Escalation

When the judge fails a high-consequence claim or gaps remain, one further
research round runs with the judge's issues as explicit instructions. Capped at
two rounds.

## Source tiering

The hard allowlist is replaced by tiers assigned from a static map with a
default of `unknown`.

| Tier | Contents |
| --- | --- |
| `official_primary` | Registers and legislation (nrsr.sk, slov-lex, rokovania.gov.sk, crz.gov.sk), statistics offices (ŠÚSR, Eurostat), central banks, courts, EU institutions, IMF, OECD, World Bank, and foreign equivalents (valasztas.hu, foreign ministries and statistical offices) |
| `factcheck` | demagog.sk, konspiratori.sk, AFP and Reuters fact check, EDMO members |
| `wire_quality` | TASR, SITA, Reuters, AP, AFP, BBC, DW, Politico Europe, Euractiv, the current high-FCRI Slovak press, and foreign quality press (telex.hu, hvg.hu, iDnes, Onet and equivalents) |
| `other_media` | Everything not blocked. Wikipedia sits here as pointer-only: usable to locate a primary source, never citable as evidence |
| `unknown` | Domain absent from the map. Treated exactly like `other_media` for verdict purposes, but logged so the map can grow |
| blocked | The existing disinfo list, hard-blocked as today |

Party and government sites are blocked as evidence for whether a claim is true,
but admitted as primary evidence of what someone said or published — a different
question, and the correct way to verify quote attributions.

Rules enforced in code, not prompts:

- False or Misleading requires at least one `official_primary`, `factcheck`, or
  `wire_quality` item with `stance = refutes` and `covers = whole_claim`, or
  full subfact coverage, published at or before the debate date. Otherwise the
  verdict drops to Unverified. This generalises `enforce_verdict_support`.
- A False that damages a named person requires two independent sources;
  otherwise confidence is capped and the fact ships flagged.
- `other_media` or `unknown` alone caps confidence at 0.5 and can never carry
  False.
- STVR and TA3 keep the cross-check tag and cannot be sole authority.
- Contested requires two or more comparable-tier sources in genuine conflict. A
  tier mismatch resolves to the higher tier with reduced confidence instead of
  discarding the information.

## File layout

`agents.py` is 2064 lines covering schemas, tools, crews, and orchestration. It
splits along the boundaries this design already implies:

```
src/briefing.py      DebateBriefing model + research pass
src/research.py      native genai loop: tool registry, budgets, concurrency
src/adjudicate.py    dossier -> verdict (no web access)
src/judge.py         evidence-quality review + run-level quality metric
src/verdicts.py      tier rules, confidence caps, downgrades (replaces reconcile.py)
src/tools/           search.py, page.py, susr.py, registers.py, calc.py, sources.py
src/agents.py        behavioral, moderator, extraction, critic, publisher crews
src/validation.py    + passage grounding against fetched page text
```

New settings in `config.py`: `consequence_threshold`, research tool budgets,
research concurrency, judge enable/model, research rounds cap. `max_claims`
changes meaning from selection rule to cost fuse.

## Cost

| Stage | Estimate |
| --- | --- |
| Briefing | ~$0.30 |
| Extraction | ~$1 (unchanged) |
| Research | $0.40–0.80 per claim, ~20 claims |
| Adjudication + judge | ~$0.10 per claim |
| Critic + publisher | unchanged |
| **Total** | **$12–20 per debate**, against $3.80–8.50 today |

Wall-clock should improve despite the extra work, because per-claim loops run
concurrently instead of as one serialised batch task.

## Migration order

Each step ships and is measurable on its own.

1. **Judge and baseline.** Build the evidence rubric and run `--rejudge` over
   the eleven stored reports. Establishes the quality number every later step is
   scored against, before anything changes.
2. **Briefing.** Independent of the rest and directly targets the main failure.
   Wire it into extraction with consequence-based adaptive selection. Measurable
   immediately by whether trivia still occupies fact-check slots.
3. **Tooling.** `open_page` with real extraction and PDF support, tiered search
   without silent discarding, register and calc tools.
4. **Research loop and adjudicator.** Replace phase B, retire the manager crew,
   behind a config flag so the old path stays runnable for A/B on one debate.
5. **Calibration.** Numeric tolerance, confidence gating in scoring,
   `checkability` and `unverified_reason`, Slovak claim text.

## Risks

- **Rate limits.** Concurrent per-claim loops will provoke Vertex 429s. The
  existing backoff must wrap the native client, and concurrency stays
  configurable.
- **Judge self-agreement.** A model grading its own pipeline is biased toward
  approval. Mitigated by a strict rubric, a separate context excluding the
  researcher's reasoning, and the option to run the judge on a different model.
- **Passage grounding false negatives.** JavaScript-rendered pages return no
  matching text. Degrades to a warning when the fetch was inconclusive.
- **Baseline comparability.** Moving claim text to Slovak breaks comparison with
  stored reports, which is why the baseline is captured in step 1.
- **Tier map maintenance.** An unknown-by-default tier is safe but will leave
  good foreign sources under-weighted until the map grows. The judge's
  independence axis surfaces this.

## Testing

Unit coverage, all offline against fixture dossiers:

- Tier assignment and the tier rules (False without an adequate source, damaging
  False without independence, `other_media` confidence cap).
- Verdict downgrade paths driven by judge output.
- Passage grounding, including the inconclusive-fetch degradation.
- Numeric tolerance (hedged within 10%, unhedged within 2%, beyond both).
- Adaptive selection: threshold, per-speaker floor, cost fuse, non-empirical
  exclusion.

`tests/test_validation.py` and `tests/test_scoring.py` extend rather than get
replaced.

## Definition of done

- A briefing is produced for every debate and visibly drives claim selection:
  no definitional or trivia claims occupy fact-check slots.
- Every published verdict carries dated, quoted, tier-labelled evidence, and
  every passage is verified to occur in the fetched source.
- False and Misleading verdicts without adequate-tier whole-claim evidence are
  impossible by construction, not by prompt instruction.
- The judge's run-level quality score improves against the step-1 baseline on
  the stored reports.
- Cost stays at or under roughly 3x the current per-debate spend.
