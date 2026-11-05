# Spec: claim interpretation before fact-checking

## Problem

Checkers often refute a claim the speaker never made. An audit of the 62
False/Misleading verdicts in the 15 reports from the current pipeline (620752
onward) found about 15 wrong and 14 doubtful. The largest single pattern
(about 10 cases) is that extraction rewords the claim and the verdict is judged
against the rewording:

- Drucker said schools "dostávajú informácie"; the claim added "automatically"
  (9iWrH8dOX9k#6).
- Šimečka said his mother "sa nedopustila trestného činu"; the claim became "nie
  je obvinená" (mcsMbNlpsRw#26).
- Viskupič said Bratislava VW "prehráva internú súťaž" for new models (model
  allocation); `context` said the claim was about production start, and the
  checker refuted it with Passat/Superb/Q9/Cayenne production starts
  (Oo4HPBwi62g#12).

Checkers build queries from claim, quote and the one-sentence `context`, never
see the transcript, and treat `context` as binding. The fidelity gate
(`src/fidelity.py`) catches 12 of the 15 wrong accusations after the fact, but
it takes the rationale's findings as given, so it cannot see that the checker
researched the wrong thing. Three misses remain:

| Case | What was said | What was checked |
| --- | --- | --- |
| ta3-1074913#9 (Šimečka, 09:04) | Transaction-tax abolition **for živnostníci** passed with opposition and SNS votes (passed Sept 2025) | Full abolition of the tax |
| 2898725#4 (Karas, 20:21) | Over the 3 consolidation packages total debt grew by 30 bn since the government took office | 7.4 bn package total and a cumulative-government debt figure; manager flipped Unverified to False/fabrication |
| 623275#17 (Remišová, 50:05) | All EU money "my" (the government) negotiated for 2021–2027 was 24 bn | "my" read as her ministry only |

Only a better starting point fixes these: an explicit reading of what was
asserted, made before any search, from the transcript around the quote.

## Goals

1. Every selected claim reaches the checkers with an explicit, faithful reading
   of what the speaker asserted, with qualifiers kept and referents resolved.
2. Claim text that drifted from the quote is corrected before checking, with
   the original kept for audit.
3. Checkers and the manager know what would and would not refute the claim, and
   see related claims of the same debate.
4. Interpretation can never become a channel for background facts or model
   memory to act as evidence.
5. The effect is measurable on the existing gold set without a full pipeline
   rerun.

## Non-goals

- Cross-debate memory or a debate-level fact table. Using earlier reports (e.g.
  617000#14 for the transaction-tax case) would let prior verdicts act as
  evidence.
- Changing selection, scoring, the sub-fact rule, or provenance enforcement.
- Re-checking stored reports in production. They stay as they are.
- The per-claim research loop of `docs/factcheck-quality-spec.md`.

## Relationship to `docs/factcheck-quality-spec.md`

That spec plans a per-claim research loop whose first step, Plan, restates the
claim as one precise question, resolves vague references through
`entity_index`, and decomposes it into subfacts, followed by a separate
adjudicator that receives the `interpretation` but not the briefing.

This spec is a smaller step toward it and replaces the restate-and-resolve part
of Plan. The reading is produced once per debate, by a model with no tools that
never sees a verdict, from the transcript window rather than from the claim
text. A future research loop consumes `interpretation`, `referents` and
`refutation_would_be` as its plan input instead of producing them, and its
adjudicator receives them as that spec already prescribes. Subfact
decomposition stays with the checker, as today.

## Architecture

```
Phase A   Extract                 unchanged
          Select                  unchanged
Phase A2  Interpret (new)         one batched pass over selected claims -> ClaimReading
          apply_readings (new)    code guards, claim rewrite on drift
Phase B   Grounded + specialists  receive interpretation, not_asserted, refutation_would_be, related claims
Phase B2  Manager rounds 1-3      receive related claims with current verdicts
          Reconcile, validation, accusation guard
          Fidelity gate           also receives interpretation and not_asserted
```

### Phase A2 — Interpretation pass

New module `src/interpretation.py`, built like `src/fidelity.py`: pure
functions with the LLM injected as a callable, so everything except the call
itself is testable offline. Called from `run_analysis` right after selection,
behind `settings.interpretation_enabled`. It uses `generate_json` from
`src/llm.py` with `settings.judge_model or settings.gemini_model`, like the
fidelity gate.

**Input per claim:** id, speaker, claim, quote, `time_reference`,
`time_window`, and a transcript window of `settings.interpretation_window`
lines (default 6) before and after the quoted line. The window is located in
code: each line of the claim's speaker is scored against the quote with
`validation.best_window` and the best one wins, with all lines as a fallback.
A score below the quote-grounding threshold counts as no match. If no line matches, the window is empty and the reading relies on
the quote alone. `fidelity.transcript_window` stays as is; it works from a
fact's timestamp, which does not exist yet at this stage.

**Input once per call:**

- Sibling list: every selected claim as `id, speaker, claim`, so the model can
  link related claims and read a claim in light of what the speaker said
  elsewhere.
- From the briefing: only participants with their office at the debate date
  (`entity_index`) and the glossary. No timeline, disputes, or stakes. These
  exist only to resolve referents ("my", "oni", "minister financií", "keď
  preberali vládu").
- The debate date.

Claims are sent in chunks of `settings.interpretation_chunk` (default 20) to
keep the structured output below the length where Gemini JSON has been cut
short before. Every chunk gets the full sibling list.

**Instructions, in substance:** you do not check facts and do not use your own
knowledge of them. Read the quote as a listener in the studio would understand
it from the window. Write the proposition the speaker asserted, keeping the
speaker's own qualifiers ("pre živnostníkov", "za posledné 3 roky", "my") and
adding none ("automaticky", "iba", "všetci", "definitívne"). Compare the
extracted claim with the quote and window and list every drift. Name the
stronger or different readings a checker might wrongly test. Say what evidence
would refute the proposition. Never state whether the claim is true.

The new `ExtractedClaim` fields appear in the extractor's output schema, so
`run_analysis` clears them right after extraction (`clear_interpretation`);
only the pass may fill them. Phase B and B2 move from `run_analysis` into
`check_claims(claims, settings, debate_date, *, pool, pool_facts)`, which the
replay reuses.

### Output model

```python
class DriftKind(str, Enum):
    ADDED_QUALIFIER = "added_qualifier"
    DROPPED_QUALIFIER = "dropped_qualifier"
    WRONG_REFERENT = "wrong_referent"
    CHANGED_METRIC = "changed_metric"
    CHANGED_PERIOD = "changed_period"


class Drift(BaseModel):
    kind: DriftKind
    detail: str          # Slovak, one sentence


class ReferentOrigin(str, Enum):
    TRANSCRIPT = "transcript"
    BRIEFING = "briefing"


class Referent(BaseModel):
    text: str            # as spoken, e.g. "my"
    resolved: str        # e.g. "vláda SR, v ktorej bola Remišová ministerkou"
    origin: ReferentOrigin


class ClaimReading(BaseModel):
    claim_id: int
    interpretation: str              # Slovak, the exact asserted proposition
    drift: list[Drift] = []
    claim: str = ""                  # faithful claim text; expected when drift is non-empty
    referents: list[Referent] = []
    not_asserted: list[str] = []     # stronger/different readings the speaker did not make
    refutation_would_be: str = ""    # shape of refuting evidence: metric, period, actor, stage
    related_claim_ids: list[int] = []


class ClaimReadings(BaseModel):
    readings: list[ClaimReading] = []
```

Expected readings for the three misses:

- ta3-1074913#9: drift `dropped_qualifier` ("pre živnostníkov"); claim
  rewritten to the exemption for živnostníci; `not_asserted`: full abolition of
  the transaction tax.
- 2898725#4: interpretation that gross public debt rose by about 30 bn between
  the government taking office and the debate, while the three packages
  collected over 11 bn; referent "keď preberali" resolved from the briefing;
  `not_asserted`: the packages themselves total 30 bn, the packages caused 30
  bn of debt; `refutation_would_be`: debt at takeover and at the debate whose
  difference is clearly not about 30 bn.
- 623275#17: referent "my" resolved to the government Remišová served in, not
  her ministry; `refutation_would_be`: the total EU allocation to Slovakia for
  2021–2027 being clearly different from 24 bn.

### `apply_readings` — guards in code

`apply_readings(claims, readings, windows, briefing_text) -> notes` mutates the
selected `ExtractedClaim` list in place:

1. A reading whose `claim_id` is not a selected claim is ignored with a note. A
   selected claim without a reading keeps its legacy fields and gets a note.
2. **Number guard.** Every digit token (normalized: "30", "2021", "11,5" ->
   "11.5") in `interpretation`, `claim`, `not_asserted` and
   `refutation_would_be` must occur in the quote, the window, `time_window`,
   the debate date, or the `resolved` text of a briefing-origin referent. A
   briefing-origin referent's `resolved` text is itself accepted only if each
   of its digit tokens occurs in the referent context the pass received
   (participants, `entity_index`, glossary terms and definitions). Glossary
   `status` is left out of that context because it states facts. One failing token rejects the whole
   reading for that claim: legacy fields are kept and a note names the token.
   Numbers written as words are not caught; the prompt rule and the fidelity
   gate cover those.
3. A reading with non-empty `drift` and empty `claim` is accepted, but the claim
   text is not rewritten, and a note records this.
4. On rewrite, the old text moves to `claim_extracted` and `claim` takes the
   reading's text. `quote`, `speaker`, `category`, `checkability`,
   `consequence` and the time fields are never changed by a reading.
5. `related_claim_ids` are filtered to selected ids other than the claim itself.

New optional fields on `ExtractedClaim`, all defaulting to empty:
`interpretation`, `claim_extracted`, `not_asserted`, `refutation_would_be`,
`referents`, `related_claim_ids`. They therefore appear in `_claims_json`
without further plumbing. `drift` is not stored on the claim. It is reported
through notes ("Interpretation rewrote claim N: dropped_qualifier — ...").

If the whole pass raises, `run_analysis` continues with the unchanged claims and
adds a note, like a briefing failure.

### Phase B — checkers

`_CHECK_RULES` gains a block that applies whenever a claim carries an
`interpretation`:

- **Order of authority.** The quote is what was said. The interpretation is the
  binding reading of the quote. `context` is background on the debate moment.
  If you believe the interpretation misreads the quote, judge the quote and say
  so in the rationale.
- **Queries** are built from the interpretation, the quote and
  `refutation_would_be`. The existing rule against adding entities from memory
  stays. Referents resolved in `referents` count as present in the claim.
- **Sub-facts** decompose the interpretation. A sub-fact that tests a
  `not_asserted` reading is never refuted or distorted; drop it.
- **Accusations.** False or Misleading requires evidence of the shape
  `refutation_would_be` describes: same metric, period, actor and stage.
  Evidence about a `not_asserted` reading is unusable.
- **Related claims** are read-only context and are not to be verdicted here.

The grounded checker already sees all selected claims. Specialists see only
their category, so `_check_task_description` gets an optional
`related: list[ExtractedClaim]` argument rendered as a read-only "RELATED
CLAIMS" block (id, speaker, claim, interpretation) for the related ids that fall
outside the specialist's own list.

### Phase B2 — manager

`build_manager_crew` gets a read-only "RELATED CLAIMS AND CURRENT VERDICTS"
block in every round: for each related id outside the round's claims, show
claim, interpretation, current verdict, severity and the first 300 characters of
the rationale, taken from `facts_by_id`. Rounds 2–3, which today see only
escalated claims, therefore see their related claims' verdicts. Manager rules
add: if your verdict contradicts a related claim's verdict about the same
fact, reconcile them or explain the difference in the rationale.

`apply_manager_reviews` and `accusation_backed` are unchanged.

### Fidelity gate

`build_fidelity_prompt` adds, when present, "READING RECORDED BEFORE CHECKING
(may be wrong)" with `interpretation` and `not_asserted`. The gate's own
`asserted` stays its independent reading. A rule is added: if what the
rationale's evidence refutes matches a `not_asserted` item, the reading is
`stronger_reading` or `different_referent`. The gate keeps its downgrade-only
contract. `VerifiedFact` must carry the fields for this (see Persistence).

### Persistence

`VerifiedFact` gains `interpretation: str = ""`, `not_asserted: list[str] = []`,
`claim_extracted: str = ""` and `category: ClaimCategory | None = None`, copied
in `reconcile._build_fact`. Persisting `category` makes future replays possible
without re-classification. Old reports load unchanged because every new field
has a default. The publisher, reel generator and scoring ignore the new fields.

## Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `interpretation_enabled` | `False` | Run Phase A2. Switched to `True` once the replay acceptance passes. |
| `interpretation_window` | `6` | Transcript lines before and after the quoted line. |
| `interpretation_chunk` | `20` | Claims per interpretation call. |

## Cost

One extra structured call per chunk, with windows of 13 lines for up to 40
claims (about 15–25 k input tokens per call), on the judge model: about $0.05–0.15
per debate. This is negligible against today's $4–9. The checker and manager
prompts grow by a few hundred tokens per claim.

## Evaluation and replay

The gold set `data/eval/fidelity_gold.json` replays stored rationales only, so
it cannot show whether a better reading changes what the checker researches. A
full rerun of all 15 debates would cost about $90 and many hours. Instead, a
new command re-interprets and re-checks only the gold claims:

```
python -m src.eval interpret-replay [--labels downgrade,keep] [--only ID#N ...] [--runs N] [--no-check]
```

Per gold report:

1. Load stored facts, `data/transcripts/<id>.corrected.txt`, the stored
   `briefing` and `debate_date`.
2. Rebuild `ExtractedClaims` from **all** stored facts (id = index + 1, claim,
   speaker, quote). Stored facts carry no category, so one small classifier
   call per report assigns `category`. The time fields stay empty. This is a
   documented deviation from a live run; the interpretation reads the period
   from the quote and window instead.
3. Run the interpretation pass and `apply_readings` over all claims, so related
   claims and siblings exist.
4. Re-check only the target claims through `check_claims` (grounded checker
   plus specialists, reconcile, manager rounds), copy each stored fact's
   `timestamp` onto its new fact, then run the fidelity gate on the re-checked
   facts with the report's other stored facts as siblings
   (`audit_accusations(..., siblings=...)`). `enforce_accusation_support` is
   skipped: stored transcripts no longer carry the edit-to-line mapping it
   needs, and the targets' quotes, which are unchanged, already passed it.
5. Record per case: label, stored verdict, new verdict, reading (interpretation,
   drift, `not_asserted`), and rationale.

`--no-check` stops after step 3 and prints the readings. At about $0.05 per
report this is the inner loop for prompt work: it should show the expected
readings for the three misses before any re-check is paid for.

Output goes to `data/eval/interpret_replay.json` with a summary per label, plus
the cost from `TRACKER`.

**Replay set:** the 3 misses, the 12 downgrade cases the gate already fixes, and
all 33 keep cases: 48 claims across the gold reports, estimated at $3–5 and
1–1.5 h with Vertex 429 backoff. The 3 misses run twice (`--runs 2`) to expose
variance.

**Acceptance:**

- In both runs, all 3 misses end up neither False nor Misleading, or they stay
  False/Misleading and a human reviewer marks the new reading and rationale as
  sound in the replay output.
- At most 1 of the 33 keep cases ends up Unverified or True (the gate alone
  already loses 1).
- None of the 12 gate-fixed cases is False or Misleading again.
- In `--no-check` mode, no reading of a keep case adds a qualifier or a number
  that the quote does not contain.

## Rollout

1. Ship behind `interpretation_enabled = False`. Live runs are unchanged.
2. Iterate the prompt with `--no-check` on the three misses and a sample of keep
   cases.
3. Run the full replay. If it meets acceptance, switch the default to `True`
   and record the result in this spec.
4. Stored reports are not migrated or re-checked. New reports carry the new
   fields; old ones load with defaults.

The fidelity gate, provenance enforcement, quote grounding, the sub-fact rule
and `enforce_accusation_support` stay as backstops.

## Testing

All offline, TDD, `.venv/bin/python -m pytest`:

- Window location: quote on the speaker's line, quote across two lines, no
  match (empty window), window clipped at the transcript edges.
- Number guard: digits from the quote/window/`time_window` accepted, a digit
  from nowhere rejects the reading, a briefing-origin referent's digits accepted
  only when present in the briefing, decimal comma normalization.
- `apply_readings`: rewrite with `claim_extracted` kept, drift without claim
  text, missing reading, unknown id, filtered `related_claim_ids`, protected
  fields unchanged.
- Chunking: 45 claims produce 3 calls, each carrying the full sibling list.
- Prompt content: the interpretation prompt carries only `entity_index` and
  glossary from the briefing (no timeline or disputes); `_CHECK_RULES` carries
  the authority-order block; specialist and manager prompts carry the related
  block with current verdicts in rounds 2–3.
- Fidelity prompt includes the recorded reading when present and is unchanged
  when absent.
- `run_analysis` wiring: a flag-off run makes no interpretation call; a raising
  pass leaves claims unchanged and adds a note.
- Legacy load: stored reports in `data/reports` still validate as
  `AnalysisReport`.
- Replay scoring with a fake LLM and fixture report: per-label summary, stale
  case detection, `--no-check` path makes no checker call.

## Definition of done

- With the flag on, every selected claim in a live run carries an
  interpretation or a note explaining why not.
- No interpretation contains a number absent from its quote, window, time window
  or a briefing-backed referent. This is enforced in code.
- The replay acceptance above is met and recorded in this spec.
- The default is switched to `True`.
