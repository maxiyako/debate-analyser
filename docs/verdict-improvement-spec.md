# Spec: making the debate verdict trustworthy

## Problem statement

The pipeline produces a defensible *fair-play scorecard* and then publishes it as
*who won the debate*. Those are different claims, and the gap between them is
where the current output is weakest.

Evidence from the eleven stored reports, rescored against their transcripts:

| Symptom | Measurement |
| --- | --- |
| Verdict is not reproducible | Two runs of episode 605345 give "tie, margin 4.7" and "Šimečka wins, margin 9.2". Two runs of 602993 give margin 5.0 and 12.6 |
| Behavioral evidence is too sparse to score | 26–42 moderator question turns per episode, but only 0–3 `question_dodging` findings and 0–3 `manipulation` findings |
| Foul rates therefore cannot separate anyone | Manipulation scores cluster at 94–100 across every speaker, so the 25% weight is inert |
| Truthfulness rests on unequal denominators | 591624: Krúpa 1 checked claim vs Laššáková 4. 604147: Gröhling 6 vs Danko 14 |
| Facts are silently dropped | 604147 attributes 20 of 31 facts to a scoreboard row; 11 vanish on speaker-label mismatch |
| One subjective integer decides close debates | Civility is a single 0–10 LLM rating carrying 20% weight, and it decided 592879, 599134, and 605345 |
| Systematic tilt is unmeasured | 10 of 11 stored winners are opposition politicians. This may be a real property of incumbents defending a record, or it may be model bias. Nothing in the pipeline can tell the difference |
| Transcript correction can fail silently | 599134's corrected transcript still labels everyone `Speaker A/B/…`; the moderator (95 turns) is never identified, and the report only recovers guest names by writing them as `Simona Petrík (Speaker E)` |

The recent scorer changes (moderator excluded, small-sample prior on
truthfulness, salience weighting, 5-point win margin, ties allowed) remove the
most embarrassing failures. They do not address reproducibility, evidence
sparsity, or the absence of any ground truth.

## Goals

1. Separate two published artifacts: a **fair-play scorecard** (accuracy and
   conduct, auditable per claim) and an **argument-quality judgment** (who
   actually answered and who carried the clash). Never let the first masquerade
   as the second.
2. Make the verdict reproducible: the same transcript scored twice must reach the
   same published conclusion.
3. Make every published number carry its uncertainty, and allow the pipeline to
   publish "too close to call".
4. Be able to demonstrate, with numbers, that the tool is not partisan.

## Non-goals

- Predicting who the audience found persuasive.
- Scoring policy correctness. The tool judges accuracy and conduct, not whether
  a proposal is good.

---

## Phase 1 — Ground truth and evaluation harness

Nothing below can be validated without this, so it comes first.

**Build `data/gold/<episode>.json`** for at least four episodes spanning a
coalition-heavy and an opposition-heavy panel. Hand-labelled by a human, with a
second reviewer on at least two episodes:

- Every checkable claim in the transcript with speaker, verbatim quote, and
  verdict. Recording claims the pipeline *missed* is the point; extraction recall
  is currently unmeasured.
- Per moderator question: which speaker was asked, and whether the answer was
  direct, partial, or evasive.
- Per speaker: norm violations with quotes.
- A holistic human judgment: winner or tie, plus one sentence of reasoning.

**Add `python -m src.eval --gold data/gold` reporting:**

| Metric | Target |
| --- | --- |
| Claim extraction recall against gold | ≥ 0.70 |
| Verdict agreement on shared claims (Cohen's κ) | ≥ 0.60 |
| Speaker attribution accuracy on extracted claims | ≥ 0.95 |
| Winner agreement with human label (tie counts as agreement when human said tie) | ≥ 0.75 |
| Rerun verdict stability (same transcript, 3 runs) | 3/3 identical published conclusion |

Acceptance: the harness runs offline against stored reports, prints a table, and
exits non-zero when a target regresses. Wire it into CI.

## Phase 2 — Reproducibility

The verdict currently depends on which of several plausible LLM samples came
back. Fix it by sampling deliberately instead of accidentally.

- Run the behavioral and extraction phase **K = 3 times** (configurable,
  default 3) at the existing temperature.
- Merge claims across runs by fuzzy quote match. Keep a claim when it appears in
  a majority of runs; record `runs_seen` on each claim. Claims seen once are
  extracted but flagged low-confidence and excluded from scoring.
- For per-speaker conduct ratings, take the **median** across runs and record the
  spread.
- Propagate spread into the final score as a range. Publish a winner only when
  the leader's lower bound exceeds the runner-up's upper bound.

This costs roughly 3× the phase-A token spend. `src/costs.py` already tracks
spend, so make K a documented cost/confidence dial rather than a hidden default.

Acceptance: three consecutive runs of one episode publish the same conclusion,
and `DebateVerdict` carries a `score_interval` per speaker.

## Phase 3 — Turn-level conduct scoring

This is the highest-value change. Replace whole-debate foul lists with
observations per exchange, which raises the sample size by an order of magnitude.

**Segment the transcript into question blocks** in code: a moderator turn
containing a question, plus the guest turns that follow until the next moderator
question. Episodes yield 26–42 such blocks.

**For each block, the analyst returns a small structured judgment:**

```
{block_id, addressee, answered: direct|partial|evasive,
 evasion_kind: topic_switch|counterattack|non_answer|filibuster|none,
 norm_violations: [{kind, quote}],
 fallacies: [{kind, quote}]}
```

Scoring consequences:

- **Responsiveness** becomes `direct / (direct + partial + evasive)` over blocks
  addressed to that speaker — a real rate with a real denominator, replacing
  "dodge count per 1000 words".
- **Conduct** becomes violations per block, so a speaker who is hostile in two
  exchanges out of thirty is scored differently from one hostile throughout.
- **Civility** stops being a single opaque integer. Derive it from counted,
  quoted violations by kind, each with a documented weight. Keep the sociological
  rubric already in the analyst backstory — assertiveness is not incivility — but
  apply it per exchange.

Guard against label noise: when a block's addressee cannot be determined, or the
transcript has no labelled moderator (as in 599134), mark the block unscorable
and report the unscorable fraction. Refuse to publish conduct scores when more
than 25% of blocks are unscorable.

Acceptance: median ≥ 20 scored blocks per episode; responsiveness and civility
scores spread over at least 30 points across speakers in the gold set instead of
clustering at 94–100.

## Phase 4 — Balanced and honest fact-checking

- **Per-speaker extraction quotas.** Replace the global top-N salience cut with a
  per-speaker cut, so no speaker enters scoring with one checked claim. Log the
  claims dropped per speaker.
- **Minimum coverage gate.** Truthfulness contributes to the final score only at
  ≥ 5 checked claims for that speaker; below that, report it as indicative and
  renormalize the other weights (the mechanism already exists for `None` scores).
- **Publish coverage.** Every scorecard states checked, unverified, and contested
  counts per speaker plus the number of extracted claims that never reached a
  verdict. A reader must be able to see that Gröhling was checked on 6 claims and
  Danko on 14.
- **Stop dropping facts silently.** When a fact's speaker does not resolve to a
  roster row, surface it in `critic_notes` and in the eval metrics rather than
  discarding it inside `score_report`.
- **Asymmetric-exposure note.** Incumbents cite more checkable numbers, so they
  accumulate more chances to be wrong. State this in the method notes attached to
  every published scorecard.

## Phase 5 — Argument quality, judged separately and blind

Only after the above does "who won" become answerable.

- Group the question blocks into **topic segments**.
- Run a dedicated judge over each segment with **speaker identity and party
  removed** — relabel to "Speaker 1" / "Speaker 2", strip party names and
  honorifics from the text handed to the judge. Political priors are the main
  bias risk and blinding is the cheapest control.
- Per segment the judge records: who bore the burden of proof, whether the
  central objection was answered, whether assertions were substantiated, and a
  segment outcome of speaker 1 / speaker 2 / tie, with quotes.
- **Swap-test**: run each segment twice with the speaker order reversed. When the
  outcome flips, the segment is a tie.
- Overall argument-quality outcome = segment tally, published as "won 4 of 9
  segments, 3 ties" rather than a single number.

The published verdict then has two independent parts, and the pipeline reports
when they disagree — a speaker can be more accurate and still lose the argument,
and saying so is more useful than a blended score that hides it.

## Phase 6 — Bias audit

- Add `python -m src.eval --bias` over all stored reports: winner rate, mean
  truthfulness, and mean conduct score broken down by coalition vs opposition,
  government vs opposition role, and speaker gender.
- Publish the corpus-level breakdown alongside the reports. The current 10-of-11
  opposition tilt must be stated publicly, with the asymmetric-exposure caveat,
  rather than discovered by a reader.
- Rerun the blinded judge on a sample with party labels restored; a large
  divergence between blinded and unblinded outcomes is direct evidence of model
  bias and should block release.

## Phase 7 — Output hygiene

- **Render the post deterministically.** Build the scoreboard text in code from
  `DebateVerdict`. Leave the LLM only the opening line and the claim
  descriptions, and validate its output against the verdict before writing. The
  publisher currently re-narrates numbers it should only be echoing.
- **Enforce one output language.** Four of the eleven stored reports are in
  English. The critic task now carries a Slovak instruction with an explicit
  translate directive, so those are probably pre-fix artifacts — but nothing
  checks. Add a language assertion on the finished report and regenerate on
  failure, so the question does not depend on reading the files by hand.
- **Validate the report shape.** 605345 shipped with an empty `summary`. Empty
  required prose should fail the run.
- **Name the artifact honestly** in the post: "fair-play scorecard" for accuracy
  and conduct; "argument quality" for the blinded judgment; never "winner of the
  debate" without both.
- **Fix the red flag.** It currently uses absolute fabrication plus manipulation
  counts, so it tracks talking time, and in 592879 it landed on the same person
  the post called the winner. Use rates, and suppress the badge when the flagged
  speaker is also the leader.

## Phase 8 — Upstream transcript quality

Everything above inherits ASR and diarization errors.

- Maintain a named-entity correction list (TISZA, Péter Magyar, party names) and
  apply it before analysis.
- Split the merged cross-talk lines that currently put moderator and guest speech
  in one turn; they corrupt both word-share and attribution.
- **Gate on speaker naming.** In 599134 the corrector left every line as
  `Speaker A/B/…` and the moderator's 95 turns were never identified; the report
  papered over it with labels like `Simona Petrík (Speaker E)`, and scoring
  matched them only because `_speaker_key` happens to parse the parenthesised id.
  Assert that the moderator and every guest resolve to a real name, and fail the
  run rather than scoring an unlabelled transcript.
- Report a diarization confidence summary per episode and set a floor for
  publishing.

---

## Sequencing and effort

| Phase | Effort | Unblocks |
| --- | --- | --- |
| 1 Ground truth + eval | Large, mostly human labelling | Everything; no other phase is verifiable without it |
| 2 Reproducibility | Small code, 3× phase-A cost | Trustworthy margins |
| 3 Turn-level conduct | Medium; prompt and schema rewrite | Conduct scores that discriminate |
| 4 Fact balance | Small | Honest denominators |
| 5 Blinded argument judge | Medium | The actual "who won" question |
| 6 Bias audit | Small once 1 and 5 exist | Public credibility |
| 7 Output hygiene | Small | Stops the post overclaiming |
| 8 Transcript quality | Medium | Raises the ceiling on all of it |

Phases 2, 4, and 7 are cheap and independent; they can land while the gold set
is being labelled. Phase 3 should not ship without Phase 1, because its whole
claim is better measurement and there would be no way to show it.

## Definition of done

- The eval harness passes its targets on the gold set and runs in CI.
- Three runs of an episode publish the same conclusion.
- Every published score carries an interval and a coverage statement.
- Fair-play and argument-quality verdicts are published separately, the second
  produced blind to speaker identity and validated by a swap test.
- The corpus-level coalition/opposition breakdown is published with the reports.
