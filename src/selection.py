"""Consequence-based adaptive claim selection (replaces the top-N salience cut).

Order of operations:
  1. drop non-empirical claims (opinion / prediction / definitional) — reported;
  2. merge near-duplicate claims by the same speaker (repeats recorded);
  3. keep claims with consequence >= threshold, plus every speaker's top
     `floor_per_speaker` claims so no speaker reaches scoring on a single check;
  4. if still above the cost fuse, cut lowest priority — never below the floor.

`salience` on returned claims is set to `consequence`, so scoring (which weights
errors by salience/3) keeps working unchanged.

Imported lazily from `run_analysis` (this module imports `src.agents`).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from src.agents import Checkability, ClaimUsage, ExtractedClaim
from src.validation import normalize

_ROLE_MULT = {
    ClaimUsage.ATTACK: 1.15,
    ClaimUsage.COUNTERARGUMENT: 1.15,
    ClaimUsage.DEFLECTION: 1.10,
    ClaimUsage.SUPPORTING: 1.0,
    ClaimUsage.OTHER: 1.0,
}


def priority(c: ExtractedClaim) -> float:
    """Consequence first; rhetorical role and repetition are secondary multipliers."""
    return c.consequence * _ROLE_MULT.get(c.usage, 1.0) * (1.0 + 0.1 * (c.repeats - 1))


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(normalize(a).split()), set(normalize(b).split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def merge_duplicates(
    claims: list[ExtractedClaim], min_jaccard: float = 0.7
) -> tuple[list[ExtractedClaim], list[str]]:
    notes: list[str] = []
    survivors: list[ExtractedClaim] = []
    for c in sorted(claims, key=lambda c: -priority(c)):
        twin = next(
            (
                s
                for s in survivors
                if s.speaker == c.speaker and _jaccard(s.claim, c.claim) >= min_jaccard
            ),
            None,
        )
        if twin is None:
            survivors.append(c)
            continue
        idx = survivors.index(twin)
        survivors[idx] = twin.model_copy(update={"repeats": twin.repeats + c.repeats})
        notes.append(
            f'Merged duplicate claim #{c.id} into #{twin.id} ({c.speaker}): "{c.claim[:60]}"'
        )
    return sorted(survivors, key=lambda c: c.id), notes


def select_claims(
    claims: list[ExtractedClaim],
    *,
    threshold: int,
    floor_per_speaker: int,
    fuse: int,
) -> tuple[list[ExtractedClaim], list[str]]:
    notes: list[str] = []

    excluded = [c for c in claims if c.checkability != Checkability.EMPIRICAL]
    if excluded:
        notes.append(
            "Excluded non-empirical claims (not fact-checked, not scored): "
            + "; ".join(
                f'#{c.id} ({c.checkability.value}) "{c.claim[:60]}"' for c in excluded
            )
        )
    empirical = [c for c in claims if c.checkability == Checkability.EMPIRICAL]

    merged, merge_notes = merge_duplicates(empirical)
    notes.extend(merge_notes)

    by_speaker: dict[str, list[ExtractedClaim]] = defaultdict(list)
    for c in merged:
        by_speaker[c.speaker].append(c)
    protected: set[int] = set()
    for lst in by_speaker.values():
        ranked = sorted(lst, key=lambda c: (-priority(c), c.id))
        protected.update(c.id for c in ranked[:floor_per_speaker])

    kept = [c for c in merged if c.consequence >= threshold or c.id in protected]
    kept_ids = {c.id for c in kept}
    below = [c for c in merged if c.id not in kept_ids]
    if below:
        notes.append(
            f"Below consequence threshold {threshold} (not checked): "
            + "; ".join(f'#{c.id} (c{c.consequence}) "{c.claim[:60]}"' for c in below)
        )

    if len(kept) > fuse:
        removable = sorted(
            (c for c in kept if c.id not in protected), key=lambda c: (priority(c), -c.id)
        )
        cut: list[ExtractedClaim] = []
        for c in removable:
            if len(kept) - len(cut) <= fuse:
                break
            cut.append(c)
        cut_ids = {c.id for c in cut}
        kept = [c for c in kept if c.id not in cut_ids]
        notes.append(
            f"Claim cap (cost fuse {fuse}): dropped {len(cut)} lowest-priority claims: "
            + "; ".join(f'#{c.id} (c{c.consequence}) "{c.claim[:60]}"' for c in cut)
        )
        per_speaker = Counter(c.speaker for c in cut)
        notes.append(
            "Unchecked claims per speaker (dropped by cap): "
            + ", ".join(f"{s}: {n}" for s, n in per_speaker.most_common())
        )

    out = [c.model_copy(update={"salience": max(1, min(5, c.consequence))}) for c in kept]
    return sorted(out, key=lambda c: c.id), notes
