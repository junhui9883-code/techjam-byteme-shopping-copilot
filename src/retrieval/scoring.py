"""Stage 2 of retrieval: rerank the candidate pool. This is where rank is won.

Owner: retrieval

Per CLAUDE.md section 1, moving a hit from rank 10 to rank 1 is worth 0.27
while a whole extra turn costs 0.02. This module is therefore the highest-value
surface in the repo, and it is deliberately three additive signals rather than
one clever one, so each can be ablated independently:

  1. Field-weighted BM25 over every disclosed constraint.
  2. An exact-phrase bonus, because the hidden brief is lifted near-verbatim
     from the target's own spec sheet.
  3. Price proximity, because the brief always appends `budget around $<price>`.

CAUTION (see CLAUDE.md section 5): signal 2 is a *bonus term*, never a lookup
key. If the organizer paraphrases customer messages it degrades to zero and
signals 1 and 3 must carry the session on their own.
"""

from __future__ import annotations

import re

from .index import CatalogIndex
from .text import terms

# Okapi BM25 free parameters. b is below the 0.75 default because catalog
# document lengths are wildly uneven (a details dict can dwarf a title) and
# heavy length normalisation was punishing richly-specified products.
BM25_K1 = 1.4
BM25_B = 0.6

# Query-side weights: an explicitly disclosed constraint outranks the
# turn-1 category guess.
W_CATEGORY = 1.2
W_CONSTRAINT = 2.0

# Phrase bonuses. A full constraint found verbatim in the product text is the
# single strongest evidence available; a 40-char prefix match is the partial
# credit case. Constraints of 6 chars or fewer are too generic to trust.
MIN_PHRASE_LEN = 6
PHRASE_PREFIX_LEN = 40
BONUS_PHRASE_EXACT = 14.0
BONUS_PHRASE_PREFIX = 7.0

# Third tier, below the two substring tiers: weighted lexical overlap.
# CLAUDE.md section 5 prescribes exactly this -- keep exact matching as a bonus
# term, but put a scorer underneath it that survives rewording. A contiguous
# substring match is destroyed by a single inserted word ("100% cotton" vs
# "100% um, cotton"), whereas token overlap degrades smoothly. Awarded in
# proportion to how much of the constraint is present, so it can never
# outrank a genuine exact match.
BONUS_PHRASE_OVERLAP = 8.0
MIN_OVERLAP_TERMS = 2

# Price proximity. The brief states the target's own price, so a near-exact
# match is very strong evidence; being far off is mild evidence against.
PRICE_NEAR = 0.02
PRICE_LOOSE = 0.15
BONUS_PRICE_NEAR = 10.0
BONUS_PRICE_LOOSE = 4.0
PENALTY_PRICE_FAR = -2.0
PRICE_PENALTY_CAP = 3

# Popularity prior. Shoppers search for products people actually buy, and the
# catalog is dominated by long-tail items nobody reviews. Measured on the
# public set: the median target sits at the 99th percentile of review count
# (6,846 reviews against a catalog median of 12).
#
# CAUTION: this is a PRIOR, not evidence. It must stay small enough that it only
# separates candidates the text cannot, or it degenerates into "always return
# the bestseller" and stops answering the question that was asked.
BONUS_POPULARITY = 0.0

_WS_RE = re.compile(r"\s+")


class RankParams:
    """Tunable ranker weights, so they can be swept instead of hand-guessed.

    Defaults are the historical values. Every field is overridable from the
    Agent class, which lets `--set` drive a sweep without editing code.
    """

    __slots__ = ("k1", "b", "w_category", "w_constraint", "phrase_exact",
                 "phrase_prefix", "phrase_overlap", "price_near_bonus",
                 "price_loose_bonus", "price_far_penalty", "popularity")

    def __init__(self, k1=BM25_K1, b=BM25_B, w_category=W_CATEGORY,
                 w_constraint=W_CONSTRAINT, phrase_exact=BONUS_PHRASE_EXACT,
                 phrase_prefix=BONUS_PHRASE_PREFIX,
                 phrase_overlap=BONUS_PHRASE_OVERLAP,
                 price_near_bonus=BONUS_PRICE_NEAR,
                 price_loose_bonus=BONUS_PRICE_LOOSE,
                 price_far_penalty=PENALTY_PRICE_FAR,
                 popularity=BONUS_POPULARITY):
        self.k1 = k1; self.b = b
        self.w_category = w_category; self.w_constraint = w_constraint
        self.phrase_exact = phrase_exact; self.phrase_prefix = phrase_prefix
        self.phrase_overlap = phrase_overlap
        self.price_near_bonus = price_near_bonus
        self.price_loose_bonus = price_loose_bonus
        self.price_far_penalty = price_far_penalty
        self.popularity = popularity


DEFAULT_PARAMS = RankParams()


def score(index: CatalogIndex, pid: str, category: str,
          constraints: list[str], budget: float | None,
          fallback_text: list[str] | None = None, overlap: bool = True,
          weights: list[float] | None = None,
          params: "RankParams | None" = None) -> float:
    """Relevance of one product against the accumulated session state."""
    params = params or DEFAULT_PARAMS
    if not category and not constraints and fallback_text:
        # Nothing parsed: rank on the raw transcript rather than not at all.
        return (_bm25(index, pid, " ".join(fallback_text), [], None, params)
                + params.popularity * index.pop.get(pid, 0.0))
    return (
        _bm25(index, pid, category, constraints, weights, params)
        + _phrase_bonus(index, pid, constraints, overlap, weights, params)
        + _price_bonus(index, pid, budget, params)
        + params.popularity * index.pop.get(pid, 0.0)
    )


def _bm25(index: CatalogIndex, pid: str, category: str, constraints: list[str],
          weights: list[float] | None = None,
          params: "RankParams | None" = None) -> float:
    params = params or DEFAULT_PARAMS
    doc_len = index.dl[pid]
    tf = index.tf[pid]
    total = 0.0
    # Per-constraint weights let an override demote what it supersedes without
    # discarding it (src/dialogue/state.py demote_superseded).
    scaled = [(c, params.w_constraint * _w(weights, i)) for i, c in enumerate(constraints)]
    for text, weight in [(category, params.w_category)] + scaled:
        for term in set(terms(text)):
            freq = tf.get(term, 0.0)
            if freq:
                denominator = freq + params.k1 * (1 - params.b + params.b * doc_len / index.avgdl)
                total += weight * index.idf.get(term, 0.0) * (freq * (params.k1 + 1)) / denominator
    return total


def _w(weights: list[float] | None, index: int) -> float:
    """Weight for constraint `index`; 1.0 when unset."""
    if weights is None or index >= len(weights):
        return 1.0
    return weights[index]


def _phrase_bonus(index: CatalogIndex, pid: str, constraints: list[str],
                  overlap: bool = True, weights: list[float] | None = None,
                  params: "RankParams | None" = None) -> float:
    params = params or DEFAULT_PARAMS
    fields = index.ftext[pid]
    # Only the three spec-bearing fields. Matching a constraint inside
    # `description` marketing copy is far weaker evidence.
    blob = fields["features"] + " " + fields["details"] + " " + fields["title"]
    blob_terms: set[str] | None = None
    total = 0.0
    for position, constraint in enumerate(constraints):
        scale = _w(weights, position)
        normalised = _WS_RE.sub(" ", constraint.lower()).strip()
        if len(normalised) <= MIN_PHRASE_LEN:
            continue
        if normalised in blob:
            total += params.phrase_exact * scale
        elif normalised[:PHRASE_PREFIX_LEN] in blob:
            total += params.phrase_prefix * scale
        elif overlap:
            # Neither substring tier fired. Fall back to how much of the
            # constraint's vocabulary the product actually carries.
            wanted = set(terms(normalised))
            if len(wanted) < MIN_OVERLAP_TERMS:
                continue
            if blob_terms is None:
                blob_terms = set(terms(blob))
            total += params.phrase_overlap * scale * (len(wanted & blob_terms) / len(wanted))
    return total


def _price_bonus(index: CatalogIndex, pid: str, budget: float | None,
                 params: "RankParams | None" = None) -> float:
    params = params or DEFAULT_PARAMS
    if budget is None:
        return 0.0
    price = index.price.get(pid)
    if price is None:
        return 0.0
    delta = abs(price - budget) / max(budget, 1.0)
    if delta < PRICE_NEAR:
        return params.price_near_bonus
    if delta < PRICE_LOOSE:
        return params.price_loose_bonus
    return params.price_far_penalty * min(delta, PRICE_PENALTY_CAP)
