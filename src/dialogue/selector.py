"""Candidate-aware question selection (question-value estimation).

Owner: dialogue

The shipped priority policy (src/dialogue/ask.py) is REACTIVE: it walks a fixed
order and only changes course once the customer says a slot is empty. Every
wasted ask is discovered after the fact, by spending a turn on it.

This module is PROACTIVE. Before asking, it looks at the products still in
contention and estimates how much each attribute would actually split them:

    value(a) = H(distribution of a over the live candidates) x coverage(a)

Entropy alone is not enough. An attribute can look beautifully balanced across
the six products that happen to mention it while ninety-four say nothing, and
asking it would then discriminate almost nothing. Scaling by coverage -- the
fraction of candidates that expose the attribute at all -- penalises exactly
that case.

The two terms encode the two ways a question can be worthless:

    coverage 0  -> nobody states it; the answer cannot be matched against
                   anything, so the turn is spent for nothing.
    entropy  0  -> everybody states the same value; we already know the
                   answer, so it eliminates no candidate.

When no attribute clears MIN_VALUE the specific-attribute space is exhausted
in an information-theoretic sense, and we fall back to open-ended asking. That
is the same conclusion ask.py reaches from the customer's replies -- this
module simply reaches it a turn earlier, without paying for the evidence.
"""

from __future__ import annotations

import math
from collections import Counter

from ..retrieval.index import CatalogIndex
from .ask import FALLBACK_ATTRIBUTE
from .classify import (COLORS, MATERIALS, SIZE_WORDS, STYLE_WORDS,
                       USE_CASE_WORDS)
from .state import SessionState

# Vocabularies shared with classify.py so that what we ASK about and what we
# later RECOGNISE in a reply cannot drift apart.
VOCABULARIES: dict[str, tuple[str, ...]] = {
    "material": MATERIALS,
    "color": COLORS,
    "size": SIZE_WORDS,
    "style": STYLE_WORDS,
    "use_case": USE_CASE_WORDS,
}

# Price bands for the budget attribute. Coarse on purpose: the customer answers
# with a single number, so finer bands would inflate entropy without making the
# question any more discriminating.
PRICE_BANDS = (20.0, 40.0, 60.0, 100.0, 200.0)

# How many top candidates to inspect. The reranker's order is meaningful, so
# the head of the list is what the question actually needs to separate; scanning
# all 400 would measure the tail we are never going to return.
INSPECT = 60

# Below this, asking a specific attribute is not worth a turn.
MIN_VALUE = 0.25

# Never estimate on a pool too small to have a distribution.
MIN_POOL = 8


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def _price_band(price: float) -> str:
    for edge in PRICE_BANDS:
        if price < edge:
            return f"<{edge:g}"
    return f">={PRICE_BANDS[-1]:g}"


def attribute_value(index: CatalogIndex, pool: list[str], attribute: str) -> float:
    """Expected discriminating power of asking `attribute`, in bits x coverage."""
    sample = pool[:INSPECT]
    if len(sample) < MIN_POOL:
        return 0.0
    counts: Counter = Counter()

    if attribute == "budget":
        for pid in sample:
            price = index.price.get(pid)
            if price is not None:
                counts[_price_band(price)] += 1
    else:
        vocabulary = VOCABULARIES.get(attribute)
        if not vocabulary:
            return 0.0
        for pid in sample:
            fields = index.ftext.get(pid)
            if not fields:
                continue
            blob = fields["title"] + " " + fields["features"] + " " + fields["details"]
            for word in vocabulary:
                if word in blob:
                    counts[word] += 1
                    break        # first match only: one value per product

    coverage = sum(counts.values()) / len(sample)
    return _entropy(counts) * coverage


def next_ask_candidate_aware(state: SessionState, index: CatalogIndex,
                             pool: list[str], min_value: float = MIN_VALUE) -> str:
    """Pick the attribute that best splits the live candidates.

    Falls back to open-ended asking when nothing clears `min_value`, and records
    the estimate on the state so the choice can be explained rather than
    asserted.
    """
    askable = [
        a for a in ("material", "color", "budget", "style", "size", "use_case")
        if a not in state.asked and a not in state.dead_attributes
    ]
    if not askable:
        return FALLBACK_ATTRIBUTE

    scored = sorted(
        ((attribute_value(index, pool, a), a) for a in askable), reverse=True)
    state.last_question_values = [(a, round(v, 4)) for v, a in scored]

    best_value, best_attribute = scored[0]
    if best_value < min_value:
        return FALLBACK_ATTRIBUTE
    state.asked.append(best_attribute)
    return best_attribute
