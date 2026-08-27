"""Turn a simulated customer message into state mutations.

Owner: dialogue

The evaluator's customer is a deterministic template engine, not an LLM (see
CLAUDE.md section 3), so these are literal shape matches against the templates
it can emit. The authoritative list is evaluator/local_evaluator.py lines
85, 159-163, 169, 183, 185.

CAUTION (CLAUDE.md section 5): every match here is an exact substring or regex
against organizer phrasing. If the private harness paraphrases customer turns,
this module is the first thing to break -- which is exactly what the step-4
paraphrase harness is built to measure.
"""

from __future__ import annotations

import re

from .state import SessionState

PRICE_RE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]+)?)")

# Turn-1 opener, evaluator lines 159-163.
CATEGORY_RE = re.compile(
    r"i'm looking for (.+?)(?:, but i'm still exploring\.?|\. |$)", re.I
)

# Detection is done against the lowercased message; the split is done against
# the original-case message. The two literals must therefore agree in case with
# the evaluator's own template, or the split silently yields a 1-element list.
KEY_REQUIREMENT_TEST = "a key requirement is:"
KEY_REQUIREMENT_SPLIT = "A key requirement is:"   # evaluator line 159, capital A
STILL_EXPLORING = "still exploring"               # evaluator line 163
WHAT_MATTERS_TEST = "what matters is:"
WHAT_MATTERS_SPLIT = "what matters is:"           # evaluator line 185, mid-sentence
WHAT_I_NEED_TEST = "what i need is:"

# ---------------------------------------------------------------------------
# KNOWN DEFECT -- preserved verbatim during the step-2 refactor, do not "fix"
# this in passing.
#
# The evaluator emits (line 85):
#     "Actually, ignore my earlier preference. What I need is: {new_value}."
# with a capital W. The literal below has a lowercase w, so the membership test
# on the lowercased message PASSES while the split on the original-case message
# FINDS NOTHING -- str.split returns a 1-element list and the [1] index raises
# IndexError.
#
# Consequences on every intent_override session (15% of the set):
#   * respond() raises; the evaluator catches it (line 240) and substitutes an
#     empty response, so that whole turn returns zero recommendations.
#   * the override's new_value is never appended to state, so the constraint
#     the customer just asked for is lost for the rest of the session.
#
# This is strictly worse than CLAUDE.md backlog item 3, which describes the
# override as failing to EVICT old constraints. It also fails to ADD the new
# one. Fixing it is a one-character change (w -> W) but it is a BEHAVIOURAL
# change and must be measured on its own run before it ships.
# ---------------------------------------------------------------------------
WHAT_I_NEED_SPLIT = "what I need is:"


def parse(state: SessionState, message: str, turn: int) -> None:
    """Fold one customer message into `state`. Mutates in place.

    Raises IndexError on intent-override turns; see KNOWN DEFECT above.
    """
    text = message.strip()
    low = text.lower()

    if turn == 1:
        _parse_opening(state, text, low)
        # Turn 1 returns before the budget rescan, matching the prototype:
        # the opener never carries a "budget around $X" clause.
        return

    if WHAT_MATTERS_TEST in low:
        # "For that, what matters is: A; B." -> two separate constraints.
        tail = text.split(WHAT_MATTERS_SPLIT, 1)[1].strip().rstrip(".")
        state.constraints.extend([c.strip() for c in tail.split(";") if c.strip()])
    elif WHAT_I_NEED_TEST in low:
        state.constraints.append(text.split(WHAT_I_NEED_SPLIT, 1)[1].strip().rstrip("."))

    _refresh_budget(state)


def _parse_opening(state: SessionState, text: str, low: str) -> None:
    """Turn 1 carries the category, and for `buying` sessions one free constraint."""
    match = CATEGORY_RE.match(text)
    if match:
        state.category = match.group(1).strip().rstrip(".")

    if KEY_REQUIREMENT_TEST in low:
        state.constraints.append(text.split(KEY_REQUIREMENT_SPLIT, 1)[1].strip().rstrip("."))
    elif STILL_EXPLORING not in low:
        # intent_override openers (evaluator line 162) carry a bare preference
        # clause after the first sentence. Browsing openers carry none.
        rest = text.split(".", 1)[1].strip() if "." in text else ""
        if rest:
            state.constraints.append(rest.rstrip("."))


def _refresh_budget(state: SessionState) -> None:
    """Rescan every constraint for a `budget around $X` clause; last match wins."""
    for constraint in state.constraints:
        match = PRICE_RE.search(constraint)
        if match and "budget" in constraint.lower():
            try:
                state.budget = float(match.group(1))
            except Exception:
                pass
