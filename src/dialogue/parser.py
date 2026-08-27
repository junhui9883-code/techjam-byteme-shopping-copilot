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

# Markers are matched case-insensitively via _after(), so a capitalisation
# change in the organizer's templates cannot silently break parsing. The
# previous code tested a lowercased message but split the original-case
# message on a case-specific literal; when the two disagreed (evaluator line
# 85 emits "What I need is:" with a capital W) str.split returned a 1-element
# list and the [1] index raised IndexError on every intent_override session.
KEY_REQUIREMENT = "a key requirement is:"
STILL_EXPLORING = "still exploring"            # evaluator line 163
WHAT_MATTERS = "what matters is:"              # evaluator line 185
WHAT_I_NEED = "what i need is:"                # evaluator line 85

# "I don't have an additional preference for <attr>."          evaluator L183
NO_ADDITIONAL_RE = re.compile(r"don'?t have an additional preference for (\w+)", re.I)
# "I don't have a preference for <attr>; please use your judgment."  L169
# Fires at most once per session and only for `boundary` scenarios, so it is
# a free, reliable scenario detector as well as a "stop asking" signal.
NO_PREFERENCE_RE = re.compile(r"don'?t have a preference for (\w+); please use your judgment", re.I)


def _after(text: str, low: str, marker: str) -> str | None:
    """Text following `marker`, matched case-insensitively. None if absent.

    `low` must be `text.lower()`; both are passed in so callers that already
    lowercased do not do it twice per turn.
    """
    position = low.find(marker)
    if position < 0:
        return None
    return text[position + len(marker):].strip().rstrip(".")


def parse(state: SessionState, message: str, turn: int) -> None:
    """Fold one customer message into `state`. Mutates in place."""
    text = message.strip()
    low = text.lower()

    if turn == 1:
        _parse_opening(state, text, low)
        # Turn 1 returns before the budget rescan, matching the prototype:
        # the opener never carries a "budget around $X" clause.
        return

    boundary = NO_PREFERENCE_RE.search(text)
    if boundary:
        state.dead_attributes.add(boundary.group(1).lower())
        state.boundary_signal = True
    exhausted = NO_ADDITIONAL_RE.search(text)
    if exhausted:
        state.dead_attributes.add(exhausted.group(1).lower())

    tail = _after(text, low, WHAT_MATTERS)
    if tail is not None:
        # "For that, what matters is: A; B." -> two separate constraints.
        state.constraints.extend([c.strip() for c in tail.split(";") if c.strip()])
    else:
        new_value = _after(text, low, WHAT_I_NEED)
        if new_value:
            state.constraints.append(new_value)

    _refresh_budget(state)


def _parse_opening(state: SessionState, text: str, low: str) -> None:
    """Turn 1 carries the category, and for `buying` sessions one free constraint."""
    match = CATEGORY_RE.match(text)
    if match:
        state.category = match.group(1).strip().rstrip(".")

    requirement = _after(text, low, KEY_REQUIREMENT)
    if requirement:
        state.constraints.append(requirement)
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
