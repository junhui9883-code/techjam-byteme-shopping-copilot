"""Which attribute to elicit next.

Owner: dialogue

Per CLAUDE.md section 3 this is the single biggest lever in the competition:
the official starter scores 0.107 purely because it returns ask_attribute=None
every turn and the simulated customer therefore discloses nothing.

Two policies exist today, both fixed:

  "priority" -- walk a hand-ordered list. Currently shipped. Scores 0.786.
  "other"    -- exploit the `attribute == "other"` bypass in customer_reply,
                which returns the next two undisclosed constraints regardless
                of type. Scores 0.854 but is deliberately NOT shipped as-is;
                see CLAUDE.md section 4 for the reasoning.

CLAUDE.md backlog item 2 replaces both with question-value estimation over the
live candidate set. This module is where that lands.
"""

from __future__ import annotations

from .state import SessionState

# Hand-ordered by expected disclosure value: material and color are the two
# fields intent_card() regex-prepends, so they are the most likely to be
# present in the hidden brief.
ASK_ORDER = [
    "material", "color", "budget", "style", "size",
    "use_case", "feature", "brand", "category", "other",
]

FALLBACK_ATTRIBUTE = "other"


def next_ask(state: SessionState, policy: str = "priority") -> str:
    """Pick the next attribute to elicit and record it on `state`.

    NOTE: the "other" policy is stateless and never marks anything as asked,
    so it can be selected every turn. That is what makes it drain the brief.
    """
    if policy == FALLBACK_ATTRIBUTE:
        return FALLBACK_ATTRIBUTE
    for attribute in ASK_ORDER:
        if attribute not in state.asked:
            state.asked.append(attribute)
            return attribute
    return FALLBACK_ATTRIBUTE
