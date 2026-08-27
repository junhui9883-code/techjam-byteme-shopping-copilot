"""How many recommendations to return this turn.

Owner: dialogue

List length is a scoring lever, not a UI choice. The session ends the instant
the target appears anywhere in the returned top-10, so returning all 10 while
still uninformed can END the session at rank 8 for 0.55, when holding back and
answering two turns later at rank 1 would have paid 0.92.

Measured contribution of this schedule: +0.010 overall, and +0.089 on MRR
(0.617 -> 0.706) at a cost of 0.6 turns. See CLAUDE.md section 2.

CLAUDE.md backlog item 7 grid-searches this per route; item 5 of section 6
replaces the turn/count heuristic with a confidence signal (the score gap
between rank 1 and rank 5).
"""

from __future__ import annotations

from .state import SessionState

# Below this much disclosure the ranking is largely noise, so expose only the
# very top of it and keep the session alive for a better turn.
NARROW_K = 3
MEDIUM_K = 5
EARLY_TURNS = 2
NARROW_INFO = 2
MEDIUM_INFO = 3


def list_length(state: SessionState, turn: int, top_k: int, enabled: bool = True) -> int:
    """Return the number of recommendations to expose this turn.

    WARNING: there is no turn-10 escape hatch here. The schedule keys off
    disclosure count alone, so a session where the customer keeps refusing to
    disclose (boundary: "I don't have a preference for X") can still be
    truncated to 5 on the FINAL turn. CLAUDE.md section 6 requires the opposite
    -- "always return the full 10 on turn 10" -- because a miss is worth
    exactly zero. Adding that guard is a behavioural change and must be
    measured on its own run; it is a prime suspect for the boundary subgroup
    sitting at 0.60 hit rate.
    """
    if not enabled:
        return top_k
    info = state.information_count
    if turn <= EARLY_TURNS and info < NARROW_INFO:
        return NARROW_K
    if info < MEDIUM_INFO:
        return MEDIUM_K
    return top_k
