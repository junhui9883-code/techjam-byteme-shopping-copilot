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
MEDIUM_INFO = 4   # swept: 3 -> 0.8541, 4 -> 0.8556, 5 -> 0.8513

# The session is scored over at most 10 turns (evaluator MAX_TURNS). On the
# last one there is no future turn to save the list for, and a miss is worth
# exactly zero, so always expose everything. CLAUDE.md section 6.
FINAL_TURN = 10


def list_length(state: SessionState, turn: int, top_k: int, enabled: bool = True,
                early_turns: int = EARLY_TURNS, narrow_info: int = NARROW_INFO,
                medium_info: int = MEDIUM_INFO) -> int:
    """Return the number of recommendations to expose this turn.

    Holding back a short list is only ever a bet that a LATER turn will pay
    more. On the final turn that bet cannot pay off, so the guard below always
    returns the full list there regardless of how little was disclosed.

    Without it, a customer who keeps refusing to disclose (boundary:
    "I don't have a preference for X") never clears MEDIUM_INFO and is still
    truncated to 5 on turn 10, throwing away ranks 6-10 for nothing.
    """
    if not enabled:
        return top_k
    if turn >= FINAL_TURN:
        return top_k
    info = state.information_count
    if turn <= early_turns and info < narrow_info:
        return NARROW_K
    if info < medium_info:
        return MEDIUM_K
    return top_k
