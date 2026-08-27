"""Per-session conversational state.

Owner: dialogue

One instance per session_id, created by Agent.reset(). Everything the ranker
needs about "what the customer has told us so far" lives here and nowhere else.

Currently constraints are an untyped ordered list. CLAUDE.md backlog items 3
and 4 replace this with typed slots supporting add / evict / mark_unavailable,
which is what intent-override eviction and boundary handling both need. The
class boundary exists now so that change lands in one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    """Accumulated disclosures for a single shopping session.

    Attributes
    ----------
    category    : the turn-1 product category, e.g. "a winter jacket"
    constraints : disclosed constraint strings, in disclosure order
    asked       : attributes already used as `ask_attribute`, in ask order
    budget      : parsed target price, or None if never disclosed
    profile     : the user_profile handed to reset(); not yet used in ranking
                  (CLAUDE.md backlog item 8)
    """

    category: str = ""
    constraints: list[str] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    budget: float | None = None
    profile: dict = field(default_factory=dict)

    # Attributes the customer has told us yield nothing more. Two distinct
    # signals, both meaning "stop spending turns here":
    #   "I don't have an additional preference for X."        -> exhausted
    #   "I don't have a preference for X; use your judgment." -> refused
    # The second also identifies the session as `boundary` (evaluator line
    # 169); it fires at most once per session.
    dead_attributes: set[str] = field(default_factory=set)
    boundary_signal: bool = False

    @property
    def dead_ask_count(self) -> int:
        """How many asks have come back empty. Evidence that the specific
        attribute space is exhausted and only `other` will yield anything."""
        return len(self.dead_attributes)

    @property
    def information_count(self) -> int:
        """How much the customer has actually disclosed. Drives truncation."""
        return len(self.constraints)
