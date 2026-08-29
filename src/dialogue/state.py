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

    # Raw customer turns, unparsed. This is the paraphrase safety net, and it
    # comes from PR #1 (Jun Hui) -- kept after the step-4 harness showed that
    # template matching, not the ranker, is where robustness dies: rewording
    # only the template scaffolding costs 91% of the score, while swapping
    # synonyms inside constraints costs 0.1%. When the parser recognises
    # nothing, these raw terms are all the signal there is.
    transcript: list[str] = field(default_factory=list)

    # Index of the constraint taken from the turn-1 bare-preference clause, if
    # any. intent_override sessions open with "I'm looking for X. <old_value>"
    # and later revoke exactly that clause, so knowing which constraint it was
    # makes eviction precise instead of a guess.
    opener_preference: int | None = None
    # Per-constraint ranking weight, parallel to `constraints`. Overrides
    # demote rather than delete: see demote_superseded().
    weights: list[float] = field(default_factory=list)
    # Audit trail: what was evicted and why. Surfaced in the response message
    # so the behaviour is demonstrable, not just claimed.
    evicted: list[str] = field(default_factory=list)

    def add(self, constraint: str, weight: float = 1.0) -> None:
        """Append a constraint and its ranking weight, keeping them in step."""
        self.constraints.append(constraint)
        self.weights.append(weight)

    def weight_of(self, index: int) -> float:
        """Weight for constraint `index`, defaulting to 1.0 if unset."""
        return self.weights[index] if index < len(self.weights) else 1.0

    def demote_superseded(self, types: set[str], factor: float) -> list[str]:
        """Reduce the influence of constraints the new requirement supersedes.

        Demote rather than delete. The public evaluator builds an override's
        old_value from the TARGET's own soft preferences and new_value from the
        same target's hard constraints (local_evaluator.py behavior_for), so the
        "revoked" preference is still TRUE of the product being sought.
        Deleting it discards correct evidence and measurably costs score.

        Demotion satisfies both readings: the new requirement dominates the
        ranking, as a customer reversing themselves expects, while the old one
        keeps a residual vote instead of being thrown away. `factor` is swept,
        not assumed -- 1.0 is a no-op, 0.0 is full eviction.

        Only same-type constraints are touched. Changing your mind about colour
        does not retract your budget.
        """
        from .classify import classify_all

        while len(self.weights) < len(self.constraints):
            self.weights.append(1.0)

        touched: list[str] = []
        for index, constraint in enumerate(self.constraints):
            superseded = bool(classify_all(constraint) & types)
            if superseded or index == self.opener_preference:
                self.weights[index] *= factor
                touched.append(constraint)
        self.evicted.extend(touched)
        self.opener_preference = None
        return touched

    @property
    def parsed_nothing(self) -> bool:
        """True when template parsing has yielded no usable signal at all --
        the symptom of an unrecognised (paraphrased) message shape."""
        return not self.constraints and not self.category

    @property
    def dead_ask_count(self) -> int:
        """How many asks have come back empty. Evidence that the specific
        attribute space is exhausted and only `other` will yield anything."""
        return len(self.dead_attributes)

    @property
    def information_count(self) -> int:
        """How much the customer has actually disclosed. Drives truncation."""
        return len(self.constraints)
