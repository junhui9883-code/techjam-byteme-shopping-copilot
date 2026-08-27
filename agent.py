"""TechJam 2026 / Problem 4 -- Shopping Copilot.

Submission entry point. This file implements the required Agent interface and
NOTHING else: every decision lives in src/retrieval (what to return) or
src/dialogue (what to ask, how much to show).

Offline by construction -- stdlib only, no network, no hosted LLM, no vector
DB. See CLAUDE.md section 5 for why that is non-negotiable.

Required interface (docs/api_contract):
    reset(session_id, user_profile) -> None
    respond(session_id, user_message, turn, top_k) -> dict

Pipeline, once per turn:
    parse    src/dialogue/parser    message -> SessionState mutations
    recall   src/retrieval/recall   50,000 -> 400 candidates via FTS5
    rank     src/retrieval/scoring  400 -> ordered, BM25 + phrase + price
    truncate src/dialogue/truncation how many of them to actually expose
    ask      src/dialogue/ask       which attribute to elicit next

Measured on the 200 public sessions with the unmodified official evaluator:
    ASK_POLICY="priority", TRUNCATE=True   ->  Hit@10 0.910  MRR 0.706  MTTC 5.05  Score 0.786
    ASK_POLICY="priority", TRUNCATE=False  ->  Hit@10 0.920  MRR 0.617  MTTC 4.45  Score 0.776
    ASK_POLICY="other",    TRUNCATE=True   ->  Hit@10 0.950  MRR 0.728  MTTC 2.98  Score 0.854
    official weak BM25 baseline            ->  Hit@10 0.125  MRR 0.068  MTTC 9.81  Score 0.107

Reproduce: python3 -m evaluator.local_evaluator --output runs/day0.json
"""

from __future__ import annotations

from src.dialogue.ask import next_ask
from src.dialogue.parser import parse
from src.dialogue.state import SessionState
from src.dialogue.truncation import list_length
from src.retrieval.index import CatalogIndex
from src.retrieval.recall import candidates
from src.retrieval.scoring import score


class Agent:
    """Stateful multi-turn shopping agent over a frozen 50,000-product catalog."""

    # Elicitation policy: "priority" (shipped) or "other" (see src/dialogue/ask).
    ASK_POLICY = "priority"
    # Dynamic list-length schedule (see src/dialogue/truncation).
    TRUNCATE = True

    def __init__(self, catalog_path: str = "data/catalog.jsonl") -> None:
        # ~15s, once per process. The evaluator constructs one Agent for all
        # 200 sessions, so this cost is amortised and not on the per-turn path.
        self.index = CatalogIndex(catalog_path)
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Begin a new session. Discards any state under this session_id."""
        self._sessions[session_id] = SessionState(profile=user_profile or {})

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        # setdefault, not [], so a respond() without a preceding reset() still
        # works rather than raising KeyError.
        state = self._sessions.setdefault(session_id, SessionState())

        parse(state, user_message, turn)

        pool = candidates(self.index, state.category, state.constraints)
        # sorted() is stable, so products the ranker cannot separate keep their
        # FTS5 recall order. Keeps ties reproducible across runs.
        ranked = sorted(
            pool,
            key=lambda pid: -score(self.index, pid, state.category,
                                   state.constraints, state.budget),
        )

        k = list_length(state, turn, top_k, enabled=self.TRUNCATE)
        recommendations = [{"parent_asin": pid} for pid in ranked[:k]]

        # Chosen after ranking; the attribute asked this turn only affects the
        # customer's NEXT message, never this turn's recommendations.
        ask = next_ask(state, self.ASK_POLICY)

        return {
            "message": f"Could you tell me about your {ask} preference?",
            "ask_attribute": ask,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
