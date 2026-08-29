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


QUESTION_TEXT = {
    "material": "Do you have a preferred material, such as cotton, leather, or wool?",
    "color": "Do you have a colour in mind?",
    "budget": "What price range would you like to stay within?",
    "style": "What style or fit are you looking for?",
    "size": "What size or fit would work best for you?",
    "use_case": "What will you mainly be using it for?",
    "feature": "Is there a feature that matters most to you?",
    # `other` is meaningful to the evaluator but is not customer-friendly
    # language. Phrase it as an open follow-up for manual demonstrations too.
    "other": "What else matters most to you—fit, activity, style, features, or budget?",
}


class Agent:
    """Stateful multi-turn shopping agent over a frozen 50,000-product catalog."""

    # Elicitation policy: "priority" (shipped) or "other" (see src/dialogue/ask).
    ASK_POLICY = "priority"
    # Dynamic list-length schedule (see src/dialogue/truncation).
    TRUNCATE = True
    # Dead asks tolerated before falling back to `other` (see src/dialogue/ask).
    OTHER_FALLBACK_AFTER = 1
    # Truncation schedule knobs (see src/dialogue/truncation), swept per route.
    TRUNC_EARLY_TURNS = 2
    TRUNC_NARROW_INFO = 2
    TRUNC_MEDIUM_INFO = 4
    # Rank on the raw transcript when template parsing recognises nothing.
    # Paraphrase insurance; see src/eval/paraphrase.py.
    RAW_FALLBACK = True
    # Weight applied to constraints an override supersedes. 1.0 = accumulate
    # (old behaviour), 0.0 = full eviction. Swept; see the run notes.
    OVERRIDE_DEMOTE = 0.3
    # FTS5 recall weight set (see src/retrieval/recall.py WEIGHT_SETS).
    RECALL_WEIGHTS = "default"
    # Candidate pool size handed to the reranker (see src/retrieval/recall.py).
    RECALL_LIMIT = 400
    # Lexical-overlap tier beneath the exact-phrase bonus (paraphrase insurance).
    PHRASE_OVERLAP = True

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

        parse(state, user_message, turn, demote=self.OVERRIDE_DEMOTE)

        fallback = state.transcript if (self.RAW_FALLBACK and state.parsed_nothing) else None
        pool = candidates(self.index, state.category, state.constraints,
                          limit=self.RECALL_LIMIT,
                          fallback_text=fallback, weights=self.RECALL_WEIGHTS)
        # sorted() is stable, so products the ranker cannot separate keep their
        # FTS5 recall order. Keeps ties reproducible across runs.
        ranked = sorted(
            pool,
            key=lambda pid: -score(self.index, pid, state.category,
                                   state.constraints, state.budget, fallback,
                                   self.PHRASE_OVERLAP, state.weights),
        )

        k = list_length(state, turn, top_k, enabled=self.TRUNCATE,
                        early_turns=self.TRUNC_EARLY_TURNS,
                        narrow_info=self.TRUNC_NARROW_INFO,
                        medium_info=self.TRUNC_MEDIUM_INFO)
        recommendations = [{"parent_asin": pid} for pid in ranked[:k]]

        # Chosen after ranking; the attribute asked this turn only affects the
        # customer's NEXT message, never this turn's recommendations.
        ask = next_ask(state, self.ASK_POLICY, self.OTHER_FALLBACK_AFTER)

        return {
            "message": QUESTION_TEXT.get(ask, "What matters most to you in this item?"),
            "ask_attribute": ask,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
