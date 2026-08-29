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
from src.dialogue.selector import next_ask_candidate_aware
from src.dialogue.parser import parse
from src.dialogue.state import SessionState
from src.dialogue.truncation import list_length
from src.retrieval.index import CatalogIndex
from src.retrieval.recall import candidates
from src.retrieval.scoring import RankParams, score


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
    # Minimum expected discriminating power before a specific attribute is
    # worth a turn (ASK_POLICY="candidate_aware"; see src/dialogue/selector.py).
    MIN_QUESTION_VALUE = 0.25
    # Ranker weights. Never swept before; MRR is 0.3 of the score and 62 of our
    # 191 hits land below rank 1, so this is where the remaining headroom is.
    RANK_K1 = 1.1          # swept: 1.4->0.8553, 1.1->0.8641
    RANK_B = 0.6
    RANK_W_CATEGORY = 1.2
    RANK_W_CONSTRAINT = 2.0
    RANK_PHRASE_EXACT = 20.0   # held-out: all 3 folds agree on 20 (40 was tuned pre-popularity)
    RANK_PHRASE_PREFIX = 7.0
    RANK_PHRASE_OVERLAP = 8.0
    RANK_PRICE_NEAR = 10.0
    RANK_PRICE_LOOSE = 4.0
    RANK_PRICE_FAR = -2.0
    # Multiplier applied to the popularity prior while the customer has
    # disclosed NOTHING (turn-1 browsing, "still exploring"). 1.0 = flat.
    # Two opposing intuitions, so it is measured rather than argued:
    #   <1  the prior crowds out the little text signal there is
    #   >1  with no constraints stated, popularity is the only signal we have
    POPULARITY_UNINFORMED_SCALE = 1.0
    # How the popularity prior is combined with text evidence.
    #   "global"    -- additive term, popularity normalised once against the
    #                  catalog maximum. Absolute scale, so its influence varies
    #                  with how strong the text scores happen to be.
    #   "per_query" -- min-max normalise BOTH signals within the retrieved
    #                  candidate set, then blend. Adapted from Jun Hui's
    #                  codex-improvements branch. Scale-free, so the blend
    #                  stays balanced whatever the absolute scores look like.
    RANK_POPULARITY_MODE = "per_query"
    # Weight on popularity in per_query mode (0 = text only, 1 = popularity only).
    RANK_POPULARITY_BLEND = 0.27   # held-out: all 3 folds agree, zero penalty
    # Popularity prior weight (see src/retrieval/scoring.py). 0.0 = disabled.
    RANK_POPULARITY = 28.0   # swept: 0->0.8641, 10->0.8857, 25->0.9051, 28->0.9055, 40->0.9050
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

    def _rank_per_query(self, pool, state, params, fallback):
        """Rank by a scale-free blend of text evidence and popularity.

        Both signals are min-max normalised across the retrieved candidates, so
        neither can dominate merely by having larger raw numbers. The text
        scorer is called with the popularity term switched off, otherwise the
        prior would be counted twice.
        """
        text_params = RankParams(
            k1=params.k1, b=params.b, w_category=params.w_category,
            w_constraint=params.w_constraint, phrase_exact=params.phrase_exact,
            phrase_prefix=params.phrase_prefix, phrase_overlap=params.phrase_overlap,
            price_near_bonus=params.price_near_bonus,
            price_loose_bonus=params.price_loose_bonus,
            price_far_penalty=params.price_far_penalty,
            popularity=0.0)

        text = {pid: score(self.index, pid, state.category, state.constraints,
                           state.budget, fallback, self.PHRASE_OVERLAP,
                           state.weights, text_params) for pid in pool}
        pop = {pid: self.index.pop.get(pid, 0.0) for pid in pool}
        if not pool:
            return []

        t_lo, t_hi = min(text.values()), max(text.values())
        p_lo, p_hi = min(pop.values()), max(pop.values())
        t_span = (t_hi - t_lo) or 1.0
        p_span = (p_hi - p_lo) or 1.0
        blend = self.RANK_POPULARITY_BLEND

        def combined(pid: str) -> float:
            return ((1.0 - blend) * ((text[pid] - t_lo) / t_span)
                    + blend * ((pop[pid] - p_lo) / p_span))

        return sorted(pool, key=lambda pid: -combined(pid))

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Begin a new session. Discards any state under this session_id."""
        self._sessions[session_id] = SessionState(profile=user_profile or {})

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        # setdefault, not [], so a respond() without a preceding reset() still
        # works rather than raising KeyError.
        state = self._sessions.setdefault(session_id, SessionState())

        parse(state, user_message, turn, demote=self.OVERRIDE_DEMOTE)

        params = RankParams(
            k1=self.RANK_K1, b=self.RANK_B,
            w_category=self.RANK_W_CATEGORY, w_constraint=self.RANK_W_CONSTRAINT,
            phrase_exact=self.RANK_PHRASE_EXACT,
            phrase_prefix=self.RANK_PHRASE_PREFIX,
            phrase_overlap=self.RANK_PHRASE_OVERLAP,
            price_near_bonus=self.RANK_PRICE_NEAR,
            price_loose_bonus=self.RANK_PRICE_LOOSE,
            price_far_penalty=self.RANK_PRICE_FAR,
            popularity=(self.RANK_POPULARITY * self.POPULARITY_UNINFORMED_SCALE
                        if not state.constraints else self.RANK_POPULARITY))
        fallback = state.transcript if (self.RAW_FALLBACK and state.parsed_nothing) else None
        pool = candidates(self.index, state.category, state.constraints,
                          limit=self.RECALL_LIMIT,
                          fallback_text=fallback, weights=self.RECALL_WEIGHTS)
        # sorted() is stable, so products the ranker cannot separate keep their
        # FTS5 recall order. Keeps ties reproducible across runs.
        if self.RANK_POPULARITY_MODE == "per_query":
            ranked = self._rank_per_query(pool, state, params, fallback)
        else:
            ranked = sorted(
                pool,
                key=lambda pid: -score(self.index, pid, state.category,
                                       state.constraints, state.budget, fallback,
                                       self.PHRASE_OVERLAP, state.weights, params),
            )

        k = list_length(state, turn, top_k, enabled=self.TRUNCATE,
                        early_turns=self.TRUNC_EARLY_TURNS,
                        narrow_info=self.TRUNC_NARROW_INFO,
                        medium_info=self.TRUNC_MEDIUM_INFO)
        recommendations = [{"parent_asin": pid} for pid in ranked[:k]]

        # Chosen after ranking; the attribute asked this turn only affects the
        # customer's NEXT message, never this turn's recommendations.
        if self.ASK_POLICY == "candidate_aware":
            ask = next_ask_candidate_aware(state, self.index, pool, self.MIN_QUESTION_VALUE)
        else:
            ask = next_ask(state, self.ASK_POLICY, self.OTHER_FALLBACK_AFTER)

        return {
            "message": QUESTION_TEXT.get(ask, "What matters most to you in this item?"),
            "ask_attribute": ask,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
