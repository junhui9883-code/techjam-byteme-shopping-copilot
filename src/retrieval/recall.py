"""Stage 1 of retrieval: narrow 50,000 products to a rerankable candidate pool.

Owner: retrieval

FTS5 does the cheap work. Its BM25 is only good enough to decide *membership*
of the pool, never the final order -- src/retrieval/scoring.py owns the order.
"""

from __future__ import annotations

from .index import CatalogIndex
from .text import terms

CANDIDATE_LIMIT = 400

# Per-column BM25 weights for the recall query, aligned to the FTS5 column
# order (pid, title, categories, features, details, store, description).
# pid is weighted 0.0 because it is an opaque identifier.
_RECALL_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)

# Named alternatives, swept via `--set RECALL_WEIGHTS=<name>`.
#   codex : proposed by Jun Hui on branch codex-improvements. Boosts `features`
#           2.5 -> 6.0 on the reasoning that intent_card() is generated from
#           `features` and `details`, so the brief's wording should live there.
WEIGHT_SETS = {
    "default": (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0),
    "codex":   (0.0, 5.0, 5.0, 6.0, 3.0, 2.0, 1.5),
    "features-heavy": (0.0, 4.0, 3.0, 8.0, 5.0, 1.0, 0.5),
    "spec-only":      (0.0, 3.0, 2.0, 8.0, 8.0, 1.0, 0.5),
}

# FTS5 degrades badly on very long disjunctions, and the intent card can only
# ever disclose ~4 short strings, so a 60-term ceiling costs us nothing.
MAX_QUERY_TERMS = 60

def _sql_for(weights: tuple[float, ...]) -> str:
    return ("SELECT pid FROM p WHERE p MATCH ? ORDER BY bm25(p,"
            + ",".join(str(w) for w in weights) + ") LIMIT ?")


_SQL = _sql_for(_RECALL_WEIGHTS)


def candidates(index: CatalogIndex, category: str, constraints: list[str],
               limit: int = CANDIDATE_LIMIT, fallback_text: list[str] | None = None,
               weights: str = "default") -> list[str]:
    """Return up to `limit` candidate pids for the accumulated session state.

    With nothing disclosed yet (turn 1 of a browsing session) there is no query
    to run, so we fall back to catalog order. Those turns are noise by
    construction and the ranker cannot do better than chance on them.
    """
    # `fallback_text` is the raw transcript, used only when template parsing
    # produced nothing. Catalog order (the previous behaviour) is worth about
    # nothing, so any lexical signal beats it.
    query = " ".join([category] + constraints)
    if not query.strip() and fallback_text:
        query = " ".join(fallback_text)
    # dict.fromkeys dedupes while preserving first-seen order, which keeps the
    # query string stable across runs.
    query_terms = list(dict.fromkeys(terms(query)))[:MAX_QUERY_TERMS]
    if not query_terms:
        return list(index.ids[:limit])

    expression = " OR ".join(f'"{t}"' for t in query_terms)
    sql = _SQL if weights == "default" else _sql_for(WEIGHT_SETS[weights])
    rows = index.conn.execute(sql, (expression, limit)).fetchall()
    return [r[0] for r in rows]
