"""Tokenisation and field flattening shared by the index and the ranker.

Owner: retrieval

Everything here is intentionally stdlib-only and deterministic. These helpers
run once per catalog field at build time (50k products x 6 fields), so they are
kept allocation-cheap rather than clever.
"""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Deliberately small. Aggressive stopword removal hurts here because the hidden
# brief is lifted near-verbatim from a product spec sheet, so short function
# words still carry phrase-alignment signal.
STOP = set(
    "a an and are as at be but by for from i in is it me my of on or please "
    "some that the this to want with would you looking need key requirement "
    "matters".split()
)

# Catalog fields we index, and their term-frequency weights in the BM25 route.
# Order matters: it is the column order of the FTS5 table.
FIELDS = ("title", "features", "details", "categories", "store", "description")
FIELD_W = {
    "title": 3.0,
    "features": 2.5,
    "details": 2.0,
    "categories": 1.5,
    "store": 1.0,
    "description": 1.0,
}


def flatten(value: object) -> str:
    """Collapse an arbitrary catalog value into one searchable string.

    Catalog fields are heterogeneous: `title` is a string, `features` a list,
    `details` a dict of spec-sheet key/value pairs. Dict keys are kept because
    the intent card is generated from the same key/value pairs.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def terms(text: str) -> list[str]:
    """Lowercased content tokens, single characters and stopwords dropped."""
    return [w.lower() for w in TOKEN_RE.findall(text) if len(w) > 1 and w.lower() not in STOP]
