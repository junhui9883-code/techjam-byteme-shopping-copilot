"""Classify a constraint string into an attribute type.

Owner: dialogue

Used by override eviction to decide WHICH prior constraints a new requirement
supersedes. When a customer says "actually, make them casual white sneakers",
the colour and use-case they stated earlier are revoked -- but a budget they
mentioned is not, and blindly clearing all state would throw it away.

This deliberately reimplements the classification rather than importing from
evaluator/. The agent must not depend on evaluator internals: they are the
organizer's, we are forbidden from modifying them, and the private harness may
not expose the same module. Keeping our own copy also means a change there
cannot silently alter our behaviour.

The vocabulary mirrors the observable behaviour of the public evaluator's
`classify_constraint`, which is total over these seven types:

    budget, material, color, size, style, use_case, feature

`feature` is the catch-all. `brand` and `category` never appear -- no
constraint can classify as either, which is why src/dialogue/ask.py excludes
them from ASK_ORDER.
"""

from __future__ import annotations

import re

BUDGET_RE = re.compile(r"budget|\$|\bunder\b|<=|\bprice\b|\bcheap", re.I)

MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "denim", "linen", "mesh", "suede", "canvas",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray",
    "grey", "purple", "yellow", "orange", "navy", "beige", "tan", "ivory",
)
SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow", "fit", "length")
STYLE_WORDS = ("department", "style", "sleeve", "neck", "casual", "formal",
               "classic", "modern", "vintage")
USE_CASE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work",
                  "travel", "training", "walking", "summer", "everyday")

_MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIALS) + r")\b", re.I)
_COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)
_SIZE_RE = re.compile(r"\b(" + "|".join(SIZE_WORDS) + r")\b", re.I)
_STYLE_RE = re.compile(r"\b(" + "|".join(STYLE_WORDS) + r")\b", re.I)
_USE_CASE_RE = re.compile(r"\b(" + "|".join(USE_CASE_WORDS) + r")\b", re.I)

FALLBACK_TYPE = "feature"


def classify(value: str) -> str:
    """Single best attribute type for a constraint. Never raises."""
    types = classify_all(value)
    return next(iter(types)) if types else FALLBACK_TYPE


def classify_all(value: str) -> set[str]:
    """EVERY attribute type a constraint touches.

    A single override clause routinely revokes more than one thing -- "casual
    white sneakers" is both a style and a colour -- so eviction keys off the
    full set, not just the first match. Returned in a stable order via the
    checks below so behaviour does not depend on set iteration order.
    """
    if not value:
        return set()
    found: set[str] = set()
    if BUDGET_RE.search(value):
        found.add("budget")
    if _MATERIAL_RE.search(value):
        found.add("material")
    if _COLOR_RE.search(value):
        found.add("color")
    if _SIZE_RE.search(value):
        found.add("size")
    if _STYLE_RE.search(value):
        found.add("style")
    if _USE_CASE_RE.search(value):
        found.add("use_case")
    return found or {FALLBACK_TYPE}
