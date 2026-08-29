"""Catalog index: built once per process, read-only thereafter.

Owner: retrieval

Two parallel representations of the same 50,000 products, because the two
retrieval stages want different things:

  1. An in-memory SQLite FTS5 table, used only for cheap candidate *recall*
     (narrow 50,000 -> 400).
  2. Field-weighted term frequencies + IDF + parsed prices + a lowercased text
     blob per product, used for *reranking* those 400 candidates.

Build cost is dominated by tokenising 300k fields. Nothing here touches the
network and nothing is persisted to disk.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

from .text import FIELDS, FIELD_W, flatten, terms

# FTS5 column order is (pid, *FIELDS); pid is UNINDEXED so it never matches.
_CREATE_TABLE = (
    "CREATE VIRTUAL TABLE p USING fts5("
    "pid UNINDEXED, title, categories, features, details, store, description, "
    "tokenize='unicode61 remove_diacritics 2')"
)
_INSERT_BATCH = 2000


class CatalogIndex:
    """Immutable view over the frozen catalog.

    Attributes
    ----------
    conn    : sqlite3 connection holding the FTS5 recall table
    ids     : product ids in catalog file order (also the no-query fallback order)
    price   : pid -> float price, or None when absent/unparseable
    ftext   : pid -> {field: lowercased text} for phrase checks
    tf      : pid -> {term: field-weighted frequency}
    dl      : pid -> unweighted token count (BM25 document length)
    idf     : term -> inverse document frequency
    avgdl   : mean document length over the catalog
    pop     : pid -> popularity prior in [0, 1], from log1p(rating_number)
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.conn = sqlite3.connect(":memory:")
        self.ids: list[str] = []
        self.price: dict[str, float | None] = {}
        self.ftext: dict[str, dict[str, str]] = {}
        self.tf: dict[str, dict[str, float]] = {}
        self.dl: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.pop: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        cur = self.conn.cursor()
        cur.execute(_CREATE_TABLE)

        batch: list[tuple] = []
        df: defaultdict[str, int] = defaultdict(int)

        with self.catalog_path.open(encoding="utf-8") as fh:
            for line in fh:
                product = json.loads(line)
                pid = str(product["parent_asin"])
                fx = {f: flatten(product.get(f)) for f in FIELDS}

                batch.append((
                    pid,
                    fx["title"], fx["categories"], fx["features"],
                    fx["details"], fx["store"], fx["description"],
                ))
                self.ids.append(pid)
                self.ftext[pid] = {f: fx[f].lower() for f in FIELDS}
                self.price[pid] = _parse_price(product.get("price"))
                self.pop[pid] = _rating_count(product.get("rating_number"))

                # Field-weighted term frequencies. `n` counts raw tokens (not
                # weighted) so document length stays a true length.
                tf: defaultdict[str, float] = defaultdict(float)
                n = 0
                for field in FIELDS:
                    weight = FIELD_W[field]
                    for term in terms(fx[field]):
                        tf[term] += weight
                        n += 1
                self.tf[pid] = tf
                self.dl[pid] = max(n, 1)
                for term in tf:
                    df[term] += 1

                if len(batch) >= _INSERT_BATCH:
                    cur.executemany("INSERT INTO p VALUES (?,?,?,?,?,?,?)", batch)
                    batch.clear()

        if batch:
            cur.executemany("INSERT INTO p VALUES (?,?,?,?,?,?,?)", batch)
        self.conn.commit()

        total = len(self.ids)
        self.idf = {t: math.log(1 + (total - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.avgdl = sum(self.dl.values()) / max(total, 1)

        # Popularity prior. Review counts are extremely heavy-tailed -- the
        # catalog median is 12 while the public set's median TARGET has 6,846 --
        # so raw counts would let one blockbuster dominate every ranking. log1p
        # compresses that, and dividing by the observed maximum puts the prior
        # in [0, 1] so its weight is interpretable against the other signals.
        largest = max(self.pop.values(), default=0.0) or 1.0
        self.pop = {pid: value / largest for pid, value in self.pop.items()}


def _rating_count(raw: object) -> float:
    """log1p of the review count; 0.0 when absent or unparseable."""
    if not isinstance(raw, (int, float)) or raw <= 0:
        return 0.0
    return math.log1p(float(raw))


def _parse_price(raw: object) -> float | None:
    """Catalog prices arrive as floats, '$12.99' strings, or junk. Never raise."""
    if raw in (None, "", "None"):
        return None
    try:
        return float(str(raw).replace("$", ""))
    except Exception:
        return None
