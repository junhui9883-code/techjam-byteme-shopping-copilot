"""Popularity prior: it must break ties, never override evidence.

The prior is the largest single scoring gain we have (+0.041), and it is also
the easiest to get wrong. Weighted too heavily it stops being a tie-breaker and
becomes "always return the bestseller", which would score well on a public set
whose targets happen to be popular and fail the moment a customer wants
something niche.

These tests pin the two properties that keep it honest:

  1. At weight 0 it is EXACTLY inert -- the ranking is byte-identical to the
     ranking without it. This is the control the whole weight sweep was
     measured against; if it ever stops holding, every swept number is void.
  2. Evidence beats popularity. A product matching a stated requirement
     verbatim must outrank a far more popular product that does not, even at
     an extreme review-count ratio.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent import Agent

# A niche product that answers the question, and a blockbuster that does not.
# The review gap is deliberately absurd (1 vs 10,000,000) so the test fails
# loudly if the prior is ever allowed to dominate.
EXACT_MATCH_UNPOPULAR = {
    "parent_asin": "NICHE001",
    "title": "Merino Wool Hiking Sock",
    "features": ["merino wool cushioned crew", "reinforced heel"],
    "description": ["Warm hiking sock"],
    "price": 24.99,
    "categories": ["Clothing", "Socks"],
    "details": {"Material": "Merino Wool"},
    "average_rating": 4.1,
    "rating_number": 1,
    "store": "Small Maker",
}
POPULAR_MISMATCH = {
    "parent_asin": "BLOCK001",
    "title": "Plastic Phone Case",
    "features": ["clear polycarbonate shell"],
    "description": ["Protective case"],
    "price": 9.99,
    "categories": ["Electronics", "Cases"],
    "details": {"Material": "Polycarbonate"},
    "average_rating": 4.8,
    "rating_number": 10_000_000,
}
FILLER = [
    {
        "parent_asin": f"FILL{i:03d}",
        "title": f"Generic Item {i}",
        "features": [f"generic feature {i}"],
        "description": ["Filler"],
        "price": 15.0 + i,
        "categories": ["Clothing", "Misc"],
        "details": {"Material": "Cotton"},
        "average_rating": 4.0,
        "rating_number": 100 * (i + 1),
        "store": "Filler Store",
    }
    for i in range(10)
]


def _catalog(products) -> Path:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    with handle:
        for product in products:
            handle.write(json.dumps(product) + "\n")
    return Path(handle.name)


class PopularityPriorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = _catalog([EXACT_MATCH_UNPOPULAR, POPULAR_MISMATCH] + FILLER)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.path.unlink(missing_ok=True)

    def _rank(self, popularity: float) -> list[str]:
        agent = Agent(self.path)
        agent.RANK_POPULARITY = popularity
        agent.reset("s", {})
        response = agent.respond(
            "s",
            "I'm looking for Clothing Socks. A key requirement is: merino wool cushioned crew.",
            1, 10)
        return [item["parent_asin"] for item in response["recommendations"]]

    def test_zero_weight_is_exactly_inert(self) -> None:
        """The control the weight sweep rests on: 0.0 must change nothing.

        Compared against a scorer called with the prior absent entirely, not
        merely against another run at weight 0 -- otherwise the test would pass
        even if the prior were being added twice.
        """
        from src.retrieval.index import CatalogIndex
        from src.retrieval.scoring import RankParams, score

        index = CatalogIndex(self.path)
        constraints = ["merino wool cushioned crew"]
        for pid in index.ids:
            with_zero = score(index, pid, "Clothing Socks", constraints, None,
                              params=RankParams(popularity=0.0))
            without = score(index, pid, "Clothing Socks", constraints, None,
                            params=RankParams())
            self.assertEqual(with_zero, without, f"prior not inert at 0.0 for {pid}")

    def test_evidence_beats_popularity_at_shipped_weight(self) -> None:
        """A verbatim requirement match must outrank a 10M-review mismatch."""
        ranking = self._rank(Agent.RANK_POPULARITY)
        self.assertIn("NICHE001", ranking)
        self.assertLess(
            ranking.index("NICHE001"),
            ranking.index("BLOCK001") if "BLOCK001" in ranking else len(ranking),
            "a hugely popular non-match outranked an exact requirement match")

    def test_evidence_still_wins_well_above_the_shipped_weight(self) -> None:
        """Headroom check: the shipped weight is not sitting on a cliff edge."""
        ranking = self._rank(Agent.RANK_POPULARITY * 1.5)
        self.assertIn("NICHE001", ranking)
        self.assertEqual(ranking[0], "NICHE001",
                         "prior overtakes exact evidence at 1.5x the shipped weight")

    def test_prior_orders_otherwise_equal_candidates(self) -> None:
        """The prior must actually DO something when the text cannot separate."""
        agent = Agent(self.path)
        agent.RANK_POPULARITY = Agent.RANK_POPULARITY
        agent.reset("t", {})
        # A query matching only the interchangeable filler items.
        ranked = [item["parent_asin"] for item in agent.respond(
            "t", "I'm looking for Clothing Misc. A key requirement is: generic feature.",
            1, 10)["recommendations"]]
        fillers = [p for p in ranked if p.startswith("FILL")]
        self.assertGreaterEqual(len(fillers), 2, "expected filler items in the ranking")
        index = agent.index
        self.assertGreaterEqual(
            index.pop[fillers[0]], index.pop[fillers[-1]],
            "among textually equivalent items the more reviewed one should rank higher")

    def test_missing_rating_data_is_safe(self) -> None:
        """Absent or junk review counts must not raise or poison the prior."""
        odd = [
            {**EXACT_MATCH_UNPOPULAR, "parent_asin": "ODD001", "rating_number": None},
            {**EXACT_MATCH_UNPOPULAR, "parent_asin": "ODD002", "rating_number": "many"},
            {**EXACT_MATCH_UNPOPULAR, "parent_asin": "ODD003", "rating_number": -5},
        ]
        path = _catalog(odd)
        try:
            agent = Agent(path)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in agent.index.pop.values()))
            agent.reset("u", {})
            agent.respond("u", "I'm looking for socks.", 1, 10)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
