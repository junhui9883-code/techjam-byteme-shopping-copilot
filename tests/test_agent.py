from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent import Agent


PRODUCTS = [
    {
        "parent_asin": "TEST001",
        "title": "Blue Cotton Running Shirt",
        "features": ["100% Cotton", "Lightweight running shirt"],
        "description": ["Comfortable athletic top"],
        "price": 29.99,
        "categories": ["Men", "Clothing", "Shirts"],
        "details": {"Department": "Mens"},
        "average_rating": 4.5,
        "rating_number": 100,
        "store": "Test Store",
    },
    {
        "parent_asin": "TEST002",
        "title": "Black Leather Belt",
        "features": ["Full grain leather"],
        "description": ["Everyday belt"],
        "price": 39.99,
        "categories": ["Men", "Accessories", "Belts"],
        "details": {"Material": "Leather"},
        "average_rating": 4.2,
        "rating_number": 50,
        "store": "Test Store",
    },
]


class AgentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        with handle:
            for product in PRODUCTS:
                handle.write(json.dumps(product) + "\n")
        cls.catalog_path = Path(handle.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.catalog_path.unlink(missing_ok=True)

    def setUp(self) -> None:
        self.agent = Agent(self.catalog_path)

    def test_response_follows_contract(self) -> None:
        self.agent.reset("session-1", {})
        response = self.agent.respond(
            "session-1",
            "I'm looking for Clothing Shirts. A key requirement is: cotton.",
            1,
            10,
        )
        self.assertIsInstance(response["message"], str)
        self.assertIn(response["ask_attribute"], {
            "category", "material", "color", "size", "style", "brand",
            "budget", "feature", "use_case", "other", None,
        })
        self.assertLessEqual(len(response["recommendations"]), 10)
        self.assertTrue(all("parent_asin" in item for item in response["recommendations"]))

    def test_conversation_accumulates_information(self) -> None:
        self.agent.reset("session-2", {})
        self.agent.respond(
            "session-2",
            "I'm looking for Clothing Shirts, but I'm still exploring.",
            1,
            10,
        )
        self.agent.respond(
            "session-2",
            "For that, what matters is: cotton; lightweight.",
            2,
            10,
        )
        state = self.agent._sessions["session-2"]
        self.assertEqual(state.category, "Clothing Shirts")
        self.assertIn("cotton", state.constraints)
        self.assertIn("lightweight", state.constraints)

    def test_respond_without_reset_is_safe(self) -> None:
        response = self.agent.respond(
            "new-session",
            "I'm looking for Accessories Belts, but I'm still exploring.",
            1,
            10,
        )
        self.assertIsInstance(response, dict)
        self.assertIn("new-session", self.agent._sessions)

    def test_recommendations_are_deterministic(self) -> None:
        first = Agent(self.catalog_path)
        second = Agent(self.catalog_path)
        message = "I'm looking for Clothing Shirts. A key requirement is: cotton."
        first.reset("a", {})
        second.reset("b", {})
        first_result = first.respond("a", message, 1, 10)["recommendations"]
        second_result = second.respond("b", message, 1, 10)["recommendations"]
        self.assertEqual(first_result, second_result)


if __name__ == "__main__":
    unittest.main()

class TopKContractTest(unittest.TestCase):
    """Regression guard: the agent must never return more than top_k items.

    NARROW_K and MEDIUM_K in src/dialogue/truncation.py are absolute constants.
    Before they were clamped, `respond(..., top_k=1)` returned 3 and
    `top_k=2` returned 3 -- a contract violation. The official evaluator always
    passes top_k=10 so it never surfaced there, but the submission rules state
    the organizer "reserves the right to run your submission under CPU, memory,
    timeout, and network restrictions", and nothing fixes top_k at 10.
    """

    def test_never_exceeds_top_k(self):
        from agent import Agent
        agent = Agent("data/catalog.jsonl")
        agent.reset("contract", {})
        message = "I'm looking for a jacket, but I'm still exploring."
        for top_k in (0, 1, 2, 3, 5, 10, 20):
            with self.subTest(top_k=top_k):
                out = agent.respond("contract", message, 1, top_k)
                self.assertLessEqual(len(out["recommendations"]), max(top_k, 0))
