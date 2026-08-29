from __future__ import annotations

import unittest

from src.llm.gemini_parser import EMPTY_STATE, normalise_state, state_constraints


class GeminiParserHelperTest(unittest.TestCase):
    def test_normalise_state_rejects_unknown_fields(self) -> None:
        state = normalise_state({
            "intent": "buying",
            "category": "running shoes",
            "features": ["waterproof"],
            "unexpected": "do not keep me",
        })
        self.assertEqual(state["intent"], "buying")
        self.assertEqual(state["category"], "running shoes")
        self.assertNotIn("unexpected", state)
        self.assertEqual(set(state), set(EMPTY_STATE))

    def test_normalise_state_sanitises_numbers(self) -> None:
        state = normalise_state({"budget_max": "120", "confidence": 5})
        self.assertEqual(state["budget_max"], 120.0)
        self.assertEqual(state["confidence"], 1.0)

    def test_state_constraints_translates_slots(self) -> None:
        state = normalise_state({
            "brand": "Nike",
            "color": "black",
            "budget_max": 100,
            "features": ["waterproof"],
        })
        self.assertEqual(
            state_constraints(state),
            ["brand: Nike", "color: black", "waterproof", "budget around $100.0"],
        )


if __name__ == "__main__":
    unittest.main()
