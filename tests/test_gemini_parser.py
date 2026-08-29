from __future__ import annotations

import unittest

from src.llm.gemini_parser import EMPTY_STATE, normalise_state, state_constraints
from src.llm.session_memory import SessionMemory


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

    def test_memory_keeps_separate_recipient_contexts(self) -> None:
        memory = SessionMemory()
        memory.apply({
            "recipient": "self",
            "context_action": "new",
            "intent": "buying",
            "category": "bag",
            "color": "white",
        }, "Find a white bag for me")
        memory.apply({
            "recipient": "brother",
            "context_action": "switch",
            "intent": "buying",
            "category": "bag",
            "color": "black",
        }, "Find a black bag for my brother")
        restored = memory.apply({
            "recipient": "self",
            "context_action": "resume",
            "intent": "buying",
            "category": "bag",
            "color": "white",
        }, "Go back to the white bag for me")

        self.assertEqual(restored["color"], "white")
        self.assertEqual(memory.contexts["brother|bag"]["color"], "black")
        self.assertEqual(len(memory.contexts), 2)

    def test_only_explicit_profile_updates_become_long_term(self) -> None:
        memory = SessionMemory()
        memory.apply({
            "recipient": "self",
            "category": "bag",
            "color": "white",
            "profile_updates": ["Usually prefers white bags"],
        }, "I usually prefer white bags")
        memory.apply({
            "recipient": "brother",
            "category": "bag",
            "color": "black",
        }, "My brother wants a black bag")

        self.assertEqual(
            memory.user_profile["stable_preferences"],
            ["Usually prefers white bags"],
        )


if __name__ == "__main__":
    unittest.main()
