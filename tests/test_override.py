"""Intent-override handling: supersede rather than accumulate.

The organizer's OVERRIDE EXAMPLE slide distinguishes a weak agent, which
"appends contradictory words", from a strong one, which replaces the revoked
requirement and reranks. These tests pin that behaviour so it cannot silently
regress back to appending.

They also pin the two judgement calls the implementation rests on, both of
which are easy to "simplify" into bugs later:

  1. Demotion, not deletion. The public evaluator builds an override's
     old_value from the target's own soft preferences and new_value from the
     same target's hard constraints (local_evaluator.py behavior_for), so the
     revoked preference is still TRUE of the product being sought. Deleting it
     discards correct evidence and measurably costs score; demoting keeps a
     residual vote while letting the new requirement dominate.

  2. Type-scoped, not global. Changing your mind about colour does not retract
     your budget.
"""

from __future__ import annotations

import unittest

from src.dialogue.classify import classify_all
from src.dialogue.parser import parse
from src.dialogue.state import SessionState

# The slide's own example.
OPENER = "I'm looking for running shoes. black running shoes"
OVERRIDE = "Actually, ignore my earlier preference. What I need is: casual white sneakers."
DISCLOSURE = "For that, what matters is: budget around $60."

SHIPPED_DEMOTE = 0.3


def _weight_of(state: SessionState, needle: str) -> float | None:
    """Ranking weight of the constraint containing `needle`, or None if absent."""
    for constraint, weight in zip(state.constraints, state.weights):
        if needle in constraint.lower():
            return weight
    return None


class OverrideDemotionTest(unittest.TestCase):
    def _session(self, demote: float, messages) -> SessionState:
        state = SessionState()
        for turn, message in messages:
            parse(state, message, turn, demote=demote)
        return state

    def test_slide_example_new_requirement_dominates(self) -> None:
        """The revoked requirement must not outrank the new one."""
        state = self._session(SHIPPED_DEMOTE, [(1, OPENER), (3, OVERRIDE)])

        revoked = _weight_of(state, "black running shoes")
        current = _weight_of(state, "casual white sneakers")
        self.assertIsNotNone(revoked, "the old constraint should still be present")
        self.assertIsNotNone(current, "the new requirement must be recorded")
        self.assertLess(revoked, current, "the revoked requirement still outranks the new one")

    def test_revoked_requirement_is_demoted_not_deleted(self) -> None:
        """Deletion costs score; the old value is still true of the target."""
        state = self._session(SHIPPED_DEMOTE, [(1, OPENER), (3, OVERRIDE)])
        self.assertIn("black running shoes", state.constraints)
        self.assertAlmostEqual(_weight_of(state, "black"), SHIPPED_DEMOTE, places=6)

    def test_override_is_recorded_for_explanation(self) -> None:
        """What was superseded is auditable, so the behaviour can be shown."""
        state = self._session(SHIPPED_DEMOTE, [(1, OPENER), (3, OVERRIDE)])
        self.assertTrue(any("black" in item.lower() for item in state.evicted))

    def test_unrelated_constraint_survives(self) -> None:
        """A colour/style override must not retract a stated budget."""
        state = self._session(
            SHIPPED_DEMOTE, [(1, OPENER), (2, DISCLOSURE), (3, OVERRIDE)])
        budget = _weight_of(state, "budget")
        self.assertIsNotNone(budget, "budget was dropped by an unrelated override")
        self.assertEqual(budget, 1.0, "budget was demoted by an unrelated override")

    def test_demote_one_is_a_no_op(self) -> None:
        """The control the sweep was measured against: 1.0 must change nothing."""
        state = self._session(1.0, [(1, OPENER), (3, OVERRIDE)])
        self.assertEqual(state.weights, [1.0, 1.0])
        self.assertEqual(state.evicted, [])

    def test_full_eviction_removes_the_revoked_requirement(self) -> None:
        """demote=0.0 is the documented ablation, not the shipped setting."""
        state = self._session(0.0, [(1, OPENER), (3, OVERRIDE)])
        self.assertEqual(_weight_of(state, "black running shoes"), 0.0)

    def test_weights_stay_parallel_to_constraints(self) -> None:
        """The ranker indexes weights by constraint position; drift is silent."""
        for demote in (0.0, SHIPPED_DEMOTE, 1.0):
            with self.subTest(demote=demote):
                state = self._session(
                    demote, [(1, OPENER), (2, DISCLOSURE), (3, OVERRIDE)])
                self.assertEqual(len(state.weights), len(state.constraints))

    def test_override_without_preceding_state_is_safe(self) -> None:
        """An override on turn 1, with nothing to supersede, must not raise."""
        state = SessionState()
        parse(state, OVERRIDE, 1, demote=SHIPPED_DEMOTE)
        parse(state, OVERRIDE, 2, demote=SHIPPED_DEMOTE)
        self.assertEqual(len(state.weights), len(state.constraints))


class ConstraintClassifierTest(unittest.TestCase):
    """classify_all decides WHAT an override supersedes, so its breadth matters."""

    def test_multi_type_constraint(self) -> None:
        self.assertEqual(classify_all("black running shoes"), {"color", "use_case"})

    def test_override_shares_a_type_with_what_it_revokes(self) -> None:
        """If these did not intersect, nothing would ever be demoted."""
        self.assertTrue(
            classify_all("casual white sneakers") & classify_all("black running shoes"))

    def test_budget_is_its_own_type(self) -> None:
        self.assertEqual(classify_all("budget around $60"), {"budget"})

    def test_unknown_text_falls_back_to_feature(self) -> None:
        self.assertEqual(classify_all("machine washable"), {"feature"})

    def test_empty_input_is_safe(self) -> None:
        self.assertEqual(classify_all(""), set())


if __name__ == "__main__":
    unittest.main()
