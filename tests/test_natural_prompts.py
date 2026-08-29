from __future__ import annotations

import unittest

from src.eval.natural_prompts import naturalize


class NaturalPromptTest(unittest.TestCase):
    def test_buying_prompt_keeps_category_and_constraint(self) -> None:
        original = "I'm looking for running shoes. A key requirement is: waterproof."
        rewritten = naturalize(original)
        self.assertIn("running shoes", rewritten)
        self.assertIn("waterproof", rewritten)
        self.assertNotIn("A key requirement is:", rewritten)

    def test_disclosure_uses_natural_wording(self) -> None:
        original = "For that, what matters is: cotton; color: blue."
        rewritten = naturalize(original)
        self.assertIn("cotton; color: blue", rewritten)
        self.assertNotIn("For that, what matters is:", rewritten)

    def test_unknown_messages_are_not_modified(self) -> None:
        original = "I need an inexpensive jacket for a rainy commute."
        self.assertEqual(naturalize(original), original)


if __name__ == "__main__":
    unittest.main()
