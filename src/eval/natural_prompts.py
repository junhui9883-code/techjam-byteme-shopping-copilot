"""Non-official natural-language stress test and interactive chat.

The public evaluator uses a deterministic template customer.  This module
rewrites only the customer-facing wording into more natural sentences while
preserving the same hidden targets, scenario timing, and scoring rules.  Its
numbers are therefore *robustness diagnostics*, never official scores.

Run a benchmark:
    python3 -m src.eval.natural_prompts --benchmark
"""

from __future__ import annotations

import argparse
import hashlib
import re

from agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl


_BUYING = re.compile(r"^I'm looking for (.+?)\. A key requirement is: (.+?)\.$", re.I)
_BROWSING = re.compile(r"^I'm looking for (.+?), but I'm still exploring\.$", re.I)
_DISCLOSURE = re.compile(r"^For that, what matters is: (.+?)\.$", re.I)
_OVERRIDE = re.compile(r"^Actually, ignore my earlier preference\. What I need is: (.+?)\.$", re.I)
_BOUNDARY = re.compile(r"^I don't have a preference for (.+?); please use your judgment\.$", re.I)
_NO_MORE = re.compile(r"^I don't have an additional preference for (.+?)\.$", re.I)


def _variant(message: str, choices: tuple[str, ...]) -> str:
    """Choose a stable wording without using evaluator session IDs."""
    digest = hashlib.sha256(message.encode("utf-8")).digest()
    return choices[int.from_bytes(digest[:2], "big") % len(choices)]


def naturalize(message: str) -> str:
    """Render one known evaluator utterance in ordinary customer language.

    Values such as a material or budget remain unchanged.  Only the template
    framing is replaced, so a resulting score measures language robustness
    rather than a changed target intent.
    """
    text = message.strip()
    if match := _BUYING.match(text):
        category, constraint = match.groups()
        return _variant(text, (
            f"Hi! I'm shopping for {category} and really need {constraint}.",
            f"Could you help me find {category}? The main thing is {constraint}.",
            f"I'm after {category}; it has to have {constraint}.",
        ))
    if match := _BROWSING.match(text):
        category = match.group(1)
        return _variant(text, (
            f"I'm browsing for {category} but haven't settled on the details yet.",
            f"I'd like to look at some {category}. I'm still figuring out what suits me.",
            f"Can you show me {category}? I'm open to ideas at the moment.",
        ))
    if match := _DISCLOSURE.match(text):
        values = match.group(1)
        return _variant(text, (
            f"I'd really like {values}, if possible.",
            f"The details I care about are {values}.",
            f"For me, {values} would make the biggest difference.",
        ))
    if match := _OVERRIDE.match(text):
        value = match.group(1)
        return _variant(text, (
            f"Actually, I've changed my mind — I need {value} instead.",
            f"Let's forget the earlier preference. Please prioritise {value}.",
            f"On second thought, {value} is what I need most.",
        ))
    if match := _BOUNDARY.match(text):
        attribute = match.group(1)
        return _variant(text, (
            f"I don't mind about {attribute}; use your best judgment.",
            f"No strong preference on {attribute}. You can decide.",
        ))
    if match := _NO_MORE.match(text):
        attribute = match.group(1)
        return _variant(text, (
            f"Nothing else comes to mind for {attribute}.",
            f"I don't have any more thoughts about {attribute}.",
        ))
    if text == "Those options are not quite right yet. Ask me about one specific attribute.":
        return "Those do not feel right yet. Could you ask me something more specific?"
    return text


class NaturalPromptAgent:
    """Agent proxy used only by the non-official natural-language benchmark."""

    def __init__(self, inner: Agent) -> None:
        self.inner = inner
        self.rewritten = 0

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        rewritten = naturalize(user_message)
        self.rewritten += int(rewritten != user_message)
        return self.inner.respond(session_id, rewritten, turn, top_k)


def run_benchmark(catalog: str, dataset: str) -> dict:
    """Evaluate natural customer wording with unchanged targets and simulator."""
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)
    wrapper = NaturalPromptAgent(Agent(catalog))
    result = evaluate(wrapper, samples, catalog_ids, categories, products)
    result["natural_prompt_diagnostic"] = {
        "rewritten_customer_turns": wrapper.rewritten,
        "warning": "Non-official robustness diagnostic; do not report as the competition score.",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Natural-language diagnostics for the Shopping Copilot")
    parser.add_argument("--benchmark", action="store_true", required=True,
                        help="score natural customer wording (non-official)")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    result = run_benchmark(args.catalog, args.dataset)
    print("NATURAL-PROMPT DIAGNOSTIC — NOT AN OFFICIAL SCORE")
    print(f"rewritten customer turns : {result['natural_prompt_diagnostic']['rewritten_customer_turns']}")
    print(f"hit@10                  : {result['hit_rate_at_10']:.6f}")
    print(f"mrr                     : {result['mrr']:.6f}")
    print(f"mttc                    : {result['mttc']:.6f}")
    print(f"technical score         : {result['recommended_technical_score']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
