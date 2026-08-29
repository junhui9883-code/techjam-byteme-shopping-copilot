"""Chat with ByteMe using free-form prompts and an optional Gemini parser."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.text import terms

from src.llm.gemini_parser import (
    EMPTY_STATE,
    GeminiParseError,
    GeminiShoppingParser,
    state_constraints,
)
from src.retrieval.index import CatalogIndex
from src.retrieval.recall import candidates
from src.retrieval.scoring import score


def short(value: object, limit: int = 88) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"

def category_first_candidates(
    index: CatalogIndex,
    category: str,
    constraints: list[str],
    transcript: list[str],
    limit: int = 400,
) -> list[str]:
    # Original broad retrieval remains available as a fallback.
    broad_pool = candidates(
        index,
        category,
        constraints,
        limit=limit,
        fallback_text=transcript,
    )

    wanted_terms = set(terms(category))
    if not wanted_terms:
        return broad_pool

    # Retrieve using the category alone so other preferences cannot overpower it.
    category_pool = candidates(
        index,
        category,
        [],
        limit=limit,
    )

    matching_products: list[str] = []

    for pid in category_pool:
        fields = index.ftext[pid]
        category_text = fields["title"] + " " + fields["categories"]
        product_terms = set(terms(category_text))

        # For "running shoes", the product must contain both "running" and "shoes".
        if wanted_terms.issubset(product_terms):
            matching_products.append(pid)

    # Use strict category matching when enough products are available.
    # Otherwise retain the original broad fallback.
    return matching_products if len(matching_products) >= 5 else broad_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive ByteMe shopping assistant")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("Building the local product index…")
    index = CatalogIndex(args.catalog)
    state = dict(EMPTY_STATE)
    transcript: list[str] = []

    try:
        llm = GeminiShoppingParser()
        print("Gemini understanding: ON")
    except GeminiParseError as exc:
        llm = None
        print(f"Gemini understanding: OFF ({exc})")
        print("Raw lexical fallback is active.")

    print("Type a shopping request, 'state' to inspect memory, or 'quit' to exit.\n")

    while True:
        message = input("You: ").strip()
        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            break
        if message.lower() == "state":
            print(state, "\n")
            continue

        transcript.append(message)
        if llm is not None:
            try:
                state = llm.parse(message, state)
            except GeminiParseError as exc:
                print(f"[Gemini unavailable this turn: {exc}]")
                state["features"] = [" ".join(transcript)]
        else:
            state["features"] = [" ".join(transcript)]

        constraints = state_constraints(state)
        category = str(state.get("category") or "")
        pool = category_first_candidates(
        index,
        category,
        constraints,
        transcript,
        limit=400,
        )
        budget_max = state.get("budget_max")

        # A stated maximum budget is a hard limit.
        # Products with missing prices remain available as fallback candidates.
        if budget_max is not None:
            pool = [
                pid
                for pid in pool
                if index.price.get(pid) is None
                or index.price[pid] <= budget_max
            ]

        ranked = sorted(
            pool,
            key=lambda pid: (
                # When a budget exists, rank verified prices before unknown prices.
                1
                if budget_max is not None and index.price.get(pid) is None
                else 0,
                -score(
                    index,
                    pid,
                    category,
                    constraints,
                    budget_max,
                    transcript,
                    True,
                ),
            ),
        )

        print(f"\nByteMe understood: intent={state['intent']}, category={state['category']}")
        print("Current requirements:", constraints or "none yet")
        print("Recommendations:")
        for rank, pid in enumerate(ranked[: max(1, min(args.top_k, 10))], 1):
            title = index.ftext[pid]["title"]
            price = index.price.get(pid)
            price_text = "price unavailable" if price is None else f"${price:.2f}"
            print(f"  {rank}. {short(title)} — {price_text} [{pid}]")

        question = str(state.get("clarification_question") or "").strip()
        if question:
            print(f"Question: {question}")
        print()


if __name__ == "__main__":
    main()

