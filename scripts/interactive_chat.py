"""Chat with ByteMe using free-form prompts and an optional Gemini parser."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.gemini_parser import (
    EMPTY_STATE,
    GeminiParseError,
    GeminiShoppingParser,
    state_constraints,
)
from src.llm.session_memory import SessionMemory
from src.retrieval.index import CatalogIndex
from src.retrieval.recall import candidates
from src.retrieval.scoring import score
from src.retrieval.text import terms


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
    broad_pool = candidates(
        index, category, constraints, limit=limit, fallback_text=transcript
    )
    wanted_terms = set(terms(category))
    if not wanted_terms:
        return broad_pool

    category_pool = candidates(index, category, [], limit=limit)
    matching_products = []
    for pid in category_pool:
        fields = index.ftext[pid]
        product_terms = set(terms(fields["title"] + " " + fields["categories"]))
        if wanted_terms.issubset(product_terms):
            matching_products.append(pid)
    return matching_products if len(matching_products) >= 5 else broad_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive ByteMe shopping assistant")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("Building the local product index…")
    index = CatalogIndex(args.catalog)
    memory = SessionMemory()

    try:
        llm = GeminiShoppingParser()
        print("Gemini understanding: ON")
    except GeminiParseError as exc:
        llm = None
        print(f"Gemini understanding: OFF ({exc})")
        print("Raw lexical fallback is active.")

    print(
        "Type a shopping request, 'state' for the active goal, 'memory' for all "
        "goals, 'new' to start over, or 'quit' to exit.\n"
    )

    while True:
        message = input("You: ").strip()
        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            break
        if message.lower() in {"new", "reset"}:
            memory.new_request()
            print("Started a new shopping request; long-term preferences were retained.\n")
            continue
        if message.lower() == "state":
            print(memory.active_state(), "\n")
            continue
        if message.lower() == "memory":
            print(memory.inspect(), "\n")
            continue

        current_state = memory.active_state()
        if llm is not None:
            try:
                parsed = llm.parse(
                    message,
                    current_state,
                    saved_contexts=memory.summaries(),
                    user_profile=memory.user_profile,
                )
            except GeminiParseError as exc:
                print(f"[Gemini unavailable this turn: {exc}]")
                parsed = current_state
                parsed["features"] = list(parsed.get("features", [])) + [message]
        else:
            parsed = current_state
            parsed["features"] = list(parsed.get("features", [])) + [message]

        state = memory.apply(parsed, message)
        transcript = memory.active_transcript()

        constraints = state_constraints(state)
        category = str(state.get("category") or "")
        pool = category_first_candidates(
            index, category, constraints, transcript, limit=400
        )

        budget_max = state.get("budget_max")
        if budget_max is not None:
            pool = [
                pid for pid in pool
                if index.price.get(pid) is None or index.price[pid] <= budget_max
            ]
        ranked = sorted(
            pool,
            key=lambda pid: (
                1 if budget_max is not None and index.price.get(pid) is None else 0,
                -score(
                    index, pid, category, constraints, budget_max, transcript, True
                ),
            ),
        )

        print(
            f"\nByteMe understood: recipient={state['recipient']}, "
            f"intent={state['intent']}, category={state['category']}, "
            f"context={state['context_action']}"
        )
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
