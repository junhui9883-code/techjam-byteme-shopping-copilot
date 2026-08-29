"""Print one official public session as a readable end-to-end demo."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluator.local_evaluator import (
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent


def shorten(text: str, limit: int = 82) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "â€¦"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a readable TechJam demo session")
    parser.add_argument("--sample", default="public_0003")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    samples = {item["sample_id"]: item for item in load_jsonl(args.dataset)}
    if args.sample not in samples:
        raise SystemExit(f"Unknown sample: {args.sample}")

    sample = samples[args.sample]
    catalog_ids, categories, products = catalog_index(args.catalog)
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": card, "behavior": behavior}

    agent = Agent(args.catalog)
    session_id = f"demo_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(
        effective_sample,
        coarse_category(categories.get(target, [])),
        disclosed,
    )

    print("\nBYTE ME â€” SHOPPING COPILOT DEMO")
    print(f"Session: {sample['sample_id']} | Scenario: {sample['scenario_type']}")
    print("The target remains hidden until the agent finds it.\n")

    for turn in range(1, 11):
        print(f"TURN {turn}")
        print(f"Customer: {user_message}")
        response = agent.respond(session_id, user_message, turn, TOP_K)
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)

        print(f"Agent: {response['message']}")
        print(f"Structured ask: {response.get('ask_attribute')}")
        print("Recommendations:")
        for rank, asin in enumerate(ranked, 1):
            title = shorten(str(products[asin].get("title") or "Untitled product"))
            print(f"  {rank}. {title} [{asin}]")

        if override_applied and target in ranked:
            rank = ranked.index(target) + 1
            print("\nâœ“ CONVERSION")
            print(f"Hidden target found at rank {rank} on turn {turn}:")
            print(f"  {products[target].get('title')} [{target}]\n")
            return

        if turn == 10:
            break

        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(
                override.get("message", "Actually, please ignore my earlier preference.")
            )
        else:
            user_message, boundary_used = customer_reply(
                effective_sample,
                response.get("ask_attribute"),
                disclosed,
                boundary_used,
            )

        print()

    print("\nâœ— Target not found within 10 turns.")
    print(f"Hidden target was: {products[target].get('title')} [{target}]\n")


if __name__ == "__main__":
    main()