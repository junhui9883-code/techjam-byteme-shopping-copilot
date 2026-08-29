"""Held-out validation for tuned parameters.

Owner: eval

Every weight we ship was chosen by sweeping against all 200 public sessions and
keeping the maximum. That is fitting on the test set: the winning value is
partly chosen for noise it happens to exploit, so the reported number is
optimistic and will not fully survive the private 800.

This splits the public set in two, tunes on one half and reports the score on
the other half only. The gap between "best on the tuning half" and "same value
scored on the held-out half" is a direct estimate of how much of a gain is real
rather than fitted.

The split is deterministic (by SHA-256 of sample_id, not by position or by an
unseeded shuffle) so the same halves come back on every machine and run.

    python3 -m src.eval.holdout --param RANK_POPULARITY --values 0,10,20,25,28,32,40
"""

from __future__ import annotations

import argparse
import hashlib
import sys

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

from agent import Agent


def split(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic halves. Position-based splits can align with dataset
    ordering (the public set is grouped by scenario), which would make the two
    halves systematically different rather than merely disjoint."""
    a, b = [], []
    for sample in samples:
        digest = hashlib.sha256(str(sample["sample_id"]).encode()).digest()
        (a if digest[0] % 2 == 0 else b).append(sample)
    return a, b


def main() -> int:
    parser = argparse.ArgumentParser(description="Held-out validation of a swept parameter")
    parser.add_argument("--param", required=True, help="Agent attribute to sweep")
    parser.add_argument("--values", required=True, help="comma-separated values")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    if not hasattr(Agent, args.param):
        raise SystemExit(f"Agent has no attribute {args.param!r}")
    values = [float(v) for v in args.values.split(",")]

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    fold_a, fold_b = split(samples)
    print(f"[holdout] fold A: {len(fold_a)} sessions   fold B: {len(fold_b)} sessions")
    for name, fold in (("A", fold_a), ("B", fold_b)):
        counts: dict[str, int] = {}
        for s in fold:
            counts[s["scenario_type"]] = counts.get(s["scenario_type"], 0) + 1
        print(f"           fold {name} mix: {counts}")

    # One index build, reused for every value: the parameter is an instance
    # attribute, so nothing about the index depends on it.
    agent = Agent(args.catalog)

    print(f"\n{args.param:>22}{'fold A':>12}{'fold B':>12}{'full':>12}")
    print("-" * 58)
    results: dict[float, tuple[float, float, float]] = {}
    for value in values:
        setattr(agent, args.param, value)
        scores = []
        for fold in (fold_a, fold_b, samples):
            result = evaluate(agent, fold, catalog_ids, categories, products)
            scores.append(result["recommended_technical_score"])
        results[value] = tuple(scores)
        print(f"{value:>22.1f}{scores[0]:>12.6f}{scores[1]:>12.6f}{scores[2]:>12.6f}")

    best_a = max(results, key=lambda v: results[v][0])
    best_b = max(results, key=lambda v: results[v][1])
    best_full = max(results, key=lambda v: results[v][2])

    print("\n[holdout] verdict")
    print(f"  best on fold A            : {best_a:g}  (A={results[best_a][0]:.6f})")
    print(f"  that value scored on B    : {results[best_a][1]:.6f}")
    print(f"  best achievable on B      : {results[best_b][1]:.6f}  (at {best_b:g})")
    penalty = results[best_b][1] - results[best_a][1]
    print(f"  overfitting penalty       : {penalty:.6f}"
          f"   {'(negligible)' if penalty < 0.005 else '(SIGNIFICANT)'}")
    print(f"  best on the full set      : {best_full:g}")
    agree = "AGREE" if best_a == best_b == best_full else "DISAGREE"
    print(f"  A / B / full choices      : {best_a:g} / {best_b:g} / {best_full:g}  -> {agree}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
