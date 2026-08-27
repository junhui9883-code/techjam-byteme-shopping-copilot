"""One command per experiment.

Owner: eval

    python3 -m src.eval.run --name my-experiment

Runs the OFFICIAL evaluator unmodified against the current agent, writes the
full metric block to runs/<name>.json, and prints a diff against the baseline
(runs/day0.json) showing the delta per scenario plus which individual sessions
were fixed or broken.

CLAUDE.md section 8 is the rule this exists to enforce: every behavioural
change is measured before it is kept, and no unmeasured change is merged.

Ablations (CLAUDE.md backlog item 9) are one command too -- --set overrides any
Agent class attribute for the run:

    python3 -m src.eval.run --name ablation-no-truncation --set TRUNCATE=false
    python3 -m src.eval.run --name ablation-other-policy  --set ASK_POLICY=other

Gate a change in CI with --gate, which exits non-zero if the score regresses:

    python3 -m src.eval.run --name candidate --gate
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

from agent import Agent

from .compare import format_diff, load_run

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
DEFAULT_BASELINE = RUNS_DIR / "day0.json"
DEFAULT_CATALOG = "data/catalog.jsonl"
DEFAULT_DATASET = "data/public_set.jsonl"


def _coerce(raw: str) -> object:
    """Parse a --set value into bool / int / float / str, in that order."""
    low = raw.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def _git_commit() -> str | None:
    """Record which commit produced a run. Reproducibility is a stated
    requirement (CLAUDE.md section 8); a run whose code we cannot identify is
    not evidence of anything."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        # A dirty tree means the run is not reproducible from the commit alone.
        return commit + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def build_agent(overrides: dict[str, object], catalog: str) -> Agent:
    agent = Agent(catalog)
    # Instance attributes shadow the class attributes, so this scopes the
    # override to this run only and never mutates the shipped defaults.
    for key, value in overrides.items():
        setattr(agent, key, value)
    return agent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one measured experiment")
    parser.add_argument("--name", required=True, help="experiment name -> runs/<name>.json")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="run to diff against")
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override an Agent class attribute")
    parser.add_argument("--note", default="", help="free text recorded in the run file")
    parser.add_argument("--no-write", action="store_true", help="run and diff without saving")
    parser.add_argument("--gate", action="store_true",
                        help="exit 1 if the technical score regresses against the baseline")
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    for item in args.overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        key = key.strip()
        # Validate now: loading the catalog costs ~4s and it is maddening to
        # wait for it only to be told a flag was misspelled.
        if not hasattr(Agent, key):
            known = ", ".join(sorted(
                k for k in vars(Agent) if k.isupper() and not k.startswith("_")))
            raise SystemExit(f"--set {key}: Agent has no attribute {key!r} (known: {known})")
        overrides[key] = _coerce(value)

    print(f"[run] experiment : {args.name}")
    if overrides:
        print(f"[run] overrides  : {overrides}")

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)

    t0 = time.perf_counter()
    agent = build_agent(overrides, args.catalog)
    build_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = evaluate(agent, samples, catalog_ids, categories, products)
    eval_seconds = time.perf_counter() - t1

    # Metadata block. Latency and token cost are required disclosures for the
    # writeup (CLAUDE.md section 10 / backlog item 10).
    result["run_metadata"] = {
        "name": args.name,
        "note": args.note,
        "overrides": {k: v for k, v in overrides.items()},
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "index_build_seconds": round(build_seconds, 3),
        "eval_seconds": round(eval_seconds, 3),
        "seconds_per_session": round(eval_seconds / max(len(samples), 1), 4),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    print(f"[run] index built in {build_seconds:.1f}s, "
          f"{len(samples)} sessions evaluated in {eval_seconds:.1f}s "
          f"({eval_seconds / max(len(samples), 1) * 1000:.0f} ms/session)")

    if not args.no_write:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RUNS_DIR / f"{args.name}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[run] wrote {out_path.relative_to(REPO_ROOT)}")

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"[run] baseline {baseline_path} not found — skipping diff")
        print(f"[run] technical_score = {result['recommended_technical_score']}")
        return 0

    baseline = load_run(baseline_path)
    print(format_diff(baseline, result, baseline_path.stem, args.name))

    delta = result["recommended_technical_score"] - baseline["recommended_technical_score"]
    print()
    if abs(delta) < 5e-7:
        print(f"[run] VERDICT: no change vs {baseline_path.stem} "
              f"({result['recommended_technical_score']:.6f})")
    elif delta > 0:
        print(f"[run] VERDICT: IMPROVED {delta:+.6f} vs {baseline_path.stem} "
              f"-> {result['recommended_technical_score']:.6f}")
    else:
        print(f"[run] VERDICT: REGRESSED {delta:+.6f} vs {baseline_path.stem} "
              f"-> {result['recommended_technical_score']:.6f}")

    if args.gate and delta < -5e-7:
        print("[run] --gate: regression, exiting 1")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
