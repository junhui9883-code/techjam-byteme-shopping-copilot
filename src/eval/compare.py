"""Diff two evaluation runs.

Owner: eval

The aggregate score hides regressions: a change can add three hits in `buying`
while quietly breaking one in `boundary` and still look like a win. Every diff
here is therefore reported at three levels:

  1. Overall metrics with deltas.
  2. Per-scenario metrics with deltas (the subgroup view CLAUDE.md tracks).
  3. Session churn -- which individual sample_ids were FIXED and which were
     BROKEN. This is the one that catches silent regressions.

Usable standalone:
    python3 -m src.eval.compare runs/day0.json runs/my-experiment.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Overall metrics, in report order, with the direction that counts as better.
# mttc is the only one where lower is better -- but note that per CLAUDE.md
# section 1 a worse MTTC is an acceptable price for better rank, so it is
# reported without judgement.
OVERALL_METRICS = [
    ("recommended_technical_score", "score", "up"),
    ("hit_rate_at_10", "hit@10", "up"),
    ("mrr", "mrr", "up"),
    ("mttc", "mttc", "down"),
    ("efficiency", "efficiency", "up"),
]
SCENARIO_METRICS = [
    ("hit_rate_at_10", "hit@10", "up"),
    ("mrr", "mrr", "up"),
    ("mttc", "mttc", "down"),
]

# Deltas below this are formatted as a flat "=" so rounding noise does not read
# as signal.
EPSILON = 5e-7


def load_run(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sessions_by_id(run: dict) -> dict[str, dict]:
    return {s["sample_id"]: s for s in run.get("sessions", [])}


def session_churn(baseline: dict, candidate: dict) -> dict[str, list[str]]:
    """Classify every session that changed outcome between two runs.

    fixed    : missed in baseline, hit in candidate
    broken   : hit in baseline, missed in candidate   <- the important one
    improved : hit in both, but ranked better
    worsened : hit in both, but ranked worse
    """
    base, cand = _sessions_by_id(baseline), _sessions_by_id(candidate)
    out: dict[str, list[str]] = {"fixed": [], "broken": [], "improved": [], "worsened": []}
    for sid, b in base.items():
        c = cand.get(sid)
        if c is None:
            continue
        if not b["hit"] and c["hit"]:
            out["fixed"].append(sid)
        elif b["hit"] and not c["hit"]:
            out["broken"].append(sid)
        elif b["hit"] and c["hit"]:
            # Higher reciprocal_rank == better rank.
            if c["reciprocal_rank"] > b["reciprocal_rank"] + EPSILON:
                out["improved"].append(sid)
            elif c["reciprocal_rank"] < b["reciprocal_rank"] - EPSILON:
                out["worsened"].append(sid)
    return out


def _fmt_delta(delta: float | None, direction: str) -> str:
    if delta is None:
        return "     n/a"
    if abs(delta) < EPSILON:
        return "       ="
    better = (delta > 0) if direction == "up" else (delta < 0)
    return f"{delta:+8.4f}{'' if better else ' !'}"


def _get(block: dict, key: str) -> float | None:
    value = block.get(key)
    return None if value is None else float(value)


def format_diff(baseline: dict, candidate: dict,
                baseline_label: str, candidate_label: str) -> str:
    lines: list[str] = []
    w = 32
    # Run names can be long; clip so the columns never collide.
    b_label = baseline_label[:13]
    c_label = candidate_label[:13]

    lines.append("")
    lines.append(f"{'':{w}}{b_label:>14}{c_label:>14}{'delta':>12}")
    lines.append("-" * (w + 40))

    for key, label, direction in OVERALL_METRICS:
        b, c = _get(baseline, key), _get(candidate, key)
        delta = None if (b is None or c is None) else c - b
        bs = "n/a" if b is None else f"{b:.4f}"
        cs = "n/a" if c is None else f"{c:.4f}"
        lines.append(f"{label:{w}}{bs:>14}{cs:>14}{_fmt_delta(delta, direction):>12}")

    lines.append("")
    lines.append("per scenario")
    lines.append("-" * (w + 40))
    b_scen = baseline.get("scenario_metrics", {})
    c_scen = candidate.get("scenario_metrics", {})
    for name in sorted(set(b_scen) | set(c_scen)):
        b_block, c_block = b_scen.get(name, {}), c_scen.get(name, {})
        n = b_block.get("sample_count", c_block.get("sample_count", "?"))
        lines.append(f"  {name}  (n={n})")
        for key, label, direction in SCENARIO_METRICS:
            b, c = _get(b_block, key), _get(c_block, key)
            delta = None if (b is None or c is None) else c - b
            bs = "n/a" if b is None else f"{b:.4f}"
            cs = "n/a" if c is None else f"{c:.4f}"
            lines.append(f"{'    ' + label:{w}}{bs:>14}{cs:>14}{_fmt_delta(delta, direction):>12}")

    churn = session_churn(baseline, candidate)
    lines.append("")
    lines.append("session churn")
    lines.append("-" * (w + 40))
    if not any(churn.values()):
        lines.append("  no session changed outcome — behaviour is identical")
    else:
        for kind in ("fixed", "broken", "improved", "worsened"):
            ids = churn[kind]
            if not ids:
                continue
            flag = "  <-- REGRESSION" if kind in ("broken", "worsened") else ""
            shown = ", ".join(ids[:8]) + (f" (+{len(ids) - 8} more)" if len(ids) > 8 else "")
            lines.append(f"  {kind:9s} {len(ids):3d}{flag}")
            lines.append(f"            {shown}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two evaluation runs")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    baseline, candidate = load_run(args.baseline), load_run(args.candidate)
    print(format_diff(baseline, candidate,
                      Path(args.baseline).stem, Path(args.candidate).stem))


if __name__ == "__main__":
    main()
