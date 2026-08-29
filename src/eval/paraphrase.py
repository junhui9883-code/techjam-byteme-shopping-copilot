"""Paraphrase stress harness -- insurance against the risk that can zero us.

Owner: eval

CLAUDE.md section 5, Risk 1: the spec says "If natural-language paraphrasing is
added by the organizer, it cannot decide correctness." Our parser matches the
public evaluator's templates literally (see the CAUTION in
src/dialogue/parser.py) and our ranker pays a large bonus for exact constraint
phrases. Both could score 0.855 on the public set and collapse on a private set
whose customer turns are worded differently.

This module measures that exposure instead of guessing at it.

HOW IT AVOIDS TOUCHING evaluator/
---------------------------------
The evaluator drives the loop and calls `agent.respond(session_id, message,
turn, top_k)`. We wrap the AGENT, not the evaluator: ParaphrasingAgent rewrites
`message` and delegates. Nothing under evaluator/ is imported, subclassed or
modified, and the customer simulator, scoring and ground truth are untouched.

    evaluator --(template message)--> ParaphrasingAgent --(reworded)--> Agent

TWO STRESS LEVELS, MEASURED SEPARATELY
--------------------------------------
Reporting one blended number would confuse two different failures, so:

  scaffold : reword only the TEMPLATE SCAFFOLDING -- "A key requirement is:",
             "For that, what matters is:", and so on. The constraint text
             itself is left byte-identical. This isolates parser brittleness:
             can we still extract the constraint when the wrapper phrasing
             changes?

  full     : scaffold rewriting PLUS synonym substitution inside the constraint
             text. This additionally stresses the ranker, since the exact
             phrase bonus in src/retrieval/scoring.py stops firing when the
             words change.

`full` is the pessimistic bound: a real organizer paraphrase would be milder,
because rewording a spec-sheet phrase like "100% cotton" changes its meaning.

DETERMINISM
-----------
The evaluator assigns a fresh uuid4 session_id per run, so seeding on it would
make results irreproducible. The RNG is seeded from a stable SHA-256 of
(seed, turn, message) instead, so the same message at the same turn is always
reworded identically, across processes and machines.

USAGE
-----
    python3 -m src.eval.paraphrase                  # clean vs both levels
    python3 -m src.eval.paraphrase --level scaffold # one level
    python3 -m src.eval.paraphrase --save           # write runs/*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

from agent import Agent

from .compare import format_diff

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"

# --------------------------------------------------------------------------
# Scaffolding rewrites. Each key is a phrase the organizer's templates emit
# and that src/dialogue/parser.py keys off; the values are paraphrases a human
# (or an LLM rewriter) might plausibly produce. Longest keys are applied first
# so that overlapping phrases do not partially match.
# --------------------------------------------------------------------------
SCAFFOLD_REWRITES: dict[str, tuple[str, ...]] = {
    "Actually, ignore my earlier preference. What I need is:": (
        "Scratch that, what I actually want is:",
        "Forget what I said before — I really need:",
        "On second thought, ignore that. I'm after:",
    ),
    "I don't have a preference for": (
        "I really don't mind about",
        "No strong feelings on",
        "I'm easy either way on",
    ),
    "I don't have an additional preference for": (
        "Nothing else comes to mind for",
        "That's all I've got on",
        "No other thoughts about",
    ),
    "please use your judgment": (
        "you decide",
        "whatever you think is best",
        "I'll trust your call",
    ),
    "For that, what matters is:": (
        "The things that matter are:",
        "What's important to me:",
        "Mainly I care about:",
    ),
    "A key requirement is:": (
        "One thing it must have:",
        "It really needs:",
        "Important to me:",
    ),
    "but I'm still exploring": (
        "though I'm just browsing for now",
        "but I haven't made up my mind",
        "still figuring out what I want though",
    ),
    "I'm looking for": (
        "I need",
        "I'm after",
        "I want to find",
        "Trying to track down",
    ),
    "Those options are not quite right yet. Ask me about one specific attribute.": (
        "Those aren't quite it. Ask me about something specific.",
        "Not really what I had in mind — ask about one particular thing.",
    ),
}

# Content-word synonyms, used only at level="full". Deliberately conservative:
# these are near-equivalents a rewriter might pick, not meaning changes.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "lightweight": ("light", "not heavy"), "durable": ("long-lasting", "hard-wearing"),
    "comfortable": ("comfy", "easy to wear"), "waterproof": ("water-resistant", "water-repellent"),
    "breathable": ("airy", "well-ventilated"), "adjustable": ("adaptable", "customisable"),
    "soft": ("smooth", "gentle"), "warm": ("cosy", "insulating"),
    "sturdy": ("solid", "robust"), "casual": ("relaxed", "everyday"),
    "jacket": ("coat",), "sneakers": ("trainers",), "trousers": ("pants",),
    "budget": ("price range", "spend"), "around": ("about", "roughly", "approximately"),
    "large": ("big",), "small": ("compact",),
}

FILLERS = ("um,", "like,", "honestly,", "I guess", "sort of", "you know,", "basically,")

# ---------------------------------------------------------------------------
# Independently generated rewrites (optional asset).
#
# SCAFFOLD_REWRITES above were written by hand, by the same author who then had
# to decide whether to harden the parser against them -- which is circular, and
# is why they were deliberately left un-hardened. tools/generate_paraphrases.py
# produces an independent set via an external model at DEVELOPMENT TIME and
# commits it as static JSON. Nothing here makes a network call; scoring stays
# offline and deterministic. Use --source generated to switch.
# ---------------------------------------------------------------------------
GENERATED_PATH = Path(__file__).with_name("paraphrases_generated.json")


def load_rewrites(source: str = "handwritten") -> dict[str, tuple[str, ...]]:
    """Return the scaffold rewrite table for the requested source."""
    if source == "handwritten":
        return SCAFFOLD_REWRITES
    if not GENERATED_PATH.exists():
        raise SystemExit(
            f"--source {source} needs {GENERATED_PATH.name}, which is not present.\n"
            "  Generate it once:  export GEMINI_API_KEY=...\n"
            "                     python3 tools/generate_paraphrases.py")
    data = json.loads(GENERATED_PATH.read_text(encoding="utf-8"))
    generated = {k: tuple(v) for k, v in data["rewrites"].items() if v}
    if source == "generated":
        return generated
    if source == "both":
        merged = {k: tuple(v) for k, v in SCAFFOLD_REWRITES.items()}
        for phrase, options in generated.items():
            merged[phrase] = tuple(dict.fromkeys(merged.get(phrase, ()) + options))
        return merged
    raise SystemExit(f"unknown --source {source!r}")

# Probability of applying each optional transformation, per message.
P_FILLER = 0.5
P_DROP_PUNCT = 0.4
P_REORDER = 0.6
P_SYNONYM = 0.7

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def _rng_for(seed: int, turn: int, message: str) -> random.Random:
    """Stable RNG. Python's hash() is salted per process, so use SHA-256."""
    digest = hashlib.sha256(f"{seed}\0{turn}\0{message}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _rewrite_scaffolding(text: str, rng: random.Random,
                         table: dict[str, tuple[str, ...]] | None = None) -> str:
    table = SCAFFOLD_REWRITES if table is None else table
    for phrase in sorted(table, key=len, reverse=True):
        if phrase in text:
            text = text.replace(phrase, rng.choice(table[phrase]), 1)
    return text


def _swap_synonyms(text: str, rng: random.Random) -> str:
    def replace(match: re.Match) -> str:
        word = match.group(0)
        options = SYNONYMS.get(word.lower())
        if not options or rng.random() > P_SYNONYM:
            return word
        choice = rng.choice(options)
        return choice.capitalize() if word[0].isupper() else choice
    return _WORD_RE.sub(replace, text)


def _reorder_clauses(text: str, rng: random.Random) -> str:
    """Swap the order of semicolon-joined clauses.

    The evaluator joins multiple disclosed constraints with "; " (line 185), so
    this is the one reordering that actually occurs in the data. Order carries
    no meaning there, but our parser and ranker both consume constraints in
    disclosure order, so it is worth stressing.
    """
    if ";" not in text or rng.random() > P_REORDER:
        return text
    head, _, tail = text.partition(":")
    if not tail:
        return text
    trailing = "." if tail.rstrip().endswith(".") else ""
    parts = [p.strip().rstrip(".") for p in tail.split(";") if p.strip()]
    if len(parts) < 2:
        return text
    rng.shuffle(parts)
    return f"{head}: " + "; ".join(parts) + trailing


def _insert_filler(text: str, rng: random.Random) -> str:
    if rng.random() > P_FILLER:
        return text
    words = text.split()
    if len(words) < 4:
        return text
    position = rng.randrange(1, len(words))
    words.insert(position, rng.choice(FILLERS))
    return " ".join(words)


def _drop_punctuation(text: str, rng: random.Random) -> str:
    """Drop terminal periods and colons, the way hurried typing does.

    Semicolons are preserved: they separate distinct constraints and dropping
    them would merge two facts into one, which is corruption rather than
    paraphrase.
    """
    if rng.random() > P_DROP_PUNCT:
        return text
    text = text.replace(":", "")
    return text[:-1] if text.endswith(".") else text


# Named transform bundles. "none" is the control: it must reproduce the clean
# score exactly, proving the wrapper itself introduces no distortion.
LEVELS: dict[str, frozenset[str]] = {
    "none":     frozenset(),
    "scaffold": frozenset({"scaffold", "reorder", "filler", "punct"}),
    "full":     frozenset({"scaffold", "reorder", "filler", "punct", "synonym"}),
    # Single-transform ablations, to locate WHERE the brittleness lives.
    "only-scaffold": frozenset({"scaffold"}),
    "only-reorder":  frozenset({"reorder"}),
    "only-filler":   frozenset({"filler"}),
    "only-punct":    frozenset({"punct"}),
    "only-synonym":  frozenset({"synonym"}),
}


def paraphrase(message: str, turn: int, level: str = "scaffold", seed: int = 0,
               table: dict[str, tuple[str, ...]] | None = None) -> str:
    """Reword one customer message. Never raises; returns the input on error."""
    if not message or not message.strip():
        return message
    transforms = LEVELS.get(level, LEVELS["scaffold"])
    if not transforms:
        return message
    try:
        rng = _rng_for(seed, turn, message)
        text = message
        if "scaffold" in transforms:
            text = _rewrite_scaffolding(text, rng, table)
        if "reorder" in transforms:
            text = _reorder_clauses(text, rng)
        if "synonym" in transforms:
            text = _swap_synonyms(text, rng)
        if "filler" in transforms:
            text = _insert_filler(text, rng)
        if "punct" in transforms:
            text = _drop_punctuation(text, rng)
        return text
    except Exception:
        return message


class ParaphrasingAgent:
    """Transparent proxy that rewords customer messages before the agent sees
    them. Implements the same interface, so the evaluator cannot tell the
    difference and needs no modification."""

    def __init__(self, inner, level: str = "scaffold", seed: int = 0,
                 table: dict[str, tuple[str, ...]] | None = None) -> None:
        self.inner = inner
        self.level = level
        self.seed = seed
        self.table = table
        self.samples: list[tuple[str, str]] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.inner.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        stressed = paraphrase(user_message, turn, self.level, self.seed, self.table)
        if len(self.samples) < 12 and stressed != user_message:
            self.samples.append((user_message, stressed))
        return self.inner.respond(session_id, stressed, turn, top_k)


def run_level(level: str | None, catalog: str, dataset: str, seed: int,
              table: dict[str, tuple[str, ...]] | None = None
              ) -> tuple[dict, list[tuple[str, str]]]:
    samples = load_jsonl(dataset)
    catalog_ids, categories, products = catalog_index(catalog)
    agent = Agent(catalog)
    wrapper = None
    if level is not None:
        wrapper = ParaphrasingAgent(agent, level=level, seed=seed, table=table)
        agent = wrapper
    result = evaluate(agent, samples, catalog_ids, categories, products)
    return result, (wrapper.samples if wrapper else [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Paraphrase robustness stress test")
    parser.add_argument("--level", default="both",
                        help="none|scaffold|full|both|only-<transform>, or 'ablate' for all")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", choices=("handwritten", "generated", "both"),
                        default="handwritten",
                        help="which scaffold rewrite table to stress with")
    parser.add_argument("--save", action="store_true", help="write runs/paraphrase-<level>.json")
    parser.add_argument("--show-samples", type=int, default=6,
                        help="how many before/after message pairs to print")
    args = parser.parse_args()

    if args.level == "both":
        levels = ["scaffold", "full"]
    elif args.level == "ablate":
        levels = ["none", "only-filler", "only-punct", "only-reorder",
                  "only-synonym", "only-scaffold", "scaffold", "full"]
    else:
        levels = [args.level]

    table = load_rewrites(args.source)
    print(f"[paraphrase] rewrite source: {args.source} "
          f"({sum(len(v) for v in table.values())} rewrites over {len(table)} phrases)")
    print("[paraphrase] clean baseline ...")
    t = time.perf_counter()
    clean, _ = run_level(None, args.catalog, args.dataset, args.seed)
    print(f"[paraphrase] clean {clean['recommended_technical_score']:.6f} "
          f"({time.perf_counter() - t:.0f}s)")

    results: dict[str, dict] = {}
    for level in levels:
        print(f"[paraphrase] level={level} ...")
        t = time.perf_counter()
        stressed, samples = run_level(level, args.catalog, args.dataset, args.seed, table)
        results[level] = stressed
        print(f"[paraphrase] {level} {stressed['recommended_technical_score']:.6f} "
              f"({time.perf_counter() - t:.0f}s)")
        if args.show_samples and samples:
            print(f"\n  example rewrites ({level}):")
            for before, after in samples[:args.show_samples]:
                print(f"    - {before}")
                print(f"    + {after}")
        print(format_diff(clean, stressed, "clean", level))
        if args.save:
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            (RUNS_DIR / f"paraphrase-{level}.json").write_text(
                json.dumps(stressed, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 72)
    print("PARAPHRASE ROBUSTNESS SUMMARY")
    print("=" * 72)
    base = clean["recommended_technical_score"]
    print(f"  {'clean':<26}{base:>10.6f}{'':>12}")
    for level, result in results.items():
        score = result["recommended_technical_score"]
        retained = score / base * 100 if base else 0.0
        print(f"  {level:<26}{score:>10.6f}{retained:>11.1f}% retained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
