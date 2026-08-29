"""Generate independent paraphrases of the evaluator's customer templates.

DEVELOPMENT TOOL. Run once, by hand, on a laptop. NOT part of the agent and
NOT on the scoring path. The agent never imports this file and never makes a
network call; it only ever reads the static JSON this produces.

Why this exists
---------------
src/eval/paraphrase.py originally used rewrites written by hand, by the same
person who then had to decide whether the parser should be hardened against
them. That is circular: hardening a parser against your own invented sentences
measures the harness, not the agent. It is why the hand-written rewrites were
deliberately left un-hardened, and why the reported 43.3% retention is flagged
in the docs as an indicator rather than a prediction.

Having a third party write the rewrites breaks the circularity. The output is
committed as a static asset, so scoring stays fully offline, deterministic and
reproducible -- the submission rules allow prototyping with any LLM API but
warn that network access may be disabled for official scoring.

Usage
-----
    export GEMINI_API_KEY=...          # never committed; read from the env
    python3 tools/generate_paraphrases.py --n 6
    python3 -m src.eval.paraphrase --level ablate --source generated

Costs a handful of requests, once. Rerun only to refresh the asset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "src" / "eval" / "paraphrases_generated.json"

MODEL = "gemini-2.0-flash"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# The scaffolding phrases the evaluator emits and that src/dialogue/parser.py
# keys off. These are what we need independent rewordings of.
TEMPLATES = {
    "I'm looking for": "opening a shopping request, followed by a product category",
    "A key requirement is:": "introducing the one thing the customer definitely needs",
    "but I'm still exploring": "signalling they are only browsing and have not decided",
    "For that, what matters is:": "introducing a list of things that matter to them",
    "Actually, ignore my earlier preference. What I need is:":
        "reversing a previously stated preference and replacing it",
    "I don't have a preference for": "declining to state a preference for an attribute",
    "I don't have an additional preference for":
        "saying they have nothing further to add about an attribute",
    "please use your judgment": "inviting the assistant to decide on their behalf",
    "Those options are not quite right yet. Ask me about one specific attribute.":
        "rejecting the shown options and asking to be questioned about something specific",
}

PROMPT = """You are helping stress-test a shopping assistant for robustness.

Below is a phrase a simulated customer says. Write {n} alternative ways a real \
person might express the SAME meaning in a shopping conversation.

Phrase: "{phrase}"
Meaning/context: {context}

Rules:
- Preserve the meaning and the grammatical role exactly. If the phrase \
introduces something that follows it (ends with a colon), your rewrites must \
also lead into what follows.
- Vary the wording genuinely: different verbs, different register, some casual, \
some formal, some terse.
- Do NOT include the thing that would follow the phrase. Only rewrite the phrase.
- No quotes, no numbering, no commentary.
- Output exactly {n} lines, one rewrite per line."""


def call_gemini(prompt: str, api_key: str, retries: int = 4) -> str:
    """POST to Gemini. Retries on rate limit with linear backoff."""
    url = ENDPOINT.format(model=MODEL) + f"?key={api_key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 1.0, "maxOutputTokens": 512},
    }).encode()

    for attempt in range(retries):
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read())
            return payload["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as error:
            # 429 = free-tier rate limit; back off rather than give up.
            if error.code in (429, 500, 503) and attempt < retries - 1:
                wait = 8 * (attempt + 1)
                print(f"    HTTP {error.code}, retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("exhausted retries")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paraphrase assets via Gemini")
    parser.add_argument("--n", type=int, default=6, help="rewrites per phrase")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--delay", type=float, default=4.0,
                        help="seconds between calls (free tier is rate limited)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("GEMINI_API_KEY is not set in the environment.\n"
              "  export GEMINI_API_KEY=...   (never commit it)", file=sys.stderr)
        return 1

    global MODEL
    MODEL = args.model

    generated: dict[str, list[str]] = {}
    for index, (phrase, context) in enumerate(TEMPLATES.items(), 1):
        print(f"[{index}/{len(TEMPLATES)}] {phrase!r}")
        try:
            raw = call_gemini(PROMPT.format(n=args.n, phrase=phrase, context=context), api_key)
        except Exception as error:                      # noqa: BLE001
            print(f"    FAILED: {error}", file=sys.stderr)
            continue

        lines = []
        for line in raw.splitlines():
            line = line.strip().strip('"').strip()
            # Drop numbering/bullets the model may add despite instructions.
            line = line.lstrip("0123456789.-) ").strip()
            if line and line.lower() != phrase.lower():
                lines.append(line)
        if lines:
            generated[phrase] = lines[:args.n]
            for line in generated[phrase]:
                print(f"    + {line}")
        time.sleep(args.delay)

    if not generated:
        print("nothing generated", file=sys.stderr)
        return 1

    OUT_PATH.write_text(json.dumps({
        "_meta": {
            "generator": "tools/generate_paraphrases.py",
            "model": MODEL,
            "note": "Development-time asset. The agent never calls an API; "
                    "src/eval/paraphrase.py reads this file. Scoring is offline.",
        },
        "rewrites": generated,
    }, indent=2) + "\n", encoding="utf-8")
    total = sum(len(v) for v in generated.values())
    print(f"\nwrote {OUT_PATH.relative_to(REPO_ROOT)}: "
          f"{len(generated)} phrases, {total} rewrites")
    return 0


if __name__ == "__main__":
    sys.exit(main())
