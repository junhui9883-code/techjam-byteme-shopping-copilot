# How we work on this repo

Short version: **branch from `main`, put your code in `src/`, measure before you
merge.**

## Where to branch from

```bash
git checkout main && git pull        # ALWAYS start here
git checkout -b <area>/<what>        # e.g. retrieval/popularity-prior
```

`main` is the integration branch and is always the current best agent. Do not
branch from another person's feature branch unless you are deliberately building
on unmerged work — and if you do, say so in the branch name.

**Never work directly on `main`.** Open a branch, measure, then merge.

## Where the code lives — this is the important bit

`starter/agent.py` is a **three-line re-export shim**. It exists only because
`evaluator/local_evaluator.py` hardcodes `from starter.agent import Agent`, and
we are forbidden from modifying anything under `evaluator/`.

**There is no agent logic in `starter/agent.py`. Editing it does nothing.**
Three separate commits were lost this way. If you are about to change retrieval
or dialogue behaviour, the file you want is one of these:

| You want to change | Edit |
|---|---|
| how candidates are found | `src/retrieval/recall.py` |
| how candidates are ranked | `src/retrieval/scoring.py` |
| the index / field weights | `src/retrieval/index.py` |
| tokenisation, stopwords | `src/retrieval/text.py` |
| how customer messages are read | `src/dialogue/parser.py` |
| what the agent asks next | `src/dialogue/ask.py` |
| how many results to return | `src/dialogue/truncation.py` |
| session state | `src/dialogue/state.py` |
| the required interface only | `agent.py` |

`evaluator/` is **off limits**. Never edit it, never import from it in agent code.

## Measure before you merge

Every behavioural change is measured. No exceptions, no "it obviously helps".

```bash
python3 -m src.eval.run --name my-change
```

That writes `runs/my-change.json` and diffs against `runs/day0.json`, reporting
overall, per-scenario, **and per-session churn** — which individual sessions were
fixed, broken, improved or worsened. Read the churn. The aggregate hides
regressions: a change can add three hits in `buying` while quietly breaking one
in `boundary` and still look like a win.

If your change is a tunable, expose it as an `Agent` class attribute and sweep
it instead of editing code repeatedly:

```bash
python3 -m src.eval.run --name sweep --set TRUNCATE=false --no-write
```

Paste the `VERDICT` line into your PR. A change that does not beat the current
`main` score does not merge.

### Things that have already been measured — don't redo them

- **FTS5 recall weights are inert.** Four weight sets, all byte-identical
  scores. Pool size 100→1600, also inert. Recall only decides pool
  *membership*; the reranker decides everything. Tune `scoring.py`, not
  `recall.py`.
- **The `ask_attribute="other"` exploit scores 0.862860** vs our shipped
  0.856159. We are not shipping it — see the README. Don't "discover" it again.

## Tests

```bash
python3 -m unittest discover -s tests
```

`unittest`, not `pytest` — `pytest` is not a dependency and is not installed on
a clean checkout. Keep it that way; the submission must run on the standard
library alone.

## Robustness harnesses — we have two, on purpose

They answer different questions and are both worth running.

| Harness | Rewrites | Answers | Current |
|---|---|---|---|
| `src/eval/paraphrase.py` | fragments, composable transforms | *which mechanism* is fragile | 43.3% retained |
| `src/eval/natural_prompts.py` | whole sentences, natural phrasing | *how bad* with a realistic customer | 54.3% retained |

```bash
python3 -m src.eval.paraphrase --level ablate
python3 -m src.eval.natural_prompts --benchmark --control
```

Both were written independently and both show substantial degradation under
rewording, which is stronger evidence than either alone.

**Both have a null control, and you must run it.** The control puts the wrapper
in the call path but rewrites nothing; it has to reproduce the official score
exactly. Without it, a degraded number is ambiguous — it could mean the agent is
brittle, or merely that the harness itself perturbs the run.

**Do not tune the parser against these harnesses' own rewrite lists.** We wrote
those sentences; teaching the agent to recognise them measures the harness, not
robustness. Fixes must be paraphrase-agnostic (punctuation tolerance and the
raw-transcript fallback both are).

## Merging

1. Rebase or merge `main` into your branch and re-run the evaluator.
2. Confirm the score and paste it in the PR.
3. Run the tests.
4. Merge to `main`.

If two people touch the same module, the later one merges `main` first and
re-measures. The module split exists so this is rare — one owner per directory.
