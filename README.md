# ByteMe — Conversational Shopping Copilot

TechJam 2026, Problem 4. A multi-turn shopping agent that finds a hidden target
product in a frozen 50,000-item Amazon catalog within 10 turns, ranked as high
as possible.

**Technical score 0.856 on the 200 public sessions, up from the 0.107 official
baseline.** No LLM, no embeddings, no network, no vector database. Pure Python
standard library plus SQLite. The whole thing runs in 24 ms per turn on a
laptop.

> The organizer's original challenge description is preserved at
> [docs/challenge_readme.md](docs/challenge_readme.md).

---

## Results

Measured with the **unmodified** official evaluator on all 200 public sessions.
Nothing under `evaluator/` was changed at any point.

| Agent | Hit@10 | MRR | MTTC | Technical score |
|---|---|---|---|---|
| Official BM25 starter | 0.125 | 0.068 | 9.81 | 0.10671 |
| Our day-0 prototype | 0.910 | 0.706 | 5.05 | 0.785761 |
| **Shipped** | **0.955** | **0.752** | **3.35** | **0.856159** |

Per scenario, day-0 → shipped:

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 0.938 → 0.938 | 0.707 → 0.751 | 4.24 → 3.14 |
| browsing | 80 | 0.925 → 0.975 | 0.708 → 0.718 | 5.29 → 3.23 |
| intent_override | 30 | 0.900 → 0.933 | 0.789 → 0.817 | 5.53 → 4.23 |
| boundary | 10 | 0.600 → **1.000** | 0.425 → 0.842 | 8.10 → 3.40 |

Misses fell from 18 to 9 of 200.

## Reproduce

```bash
git clone <this repo> && cd techjam-byteme-shopping-copilot
curl -L -o data/catalog.jsonl.gz \
  https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
gzip -dk data/catalog.jsonl.gz   # expect exactly 50000 lines
python3 -m evaluator.local_evaluator
```

No dependencies to install. Python 3.10+ standard library only.

Every experiment is one command:

```bash
python3 -m src.eval.run --name my-experiment        # writes runs/my-experiment.json
python3 -m src.eval.run --name ablation --set TRUNCATE=false
python3 -m src.eval.compare runs/day0.json runs/best.json
python3 -m src.eval.paraphrase --level ablate       # robustness stress test
```

`run.py` diffs against `runs/day0.json` and reports **per-session churn** —
which individual sessions were fixed, broken, improved or worsened. The
aggregate hides regressions; a change can add three hits in `buying` while
quietly breaking one in `boundary` and still look like a win.

## How it works

The single most important thing we learned by reading `local_evaluator.py`
end to end: **the customer is a deterministic template engine, not an LLM, and
the hidden brief is generated from the target product's own metadata.** At most
four short strings, lifted near-verbatim from one product's spec sheet. That
makes this an entity-resolution problem wearing a dialogue costume, which is
why an offline lexical approach reaches 0.95 hit rate with no model at all.

Each turn runs five stages:

| Stage | Module | Job |
|---|---|---|
| Parse | `src/dialogue/parser.py` | customer message → state |
| Recall | `src/retrieval/recall.py` | 50,000 → 400 candidates via SQLite FTS5 |
| Rank | `src/retrieval/scoring.py` | BM25 + phrase tiers + price proximity |
| Truncate | `src/dialogue/truncation.py` | how many results to actually expose |
| Ask | `src/dialogue/ask.py` | which attribute to elicit next |

**Ranking** is three additive signals, each independently ablatable: field-
weighted BM25 over accumulated constraints; a three-tier phrase bonus (exact
substring → 40-char prefix → proportional lexical overlap); and price proximity,
since the brief always states the target's own price.

**Truncation** is a scoring lever, not a UI choice. The session ends the instant
the target appears in the returned list, so returning all 10 while still
uninformed can end a session at rank 8 for 0.55 when waiting two turns would
have paid 0.92 at rank 1. We return a short list while uninformed — but always
the full list on the final turn, because a miss is worth exactly zero.

**Asking** is the single biggest lever in the whole competition. The official
starter scores 0.107 for one reason: it returns `ask_attribute: None` every
turn, so the customer refuses to disclose anything ten times.

## Component ablation

Each row strips exactly one component from the shipped agent.

| Configuration | Score | Δ |
|---|---|---|
| **Shipped** | **0.856159** | — |
| − dynamic truncation | 0.827439 | −0.0287 |
| − lexical overlap tier | 0.855581 | −0.0006 |
| − raw-transcript fallback | 0.856159 | 0.0000 * |
| fallback after 2 dead asks (vs 1) | 0.843435 | −0.0127 |
| fallback after 3 dead asks (vs 1) | 0.835935 | −0.0203 |

\* The raw-transcript fallback only fires when template parsing recognises
nothing, which never happens on the clean public set. Its entire value is
invisible here and shows up only under the paraphrase stress test below. We
kept it on that evidence alone.

**Recall tuning is inert in this architecture — a negative result worth
recording.** FTS5 weight sets (four tried, including a teammate's tuned set)
all produce byte-identical scores, and so does pool size from 100 to 1600.
The generated SQL and the candidate pools genuinely differ (350/400 shared),
but the products that swap in and out are all irrelevant. Recall decides only
pool *membership*, and 400 is far past the point where the target is reliably
inside. The reranker decides everything that matters.

## Robustness: we tried to break our own agent

The competition spec warns that *"if natural-language paraphrasing is added by
the organizer, it cannot decide correctness."* A solution that pattern-matches
the public evaluator's exact phrasing could score well here and collapse on the
private set.

`src/eval/paraphrase.py` measures that exposure. It wraps the **agent**, not the
evaluator — a proxy implementing the same interface rewrites each customer
message and delegates, so nothing under `evaluator/` is imported or modified.
Rewrites are seeded from a stable SHA-256 of `(seed, turn, message)`, because
the evaluator assigns a fresh uuid4 session id per run.

| Stress | Score | Retained |
|---|---|---|
| clean | 0.856159 | — |
| **none (control)** | 0.856159 | **100.0%** |
| clause reordering | 0.856159 | 100.0% |
| synonym substitution | 0.856159 | 100.0% |
| punctuation dropped | 0.856660 | 100.1% |
| filler words inserted | 0.745774 | 87.1% |
| **template rewording** | **0.370891** | **43.3%** |

The `none` control reproducing the clean score *exactly* is what licenses
reading the rest: the harness itself introduces no distortion.

**This disproved our own design assumption.** We expected the exact-phrase bonus
to be the fragile part and had planned a dense-embedding route to fix it.
Measured, synonym substitution costs **0.0%**. The fragility is entirely
template matching in the parser — reword only the scaffolding and the agent
scores *below the official baseline*. We would have spent days hardening the
wrong component.

Two fixes followed directly from that measurement:

- **Punctuation tolerance.** Markers are matched with the colon optional, since
  punctuation is incidental to the phrase. Eliminated a 26% loss outright.
- **Raw-transcript fallback.** When parsing recognises nothing, rank on the raw
  customer words rather than on catalog order. Worst case went from 9.2% to
  43.3% retained, at zero cost to the clean score.

We deliberately did **not** teach the parser to recognise our own paraphrases.
Tuning an agent against its own stress harness measures the harness, not
robustness. Both fixes above are paraphrase-agnostic by construction.

## A design decision that cost us 0.007

`customer_reply` contains a bypass: `attribute == "other"` skips the type check
and returns the next two undisclosed constraints regardless of type. Two turns
of `"other"` drains the entire brief.

We found it, measured it, and **chose not to ship it**:

| Ask policy | Score |
|---|---|
| Bare `"other"` loop | 0.862860 |
| **Shipped policy** | **0.856159** |

That is 0.0067 knowingly left on the table. It buys an agent that asks real
questions. Measured ask distribution: **55% specific attributes** (material 31%,
color 20%, budget 4%), 45% open fallback; every session asks at least one
specific attribute, mean 1.79 before falling back. The conversation reads
*"what material? → what colour? → anything else?"*, not *"tell me your other
preference"* twice.

The shipped policy rests on a provable fact rather than a guess.
`classify_constraint` is total and can only return `budget`, `material`,
`color`, `size`, `style`, `use_case` or `feature`. `brand` and `category` are
legal to ask but **no constraint can ever classify as either** — asking them is
a guaranteed wasted turn. Both are excluded. That is not an exploit; it is
declining to ask a question whose answer set is provably empty. We then fall
back to open-ended asking only once the customer has explicitly said the
specific space is exhausted. Threshold measured, not assumed: 1 dead ask
(0.8562) beat 2 (0.8434) and 3 (0.8359).

## Performance and cost

Measured on an Apple Silicon laptop, single core.

| | |
|---|---|
| Index build | 4.0 s, once per process |
| Latency per turn | 23.8 ms mean, 22.7 ms p50, 51.4 ms p95, 94.9 ms max |
| 200 sessions end to end | 15.8 s |
| Memory | ~718 MB resident for the index |
| **LLM tokens** | **0 prompt, 0 completion, 0 total** |
| **API cost** | **$0.00** |
| **Network calls** | **none** |

Nothing degrades if network access is disabled at scoring time, because nothing
reaches for the network in the first place.

## Limitations

Stated plainly, because we measured them.

1. **Template dependence is our largest exposure.** 43.3% of score retained when
   customer phrasing is rewritten. If the private harness paraphrases, we expect
   a substantial drop. This is the first thing we would fix with more time.
2. **Filler words still cost 13%.** Fixable by extending the stopword list, but
   only with words our own harness inserts, so we left it rather than
   manufacture a number.
3. **`buying` is stuck at 0.938 hit rate** across 80 sessions and did not move
   through any change we made. It is the largest subgroup and the least
   understood.
4. **Faster asking costs rank on some sessions.** Three sessions now hit at turn
   3 rank 4 where they previously hit at turn 8 rank 1 — a net loss under the
   scoring function. A confidence-gated truncation schedule is the proper fix
   and is not implemented.
5. **Memory is ~718 MB.** Fine on a laptop; we do not know the scoring
   environment's limit.
6. **The paraphrase numbers are our guesses** at what an organizer might do. The
   *ranking* of which components are fragile is solid; the absolute 43.3% is an
   indicator, not a prediction.
7. **Boundary is n=10.** 0.600 → 1.000 is four sessions. Directionally real,
   statistically thin.

## Repository layout

```
agent.py                     Agent class — required interface only, no logic
starter/agent.py             re-export shim (the evaluator hardcodes this path)
src/retrieval/
  text.py                    tokenisation, field flattening, field weights
  index.py                   FTS5 table + tf/idf/lengths/prices, built once
  recall.py                  50,000 -> 400 candidates
  scoring.py                 BM25 + three-tier phrase bonus + price proximity
src/dialogue/
  state.py                   per-session state
  parser.py                  customer message -> state
  ask.py                     elicitation policy
  truncation.py              per-turn list length
src/eval/
  run.py                     one command per experiment, diffs vs baseline
  compare.py                 diff any two runs, incl. per-session churn
  paraphrase.py              robustness stress harness
runs/day0.json               shared reference baseline
```

One owner per module, so five people can work without collisions. `evaluator/`
is never modified.

## How we worked

Every behavioural change was measured before it was kept, and the harness was
itself validated against known-good results before we trusted it — it
reproduces two independently documented configurations exactly (0.776 and
0.854). Runs record the git commit, a dirty-tree marker, timestamp, build time
and per-session latency, because a run whose code cannot be identified is not
evidence of anything.

Two defects in our own day-0 prototype were found this way, not by reading:
every one of the 30 `intent_override` sessions was raising `IndexError` and
being silently swallowed by the evaluator, and six sessions were being
truncated to five results on the final turn — all six of them misses.

## Team

<!-- TODO: fill in before submission. Required by the submission rules. -->

| Member | Focus | Contributions |
|---|---|---|
| _TBD_ | retrieval | |
| _TBD_ | dialogue | |
| _TBD_ | eval | |
| _TBD_ | analysis | |
| _TBD_ | docs | |

## Data attribution

Amazon Reviews 2023, `Clothing_Shoes_and_Jewelry`. See
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). The catalog is not committed to this
repository.
