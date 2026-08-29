# ByteMe — Conversational Shopping Copilot

TechJam 2026, Problem 4. A multi-turn shopping agent that finds a hidden target
product in a frozen 50,000-item Amazon catalog within 10 turns, ranked as high
as possible.

**Technical score 0.911 on the 200 public sessions, up from the 0.107 official
baseline.** No LLM, no embeddings, no network, no vector database. Pure Python
standard library plus SQLite. The whole thing runs in 22 ms per turn on a
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
| **Shipped** | **0.985** | **0.822** | **2.41** | **0.911025** |

Per scenario, day-0 → shipped:

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 0.938 → 0.988 | 0.707 → 0.824 | 4.24 → 1.88 |
| browsing | 80 | 0.925 → 0.988 | 0.708 → 0.760 | 5.29 → 2.36 |
| intent_override | 30 | 0.900 → 0.967 | 0.789 → 0.942 | 5.53 → 3.87 |
| boundary | 10 | 0.600 → **1.000** | 0.425 → 0.950 | 8.10 → 2.70 |

Misses fell from 18 to 3 of 200. Every scenario is above 0.96 hit rate.

## Reproduce

```bash
git clone <this repo> && cd techjam-byteme-shopping-copilot
curl -L -o data/catalog.jsonl.gz \
  https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
gzip -dk data/catalog.jsonl.gz   # expect exactly 50000 lines
./verify_data.sh                 # verify against the organizer's SHA-256 sums
python3 -m evaluator.local_evaluator
```

No dependencies to install. Python 3.10+ standard library only.

`SHA256SUMS` is published as a **release asset**, not as a file in the
repository, so a plain `git clone` does not provide it. `verify_data.sh`
downloads it and checks the catalog; it exits non-zero on mismatch rather than
skipping quietly. Expected values:

```
catalog.jsonl.gz  07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8  (official)
catalog.jsonl     da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67  (ours, after gzip -d)
```

Every experiment is one command:

```bash
python3 -m src.eval.run --name my-experiment        # writes runs/my-experiment.json
python3 -m src.eval.run --name ablation --set TRUNCATE=false
python3 -m src.eval.compare runs/day0.json runs/best.json
python3 -m src.eval.paraphrase --level ablate       # robustness: which mechanism is fragile
python3 -m src.eval.natural_prompts --benchmark --control   # robustness: realistic phrasing
python3 -m unittest discover -s tests               # 10 contract + behaviour tests
python3 -m src.eval.natural_prompts --benchmark     # non-official natural wording diagnostic
```

The official benchmark uses a deterministic customer simulator.  For demo and
robustness work, `natural_prompts.py` restyles the same public-session
disclosures into ordinary customer language. Its result is a non-official
diagnostic, not a competition score.

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
| **Shipped** | **0.911025** | — |
| − popularity prior entirely | 0.864975 | −0.0461 |
| − dynamic truncation | 0.885358 | −0.0257 |
| − lexical overlap tier | 0.909750 | −0.0013 |
| − raw-transcript fallback | 0.911025 | 0.0000 * |
| popularity: global instead of per-query | 0.906275 | −0.0047 |
| override: full eviction instead of demotion | 0.898366 | −0.0127 |
| override: no demotion at all | 0.911400 | +0.0004 † |
| ask: candidate-aware selector | 0.882950 | −0.0281 |

\* The raw-transcript fallback only fires when template parsing recognises
nothing, which never happens on the clean public set. Its entire value is
invisible here and shows up only under the paraphrase stress test below. We
kept it on that evidence alone.

† Override demotion costs 0.0004. We ship it anyway: appending a requirement
the customer has explicitly revoked is the behaviour the brief names as the
weak agent, and it is not worth four ten-thousandths to exhibit it.

**Recall tuning is inert in this architecture — a negative result worth
recording.** FTS5 weight sets (four tried, including a teammate's tuned set)
all produce byte-identical scores, and so does pool size from 100 to 1600.
The generated SQL and the candidate pools genuinely differ (350/400 shared),
but the products that swap in and out are all irrelevant. Recall decides only
pool *membership*, and 400 is far past the point where the target is reliably
inside. The reranker decides everything that matters.

## The largest single gain: a popularity prior

Before building anything we checked whether the signal existed:

```
median rating_number    catalog 12      TARGETS 6,846
median target sits at the 99th percentile of catalog review count
```

Targets are overwhelmingly drawn from heavily-reviewed products, and the ranker
was ignoring the field entirely despite 100% coverage across all 50,000 items.
Adding it as a fourth signal is worth **+0.046**, the biggest change we made.

It combines the two signals the way the idea's author originally proposed:
min-max normalise text score and popularity **across the retrieved candidates**,
then blend. That is scale-free, so the mix stays balanced whatever the raw
numbers look like — and it beats normalising popularity once against the catalog
maximum by a further 0.005.

It is a **prior, not evidence**, and is weighted to only separate candidates the
text cannot. Verified, not assumed: three unrelated queries share zero products
in their top 10, results are not ordered by popularity, and a product matching a
requirement verbatim outranks a 10,000,000-review mismatch. Tests pin all of it.

**Known risk.** This exploits a property of how the public set was sampled. If
the private 800 targets were drawn the same way it transfers; if they were
sampled uniformly from the catalog it would hurt. `RANK_POPULARITY_BLEND = 0`
disables it in one line. We have asked the organizers.

## We validated our own tuning on held-out data

Swept parameters are chosen by keeping whichever value scores highest on the
public set — which is fitting on the test set. `src/eval/holdout.py` splits the
200 sessions by SHA-256 of `sample_id`, tunes on one half and reports the other.

It changed two decisions:

- **`RANK_PHRASE_EXACT` 40 → 20.** All three folds agreed, zero overfitting
  penalty. 40 had been tuned *before* the popularity prior existed; both signals
  separate candidates BM25 cannot, so keeping 40 double-counted the evidence.
- **`RANK_K1` left at 1.1.** The folds disagreed (1.25 / 0.8 / 1.1), so the value
  is not determined by the data. The whole range 0.8–1.4 spans ~0.006, so
  changing it would be moving noise around. Recorded rather than silently kept.

The blend weight that ships was chosen this way: all three folds agree on 0.27
with a zero penalty. The popularity *effect* is robust across folds with very
different scenario mixes; the third decimal of 0.911 is not. **Expect the
private 800 to score somewhat below it.**

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
| clean | 0.911025 | — |
| **none (control)** | 0.911025 | **100.0%** |
| clause reordering | 0.911025 | 100.0% |
| punctuation dropped | 0.911025 | 100.0% |
| synonym substitution | 0.911592 | 100.1% |
| filler words inserted | 0.860539 | 94.5% |
| **template rewording** | **0.402400** | **44.2%** |

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
  44.2% retained, at zero cost to the clean score.

We deliberately did **not** teach the parser to recognise our own paraphrases.
Tuning an agent against its own stress harness measures the harness, not
robustness. Both fixes above are paraphrase-agnostic by construction.

### A second, independent harness

`src/eval/natural_prompts.py` was written separately and takes a different
approach: it rewrites **whole customer sentences** into natural human phrasing
rather than substituting scaffolding fragments, so it measures realism where the
table above measures mechanism.

| Harness | Rewrites | Answers | Retained |
|---|---|---|---|
| `paraphrase.py` | fragments, composable | *which mechanism* is fragile | 44.2% |
| `natural_prompts.py` | whole sentences | *how bad* with a realistic customer | 59.3% |

Both carry a null control that reproduces 0.911025 exactly. Two independent
implementations agreeing that rewording costs roughly half the score is stronger
evidence than either number alone, and it is why we treat template dependence as
the headline limitation rather than an artefact of one person's test.

## A design decision that cost us 0.007

`customer_reply` contains a bypass: `attribute == "other"` skips the type check
and returns the next two undisclosed constraints regardless of type. Two turns
of `"other"` drains the entire brief.

We found it, measured it, and **chose not to ship it**:

| Ask policy | Score |
|---|---|
| Bare `"other"` loop | 0.913900 |
| **Shipped policy** | **0.911025** |

That is 0.0029 knowingly left on the table. It buys an agent that asks real
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
| Index build | 3.8 s, once per process |
| Latency per turn | 21.6 ms mean, 19.2 ms p50, 50.0 ms p95, 93.9 ms max |
| 200 sessions end to end | 10.4 s |
| Memory | ~725 MB resident for the index |
| **LLM tokens** | **0 prompt, 0 completion, 0 total** |
| **API cost** | **$0.00** |
| **Network calls** | **none** |

Nothing degrades if network access is disabled at scoring time, because nothing
reaches for the network in the first place.

## Limitations

Stated plainly, because we measured them.

1. **Template dependence is our largest exposure.** 44.2% of score retained when
   customer phrasing is rewritten. If the private harness paraphrases, expect a
   substantial drop. This is the first thing we would fix with more time.
2. **The popularity prior assumes the private set is sampled like the public
   one.** It is worth +0.046 and depends on targets being heavily-reviewed
   products. One line disables it if that assumption fails.
3. **Filler words still cost 5.5%.** Fixable only with words our own harness
   inserts, so left unfixed rather than manufacture a number.
4. **`browsing` is now the weakest subgroup** at 0.760 MRR across 80 sessions —
   the largest remaining pool of recoverable score.
5. **~725 MB resident.** Fine locally; the scoring environment's limit is
   unknown.
6. **Boundary is n = 10.** Directionally real, statistically thin.
7. **Tuned parameters carry residual overfitting risk.** Held-out validation
   bounds it but does not eliminate it; the effects are robust, the exact
   values are not.

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

| Member | Focus | Contributions |
|---|---|---|
| Tan Jie Yin Elicia | retrieval | Supported retrieval development and code review, including reviewing candidate retrieval behaviour and assisting with integration of the final retrieval pipeline. |
| Yap Dong Xuan, Ryan | dialogue | Developed the modular agent architecture, including conversational parsing, session state, candidate recall, reranking, adaptive questioning, dynamic truncation and paraphrase-robustness testing. |
| Xue Jingxian | eval | Led testing and results analysis by running the official evaluator, reviewing scenario-level performance, checking regressions and maintaining the experiment results. |
| Chu Ruoyuan | analysis | Coordinated the final submission, including the submission checklist, Devpost requirements, repository links, video link and deadline verification. |
| Goh Jun Hui | docs | Set up the shared GitHub workflow, reproduced the baseline, developed conversational memory and retrieval-weight experiments, tested popularity-aware reranking, independently verified the final agent and added agent contract and behaviour tests. |

## Data attribution

Amazon Reviews 2023, `Clothing_Shoes_and_Jewelry`. See
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). The catalog is not committed to this
repository.
