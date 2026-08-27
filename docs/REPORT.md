# Technical Report — ByteMe Conversational Shopping Copilot

TechJam 2026, Problem 4. Covers architecture, model choice, cost, latency, token
usage, limitations and fallback behaviour, as required by the submission rules.

All figures measured with the **unmodified** official evaluator on the 200
public sessions. Nothing under `evaluator/` was modified at any point.

---

## 1. Problem shape

Reading `evaluator/local_evaluator.py` end to end determined the architecture,
so it belongs first.

The simulated customer is a **deterministic template engine, not a language
model**. The hidden brief is produced by `intent_card(product)` from the target
product's own metadata: it flattens `features` and `details`, prepends a
regex-matched material and colour, appends `budget around $<price>`, truncates
each to 180 characters, and keeps at most two hard constraints and two soft
preferences.

The entire information budget of a session is therefore **at most four short
strings lifted near-verbatim from one product's spec sheet.**

This is entity resolution, not dialogue understanding. It is why an offline
lexical system with no model reaches 0.955 hit rate, and it is why we did not
put a language model on the scoring path.

### Scoring function

```
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50*Hit@10 + 0.30*MRR + 0.20*Efficiency
```

A hit at rank `r` on turn `t` is worth `0.5 + 0.3/r + 0.2*(11-t)/10`. One extra
turn costs 0.02; moving rank 10 → rank 1 gains 0.27. **Rank dominates speed by
more than an order of magnitude**, and that governs every design decision below.

## 2. Architecture

```
message ─▶ parse ─▶ recall ─▶ rank ─▶ truncate ─▶ ask ─▶ response
           dialogue  retrieval retrieval dialogue  dialogue
```

`agent.py` implements the required interface and contains no logic. Each stage
lives in its own module with one owner, so five people can work concurrently.

### 2.1 Index (`src/retrieval/index.py`)

Built once per process (4.0 s). Two parallel representations of the same 50,000
products, because the two retrieval stages want different things:

1. An in-memory SQLite **FTS5** table for cheap candidate recall.
2. Field-weighted term frequencies, IDF, document lengths, parsed prices, and a
   lowercased text blob per product, for reranking.

Field weights: title 3.0, features 2.5, details 2.0, categories 1.5, store 1.0,
description 1.0. Weighted toward the spec-bearing fields because that is where
`intent_card` draws from.

### 2.2 Recall (`src/retrieval/recall.py`)

FTS5 disjunction over accumulated constraint terms, capped at 60 terms,
returning 400 candidates. FTS5's own BM25 decides pool **membership** only,
never final order.

**Measured finding: this stage is not a lever.** Four weight sets and pool sizes
from 100 to 1600 all yield byte-identical scores. The pools genuinely differ
(350/400 shared between two sets, order differs), but the products that swap in
and out are all irrelevant — 400 candidates is far past the point where the
target is reliably inside. Documented so no further effort is spent here.

### 2.3 Ranking (`src/retrieval/scoring.py`)

Three additive, independently ablatable signals:

1. **Field-weighted BM25** (k1 = 1.4, b = 0.6) over every disclosed constraint.
   `b` is below the 0.75 default because catalog document lengths are extremely
   uneven — a `details` dict can dwarf a title — and heavy length normalisation
   punished richly-specified products.
2. **Three-tier phrase bonus.** Exact substring (+14.0) → 40-character prefix
   (+7.0) → **proportional lexical overlap** (up to +8.0). The third tier is
   deliberate insurance: a contiguous substring match is destroyed by a single
   inserted word, whereas token overlap degrades smoothly. Scaled so it can
   never outrank a genuine exact match.
3. **Price proximity.** The brief always states the target's own price, so a
   near-exact match is strong evidence (+10.0 within 2%), and being far off is
   mild evidence against.

### 2.4 Truncation (`src/dialogue/truncation.py`)

List length is a scoring lever, not a UI choice. The session ends the instant
the target appears in the returned list, so exposing all 10 while uninformed can
end a session at rank 8 for 0.55 when two more turns would have paid 0.92 at
rank 1.

Short list while uninformed, full list once informed, and **always the full list
on the final turn** — holding back is a bet that a later turn pays more, and on
turn 10 that bet cannot pay. Worth 0.0287.

### 2.5 Elicitation (`src/dialogue/ask.py`)

The largest single lever in the competition: the official starter scores 0.107
purely because it never sets `ask_attribute`, so the customer refuses to
disclose anything ten turns running.

Our policy rests on a provable property of the evaluator.
`classify_constraint` (`local_evaluator.py:136-151`) is total and can only
return `budget`, `material`, `color`, `size`, `style`, `use_case` or `feature`.
`brand` and `category` are in `ALLOWED_ATTRIBUTES` and are legal to ask, but **no
constraint can ever classify as either.** Asking them is a guaranteed wasted
turn. Both are excluded — not an exploit, simply declining to ask a question
whose answer set is provably empty.

We then fall back to open-ended asking only once the customer has explicitly
stated the specific space is exhausted, parsing both refusal shapes the
evaluator emits. Threshold measured: 1 dead ask (0.8562) > 2 (0.8434) > 3
(0.8359).

## 3. Model choice

**No model.** No LLM, no embeddings, no fine-tuning, no vector database.

This was a decision, not a limitation:

1. **The brief is four spec-sheet strings.** There is no natural language to
   understand; there is text to align. Lexical matching is the right tool.
2. **The submission rules state that organizer policy may disable network access
   at final scoring**, and no API keys or credits are provided. A pipeline that
   dies without a key may be scored invalid.
3. Empirically it works: 0.955 hit rate with zero parameters.

A local dense route (MiniLM + reciprocal rank fusion) was scoped and
deliberately dropped once the stress harness showed synonym substitution costs
**0.0%** — the problem it would have solved does not exist here. See §6.

## 4. Cost, latency, memory, tokens

Apple Silicon laptop, single core, 661 turns across 200 sessions.

| Metric | Value |
|---|---|
| Index build | 4.0 s (once per process, amortised) |
| Latency per turn — mean | 23.8 ms |
| Latency per turn — p50 | 22.7 ms |
| Latency per turn — p95 | 51.4 ms |
| Latency per turn — max | 94.9 ms |
| 200 sessions end to end | 15.8 s |
| Resident memory | ~718 MB |
| Prompt tokens | **0** |
| Completion tokens | **0** |
| API cost | **$0.00** |
| Network calls | **none** |

## 5. Fallback behaviour

Required disclosure, since exceptions, malformed output and timeouts all count
as misses.

- **`respond()` never raises.** Verified across 72–81 calls spanning empty,
  whitespace-only, 5000-character, punctuation-only and numeric inputs, at three
  paraphrase levels, with and without a preceding `reset()`.
- **`respond()` without `reset()`** creates a fresh session rather than raising.
- **No network dependency to fall back from.** There is no external call in the
  codebase, so there is no degraded mode — the offline path *is* the path. This
  is the specific failure mode the submission rules warn about, and we have no
  exposure to it.
- **Unparseable input**: when template parsing recognises nothing, the agent
  ranks on the raw customer transcript instead of falling back to catalog order.
- **Unknown prices** are treated as absent, never as zero.

## 6. Robustness

The spec warns that added paraphrasing "cannot decide correctness".
`src/eval/paraphrase.py` measures that exposure by wrapping the **agent** — a
proxy implementing the same interface rewrites each message and delegates, so
`evaluator/` is untouched. Rewrites are seeded from a stable SHA-256 of
`(seed, turn, message)`, since the evaluator assigns a fresh uuid4 session id
per run.

| Stress | Score | Retained |
|---|---|---|
| clean | 0.856159 | — |
| none (control) | 0.856159 | 100.0% |
| clause reordering | 0.856159 | 100.0% |
| synonym substitution | 0.856159 | 100.0% |
| punctuation dropped | 0.856660 | 100.1% |
| filler words | 0.745774 | 87.1% |
| **template rewording** | **0.370891** | **43.3%** |

The `none` control reproducing the clean score exactly is what licenses reading
the rest.

**This contradicted our own design plan.** We expected the exact-phrase bonus to
be the fragility and had scoped a dense-embedding route accordingly. Synonym
substitution costs 0.0%; the fragility is entirely template matching in the
parser. Two paraphrase-agnostic fixes followed: punctuation-tolerant markers
(eliminated a 26% loss) and the raw-transcript fallback (9.2% → 43.3%
retained, at zero clean cost).

We did **not** teach the parser our own paraphrases. That would measure the
harness rather than the agent.

## 7. Limitations

1. **Template dependence, 43.3% retained.** Largest known exposure. If the
   private harness paraphrases, expect a substantial drop.
2. **Filler words cost 13%.** Fixable only with the words our own harness
   inserts, so left unfixed rather than manufacture a number.
3. **`buying` stuck at 0.938** across 80 sessions; moved through no change we
   made. Largest subgroup, least understood.
4. **Speed/rank tension.** Three sessions now hit at turn 3 rank 4 where they
   previously hit at turn 8 rank 1 — a net loss. Confidence-gated truncation is
   the proper fix and is not implemented.
5. **~718 MB resident.** Fine locally; the scoring environment's limit is
   unknown.
6. **Boundary is n = 10.** 0.600 → 1.000 is four sessions. Directionally real,
   statistically thin.
7. **Paraphrase figures are our guesses** at organizer behaviour. The *ranking*
   of fragile components is solid; the absolute 43.3% is an indicator, not a
   prediction.

## 8. Reproducibility

- Pure standard library; no dependency versions to drift.
- `evaluator/` unmodified; the agent is reached through the re-export at
  `starter/agent.py` that the evaluator hardcodes.
- Every run records git commit (with dirty-tree marker), UTC timestamp, build
  time, eval time and per-session latency.
- The harness was validated against two independently documented configurations
  before being trusted, reproducing 0.776 and 0.854 exactly.
- Sorting is stable and tie-breaks fall back to FTS5 recall order, so runs are
  deterministic.

```bash
python3 -m evaluator.local_evaluator          # headline score
python3 -m src.eval.run --name check          # + per-session diff vs baseline
python3 -m src.eval.paraphrase --level ablate # robustness table
```

## Appendix — full ablation

| Configuration | Score | Δ vs shipped |
|---|---|---|
| **Shipped** | **0.856159** | — |
| − dynamic truncation | 0.827439 | −0.028720 |
| − lexical overlap tier | 0.855581 | −0.000578 |
| − raw-transcript fallback | 0.856159 | 0.000000 * |
| fallback after 2 dead asks | 0.843435 | −0.012724 |
| fallback after 3 dead asks | 0.835935 | −0.020224 |
| bare `"other"` exploit (not shipped) | 0.862860 | +0.006701 |
| day-0 prototype | 0.785761 | −0.070398 |
| official BM25 starter | 0.106710 | −0.749449 |

\* Fires only when parsing recognises nothing, which never occurs on the clean
public set. Its value appears solely under §6.
