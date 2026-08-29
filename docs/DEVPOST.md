# ByteMe — Conversational Shopping Copilot (Devpost draft)

> Draft for the Devpost submission. Sections map to the published judging
> criteria. Numbers are reproducible with `python3 -m evaluator.local_evaluator`.

## Inspiration

Shopping assistants usually fail in one of two ways: they interrogate you, or
they guess silently and hand you a wall of products. The interesting question
in this problem is not "can we retrieve" — it is **what is the single most
valuable question to ask next, and when should we stop asking and commit?**

## What it does

Finds a customer's hidden target product in a frozen 50,000-item catalog within
10 turns, by asking specific, answerable questions and progressively narrowing a
ranked shortlist. It handles four behaviours: a buyer with a firm requirement, a
browser with none, a customer who reverses their mind mid-conversation, and one
who declines to state a preference at all.

**Technical score 0.911 vs the 0.107 official baseline.** 98.5% of sessions end
in a hit, in 2.41 turns on average.

## How we built it

Reading the evaluator end to end came before writing any code, and it changed
the entire design. The customer is a **deterministic template engine, not an
LLM**, and the hidden brief is generated from the target product's own metadata:
at most four short strings lifted near-verbatim from one spec sheet. This is an
entity-resolution problem wearing a dialogue costume. That is why an offline
lexical system reaches 0.985 hit rate with no model at all.

Five stages per turn — parse, recall, rank, truncate, ask — each in its own
module with a single owner.

- **Recall**: SQLite FTS5, 50,000 → 400 candidates.
- **Rank**: field-weighted BM25 + a three-tier phrase bonus (exact substring →
  prefix → proportional lexical overlap) + price proximity.
- **Truncate**: list length is a scoring lever. The session ends the moment the
  target appears in the returned list, so showing all 10 while uninformed can
  end a session at rank 8 for 0.55 when waiting would have paid 0.92 at rank 1.
- **Ask**: the biggest lever in the competition. The official starter scores
  0.107 for one reason — it never sets `ask_attribute`, so the customer refuses
  to disclose anything ten times running.

### Built with

Python 3.10+ standard library, `sqlite3` (FTS5), `re`, `json`, `math`.
**No third-party packages. No LLM. No embeddings. No vector database. No
network. No API keys.** Dataset: Amazon Reviews 2023,
`Clothing_Shoes_and_Jewelry`, as supplied.

---

## Technical Execution (35%)

**Discipline, not just score.** Every behavioural change was measured before it
was kept, one command per experiment, with the result diffed against a fixed
baseline at three levels: overall, per scenario, and **per session** — which
individual sessions were fixed, broken, improved or worsened. The aggregate
hides regressions; a change can add three hits in one scenario while quietly
breaking another and still look like a win.

We validated the harness itself against two independently documented
configurations before trusting it (reproducing 0.776 and 0.854 exactly).

That discipline found two defects in our own prototype that reading had missed:

- **All 30 `intent_override` sessions were raising `IndexError`** every time the
  customer reversed their preference. The evaluator silently swallowed it and
  substituted an empty response, so the turn returned nothing *and* the new
  requirement was never recorded. Caused by testing a lowercased string but
  splitting the original-case string on a case-specific literal.
- **Six sessions returned only five results on the final turn** — and all six
  were misses. Ranks 6–10 were withheld on a turn where withholding cannot
  possibly pay.

Component ablation (each row strips one component):

| Configuration | Score |
|---|---|
| **Shipped** | **0.911025** |
| − popularity prior | 0.864975 |
| − dynamic truncation | 0.885358 |
| − lexical overlap tier | 0.909750 |
| ask: candidate-aware selector | 0.882950 |

**A negative result we think is worth as much as a positive one.** FTS5 recall
weight tuning is completely inert in this architecture — four weight sets and
pool sizes from 100 to 1600 all produce byte-identical scores. The pools
genuinely differ; the products that swap in and out are simply all irrelevant.
Recall decides only pool *membership*, and the reranker decides everything else.
We recorded it so nobody on the team spends another day there.

## Innovation & Problem Insight (20%)

**We tried to break our own agent, and succeeded.** The spec warns that if the
organizer paraphrases customer messages, that must not decide correctness. So we
built a stress harness that rewrites every customer message before the agent
sees it — wrapping the *agent*, not the evaluator, so nothing under `evaluator/`
is touched.

It disproved our own design assumption. We had planned a dense-embedding route
because we expected the **exact-phrase bonus** to be the fragile part. Measured:

| Stress | Retained |
|---|---|
| none (control) | 100.0% |
| synonym substitution | 100.1% |
| punctuation dropped | 100.0% |
| filler words | 94.5% |
| **template rewording** | **44.2%** |

Synonyms cost **nothing**. The fragility was entirely template matching in the
parser. We would have spent days hardening the wrong component. The control
reproducing the clean score exactly is what makes the rest trustworthy.

**We also found an exploit and declined to use it.** The evaluator's
`ask_attribute="other"` bypasses its own type check and drains the customer's
entire brief in two turns. It scores **0.913900** against our shipped
**0.911025**. We left 0.0029 on the table deliberately, because an agent that
asks "tell me your other preference" twice is not a shopping assistant.

Our policy instead rests on something provable: `classify_constraint` is total
and can never return `brand` or `category`, so asking either is a guaranteed
wasted turn — we exclude both. Then we fall back to open-ended asking only once
the customer has explicitly said the specific space is exhausted. Result: 55% of
questions are specific attributes, every session asks at least one, and it
scores within 0.007 of the exploit.

## Impact & Relevance (20%)

The behaviour that matters commercially is the one that was worst: **boundary**
— customers who say "I don't mind, you choose". That is a real and common
shopping stance, and an assistant that keeps interrogating such a customer is
actively annoying. It went **0.600 → 1.000** once we parsed the refusal signal
and stopped re-asking, with MRR reaching 0.950.

Detecting refusal and adapting is the difference between an assistant and an
interrogation. The same machinery handles a customer who changes their mind
mid-session (`intent_override`, 0.900 → 0.933).

Zero marginal cost per conversation means this is deployable at catalog scale
without a per-query LLM bill — the constraint that actually decides whether a
retailer ships something like this.

## Feasibility & Practicality (15%)

| | |
|---|---|
| Index build | 3.8 s, once per process |
| Latency per turn | 21.6 ms mean, 50.0 ms p95, 93.9 ms max |
| Memory | ~725 MB resident |
| Dependencies | none beyond the standard library |
| **LLM tokens** | **0** |
| **API cost** | **$0.00** |
| **Network calls** | **none** |

This is our strongest card and it was a deliberate architectural decision, not
an accident. The submission rules note that organizer policy **may disable
network access at final scoring**, and no API keys or credits are provided. A
pipeline that dies without a key may be scored invalid. Ours has nothing to
disable: no network call exists anywhere in the codebase. It runs on a laptop,
in memory, offline, for free.

## Challenges we ran into

Coordinating five people on one file. Two team branches were built on the
original single-file agent while it was being refactored into modules, and one
of those changes — carefully reasoned BM25 weight tuning — turned out to be
inert in the new architecture. We caught it by measuring rather than by
arguing, and the module split with one owner each exists to stop it recurring.

The subtler challenge was resisting a tempting mistake: our stress harness would
show a much better number if we taught the parser to recognise the paraphrases
*we ourselves wrote*. That measures the harness, not the agent. Every robustness
fix we kept is paraphrase-agnostic by construction.

## What we learned

Read the evaluator before writing code. Measure before believing — including
your own strategy document, which in our case predicted the wrong failure mode.
And record negative results, because "don't bother tuning recall" saved more
team-hours than most of our positive changes gained points.

## What's next

1. Reduce template dependence — our largest measured exposure at 44.2% retained.
2. `buying` is stuck at 0.938 across 80 sessions and moved through none of our
   changes. Largest subgroup, least understood.
3. Confidence-gated truncation: three sessions now hit faster but at worse rank,
   a net loss under the scoring function.

## Presentation (10%)

See the README for full reproduction steps, the complete ablation table, and a
plainly-stated limitations section. Every number in this document is
reproducible with a single command against the unmodified official evaluator.
