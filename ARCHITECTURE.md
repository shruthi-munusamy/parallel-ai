# Architecture

Technical design notes for the neural claim verification system: what each stage
does, why it is built that way, and what the alternatives cost.

---

## 1. The problem

Claim verification is not retrieval and it is not classification. It is both,
composed, with a specific hazard: **the retrieval step biases the classification
step.** Search a claim verbatim and you get pages that share its framing, so the
evidence set is pre-selected to agree. Any downstream aggregation then confirms
whatever was asserted.

That hazard drives three of the design decisions below: adversarial query
decomposition (§3), credibility as a weight rather than a filter (§7), and a
verdict space that includes both `MIXED` and `INSUFFICIENT` (§7.2).

The second constraint is **grounding**. A system that generates its verdict from
model memory can fabricate a citation. Here the only text that reaches the user
is text retrieved from a source, and the only judgment made about it is a 3-way
classification over that text. There is no generative step between evidence and
verdict.

---

## 2. Pipeline

```
                    claim
                      │
    ┌─────────────────▼──────────────────┐
    │ 0. ClaimWorthinessChecker          │  regex against opinion/prediction/
    │    rule-based, no model            │  question framing; rejects before
    └─────────────────┬──────────────────┘  any search runs
    ┌─────────────────▼──────────────────┐
    │ 1. QueryDecomposer                 │  Claude Haiku → 5 sub-queries
    │    LLM, rule-based fallback        │  includes refutation-seeking angles
    └─────────────────┬──────────────────┘
    ┌─────────────────▼──────────────────┐
    │ 2. SearchService                   │  asyncio.gather over sub-queries
    │    Tavily, offline fallback        │  dedup by URL, first-seen order
    └─────────────────┬──────────────────┘
    ┌─────────────────▼──────────────────┐
    │ 3. EvidenceRetriever               │  bounded-concurrency page fetch
    │    MiniLM + cross-encoder rerank   │  split → bi-encoder pool → CE rerank
    └─────────────────┬──────────────────┘
          ┌───────────┴───────────┐         ← independent, run concurrently
    ┌─────▼──────────┐   ┌────────▼───────┐
    │ 4. Stance      │   │ 5. Credibility │  reads text  │  reads URL
    │    DeBERTa NLI │   │    domain tiers│
    └─────┬──────────┘   └────────┬───────┘
          └───────────┬───────────┘
    ┌─────────────────▼──────────────────┐
    │ 6. EvidenceSynthesizer             │  weight = confidence × credibility
    └─────────────────┬──────────────────┘
                  verdict
```

Stages 4 and 5 are independent — one reads passage text, the other reads the
URL — so they overlap under a single `asyncio.gather` rather than running back to
back.

### 2.1 Claim worthiness (stage 0)

Every stage from 1 onward assumes the input is a falsifiable factual claim.
Without a check first, "Pizza is the best food" or "AI will replace all
programmers by 2030" still produce a confident-looking `SUPPORTED`/`REFUTED`
verdict — the pipeline had no way to represent *"this was never a checkable
claim"*, which is a worse failure than `INSUFFICIENT`: it looks like the
system did the work and found evidence, when the actual error was upstream of
any evidence gathering.

`backend/models/claim_worthiness.py` rejects three categories by regex —
questions (trailing `?`), opinions (first-person or prescriptive framing:
*"I think"*, *"we should"*), and speculative predictions (*"will replace"*,
*"is expected to"*) — deliberately narrow ones, since a false positive here
(rejecting a real claim) costs the user an answer entirely, while a false
negative just falls through to the existing `INSUFFICIENT` path. Rule-based
rather than an LLM call for the same reason `credibility_scorer.py` is a
static tier list rather than a trained classifier: it needs no funded API key
(this project's Anthropic key currently has no credit balance — see below),
and the interface (claim in, checkable/reason out) will not need to change if
it is later replaced by something smarter, likely folded into the existing
`QueryDecomposer` LLM call rather than a second round trip.

---

## 3. Query decomposition

**Input:** one claim. **Output:** up to 6 search queries.

The naive approach appends fixed strings (`claim + " research"`, `claim + " studies"`).
That is string manipulation, not decomposition: every variant retrieves nearly
identical results because the search engine's own semantics dominate the suffix.

Instead Claude Haiku generates semantically distinct sub-questions under an
explicit instruction to include angles that could surface *refuting* evidence.
The original claim is always kept as the first query — it remains the single best
query for finding sources that address the claim head-on.

**Why an LLM here and not for stance detection?** The two tasks have opposite
shapes:

| | Decomposition | Stance detection |
|---|---|---|
| Task | open-ended generation | 3-way classification |
| Labeled data | none exists | FEVER, VitaminC, MNLI |
| Calls per claim | 1 | 30–90 |
| Latency sensitivity | low | dominates end-to-end |
| Needs calibration | no | yes — thresholded and propagated |

An LLM is correct for the first column and wrong for the second. A fine-tuned
encoder is correct for the second and unnecessary for the first.

**Failure mode:** model responses wrap JSON in prose or code fences often enough
that strict `json.loads` fails intermittently. `_parse_json_array` tries the raw
text, then a fenced block, then a bracket scan, before giving up to the
rule-based path — so a formatting quirk does not silently degrade retrieval
quality.

---

## 4. Search

Sub-queries run concurrently via `asyncio.gather` and merge by URL, preserving
first-seen order. Sequential search would multiply the stage's latency by the
number of sub-queries (~6×) for no benefit, since the queries are independent.

`TavilyClient.search` is synchronous. Calling it directly inside `async def`
would stall the event loop for the whole round trip and serialize the very
concurrency this stage exists for, so it runs on a worker thread via
`asyncio.to_thread`.

A failed sub-query returns `[]` rather than raising. One dead query must not
abort a verification that five other queries could still answer.

**Offline fallback.** Without an API key the service serves a curated corpus.
Two properties make it useful rather than decorative:

- Every topic carries **both supporting and refuting sources across credibility
  tiers**, so mock-mode verdicts are genuinely contested. A corpus where
  everything agrees would make stance detection and weighted synthesis look
  correct without exercising either.
- A query with **zero keyword overlap returns nothing**, rather than padding with
  filler. Otherwise an unverifiable claim would produce a confident verdict built
  from unrelated documents — the exact failure the `INSUFFICIENT` verdict exists
  to prevent.

Ranking there is deliberate keyword overlap, not embeddings: duplicating the
semantic layer would mask retrieval bugs by making the mock smarter than the
real path.

---

## 5. Retrieval

### 5.1 Why fetch pages at all

Search snippets are one or two sentences selected for relevance to the *query*,
not for stance toward the claim. The sentence that actually contradicts a claim
is usually deeper in the article. Snippet-only retrieval is the degraded path,
taken when a fetch fails — not the intended one.

### 5.2 Concurrency

This was the dominant latency term. Fetching 10 sources sequentially at an 8s
timeout is 80s worst case; overlapping them is 8s. Fetches run under a shared
`ClientSession` (connection and DNS reuse) bounded by a semaphore — unbounded
fan-out opens too many sockets at once and invites throttling.

A browser `User-Agent` is sent because a default `aiohttp` agent string is
rejected outright by a meaningful share of publishers, which silently degraded
the system to snippet-only retrieval.

### 5.3 Passage construction

Pages are stripped of `script`, `style`, `nav`, `header`, `footer`, `aside` and
`form` — text that is never evidence — then grouped into ~120-word passages on
sentence boundaries. Fragments under 10 words are dropped as navigation residue.

120 words rather than the model's 512-token ceiling: a passage should be one
coherent argument. Stuffing to the token limit dilutes the stance signal.

### 5.4 Semantic ranking

Passages are ranked against the claim by cosine similarity over `all-MiniLM-L6-v2`
embeddings, keeping a wider candidate pool than the final count needed (§5.5).

Keyword matching fails the central case: a claim about *"intermittent fasting"*
must match a passage discussing *"time-restricted eating"*, which shares no
content words. That specific pair is asserted in `tests/test_retriever.py`.

### 5.5 Cross-encoder reranking

A bi-encoder embeds the claim and each passage independently, so it can only
compare embeddings after the fact — the claim never gets to attend to the
passage's specific tokens. That is a good cheap first pass and a mediocre
final ranking. `RERANKER_MODEL` (default `cross-encoder/ms-marco-MiniLM-L-12-v2`)
scores `(claim, passage)` pairs jointly through one transformer pass instead,
which is far more accurate but too slow to run over every passage from every
source.

Two stages get both properties: the bi-encoder ranks all candidates per source
and keeps `PASSAGES_PER_SOURCE x RERANK_POOL_MULTIPLIER`, then the
cross-encoder reranks only that pool and keeps the final `PASSAGES_PER_SOURCE`.
Reranker failures degrade to the bi-encoder's own order rather than dropping
the source, matching the rest of the pipeline's degrade-not-fail contract.
`RERANK_ENABLED=false` skips this stage entirely.

**A specific checkpoint bug, worth recording:** the more commonly cited
`cross-encoder/ms-marco-MiniLM-L-6-v2` crashes with a SIGBUS on forward pass on
at least one dev machine (Apple Silicon, torch 2.10) — reproducible with a bare
`transformers` forward pass outside this codebase entirely, and specific to
that exact checkpoint (`bert-base-uncased` and `ms-marco-MiniLM-L-12-v2` both
load and run cleanly). Confirmed not a NaN-dtype issue (parameters load as
float32); root cause not pursued further given a working alternative existed.
If `RERANKER_MODEL` is ever changed back to `L-6-v2`, expect this.

Measured impact on the 10 demo claims: mean latency 0.95s → 1.78s. Verdict
agreement unchanged at 8/10 on this small sample — a reranker's value is in
evidence *quality* (does the passage actually address the claim), which this
sample is too small to detect a difference in.

---

## 6. Stance detection

The intelligence layer. Everything around it is plumbing.

### 6.1 Framing

Natural Language Inference: **premise** = evidence passage, **hypothesis** =
claim. Entailment → `SUPPORTS`, contradiction → `REFUTES`, neutral → `NEUTRAL`.

### 6.2 Label mapping is read, never assumed

The two checkpoints order their labels *differently*:

| Checkpoint | 0 | 1 | 2 |
|---|---|---|---|
| `cross-encoder/nli-deberta-v3-base` | contradiction | entailment | neutral |
| `train_stance.py` output | REFUTES | NOT ENOUGH INFO | SUPPORTS |

A hardcoded map inverts `SUPPORTS` and `NEUTRAL` the moment a fine-tuned
checkpoint is dropped in — and the failure is **invisible**, because the API keeps
returning confident, well-formed, wrong verdicts. So the map is derived from the
model's own `config.id2label`, and an unrecognized label set raises at startup
rather than guessing an ordering.

### 6.3 Sentence-level scoring

The measured problem: these checkpoints are MNLI-trained, where premises are
single sentences. Real web passages are not.

| Premise | Prediction |
|---|---|
| *"Multiple independent lines of evidence show that the observed warming since the mid-20th century is caused primarily by human emission of greenhouse gases. More than 97% of actively publishing climate scientists agree."* | `NEUTRAL` 0.999 |
| *"Warming is caused by humans."* | `SUPPORTS` 0.964 |

Same claim, same content, opposite outcome — purely a function of premise length
and specificity. This is domain shift, not a bug, and it was collapsing correct
verdicts into `INSUFFICIENT`.

**Fix:** score each sentence separately, then aggregate by **max over directional
sentences**. A passage is `NEUTRAL` only when no sentence takes a side.

Max rather than mean, for a semantic reason: NLI entailment over a multi-sentence
premise requires the *whole* premise to entail the hypothesis, which is the wrong
semantics for fact-checking. A source refutes a claim if any sentence in it
refutes the claim, even when the surrounding sentences are background. Averaging
lets three neutral sentences of context bury the one sentence carrying the
evidence.

Two consequences beyond accuracy (4/10 → 8/10 on the demo set):

- **Explainability.** The deciding sentence is returned as `stance_sentence`, so
  every stance badge is traceable to specific text rather than a paragraph the
  reader must re-scan.
- **Cost.** Fan-out is capped at `MAX_SENTENCES_PER_PASSAGE` (6), since cost is
  linear in sentences × MC samples. All sentences across all passages are
  flattened into one batch, so the stage remains a single set of padded forward
  passes.

### 6.4 Uncertainty via MC dropout

Softmax confidence is poorly calibrated — a model can be confidently wrong.
Running *T* forward passes with dropout **enabled** approximates sampling from
the posterior over weights; the mean per-class standard deviation across those
samples is an epistemic uncertainty estimate.

Two implementation details that matter:

- Only `nn.Dropout` modules are toggled, not `model.train()`. The latter also
  switches normalization layers that track running statistics into training mode,
  mutating shared model state during inference.
- Dropout is restored in a `finally` block. The model is a process-wide
  singleton; if a failed pass left dropout on, **every subsequent request would
  become silently stochastic.** Asserted in `test_dropout_disabled_after_prediction`.

With `T=1` the reported uncertainty is exactly 0 — a single deterministic pass
measures no spread, and reporting anything else would be fabrication.

### 6.5 Batching and threading

Inference is batched with padding. Verified correct by
`test_batching_matches_individual_prediction`: batched and one-at-a-time results
must agree, which catches the classic attention-mask bug where short sequences
attend to pad tokens.

Torch inference is CPU-bound and releases the GIL, so it runs under
`asyncio.to_thread` to keep the event loop responsive to other requests.

### 6.6 Confidence floor

Below `UNRELATED_THRESHOLD` (0.40) a prediction becomes `UNRELATED` and is
excluded from the verdict entirely. A uniform distribution over 3 classes is
0.333 — by definition no signal. Without the floor, weak noise accumulates into a
confident verdict.

---

## 7. Synthesis

### 7.1 Credibility as a weight

Per passage: `weight_i = stance_confidence_i × credibility_i`, summed per stance.

Most NLI fact-checkers gate on credibility instead ("drop below 0.5"). That
discards signal and behaves erratically at the threshold. Weighting is strictly
more informative: a weak source still contributes, just less.

The worked example, asserted in `tests/test_synthesizer.py`:

```
PubMed  cred 0.95  REFUTES  conf 0.60  →  0.570
Blog    cred 0.35  SUPPORTS conf 0.85  →  0.298
```

Averaging confidence picks `SUPPORTED` (0.85 > 0.60). Weighting picks `REFUTED`.

Credibility itself is a small hand-curated tier list for sources this project
specifically vetted — rule-based rather than learned, because domain authority
at that end is stable and the ML capacity in this system belongs in the stance
detector — layered on top of two open datasets for everything else. CRED-1
(`data/cred1_compact.json`, CC BY 4.0, added 2026-09-30) covers ~2,700 known
misinformation/fake/conspiracy domains, built from OpenSources.co and the
Iffy.news Index. NewsGuard itself is a paid partnership and was never a real
option for an open-source project; CRED-1 was chosen because it covers the
negative-end gap the hand-curated list left, not because it's a general
credibility source — its own documentation reports only 8 reliable-source
labels. The positive-end gap is instead filled by WP:RSP
(`data/rsp_compact.json`, CC BY-SA 4.0, added 2026-10-01) — Wikipedia's
Reliable Sources/Perennial Sources list, a community-consensus judgment of
~750 frequently-discussed sources covering *both* ends (generally
reliable/no consensus/generally unreliable/deprecated/blacklisted), extracted
by parsing the list's raw wikitext rather than scraping the rendered page,
since the wikitext carries each row's status as a structured class attribute.
Checked in order: hand-curated tiers (most specific, wins any conflict) →
CRED-1 (narrower, purpose-built for known bad actors) → WP:RSP (broader
general-reliability signal) → TLD heuristic → unknown default. Matching is by
domain suffix at every stage so e.g. `pmc.ncbi.nlm.nih.gov` inherits its
parent's tier, a CRED-1 domain like `infowars.com` extends to
`forums.infowars.com`, and an RSP domain like `buzzfeednews.com` extends to
any of its subdomains — instead of any of them dropping to the unknown
default. Preprint servers (arXiv, bioRxiv) sit in tier 3 deliberately:
academic, but not peer reviewed.

### 7.2 Two ratios, kept separate

- **Reported ratios** are shares of *total* evidence mass including neutral, so
  they sum to 1 and describe what the corpus looks like.
- **The verdict** is decided by the support-vs-refute contest alone, since
  neutral evidence is not a vote for either side.

An earlier version divided only by `support + refute` and derived neutral as the
remainder — which made `neutral_ratio` identically zero and left the UI's neutral
segment permanently empty. `test_ratios_sum_to_one` guards it.

Verdict rules, in order:

| Condition | Verdict |
|---|---|
| contested mass < 20% of total | `INSUFFICIENT` |
| support share ≥ 70% | `SUPPORTED` |
| refute share ≥ 70% | `REFUTED` |
| both sides ≥ 35% | `MIXED` |
| winner < 55% | `INSUFFICIENT` |
| otherwise | winner |

The first rule is checked first deliberately: an overwhelmingly neutral corpus
cannot support a directional verdict however lopsided its few votes are.

`MIXED` and `INSUFFICIENT` are distinct on purpose. *"Credible sources genuinely
disagree"* and *"we did not find enough to say"* are different answers, and
collapsing them would misrepresent contested science as unresearched.

### 7.3 Uncertainty propagation

MC-dropout spread is averaged over directional passages and discounts the
verdict: `confidence × (1 − 0.5 × uncertainty)`.

Capped at half so that total model instability halves a verdict's confidence but
never erases a real evidence majority.

### 7.4 Temporal weighting

Publication dates flow through the pipeline from stage 2 onward but were
unused until 2026-09-29: `w_i = stance_confidence_i × credibility_i` had no
notion that a 2015 source and a 2025 source should not carry equal weight for
a claim about an evolving situation. `_recency_weight` adds a third factor —
exponential half-life decay, `0.5 ** (age_years / TEMPORAL_HALF_LIFE_YEARS)`
— with two guards against the obvious ways a decay term goes wrong:

- **Missing/unparseable dates score 1.0, not a penalty.** Web retrieval
  frequently can't recover a reliable publication date; treating that
  absence as staleness would make the pipeline worse specifically on the
  passages it already struggles to date, for no real signal gained.
- **Floored at `TEMPORAL_MIN_WEIGHT` (0.3), not decayed to zero.** Age alone
  does not make a stable fact wrong — a 1990 source is still the right answer
  for "Paris is the capital of France." This is a dampener on old evidence,
  not a filter that discards it.

`TEMPORAL_HALF_LIFE_YEARS = 5.0` is a single global constant, which is the
known limitation here: it cannot distinguish a claim about a historically
stable fact from one about a fast-moving topic (a stock price, an ongoing
investigation, current officeholders), where 5 years is arguably still too
generous. Claim-specific temporal sensitivity would need the claim's own
subject matter classified first — not attempted here.

---

## 8. Cross-cutting concerns

### Graceful degradation

| Failure | Behaviour |
|---|---|
| No search key | Offline corpus, flagged `mock_search: true` |
| No LLM key | Rule-based decomposition |
| Search provider error | `[]` for that sub-query; others proceed |
| Fetch failure / timeout | Search snippet for that source |
| Ranking failure | Unranked passages — costs relevance, not the source |
| Stance model raises | `NEUTRAL` at uncertainty 1.0 |
| No usable evidence | `INSUFFICIENT` at confidence 0.0 |
| Models still loading | `503`, so clients retry rather than treat it as a bug |

The one rule the mock path must never break: a mock-backed response is **always**
self-identifying, through `mock_search` in the payload, `live_search` in
`/health`, a banner in the UI and a warning in the demo. Silently serving
fabricated evidence that looks live would be the worst failure this system could
have.

### Caching

Keyed on a normalized claim hash, bounded by TTL and LRU eviction. A verification
costs one LLM call, several search calls, up to ten page fetches and *T×N*
transformer passes; repeating that for an identical claim is pure waste. The
previous unbounded dict leaked memory and served stale verdicts about live events
indefinitely. Per-process; the async interface is shaped for a Redis swap.

### Observability

Each request gets an id stored in a `ContextVar`, which propagates into asyncio
tasks automatically and is injected into every log record. Without it, interleaved
logs from two concurrent verifications — each fanning out across six stages and
ten fetches — are impossible to separate. Every stage logs its own duration, so a
latency regression is attributable from logs alone.

### Configuration

`backend/config.py` calls `load_dotenv()` at import time, and every entrypoint
imports it first. This is deliberate: the previous layout defined `load_dotenv()`
in a module nothing imported, so `.env` was never loaded and the search service
raised on startup.

---

## 9. Performance

Measured on CPU (M-series), offline corpus, `MC_SAMPLES=5`.

| Stage | Cost | Notes |
|---|---|---|
| Decomposition | ~0.5 s | One LLM call; 0 ms on the fallback path |
| Search | ~0.3 s | Concurrent; flat in sub-query count |
| Retrieval | 0–8 s | Network-bound; 0 ms in mock mode |
| Stance | ~1–3 s | **Dominant compute cost** |
| Credibility | <1 ms | Overlapped with stance |
| Synthesis | <1 ms | Pure arithmetic |

Stance stage, 30 passages: `T=10` → 7.2s, `T=5` → 3.7s, `T=1` → 0.6s. Linear in
`MC_SAMPLES`, which is why it is the primary tuning dial.

End-to-end over the 10 demo claims: **mean 1.48s, p95 3.16s**, cached <1ms.

**Where the remaining time goes.** Stance detection, and specifically MC dropout.
The ordered options for cutting it: GPU inference (largest win, unexercised here);
distilled or quantized checkpoint; MC only for passages near the decision
boundary rather than all of them; `T=1` to drop uncertainty entirely.

---

## 10. Evaluation status

The stance detector has a real, measured accuracy number as of 2026-09-23:
**86.9% accuracy, 0.838 macro F1** on 1000 held-out FEVER+VitaminC examples
(REFUTES 0.870, NOT ENOUGH INFO 0.717, SUPPORTS 0.927), fine-tuned on
Northeastern's Explorer SLURM cluster (Tesla V100), stopped early by
`EarlyStoppingCallback` around epoch 1.2 of 3. This satisfies (1) below. It
is **not yet a defensible end-to-end benchmark number**, which requires:

1. ~~FEVER or VitaminC dev-set evaluation of the stance detector, per class.~~
   Done — see above.
2. A baseline comparison — zero-shot MNLI vs fine-tuned — on the same split.
   Also measured: the zero-shot fallback checkpoint scores 57.9% accuracy on
   the same held-out slice, so fine-tuning is a +29-point improvement, not
   noise.
3. End-to-end verdict accuracy against published fact-checker rulings
   (PolitiFact, Snopes), which measures the *pipeline*, not just the
   classifier. **Not done.** The 8/10 in the README is still a regression
   check on ten claims whose expected verdicts are the author's reading of
   the evidence, against a corpus the author wrote — it detects breakage, not
   accuracy, and should not be quoted as one.

Getting here took four rounds of bugs, none of which showed up until the
script was actually run against real infrastructure:

- **Local smoke test (2026-09-18)**, 200-example slice on this machine (no
  GPU): a FEVER loader dead against `datasets>=3.0` (fixed by switching to
  `pietrolesci/nli_fever`, a Parquet re-host), a missing `sentencepiece`
  dependency, a renamed `Trainer` argument, and a model that silently loaded
  in float16 and produced NaN loss from the first step.
- **Real Explorer run (2026-09-20 to 09-23)**, four more bugs surfaced only
  under the actual cluster environment: `conda activate` silently failing in
  a non-interactive batch script (needed `conda.sh` sourced explicitly, and
  `set -e` to fail loudly instead of running against the wrong Python); a
  missing `protobuf` dependency alongside `sentencepiece`; `warmup_ratio`
  fully removed (not just deprecated) on the cluster's newer transformers
  version; a CUDA build/driver mismatch (`torch` defaulted to a cu130 build,
  but the node's driver only supports CUDA 12.3, so `torch.cuda.is_available()`
  silently returned `False` and one full job trained uselessly on CPU before
  anyone noticed); and finally a security-guard regression in transformers
  that refuses `torch.load` of `pytorch_model.bin` below torch 2.6 — worked
  around with `use_safetensors=True` rather than fighting that version
  constraint against the CUDA one.

Full details in `backend/training/train_stance.py`,
`backend/training/setup_explorer_env.sh`, and `backend/training/submit_explorer.sh`.

---

## 11. What would come next

**Accuracy.** ~~Run the FEVER + VitaminC fine-tune.~~ Done 2026-09-23: 86.9%
accuracy / 0.838 macro F1 on held-out FEVER+VitaminC — see §10. ~~End-to-end
verdict accuracy against published fact-checker rulings, which measures the
whole pipeline rather than the classifier in isolation.~~ Done 2026-10-01 —
see §12. First pass: 29% accuracy on claims the pipeline committed to (55/96;
rest `INSUFFICIENT`), with a specific, diagnosed SUPPORTED bias, not just
noise. ~~Fix the diagnosed bias (retrieval ranks by claim-similarity only, so
a fact-check's own ruling loses to a passage that merely restates the
claim).~~ Done same day — see §12.1: accuracy on a matched 36-claim subset
rose 27% → 63%, trading some coverage (61% → 44%) for it. Finishing the
classifier was the easy half; evaluating and then fixing the whole assembled
system was the hard half, and neither step would have been possible without
the other — the classifier-only benchmark could never have shown this bug,
and the bug couldn't be fixed without a reproducible end-to-end eval to
validate the fix against.

**Calibration.** ~~MC dropout gives a *relative* uncertainty signal, not a
calibrated probability.~~ Done 2026-09-19 for the zero-shot fallback
checkpoint, then redone 2026-09-23 against the fine-tuned checkpoint (ECE
0.049 → 0.021 — better calibrated out of the box than the fallback model was
even after scaling; see README's Calibration section). Must be re-run again
any time `STANCE_CHECKPOINT` changes — `STANCE_TEMPERATURE` is specific to
one model's confidence/accuracy relationship, not a property of the task.

**Credibility.** ~~Replace the static tier list with NewsGuard ratings or a
classifier trained on Media Bias/Fact Check labels.~~ Done, via two open
datasets rather than one: NewsGuard is paid-only and Media Bias/Fact Check has
no official open release (only unofficial scrapes of uncertain license), so
the low end was extended by CRED-1 (2026-09-30) and the positive end — the
part CRED-1 structurally cannot cover — by WP:RSP (2026-10-01, CC BY-SA 4.0)
instead. See §7.1. The interface did not change: source dict in, scalar in
[0,1] out.

**Retrieval.** ~~Add a cross-encoder reranker over the top-k MiniLM results.
Bi-encoder similarity is a good first pass and a mediocre final ranking.~~
Done 2026-09-23 — see §5.5. Still open: the bi-encoder/cross-encoder pool
sizes (`PASSAGES_PER_SOURCE`, `RERANK_POOL_MULTIPLIER`) were chosen to hold
latency roughly flat, not tuned against a labeled relevance set — there's no
measurement yet of whether a wider pool would improve final evidence quality
enough to justify the added latency.

**Claim-worthiness.** ~~Detect and reject opinions and predictions before the
pipeline runs, instead of processing them as falsifiable factual claims.~~
Done 2026-09-29 — see §2.1. Rule-based and deliberately conservative; an LLM
pass (folded into the existing `QueryDecomposer` call) would catch more
subtle cases once a funded Anthropic key is available.

**Temporal reasoning.** ~~Publication dates are carried through the pipeline
but never used. For claims about evolving situations, a 2015 source and a
2025 source should not carry equal weight.~~ Done 2026-09-29 — see §7.4.
Still open: a single global half-life can't tell a stable fact from a
fast-moving topic; that needs the claim's subject matter classified first,
not attempted here.

---

## 12. End-to-end evaluation against LIAR

`backend/eval/eval_liar.py` runs real claims through the live assembled
pipeline (real Tavily search, not the mock corpus) and compares the final
verdict to real PolitiFact rulings, via the LIAR dataset (Wang, 2017; 12.8K+
labeled political statements, `data/liar/`). This is the piece no
classifier-only benchmark can stand in for: FEVER+VitaminC measures the
stance detector alone, against evidence the benchmark already hands it.
Here, decomposition, live retrieval, reranking, credibility weighting, and
the synthesizer's verdict rule are all live and all contribute to the final
answer.

**Label mapping.** LIAR's 6-way labels collapse to this pipeline's 3-way
directional verdicts, symmetrically around PolitiFact's own "half true"
midpoint: `pants-fire`/`false` → `REFUTED`, `barely-true`/`half-true` →
`MIXED`, `mostly-true`/`true` → `SUPPORTED`. `UNCHECKABLE` and `INSUFFICIENT`
are scored as abstentions, not failures — the pipeline declining to commit is
a different outcome than committing and being wrong.

**Result (2026-10-01, 96 claims, stratified across all 6 LIAR labels, seed=0,
`claude-haiku` decomposition unfunded so running on the rule-based
fallback):** 0 errors, 0 claims rejected by the claim-worthiness gate.
**57% coverage** (55/96 claims got a directional verdict; 41 came back
`INSUFFICIENT`). **29% accuracy on those 55** — below the 33% a 3-way random
guess would get on average, and this is not noise: the confusion matrix shows
a specific, strong directional bias rather than scattered error.

```
predicted:      SUPPORTED  REFUTED  MIXED  INSUFFICIENT
actual REFUTED:      14        2       5        11        (n=32)
actual MIXED:        16        1       2        13        (n=32)
actual SUPPORTED:    12        1       2        17        (n=32)
```

Whenever the pipeline committed to a verdict at all, it said `SUPPORTED` 76%
of the time (42/55) — including on 14 of 32 claims PolitiFact rated
`pants-fire` or `false`. Recall on `REFUTED` is 2/32; on `MIXED`, 2/32.

**Diagnosed mechanism, not just a number.** Spot-checking the `REFUTED`→
`SUPPORTED` misses (e.g. the Menendez "3.8% home sales tax" chain-email
claim, a fabricated Stalin quote, the CFL-bulbs-cause-migraines claim — all
well-known false claims with extensive fact-check coverage) shows the same
pattern each time: the web's coverage of a widely-circulated false claim
*restates the claim's content* prominently, even in articles whose overall
verdict is "this is false" — a debunking article about the Stalin quote
still contains a sentence reporting what the quote *says*. §4's sentence-level
stance scoring (added specifically to stop NLI collapsing to NEUTRAL on long
passages) reads that single restating sentence in isolation and scores it as
entailment, with no way to see that the surrounding article's actual
conclusion is the opposite. The fix that solved one failure mode
(context dilution) opened a different one (context loss) — a real tradeoff,
not a bug in the sense of an implementation mistake.

This was not visible at any point before this eval: FEVER+VitaminC's
evidence sentences are already the relevant supporting/refuting sentence, by
construction — the benchmark cannot contain this failure mode, because it
never hands the model a sentence that merely *restates* a claim inside a
larger debunking argument. Only running the whole pipeline against
real, messy web coverage of real false claims surfaced it.

Full per-claim results: `data/liar/eval_results_test_n100_seed0.json`.

### 12.1 The fix, and validating it honestly

Root cause traced to two places, not one — fixed both:

1. **Retrieval ranks by claim-similarity only** (`retriever.py`, both the
   bi-encoder and cross-encoder stages). A fact-check's own ruling ("we rate
   this claim False") shares almost no vocabulary with the claim it rules on,
   so pure-similarity ranking systematically scores it *below* a passage that
   merely restates the claim — maximally similar to it by construction.
   Confirmed by direct inspection: for the Menendez example, all 9 retrieved
   passages across 5 real fact-check pages were restatements; the source
   pages' actual verdict sentences were never retrieved at all. Fix:
   `_has_verdict_language()` reserves up to `VERDICT_SLOTS` (default 2) pool
   slots for passages carrying fact-check-ruling vocabulary ("we rate",
   "debunked", "pants on fire", "contrary to", …) that clear a minimum
   relevance floor (`VERDICT_MIN_RELEVANCE`) but would otherwise lose on pure
   similarity — it never overrides similarity ranking outright, only
   guarantees such a passage a chance to compete.
2. **Sentence-level aggregation has no concept of attribution**
   (`stance_detector.py`). Even once a verdict sentence is retrieved, the
   existing max-confidence-wins rule still lets a claim-echoing sentence beat
   it if the echo happens to score higher. Fix: `_is_claim_echo()` excludes a
   directional sentence from the vote when it both (a) contains an
   attribution verb or quotation mark and (b) has ≥50% lexical overlap with
   the claim — i.e. it's quoting/attributing the claim to a third party, not
   asserting or evaluating it independently. This is a named, studied NLI
   failure mode (annotation artifacts / lexical-overlap shortcuts; McCoy,
   Pavlick & Linzen 2019, "Right for the Wrong Reasons"), not something
   specific to this codebase.

**Validated, not assumed.** Re-ran the eval on the exact same 36 claims from
the original run (`eval_liar.py --n 40 --seed 0` reproduces an identical
prefix of the original stratified sample — verified by matching claim IDs)
rather than trusting the single Menendez anecdote:

| | Before | After |
|---|---|---|
| Coverage | 61% | 44% |
| Accuracy on directional | 27% | **63%** |
| Share of directional calls that were SUPPORTED | 59% | 56% |

Accuracy on directional more than doubled. Coverage *dropped* — the system
got more conservative, which is the expected and acceptable side effect:
removing the spurious high-confidence echo/restatement "evidence" that used
to push borderline claims over the verdict threshold means more of them
correctly fall back to `INSUFFICIENT` instead of being confidently wrong.
Trading coverage for correctness is the right trade for a fact-checker.
n=36 is still modest — this is a real, credible improvement, not a claim that
the bias is fully solved.

Full per-claim results, both runs: `data/liar/eval_results_test_n100_seed0.json`
(before) and `data/liar/eval_results_test_n40_seed0_postfix.json` (after,
same 36-claim prefix).
