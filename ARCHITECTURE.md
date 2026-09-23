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
    │ 1. QueryDecomposer                 │  Claude Haiku → 5 sub-queries
    │    LLM, rule-based fallback        │  includes refutation-seeking angles
    └─────────────────┬──────────────────┘
    ┌─────────────────▼──────────────────┐
    │ 2. SearchService                   │  asyncio.gather over sub-queries
    │    Tavily, offline fallback        │  dedup by URL, first-seen order
    └─────────────────┬──────────────────┘
    ┌─────────────────▼──────────────────┐
    │ 3. EvidenceRetriever               │  bounded-concurrency page fetch
    │    MiniLM embeddings               │  split → rank by cosine → top-k
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
embeddings, keeping the top *k*.

Keyword matching fails the central case: a claim about *"intermittent fasting"*
must match a passage discussing *"time-restricted eating"*, which shares no
content words. That specific pair is asserted in `tests/test_retriever.py`.

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

Credibility itself is a static domain tier list — rule-based rather than learned,
because no labeled credibility dataset exists at the needed scale, domain
authority is stable, and the ML capacity in this system belongs in the stance
detector. Matching is by domain suffix so `pmc.ncbi.nlm.nih.gov` inherits its
parent's tier instead of dropping to the unknown default. Preprint servers
(arXiv, bioRxiv) sit in tier 3 deliberately: academic, but not peer reviewed.

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
accuracy / 0.838 macro F1 on held-out FEVER+VitaminC — see §10. What's left is
the harder half: end-to-end verdict accuracy against published fact-checker
rulings (PolitiFact, Snopes), which measures the whole pipeline rather than
the classifier in isolation.

**Calibration.** ~~MC dropout gives a *relative* uncertainty signal, not a
calibrated probability.~~ Done 2026-09-19 for the zero-shot fallback
checkpoint, then redone 2026-09-23 against the fine-tuned checkpoint (ECE
0.049 → 0.021 — better calibrated out of the box than the fallback model was
even after scaling; see README's Calibration section). Must be re-run again
any time `STANCE_CHECKPOINT` changes — `STANCE_TEMPERATURE` is specific to
one model's confidence/accuracy relationship, not a property of the task.

**Credibility.** Replace the static tier list with NewsGuard ratings or a
classifier trained on Media Bias/Fact Check labels. The interface does not change:
source dict in, scalar in [0,1] out.

**Retrieval.** Add a cross-encoder reranker over the top-k MiniLM results.
Bi-encoder similarity is a good first pass and a mediocre final ranking.

**Claim-worthiness.** Detect and reject opinions and predictions before the
pipeline runs, instead of processing them as falsifiable factual claims.

**Temporal reasoning.** Publication dates are carried through the pipeline but
never used. For claims about evolving situations, a 2015 source and a 2025 source
should not carry equal weight.
