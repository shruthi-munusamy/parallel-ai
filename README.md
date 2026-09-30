# Neural Claim Verification

A fact-checking system that automates *lateral reading* — the method professional
fact-checkers use: take a claim, search for what independent sources say about it,
read what each source actually argues, weigh those sources by how trustworthy they
are, and report a verdict that stays traceable to the text it came from.

Every verdict is grounded in retrieved passages. Nothing is generated from model
memory, so there is no hallucination surface: if the system cannot find evidence,
it says `INSUFFICIENT` rather than inventing a conclusion.

```
POST /verify  {"claim": "Vaccines cause autism"}

REFUTED   confidence 82%   uncertainty 14%
support 11% · neutral 0% · refute 89%

REFUTES  conf 0.99  cred 0.92  cdc.gov
  scored: "Multiple large-scale studies covering more than 1.2 million
           children have found no link between vaccines and autism..."
REFUTES  conf 0.79  cred 0.92  thelancet.com
  scored: "No causal link between the MMR vaccine and autism was established."
```

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # optional: add API keys
python -m backend.main        # http://localhost:8000/docs
```

The API keys are **optional**. Without them the system runs end to end against a
built-in offline corpus and labels every response `mock_search: true`, so a demo
run can never be mistaken for a live one.

| Key | Effect when absent |
|---|---|
| `TAVILY_API_KEY` | Falls back to the offline corpus in `backend/services/mock_corpus.py` |
| `ANTHROPIC_API_KEY` | Falls back to rule-based query decomposition |

First run downloads ~750MB of model weights and takes a few minutes; afterwards
startup is about 2 seconds.

**Demo** — runs ten curated claims and prints a scored report:

```bash
python demo_script.py                 # all cases in data/test_cases.json
python demo_script.py --limit 3       # first three
python demo_script.py --claim "..."   # one ad-hoc claim
```

**Frontend**

```bash
cd frontend && npm install && npm run dev   # http://localhost:3000
```

**Tests**

```bash
python -m pytest              # 113 tests, ~30s
python -m pytest -m "not slow"  # skip the ones that load model weights
```

---

## How it works

```
claim
  │
  ├─ 0. ClaimWorthinessChecker ── rejects opinions/predictions/questions
  │                               before any search runs
  ├─ 1. QueryDecomposer ──── Claude generates 5 distinct sub-queries,
  │                          deliberately including refutation-seeking ones
  ├─ 2. SearchService ────── all sub-queries run concurrently, merged by URL
  ├─ 3. EvidenceRetriever ── pages fetched concurrently, split into passages,
  │                          ranked by embedding similarity, then reranked by
  │                          a cross-encoder for the final selection
  ├─ 4. StanceDetector ───── DeBERTa NLI per sentence, MC-dropout uncertainty
  ├─ 5. CredibilityScorer ── domain authority prior  (runs concurrently with 4)
  └─ 6. Synthesizer ─────── credibility-weighted verdict
  │
verdict + evidence + confidence + uncertainty
```

Full rationale for each stage is in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## The three decisions that matter

### DeBERTa NLI, not an LLM, for stance detection

Stance detection is a calibrated 3-way classification that runs dozens of times
per claim. A fine-tuned encoder is the right tool: it is faster, cheaper per
call, returns a probability distribution you can threshold and propagate, and
cannot hallucinate a citation. An LLM is the right tool for query decomposition —
open-ended generation, no labeled data, once per claim — which is exactly where
this system uses one.

The split is the point: **specialized model for the hot classification path,
general model for the cold generative path.**

### Credibility as a weight, not a filter

Most NLI fact-checkers treat source quality as a gate ("drop anything below
0.5"). That discards signal and behaves erratically at the boundary. Here
credibility multiplies evidence weight during aggregation:

```
weight_i = stance_confidence_i × credibility_i × recency_i
```

A PubMed paper refuting at 0.60 confidence (0.95 × 0.60 = **0.57**) outweighs a
blog supporting at 0.85 (0.35 × 0.85 = **0.30**). Averaging confidence alone
would return the opposite verdict. This case is asserted directly in
`tests/test_synthesizer.py`.

`recency_i` is an exponential half-life decay on publication date (5-year
half-life, floored at 0.3 so an old source is dampened, never discarded — an
old source can still be correct about a stable fact). Missing or unparseable
dates score 1.0, not a penalty. See `ARCHITECTURE.md` §7.4.

`credibility_i`'s high-credibility tiers are hand-curated, but the low end is
backed by [CRED-1](https://github.com/aloth/cred-1) (`data/cred1_compact.json`,
CC BY 4.0 — Alexander Loth, built from OpenSources.co and the Iffy.news Index),
an open dataset of ~2,700 known misinformation/fake/conspiracy domains. See
`ARCHITECTURE.md` §7.1 for why NewsGuard wasn't an option and why CRED-1 only
covers the low end.

### Sentence-level stance scoring

This one came out of measurement, not design. The NLI checkpoints are MNLI-trained,
where premises are single sentences. Feed them a real multi-sentence web passage
and they collapse to `NEUTRAL` with near-1.0 confidence even when one sentence
plainly decides the claim:

| Premise | Result |
|---|---|
| Full 2-sentence passage | `NEUTRAL` 0.999 |
| Same content, one sentence | `SUPPORTS` 0.964 |

So stance is scored per sentence and aggregated by **max over directional
sentences**: a source refutes a claim if *any* sentence in it refutes the claim,
even when the surrounding text is background. Averaging would let three neutral
sentences of context bury the one sentence carrying the evidence.

On the ten demo claims (measured pre-calibration, `STANCE_TEMPERATURE=1.0`,
zero-shot checkpoint) this moved agreement from 4/10 to 8/10. It also made
the system explainable — the deciding sentence is returned as
`stance_sentence` and displayed in the UI, so every badge is traceable to
specific text.

---

## Uncertainty

Raw softmax confidence is poorly calibrated — a model can be confidently
wrong — in two distinct ways, handled by two different mechanisms:

**Epistemic (does the model know?).** Each prediction runs `MC_SAMPLES`
forward passes with dropout left enabled, which approximates sampling from
the posterior over weights. The spread across those samples is reported as
`uncertainty`, separately from `confidence`. **"The model says neutral"** and
**"the model does not know"** are different statements, and a reader deserves
to see which one they got. Uncertainty discounts the final verdict
confidence, capped at 50% so model instability can never erase a genuine
evidence majority. `MC_SAMPLES` is the main latency dial in the system — cost
is linear in it.

**Calibration (when the model says 90%, is it right 90% of the time?).**
Separately, softmax confidence from an untuned model is systematically
overconfident regardless of MC dropout. `backend/training/calibrate_temperature.py`
fits a single softmax temperature (Guo et al. 2017) against a held-out
FEVER+VitaminC sample — a scalar that divides logits before softmax, which
softens confidence without changing which class wins. Fit on 2026-09-23
against the fine-tuned checkpoint (below), 1000 held-out examples:

| | Before | After |
|---|---|---|
| Expected Calibration Error | 0.049 | 0.021 |

For comparison, the zero-shot fallback checkpoint measured ECE 0.387 → 0.078
on the same test — fine-tuning made the model dramatically better calibrated
on its own, before any temperature scaling at all.

`STANCE_TEMPERATURE=1.49` is the current default (`.env`). This value is
specific to the current `STANCE_CHECKPOINT` and **must be recalibrated**
whenever that checkpoint is retrained or replaced — a different model has a
different confidence/accuracy relationship.

---

## Measured performance

**Stance detector accuracy on held-out FEVER+VitaminC** (the actual benchmark
number, not the demo regression check): fine-tuned DeBERTa-v3-base,
1000-example validation sample, measured 2026-09-23 —

| Metric | Value |
|---|---|
| Accuracy | **86.9%** |
| Macro F1 | **0.838** |
| F1, REFUTES | 0.870 |
| F1, NOT ENOUGH INFO | 0.717 |
| F1, SUPPORTS | 0.927 |

Trained on Northeastern's Explorer SLURM cluster (Tesla V100), stopped early
by `EarlyStoppingCallback` around epoch 1.2 of 3 once macro F1 stopped
improving. Full run details in `backend/training/train_stance.py` and
`ARCHITECTURE.md` §10.

CPU (M-series), offline corpus, `MC_SAMPLES=5`, fine-tuned checkpoint, cross-encoder reranking on:

| Metric | Value |
|---|---|
| Mean latency, 10 demo claims | **1.78 s** |
| p95 latency | **3.37 s** |
| Cached response | **< 1 ms** |
| Model load at startup | ~2 s (after first download) |
| Verdict agreement, 10 demo claims | **8/10** |

Mean latency was 0.95s before reranking was added (`RERANK_ENABLED=false`) —
the reranker's cost is a deliberate accuracy/latency tradeoff (§5.5 in
ARCHITECTURE.md), not a regression.

Stance stage in isolation, 30 passages:

| `MC_SAMPLES` | Time | Per passage |
|---|---|---|
| 10 | 7.2 s | 239 ms |
| 5 *(default)* | 3.7 s | 124 ms |
| 1 *(no uncertainty)* | 0.6 s | 19 ms |

Two caveats, stated plainly:

- **The 8/10 is not a benchmark** — the FEVER+VitaminC numbers above are. Ten
  demo claims, labels are the author's reading of the evidence, and the
  offline corpus was written by the author. It is a regression check against
  known cases, not an accuracy measurement.
- **Latency excludes live page fetching.** With `TAVILY_API_KEY` set, stage 3
  fetches real pages; that adds network time bounded by `FETCH_TIMEOUT_S` (8s
  default) and `FETCH_CONCURRENCY` (10), not by the number of sources.

---

## Reliability

Every stage degrades instead of failing:

| Failure | Behaviour |
|---|---|
| No search API key | Offline corpus, response flagged `mock_search: true` |
| No LLM key | Rule-based query decomposition |
| Search provider down | Returns `[]`; one bad sub-query cannot abort the request |
| Page fetch fails / times out | Falls back to the search snippet for that source |
| Stance model raises | `NEUTRAL` at uncertainty 1.0 — the synthesizer reads it as no signal |
| No usable evidence | `INSUFFICIENT` at confidence 0.0, never a guess |
| Claim is an opinion, prediction, or question | `UNCHECKABLE` at confidence 0.0 — rejected before any search runs |

Covered by `TestResilience` in `tests/test_integration.py`.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness, checkpoint, device, and which degraded modes are active |
| `POST /verify` | Verify a claim |
| `GET /cache/stats` | Hit rate, entries, evictions |
| `POST /cache/clear` | Flush the cache |

Interactive docs at `/docs`.

<details>
<summary><code>POST /verify</code> response schema</summary>

```jsonc
{
  "claim": "Vaccines cause autism",
  "verdict": "REFUTED",              // SUPPORTED | REFUTED | MIXED | INSUFFICIENT | UNCHECKABLE
  "confidence": 0.82,                // uncertainty-discounted
  "uncertainty": 0.14,               // mean MC-dropout spread
  "evidence": [
    {
      "source_url": "https://www.cdc.gov/...",
      "source_title": "Vaccines do not cause autism",
      "excerpt": "...",
      "stance": "REFUTES",           // SUPPORTS | REFUTES | NEUTRAL
      "stance_sentence": "Multiple large-scale studies...",  // what was scored
      "stance_confidence": 0.99,
      "uncertainty": 0.03,
      "credibility_score": 0.92,
      "publication_date": "2024-02-14"
    }
  ],
  "consensus_summary": "Credibility-weighted evidence contradicts the claim...",
  "processing_time_ms": 2477,
  "sources_analyzed": 4,
  "passages_analyzed": 4,
  "support_ratio": 0.11,             // these three sum to 1.0
  "refute_ratio": 0.89,
  "neutral_ratio": 0.00,
  "mock_search": false,
  "cached": false
}
```

Evidence is sorted by `stance_confidence × credibility_score` — the exact
quantity the synthesizer aggregated — so the ordering reflects what actually
drove the verdict.
</details>

---

## Configuration

All settings live in `backend/config.py` and are overridable via `.env`. The ones
worth knowing:

| Variable | Default | Effect |
|---|---|---|
| `MC_SAMPLES` | `5` | Uncertainty passes. Linear latency cost; `1` disables uncertainty |
| `SENTENCE_LEVEL_STANCE` | `true` | Per-sentence scoring (see above) |
| `UNRELATED_THRESHOLD` | `0.40` | Below this, a passage is dropped as `UNRELATED` |
| `FETCH_CONCURRENCY` | `10` | Parallel page fetches |
| `RERANK_ENABLED` | `true` | Cross-encoder reranking over bi-encoder candidates; `false` skips it |
| `STANCE_CHECKPOINT` | `./checkpoints/stance_deberta` | Fine-tuned weights; falls back to zero-shot when absent |
| `STANCE_TEMPERATURE` | `1.0` | Softmax temperature; see Calibration above. Recalibrate when `STANCE_CHECKPOINT` changes |
| `CACHE_TTL_S` | `3600` | Result cache lifetime |

---

## Fine-tuning

`backend/training/train_stance.py` fine-tunes DeBERTa-v3 on FEVER + VitaminC.
VitaminC matters specifically because it trains robustness to counterfactual
edits — *"aspirin reduces fever"* versus *"aspirin does not reduce fever"* — which
is the dominant failure mode of NLI models on real fact-checking. That failure
mode is case 7 in `data/test_cases.json`.

```bash
pip install datasets accelerate sentencepiece protobuf
python -m backend.training.train_stance
```

Drop the resulting checkpoint at `./checkpoints/stance_deberta` and the stance
detector picks it up automatically — including its label ordering, which is read
from the checkpoint's own `id2label` rather than assumed. Re-run
`backend/training/calibrate_temperature.py` afterward — `STANCE_TEMPERATURE`
is fit per checkpoint, and the current default is only valid for the specific
checkpoint currently in `./checkpoints/stance_deberta`.

The full run needs a GPU; on a SLURM cluster (e.g. Northeastern's
Explorer/Discovery), run `bash backend/training/setup_explorer_env.sh` once to
create the conda environment, then `sbatch backend/training/submit_explorer.sh`.

Before a long unattended run, smoke-test on a slice — this is also how three
dependency/API breaks and a silent float16-NaN-loss bug were caught before
they could burn hours of GPU time:

```bash
MAX_TRAIN_SAMPLES=200 MAX_VAL_SAMPLES=60 TRAIN_EPOCHS=1 EVAL_STEPS=3 \
  python -m backend.training.train_stance
```

**Run 2026-09-23** on Northeastern's Explorer SLURM cluster (Tesla V100):
86.9% accuracy / 0.838 macro F1 on 1000 held-out FEVER+VitaminC examples,
stopped early around epoch 1.2 of 3 — see Measured performance above and
`ARCHITECTURE.md` §10 for the full numbers and what a defensible benchmark
number would still require (published fact-checker rulings, not just FEVER).

---

## Project layout

```
backend/
  config.py               single source of truth; loads .env on import
  main.py                 FastAPI app, pipeline orchestration
  models/
    query_decomposer.py   claim -> sub-queries (LLM, rule-based fallback)
    retriever.py          concurrent fetch, passage split, semantic ranking
    stance_detector.py    DeBERTa NLI, MC dropout, sentence-level aggregation
    credibility_scorer.py domain authority tiers
    synthesizer.py        credibility-weighted aggregation
  services/
    search_service.py     Tavily, with offline fallback
    mock_corpus.py        offline corpus, both sides of every topic
    cache_service.py      TTL + LRU
  training/
    train_stance.py       FEVER + VitaminC fine-tuning
  utils/
    logging_config.py     request-id correlation
frontend/                 React + Vite
tests/                    113 tests
data/test_cases.json      10 curated claims
demo_script.py            scored report over those claims
```

---

## Known limitations

- **Zero-shot stance detection is the accuracy ceiling.** Sentence-level scoring
  mitigates the MNLI domain shift but does not remove it; some clear entailments
  still read as `NEUTRAL` when the claim and the source phrase a concept
  differently ("climate change" vs "the observed warming"). Fine-tuning on FEVER
  is the real fix.
- **Credibility scoring's high-credibility tiers are still a static domain
  list.** The low-credibility end is now backed by CRED-1, an open dataset
  of known misinformation domains (see `ARCHITECTURE.md` §7.1), but a real
  dataset for *reliable* sources (NewsGuard is paid-only) is still open. Either
  way, the scorer does not know that a given journal retracted a paper, or
  that a reputable outlet ran a bad story.
- **The cache is per-process.** Multi-worker deployment needs Redis; the async
  interface is already the right shape for that swap.
- **No claim-worthiness check.** Opinions and predictions are processed as if
  they were falsifiable factual claims.
