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
  ├─ 1. QueryDecomposer ──── Claude generates 5 distinct sub-queries,
  │                          deliberately including refutation-seeking ones
  ├─ 2. SearchService ────── all sub-queries run concurrently, merged by URL
  ├─ 3. EvidenceRetriever ── pages fetched concurrently, split into passages,
  │                          ranked against the claim by embedding similarity
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
weight_i = stance_confidence_i × credibility_i
```

A PubMed paper refuting at 0.60 confidence (0.95 × 0.60 = **0.57**) outweighs a
blog supporting at 0.85 (0.35 × 0.85 = **0.30**). Averaging confidence alone
would return the opposite verdict. This case is asserted directly in
`tests/test_synthesizer.py`.

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

On the ten demo claims this moved agreement from 4/10 to 8/10. It also made the
system explainable — the deciding sentence is returned as `stance_sentence` and
displayed in the UI, so every badge is traceable to specific text.

---

## Uncertainty

Softmax confidence is poorly calibrated; a model can be confidently wrong. Each
prediction runs `MC_SAMPLES` forward passes with dropout left enabled, which
approximates sampling from the posterior over weights. The spread across those
samples is reported as `uncertainty`, separately from `confidence`.

The distinction is real: **"the model says neutral"** and **"the model does not
know"** are different statements, and a reader deserves to see which one they got.
Uncertainty then discounts the final verdict confidence, capped at 50% so model
instability can never erase a genuine evidence majority.

`MC_SAMPLES` is the main latency dial in the system — cost is linear in it.

---

## Measured performance

CPU (M-series), offline corpus, `MC_SAMPLES=5`:

| Metric | Value |
|---|---|
| Mean latency, 10 demo claims | **1.48 s** |
| p95 latency | **3.16 s** |
| Cached response | **< 1 ms** |
| Model load at startup | ~2 s (after first download) |
| Verdict agreement, 10 demo claims | **8/10** |

Stance stage in isolation, 30 passages:

| `MC_SAMPLES` | Time | Per passage |
|---|---|---|
| 10 | 7.2 s | 239 ms |
| 5 *(default)* | 3.7 s | 124 ms |
| 1 *(no uncertainty)* | 0.6 s | 19 ms |

Two caveats, stated plainly:

- **The 8/10 is not a benchmark.** Ten claims, labels are the author's reading of
  the evidence, and the offline corpus was written by the author. It is a
  regression check, not an accuracy measurement. A real number requires FEVER or
  VitaminC evaluation — see below.
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
  "verdict": "REFUTED",              // SUPPORTED | REFUTED | MIXED | INSUFFICIENT
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
| `STANCE_CHECKPOINT` | `./checkpoints/stance_deberta` | Fine-tuned weights; falls back to zero-shot when absent |
| `CACHE_TTL_S` | `3600` | Result cache lifetime |

---

## Fine-tuning

`backend/training/train_stance.py` fine-tunes DeBERTa-v3 on FEVER + VitaminC.
VitaminC matters specifically because it trains robustness to counterfactual
edits — *"aspirin reduces fever"* versus *"aspirin does not reduce fever"* — which
is the dominant failure mode of NLI models on real fact-checking. That failure
mode is case 7 in `data/test_cases.json`.

```bash
pip install datasets accelerate
python -m backend.training.train_stance
```

Drop the resulting checkpoint at `./checkpoints/stance_deberta` and the stance
detector picks it up automatically — including its label ordering, which is read
from the checkpoint's own `id2label` rather than assumed.

**This has not been run yet.** No fine-tuned metrics are reported anywhere in this
repo because none have been measured.

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
- **Credibility scoring is a static domain list.** It does not know that a given
  journal retracted a paper, or that a reputable outlet ran a bad story.
- **The cache is per-process.** Multi-worker deployment needs Redis; the async
  interface is already the right shape for that swap.
- **No claim-worthiness check.** Opinions and predictions are processed as if
  they were falsifiable factual claims.
