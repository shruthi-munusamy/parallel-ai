# Parallel AI

A fact-checking system that automates *lateral reading*, the method professional fact-checkers use. Given a claim, it searches for what independent sources say about it, determines the stance each source takes, weights those sources by credibility, and returns a verdict that is traceable to the specific sentences it was derived from.

![Parallel AI demo](docs/demo.gif)

Verdicts are derived only from retrieved text. The LLM generates search queries and never produces the verdict. If the system finds no usable evidence, it returns `INSUFFICIENT` rather than guessing.

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

## Results

| | Result |
|---|---|
| Stance model accuracy, held-out FEVER + VitaminC (1,000 examples) | **86.9%** accuracy, **0.838** macro-F1 |
| Calibration (expected calibration error) | 0.049 → **0.021** after temperature scaling |
| End-to-end accuracy on LIAR claims given a verdict (36 claims, rule-based query decomposition) | 27% → **63%** after fixing a prediction bias, with coverage 61% → 44% |
| Mean latency, 10 demo claims (CPU, offline corpus) | **1.78 s** (p95 3.37 s) |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # optional: add API keys
python -m backend.main        # http://localhost:8000/docs
```

The API keys are optional. Without them, the system runs end to end on a built-in offline corpus and labels every response `mock_search: true`, so offline results cannot be mistaken for live ones.

| Key | Effect when absent |
|---|---|
| `TAVILY_API_KEY` | Falls back to the offline corpus in `backend/services/mock_corpus.py` |
| `ANTHROPIC_API_KEY` | Falls back to rule-based query decomposition |

The first run downloads about 750 MB of model weights and takes a few minutes. Subsequent startups take about 2 seconds.

**Demo.** Runs ten curated claims and prints a scored report:

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
python -m pytest                  # 177 tests
python -m pytest -m "not slow"    # skip the tests that load model weights
```

## How it works

```
claim
  │
  ├─ 0. ClaimWorthinessChecker ── rejects opinions, predictions, and questions
  │                               before any search runs
  ├─ 1. QueryDecomposer ──── Claude writes 5 distinct sub-queries,
  │                          including ones that look for refutations
  ├─ 2. SearchService ────── runs all sub-queries concurrently, merges by URL
  ├─ 3. EvidenceRetriever ── fetches pages concurrently, splits them into passages,
  │                          ranks by embedding similarity, then reranks
  │                          with a cross-encoder
  ├─ 4. StanceDetector ───── DeBERTa NLI per sentence, MC-dropout uncertainty
  ├─ 5. CredibilityScorer ── domain credibility score (runs concurrently with 4)
  └─ 6. Synthesizer ─────── credibility-weighted verdict
  │
verdict + evidence + confidence + uncertainty
```

The rationale for each stage is documented in [ARCHITECTURE.md](ARCHITECTURE.md).

## Issues and fixes

**Multi-sentence passages were classified as neutral.** The NLI models are trained on MNLI, where each premise is a single sentence. Given a multi-sentence web passage, they returned `NEUTRAL` with near-total confidence even when one sentence clearly determined the claim:

| Premise | Result |
|---|---|
| Full 2-sentence passage | `NEUTRAL` 0.999 |
| Same content, one sentence | `SUPPORTS` 0.964 |

I changed the stance detector to score each sentence separately and take the strongest directional sentence as the passage's stance. Averaging was rejected because neutral background sentences would outweigh the single sentence that carries the evidence. On the ten demo claims, before fine-tuning or calibration, this change raised agreement from 4/10 to 8/10. It also makes every verdict explainable, because the deciding sentence is returned as `stance_sentence` and displayed in the UI.

**The full pipeline was biased toward one verdict.** The stance model performed well on its own benchmark, but evaluating the complete system on 36 claims from the LIAR benchmark revealed a strong bias toward one verdict: only 27% of the claims that received a verdict were classified correctly (6 of 22). The cause was that articles debunking a false claim also restate it, and sentence-level scoring read those restatements as support; the fix reserves retrieval slots for passages containing fact-check ruling language and excludes sentences that attribute or quote the claim from the stance vote. After these changes, accuracy on claims given a verdict rose to 63% (10 of 16). Coverage fell from 61% to 44%, because borderline claims that previously received a confident but incorrect verdict now return `INSUFFICIENT`. Both runs used rule-based query decomposition. To reproduce the evaluation, download LIAR using the instructions in `data/liar/README`.

**Training failures detected before cluster runs.** Before every long cluster run, I train for one epoch on a small subset of the data. This check detected three dependency and API incompatibilities and a silent float16 error that produced NaN loss, each before it could consume hours of cluster time:

```bash
MAX_TRAIN_SAMPLES=200 MAX_VAL_SAMPLES=60 TRAIN_EPOCHS=1 EVAL_STEPS=3 \
  python -m backend.training.train_stance
```

## Design decisions

**A fine-tuned encoder for stance, an LLM for queries.** Stance detection is a 3-way classification that runs dozens of times per claim. A fine-tuned encoder is faster and cheaper per call, returns a probability distribution that can be thresholded and calibrated, and cannot fabricate a citation. Query decomposition is open-ended generation that runs once per claim and has no labeled data, which makes it a suitable task for an LLM.

**Credibility as a weight, not a filter.** Many fact-checking systems discard sources below a credibility cutoff. This discards evidence and produces unstable results near the cutoff. In this system, credibility multiplies each passage's weight instead:

```
weight_i = stance_confidence_i × credibility_i × recency_i
```

A PubMed paper refuting at 0.60 confidence (0.95 × 0.60 = **0.57**) outweighs a blog supporting at 0.85 (0.35 × 0.85 = **0.30**). Averaging confidence alone would return the opposite verdict. `tests/test_synthesizer.py` tests this case directly.

`recency_i` decays with a 5-year half-life and never drops below 0.3. An older source is therefore down-weighted but never discarded, because it can still be accurate about a stable fact. Missing or unreadable dates score 1.0 rather than being penalized.

## Uncertainty and calibration

**Model uncertainty.** Each prediction runs `MC_SAMPLES` forward passes with dropout enabled, and the spread across them is reported as `uncertainty`, separately from `confidence`. A neutral prediction and an uncertain prediction are different outcomes, and the response distinguishes between them. Uncertainty reduces the final verdict confidence by at most 50%, so model instability cannot override a clear majority of evidence. `MC_SAMPLES` is the main latency setting, and its cost grows linearly.

**Calibration.** Softmax confidence tends to be overconfident. `backend/training/calibrate_temperature.py` fits a single temperature (Guo et al., 2017) on 1,000 held-out FEVER + VitaminC examples. Dividing the logits by this temperature reduces overconfidence without changing the predicted class.

| | Before | After |
|---|---|---|
| Expected calibration error | 0.049 | 0.021 |

The zero-shot checkpoint measured 0.387 → 0.078 on the same test, so fine-tuning alone substantially improved calibration before any temperature scaling. The fitted value, `STANCE_TEMPERATURE=1.49`, is set in `.env` and applies only to the current checkpoint. It must be refitted whenever the checkpoint changes.

## Measured performance

**Stance model**, fine-tuned DeBERTa-v3-base, 1,000 held-out FEVER + VitaminC examples:

| Metric | Value |
|---|---|
| Accuracy | **86.9%** |
| Macro-F1 | **0.838** |
| F1, REFUTES | 0.870 |
| F1, NOT ENOUGH INFO | 0.717 |
| F1, SUPPORTS | 0.927 |

Trained on Northeastern's Explorer SLURM cluster (Tesla V100). Early stopping ended training around epoch 1.2 of 3, when macro-F1 stopped improving.

**Pipeline latency** on CPU (M-series), offline corpus, `MC_SAMPLES=5`, fine-tuned checkpoint, reranking on:

| Metric | Value |
|---|---|
| Mean latency, 10 demo claims | **1.78 s** |
| p95 latency | **3.37 s** |
| Cached response | **< 1 ms** |
| Model load at startup | ~2 s (after first download) |
| Verdict agreement, 10 demo claims | **8/10** |

Without reranking (`RERANK_ENABLED=false`), mean latency is 0.95 s. Verdict agreement on the 10 demo claims is 8/10 with or without the reranker. I kept it for the quality of the retrieved passages, an effect this sample is too small to measure.

**Stance stage alone**, 30 passages:

| `MC_SAMPLES` | Time | Per passage |
|---|---|---|
| 10 | 7.2 s | 239 ms |
| 5 *(default)* | 3.7 s | 124 ms |
| 1 *(no uncertainty)* | 0.6 s | 19 ms |

Two caveats:

- **The 8/10 is a regression check, not a benchmark.** I wrote the ten demo claims, their labels, and the offline corpus. The FEVER + VitaminC and LIAR results are the independent measurements.
- **Latency excludes live page fetching.** With `TAVILY_API_KEY` set, stage 3 fetches real pages, which adds network time bounded by `FETCH_TIMEOUT_S` (8 s default) and `FETCH_CONCURRENCY` (10).

## Reliability

Each stage degrades rather than failing:

| Failure | Behavior |
|---|---|
| No search API key | Offline corpus, response flagged `mock_search: true` |
| No LLM key | Rule-based query decomposition |
| Search provider down | Returns `[]`, so one bad sub-query cannot abort the request |
| Page fetch fails or times out | Falls back to the search snippet for that source |
| Stance model raises an error | `NEUTRAL` at uncertainty 1.0, which the synthesizer treats as no signal |
| No usable evidence | `INSUFFICIENT` at confidence 0.0, never a guess |
| Claim is an opinion, prediction, or question | `UNCHECKABLE` at confidence 0.0, rejected before any search runs |

`TestResilience` in `tests/test_integration.py` covers each case.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness, checkpoint, device, and which fallback modes are active |
| `POST /verify` | Verify a claim |
| `GET /cache/stats` | Hit rate, entries, evictions |
| `POST /cache/clear` | Clear the cache |

Interactive API documentation is served at `/docs`.

<details>
<summary><code>POST /verify</code> response schema</summary>

```jsonc
{
  "claim": "Vaccines cause autism",
  "verdict": "REFUTED",              // SUPPORTED | REFUTED | MIXED | INSUFFICIENT | UNCHECKABLE
  "confidence": 0.82,                // after the uncertainty discount
  "uncertainty": 0.14,               // mean MC-dropout spread
  "evidence": [
    {
      "source_url": "https://www.cdc.gov/...",
      "source_title": "Vaccines do not cause autism",
      "excerpt": "...",
      "stance": "REFUTES",           // SUPPORTS | REFUTES | NEUTRAL
      "stance_sentence": "Multiple large-scale studies...",  // the sentence that was scored
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

Evidence is sorted by `stance_confidence × credibility_score × recency`, the same weight the synthesizer uses, so the order reflects each passage's contribution to the verdict.

</details>

## Configuration

All settings are defined in `backend/config.py` and can be overridden in `.env`. The most commonly adjusted settings are:

| Variable | Default | Effect |
|---|---|---|
| `MC_SAMPLES` | `5` | Uncertainty passes. Latency grows linearly; `1` disables uncertainty |
| `SENTENCE_LEVEL_STANCE` | `true` | Per-sentence scoring (see above) |
| `UNRELATED_THRESHOLD` | `0.40` | Passages below this are dropped as `UNRELATED` |
| `FETCH_CONCURRENCY` | `10` | Parallel page fetches |
| `RERANK_ENABLED` | `true` | Cross-encoder reranking; `false` skips it |
| `STANCE_CHECKPOINT` | `./checkpoints/stance_deberta` | Fine-tuned weights; falls back to zero-shot when absent |
| `STANCE_TEMPERATURE` | `1.0` in code, `1.49` in `.env` | Softmax temperature (see Calibration). Recalibrate when the checkpoint changes |
| `CACHE_TTL_S` | `3600` | Result cache lifetime |

## Fine-tuning

`backend/training/train_stance.py` fine-tunes DeBERTa-v3 on FEVER + VitaminC. VitaminC is included because it trains the model on minimal edits such as *"aspirin reduces fever"* vs. *"aspirin does not reduce fever"*, which are the most common failure mode for NLI models in real fact-checking. Case 7 in `data/test_cases.json` tests this failure mode.

```bash
pip install datasets accelerate sentencepiece protobuf
python -m backend.training.train_stance
```

When the resulting checkpoint is placed at `./checkpoints/stance_deberta`, the stance detector loads it automatically and reads the label order from the checkpoint's own `id2label` rather than assuming a fixed order. After changing the checkpoint, rerun `backend/training/calibrate_temperature.py`, because the temperature is fitted per checkpoint.

The full run requires a GPU. On a SLURM cluster such as Northeastern's Explorer, run `bash backend/training/setup_explorer_env.sh` once to create the conda environment, set your email in `backend/training/submit_explorer.sh`, and then run `sbatch backend/training/submit_explorer.sh`.

## Project layout

```
backend/
  config.py                   settings, loaded from .env on import
  main.py                     FastAPI app and pipeline orchestration
  models/
    claim_worthiness.py       rejects opinions, predictions, and questions
    query_decomposer.py       claim -> sub-queries (LLM, rule-based fallback)
    retriever.py              concurrent fetch, passage splitting, semantic ranking
    stance_detector.py        DeBERTa NLI, MC dropout, sentence-level aggregation
    credibility_scorer.py     domain credibility (tiers, CRED-1, WP:RSP, TLD)
    synthesizer.py            credibility-weighted aggregation
  services/
    search_service.py         Tavily, with offline fallback
    mock_corpus.py            offline corpus covering both sides of every topic
    cache_service.py          TTL + LRU cache
  training/
    train_stance.py           FEVER + VitaminC fine-tuning
    calibrate_temperature.py  temperature scaling on held-out examples
    setup_explorer_env.sh     one-time conda setup on a SLURM cluster
    submit_explorer.sh        SLURM job for the full training run
  eval/
    eval_liar.py              end-to-end eval against LIAR (PolitiFact rulings)
  utils/
    logging_config.py         request-ID logging
frontend/                     React + Vite
tests/                        177 tests
data/
  test_cases.json             10 curated claims
  cred1_compact.json          CRED-1 domain credibility data
  rsp_compact.json            WP:RSP domain reliability data
  liar/                       LIAR eval results; TSV download steps in its README
demo_script.py                scored report over the curated claims
```

## Known limitations

- **Phrasing mismatches.** Sentence-level scoring reduces the MNLI domain gap but does not eliminate it. Some clear entailments are still classified as `NEUTRAL` when the claim and the source express the same idea in different terms ("climate change" vs. "the observed warming").
- **Credibility is a static domain list.** Scores come from curated tiers, CRED-1, and WP:RSP, with a TLD heuristic as the fallback. A domain-level score cannot reflect that a journal retracted a paper or that a reputable outlet published an inaccurate story.
- **The cache is per-process.** Running multiple workers would require a shared store such as Redis. The cache's async interface is designed to support that replacement.
- **Small end-to-end test sets.** The LIAR evaluation covers 36 claims and the demo set covers 10, so the end-to-end numbers are indicative rather than conclusive. A stronger benchmark would use a larger set of published fact-checker rulings.

## Credits

- Domain credibility data from [CRED-1](https://github.com/aloth/cred-1) by Alexander Loth (CC BY 4.0), built from OpenSources.co (CC BY 4.0) by Melissa Zimdars et al. and the Iffy.news Index (MIT).
- Domain reliability data derived from Wikipedia's [Reliable sources/Perennial sources](https://en.wikipedia.org/wiki/Wikipedia:Reliable_sources/Perennial_sources) (WP:RSP), text available under CC BY-SA 4.0.
