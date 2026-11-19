"""
End-to-end verdict accuracy against PolitiFact rulings from the LIAR dataset
(Wang, 2017), using live search. Download the data first; see data/liar/README.

LIAR labels map to verdicts symmetrically around "half-true":
  pants-fire, false      -> REFUTED
  barely-true, half-true -> MIXED
  mostly-true, true      -> SUPPORTED
UNCHECKABLE and INSUFFICIENT count as abstentions, not errors.

Each claim costs up to 6 Tavily searches.
"""

import argparse
import asyncio
import csv
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path

from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)

LIAR_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "liar"
MAX_CLAIM_LEN = 500  # ClaimRequest.claim max_length in backend/main.py

LABEL_TO_BUCKET = {
    "pants-fire": "REFUTED",
    "false": "REFUTED",
    "barely-true": "MIXED",
    "half-true": "MIXED",
    "mostly-true": "SUPPORTED",
    "true": "SUPPORTED",
}

# Verdicts scored against LABEL_TO_BUCKET. MIXED is scored like any other class.
DIRECTIONAL_VERDICTS = {"SUPPORTED", "REFUTED", "MIXED"}


def load_liar_split(split: str) -> list[dict]:
    path = LIAR_DIR / f"{split}.tsv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.reader(f, delimiter="\t")
        claims = []
        for row in rows:
            statement_id, label, statement = row[0], row[1], row[2]
            statement = " ".join(statement.split())
            if label not in LABEL_TO_BUCKET or len(statement) > MAX_CLAIM_LEN:
                continue
            claims.append({"id": statement_id, "label": label, "statement": statement})
        return claims


def stratified_sample(claims: list[dict], n: int, seed: int) -> list[dict]:
    """Sample about n/6 claims per label so small runs cover every label."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {}
    for c in claims:
        by_label.setdefault(c["label"], []).append(c)
    for bucket in by_label.values():
        rng.shuffle(bucket)

    per_label = max(1, n // len(by_label))
    sample = []
    for bucket in by_label.values():
        sample.extend(bucket[:per_label])
    rng.shuffle(sample)
    return sample[:n]


async def run_eval(client: TestClient, claims: list[dict], max_sources: int) -> list[dict]:
    results = []
    for i, c in enumerate(claims, 1):
        start = time.perf_counter()
        try:
            resp = client.post(
                "/verify",
                json={"claim": c["statement"], "max_sources": max_sources},
                timeout=120,
            )
            resp.raise_for_status()
            body = resp.json()
            result = {
                "id": c["id"],
                "label": c["label"],
                "expected_bucket": LABEL_TO_BUCKET[c["label"]],
                "statement": c["statement"],
                "verdict": body["verdict"],
                "confidence": body["confidence"],
                "sources_analyzed": body["sources_analyzed"],
                "passages_analyzed": body["passages_analyzed"],
                "processing_time_ms": body["processing_time_ms"],
                "mock_search": body["mock_search"],
            }
        except Exception as e:
            logger.error("Claim %s failed: %s", c["id"], e)
            result = {
                "id": c["id"], "label": c["label"],
                "expected_bucket": LABEL_TO_BUCKET[c["label"]],
                "statement": c["statement"], "verdict": "ERROR", "error": str(e),
            }
        elapsed = time.perf_counter() - start
        logger.info(
            "[%d/%d] %s -> %s (expected %s) in %.1fs",
            i, len(claims), c["label"], result["verdict"], result["expected_bucket"], elapsed,
        )
        results.append(result)
    return results


def summarize(results: list[dict]) -> dict:
    total = len(results)
    errors = [r for r in results if r["verdict"] == "ERROR"]
    unchecked = [r for r in results if r["verdict"] == "UNCHECKABLE"]
    insufficient = [r for r in results if r["verdict"] == "INSUFFICIENT"]
    directional = [r for r in results if r["verdict"] in DIRECTIONAL_VERDICTS]

    correct = sum(1 for r in directional if r["verdict"] == r["expected_bucket"])
    confusion = Counter((r["expected_bucket"], r["verdict"]) for r in directional)

    return {
        "total": total,
        "errors": len(errors),
        "unchecked": len(unchecked),
        "insufficient": len(insufficient),
        "directional": len(directional),
        "accuracy_on_directional": correct / len(directional) if directional else None,
        "coverage": len(directional) / total if total else None,
        "confusion_matrix": {f"{exp}->{pred}": n for (exp, pred), n in confusion.items()},
        "estimated_tavily_searches": total * 6,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20, help="number of claims to sample")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-sources", type=int, default=5)
    parser.add_argument("--out", default=None, help="path to write raw per-claim results JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Deferred so argparse and logging are set up before the app loads models.
    from backend.main import app

    claims = load_liar_split(args.split)
    sample = stratified_sample(claims, args.n, args.seed)
    logger.info(
        "Sampled %d/%d claims from %s split (seed=%d)",
        len(sample), len(claims), args.split, args.seed,
    )

    with TestClient(app) as client:
        health = client.get("/health").json()
        if not health.get("live_search"):
            logger.warning(
                "Pipeline appears to be running on the MOCK search corpus, not "
                "live Tavily -- this eval is meaningless without live search. "
                "Check TAVILY_API_KEY in .env."
            )
        results = asyncio.run(run_eval(client, sample, args.max_sources))

    summary = summarize(results)
    print(json.dumps(summary, indent=2))

    out_path = Path(args.out) if args.out else (
        LIAR_DIR / f"eval_results_{args.split}_n{args.n}_seed{args.seed}.json"
    )
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    logger.info("Wrote full results to %s", out_path)


if __name__ == "__main__":
    main()
