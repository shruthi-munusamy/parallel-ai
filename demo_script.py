#!/usr/bin/env python3
"""
demo_script.py - Run the verification pipeline over a set of claims and print
a readable report.

    python demo_script.py                      # all cases in data/test_cases.json
    python demo_script.py --limit 3            # first three
    python demo_script.py --claim "..."        # a single ad-hoc claim
    python demo_script.py --sources 12         # widen the evidence set

Runs in-process: it calls the pipeline directly rather than going through HTTP,
so no server needs to be running. Without API keys it uses the offline corpus
and says so, because a demo that silently shows fabricated evidence as though
it were live is worse than no demo.

Accuracy is reported against the expected verdicts in data/test_cases.json.
Those labels are the author's reading of the evidence, not a gold-standard
benchmark, and the sample is small -- the number is a sanity check, not a
publishable metric.
"""

import argparse
import asyncio
import json
import pathlib
import sys
import time

from backend.config import config
from backend.utils.logging_config import configure_logging

TEST_CASES = pathlib.Path(__file__).parent / "data" / "test_cases.json"

# ── Terminal formatting ──────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
GREY = "\033[90m"

VERDICT_COLOR = {
    "SUPPORTED": GREEN,
    "REFUTED": RED,
    "MIXED": YELLOW,
    "INSUFFICIENT": GREY,
}
STANCE_COLOR = {"SUPPORTS": GREEN, "REFUTES": RED, "NEUTRAL": YELLOW, "UNRELATED": GREY}

WIDTH = 78


def supports_color() -> bool:
    return sys.stdout.isatty()


def c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if supports_color() else text


def bar(value: float, width: int = 24) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def rule(char: str = "─") -> str:
    return char * WIDTH


# ── Report rendering ─────────────────────────────────────────────────────────


def print_result(index: int, total: int, case: dict, result: dict, elapsed: float) -> None:
    verdict = result["verdict"]
    color = VERDICT_COLOR.get(verdict, GREY)
    expected = case.get("expected_verdict")

    print()
    print(c(rule("═"), DIM))
    print(f"{c(f'[{index}/{total}]', DIM)} {c(case['claim'], BOLD)}")
    print(c(rule("═"), DIM))

    verdict_line = f"  {c(verdict, color + BOLD)}"
    if expected:
        ok = verdict == expected
        mark = c("✓ matches expected", GREEN) if ok else c(f"✗ expected {expected}", RED)
        verdict_line += f"   {mark}"
    print(verdict_line)

    print(f"  confidence   {bar(result['confidence'])} {result['confidence'] * 100:5.1f}%")
    print(f"  uncertainty  {bar(result['uncertainty'])} {result['uncertainty'] * 100:5.1f}%")

    print()
    print(
        "  evidence mix "
        + c(f"support {result['support_ratio'] * 100:.0f}%", GREEN)
        + " · "
        + c(f"neutral {result['neutral_ratio'] * 100:.0f}%", YELLOW)
        + " · "
        + c(f"refute {result['refute_ratio'] * 100:.0f}%", RED)
    )
    print(f"  {c(result['consensus_summary'], DIM)}")

    evidence = result["evidence"][:3]
    if evidence:
        print()
        print(c(f"  TOP EVIDENCE ({len(result['evidence'])} passages total)", BOLD))
        for item in evidence:
            stance_color = STANCE_COLOR.get(item["stance"], GREY)
            url = item["source_url"]
            host = url.split("/")[2] if "//" in url else url
            stance = c(f"{item['stance']:<9}", stance_color)
            print()
            print(
                f"    {stance}  conf {item['stance_confidence']:.2f}"
                f"  cred {item['credibility_score']:.2f}  {c(host, BLUE)}"
            )
            excerpt = item["excerpt"][:220].replace("\n", " ")
            print(f"      {c(excerpt, DIM)}")

    print()
    footer = (
        f"{elapsed:.2f}s · {result['sources_analyzed']} sources "
        f"· {result['passages_analyzed']} passages"
    )
    print(f"  {c(footer, DIM)}")


def print_summary(records: list[dict], total_elapsed: float) -> None:
    print()
    print(c(rule("═"), BOLD))
    print(c("  SUMMARY", BOLD))
    print(c(rule("═"), BOLD))

    scored = [r for r in records if r["expected"]]
    if scored:
        correct = sum(1 for r in scored if r["verdict"] == r["expected"])
        pct = correct / len(scored) * 100
        print(f"  Verdict agreement  {correct}/{len(scored)} ({pct:.0f}%) vs expected labels")

    latencies = sorted(r["elapsed"] for r in records)
    if latencies:
        mean = sum(latencies) / len(latencies)
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(f"  Latency            mean {mean:.2f}s · p95 {p95:.2f}s · total {total_elapsed:.1f}s")

    print()
    for r in records:
        color = VERDICT_COLOR.get(r["verdict"], GREY)
        mark = " "
        if r["expected"]:
            mark = c("✓", GREEN) if r["verdict"] == r["expected"] else c("✗", RED)
        claim = r["claim"][:44] + ("…" if len(r["claim"]) > 44 else "")
        verdict = c(f"{r['verdict']:<13}", color)
        print(f"   {mark} {claim:<46} {verdict} {r['elapsed']:5.2f}s")

    if scored:
        print()
        print(c("  Expected verdicts are the author's reading of the evidence on a", DIM))
        print(c("  10-claim sample, not a gold-standard benchmark.", DIM))


# ── Main ─────────────────────────────────────────────────────────────────────


async def run(claims: list[dict], max_sources: int) -> int:
    # Imported here so --help stays instant instead of loading torch first.
    from backend.main import run_pipeline, pipeline
    from backend.models.query_decomposer import QueryDecomposer
    from backend.models.retriever import EvidenceRetriever
    from backend.models.stance_detector import StanceDetector
    from backend.models.credibility_scorer import CredibilityScorer
    from backend.models.synthesizer import EvidenceSynthesizer
    from backend.services.search_service import SearchService
    from backend.services.cache_service import CacheService

    print(c("\n  Neural Claim Verification — demo", BOLD))
    print(c("  Loading models (first run downloads ~750MB)...", DIM))

    load_start = time.perf_counter()

    def build():
        return {
            "decomposer": QueryDecomposer(),
            "retriever": EvidenceRetriever(),
            "stance": StanceDetector(),
            "credibility": CredibilityScorer(),
            "synthesizer": EvidenceSynthesizer(),
            "search": SearchService(),
            "cache": CacheService(),
        }

    pipeline.update(await asyncio.to_thread(build))
    print(c(f"  Ready in {time.perf_counter() - load_start:.1f}s", DIM))

    stance = pipeline["stance"]
    search = pipeline["search"]
    print(
        c(
            f"  stance={stance.checkpoint} · device={stance.device} · "
            f"MC samples={stance.mc_samples}",
            DIM,
        )
    )
    if search.is_mock:
        print(
            c(
                "\n  ⚠ TAVILY_API_KEY not set — evidence comes from the offline "
                "corpus,\n    not live web search. Set it in .env for real results.",
                YELLOW,
            )
        )
    if not pipeline["decomposer"].use_llm:
        print(c("  ⚠ ANTHROPIC_API_KEY not set — rule-based query decomposition.", YELLOW))

    records = []
    suite_start = time.perf_counter()

    for i, case in enumerate(claims, 1):
        start = time.perf_counter()
        try:
            result = await run_pipeline(case["claim"], max_sources)
        except Exception as e:
            print(c(f"\n  [{i}/{len(claims)}] {case['claim']}\n    FAILED: {e}", RED))
            continue
        elapsed = time.perf_counter() - start

        # run_pipeline returns Evidence models; normalize for printing.
        result = {**result, "evidence": [e.model_dump() for e in result["evidence"]]}

        print_result(i, len(claims), case, result, elapsed)
        records.append(
            {
                "claim": case["claim"],
                "verdict": result["verdict"],
                "expected": case.get("expected_verdict"),
                "elapsed": elapsed,
            }
        )

    print_summary(records, time.perf_counter() - suite_start)

    scored = [r for r in records if r["expected"]]
    if scored and not search.is_mock:
        correct = sum(1 for r in scored if r["verdict"] == r["expected"])
        # Non-zero exit on poor agreement makes this usable as a CI gate.
        return 0 if correct / len(scored) >= 0.5 else 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo the claim verification pipeline.")
    parser.add_argument("--claim", help="Verify a single ad-hoc claim instead of the test set.")
    parser.add_argument("--limit", type=int, help="Only run the first N test cases.")
    parser.add_argument("--sources", type=int, default=10, help="Max sources per claim (default 10).")
    parser.add_argument("--quiet", action="store_true", help="Suppress pipeline logging.")
    args = parser.parse_args()

    configure_logging("WARNING" if args.quiet else config.LOG_LEVEL)

    if args.claim:
        claims = [{"claim": args.claim}]
    else:
        if not TEST_CASES.exists():
            print(f"Missing test cases file: {TEST_CASES}", file=sys.stderr)
            return 1
        with open(TEST_CASES) as f:
            claims = json.load(f)
        if args.limit:
            claims = claims[: args.limit]

    return asyncio.run(run(claims, args.sources))


if __name__ == "__main__":
    sys.exit(main())
