"""
query_decomposer.py - LLM-based claim decomposition.

Why this stage exists:
  Searching the claim verbatim returns pages that echo the claim's framing, so
  the evidence set is biased toward whichever side phrased it. Decomposition
  generates semantically distinct sub-questions, including ones aimed at
  refuting evidence, which is what makes the verdict contestable rather than a
  confirmation of the query.

  The original implementation appended fixed strings ("+ research", "+ studies")
  -- that is string manipulation, and every variant retrieves near-identical
  results because the search engine's own semantics dominate.

Design decision worth defending:
  An LLM does decomposition, a fine-tuned discriminative model does stance.
  Decomposition is open-ended generation with no labeled training data and runs
  once per claim, so it is neither latency-critical nor a good fit for a small
  fine-tune. Stance detection is a calibrated 3-way classification run dozens of
  times per claim, where a fine-tuned encoder is both more accurate and roughly
  an order of magnitude cheaper per call.
"""

import asyncio
import json
import logging
import re

from backend.config import config

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """You are a fact-checking assistant. Given a claim, generate 5 distinct search queries that would help verify or refute it.

Rules:
- Each query targets a DIFFERENT aspect of the claim
- Queries should be concise (4-8 words)
- Include queries that could find both supporting AND refuting evidence
- Do not repeat the original claim verbatim
- Return ONLY a JSON array of strings, no other text

Claim: {claim}

JSON array:"""

MAX_QUERIES = 6


class QueryDecomposer:
    def __init__(self, api_key: str | None = None):
        key = config.ANTHROPIC_API_KEY if api_key is None else api_key
        self.client = None
        self.use_llm = False

        if not key:
            logger.warning("ANTHROPIC_API_KEY not set - using rule-based decomposition")
            return

        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=key)
            self.use_llm = True
            logger.info("Query decomposer ready (LLM: %s)", config.DECOMPOSER_MODEL)
        except Exception as e:
            logger.error("Anthropic init failed (%s) - using rule-based fallback", e)

    async def decompose(self, claim: str) -> list[str]:
        if self.use_llm:
            queries = await self._llm_decompose(claim)
            if queries:
                return queries
        return self._rule_based_fallback(claim)

    async def _llm_decompose(self, claim: str) -> list[str]:
        try:
            # The SDK client is synchronous; calling it directly would block the
            # event loop for the whole round trip.
            message = await asyncio.to_thread(
                self.client.messages.create,
                model=config.DECOMPOSER_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(claim=claim)}],
            )
            raw = message.content[0].text.strip()
            queries = _parse_json_array(raw)
            if not queries:
                logger.warning("Could not parse decomposition response: %r", raw[:200])
                return []

            # The claim itself anchors the search: sub-queries broaden coverage
            # but the literal claim is still the single best query for finding
            # sources that address it head-on.
            ordered = [claim]
            seen = {claim.strip().lower()}
            for query in queries:
                key = query.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    ordered.append(query.strip())

            logger.info("Decomposed into %d queries", min(len(ordered), MAX_QUERIES))
            return ordered[:MAX_QUERIES]

        except Exception as e:
            logger.warning("LLM decomposition failed (%s) - using fallback", e)
            return []

    def _rule_based_fallback(self, claim: str) -> list[str]:
        """
        Structured fallback. Weaker than LLM decomposition because the variants
        share most of their content words, but it deliberately includes opposing
        framings so the evidence set is not purely confirmatory.
        """
        stripped = claim.rstrip("?.").strip()
        return [
            claim,
            f"evidence supporting {stripped}",
            f"evidence against {stripped}",
            f"{stripped} systematic review",
            f"{stripped} debunked",
            f"{stripped} fact check",
        ][:MAX_QUERIES]


def _parse_json_array(text: str) -> list[str]:
    """
    Extract a JSON array of strings from a model response.

    Models wrap JSON in prose or fences often enough that strict json.loads on
    the raw text fails intermittently, which would drop us to the weaker
    fallback for no real reason.
    """
    candidates = [text]

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))

    bracketed = re.search(r"\[.*\]", text, re.DOTALL)
    if bracketed:
        candidates.append(bracketed.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate.strip())
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list):
            return [str(item) for item in parsed if isinstance(item, str) and item.strip()]

    return []
