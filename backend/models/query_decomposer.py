"""
Claim decomposition into search queries.

Searching the claim verbatim returns pages that echo its framing, so an LLM
generates distinct sub-queries, including ones aimed at refuting evidence.
Falls back to fixed templates without an API key.
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

            # The literal claim stays first; it is still the best single query.
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
        """Template queries, including opposing framings."""
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
    """Extract a JSON array of strings, tolerating code fences and surrounding prose."""
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
