"""
query_decomposer.py - LLM-based claim decomposition

Why this matters:
  The original implementation appended fixed strings ("+ research", "+ studies").
  That is string manipulation, not ML. A real decomposer generates semantically
  distinct sub-questions that cover different aspects of a claim, enabling
  more comprehensive evidence retrieval.

Approach:
  Use the Anthropic API to generate 4-6 targeted search queries from a claim.
  Fallback: rule-based expansion if API is unavailable.

Design decision (interview-ready answer):
  We use an LLM for decomposition rather than a fine-tuned T5 because:
  1. No labeled decomposition data exists for this exact task
  2. LLMs generalize well to novel claim types
  3. The decomposition step is not latency-critical (runs once per claim)
  The stance detection — where we need precise, calibrated predictions —
  uses a fine-tuned discriminative model (DeBERTa), which outperforms
  generative LLMs on classification tasks.
"""

import os
import json
import logging
import anthropic

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


class QueryDecomposer:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
            self.use_llm = True
            logger.info("✓ Query decomposer ready (LLM mode)")
        else:
            self.client = None
            self.use_llm = False
            logger.warning("ANTHROPIC_API_KEY not set — using rule-based fallback")

    async def decompose(self, claim: str) -> list[str]:
        if self.use_llm:
            return await self._llm_decompose(claim)
        return self._rule_based_fallback(claim)

    async def _llm_decompose(self, claim: str) -> list[str]:
        """
        Generate semantically distinct sub-queries using Claude.
        Parse the JSON array response; fall back on parse error.
        """
        try:
            message = self.client.messages.create(
                model="claude-haiku-4-5-20251001",   # fast + cheap for this task
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": DECOMPOSE_PROMPT.format(claim=claim)
                }]
            )

            response_text = message.content[0].text.strip()

            # Strip markdown fences if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            queries = json.loads(response_text)
            # Always include the original claim as anchor
            queries = [claim] + [q for q in queries if q != claim]
            return queries[:6]

        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e} — using fallback")
            return self._rule_based_fallback(claim)

    def _rule_based_fallback(self, claim: str) -> list[str]:
        """
        Structured fallback. Better than the original (no semantic overlap)
        but clearly labeled as a fallback in code.
        """
        return [
            claim,
            f"evidence for: {claim}",
            f"evidence against: {claim}",
            f"{claim} scientific studies",
            f"{claim} expert opinion",
            f"{claim} fact check",
        ]
