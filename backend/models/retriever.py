"""
retriever.py - Passage extraction and semantic ranking.

Three stages per source:
  1. Fetch the page and strip it to readable text.
  2. Split into passages and rank them against the claim by embedding cosine
     similarity (bi-encoder), keeping a wider candidate pool than needed.
  3. Rerank that pool with a cross-encoder, keeping the final top-k.

Why semantic ranking rather than keyword matching: a claim about "intermittent
fasting" must match a passage discussing "time-restricted eating", which shares
no content words. Cosine similarity over sentence embeddings captures that;
BM25 or substring matching does not.

Why rerank at all when the bi-encoder already ranks by similarity: a
bi-encoder embeds the claim and each passage independently, so it can only
compare their embeddings after the fact -- it never lets the claim and
passage attend to each other. A cross-encoder scores them jointly through
one transformer pass, which is more accurate but too slow to run over every
passage from every source. Two stages get both: the bi-encoder's cheap recall
over everything, then the cross-encoder's expensive precision over its
pre-filtered pool.

Why fetch the page at all when search already returns a snippet: snippets are
one or two sentences chosen for relevance to the *query*, not for stance. The
sentence that actually contradicts a claim is usually deeper in the article.
Snippet-only retrieval is the degraded path, not the intended one.
"""

import asyncio
import logging
import re

import aiohttp
import torch
from bs4 import BeautifulSoup
from sentence_transformers import CrossEncoder, SentenceTransformer, util

from backend.config import config

logger = logging.getLogger(__name__)

# A fact-check's own ruling ("we rate this claim False", "contrary to some
# reports") shares almost no vocabulary with the claim it rules on, so claim-
# similarity ranking -- the only signal _rank_passages has -- systematically
# scores it *below* a passage that merely restates the claim, which is
# maximally similar to the claim by construction. Confirmed empirically via
# the end-to-end LIAR eval (ARCHITECTURE.md §12): for one claim, all 9
# retrieved passages across 5 real PolitiFact/NJ.com fact-check pages
# contained only restatements of the claim -- not one verdict sentence from
# those same pages was ever retrieved. This is a narrow keyword signal for
# "this passage likely states a conclusion," used only to reserve a couple of
# slots in the ranked pool (see VERDICT_SLOTS) -- it never overrides
# similarity ranking outright, and a passage still has to clear a minimum
# relevance floor (VERDICT_MIN_RELEVANCE) to qualify.
_VERDICT_MARKERS = re.compile(
    r"\b(we rate|rated|rating|pants[- ]on[- ]fire|half[- ]true|mostly[- ]true|"
    r"mostly[- ]false|debunks?|debunked|debunking|misleading|inaccurate|"
    r"unsubstantiated|no evidence|our ruling|fact[- ]check(?:ed|ers?)?|"
    r"hoax|fabricated|contrary to)\b",
    re.IGNORECASE,
)


def _has_verdict_language(text: str) -> bool:
    return bool(_VERDICT_MARKERS.search(text))

# Sent on fetches: a default aiohttp agent string is rejected outright by a
# meaningful share of publishers, which silently degraded us to snippet-only.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Structural elements whose text is never evidence.
_STRIP_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form", "noscript")


class EvidenceRetriever:
    def __init__(self, model_name: str | None = None):
        model_name = model_name or config.EMBEDDINGS_MODEL
        logger.info("Loading retriever embeddings: %s", model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(model_name, device=self.device)

        # Reranker is optional: a load failure degrades to bi-encoder-only
        # ranking rather than taking down retrieval entirely.
        self.reranker: CrossEncoder | None = None
        if config.RERANK_ENABLED:
            try:
                self.reranker = CrossEncoder(config.RERANKER_MODEL, device=self.device)
            except Exception as e:
                logger.warning(
                    "Reranker %s failed to load, falling back to bi-encoder-only "
                    "ranking: %s", config.RERANKER_MODEL, e,
                )

        logger.info(
            "Retriever ready on %s | reranker=%s",
            self.device, config.RERANKER_MODEL if self.reranker else "disabled",
        )

    async def extract_evidence(
        self,
        claim: str,
        sources: list[dict],
        top_k: int | None = None,
        fetch_pages: bool = True,
    ) -> list[dict]:
        """
        Returns a flat list of ranked passages, each carrying its source metadata.

        Fetches run concurrently. Sequential fetching was the dominant term in
        end-to-end latency: 10 sources x up to an 8s timeout is 80s worst case,
        versus 8s when they overlap.
        """
        if not sources:
            return []

        top_k = top_k or config.PASSAGES_PER_SOURCE

        if fetch_pages:
            pages = await self._fetch_all([s.get("url", "") for s in sources])
        else:
            # Mock mode: URLs are synthetic, so skip the network entirely and
            # rank over snippets instead of burning a timeout per source.
            pages = ["" for _ in sources]

        passages: list[dict] = []
        for source, content in zip(sources, pages):
            meta = {
                "source_url": source.get("url", ""),
                "source_title": source.get("title") or "Untitled",
                "date": source.get("date"),
            }
            snippet = source.get("snippet", "") or ""

            if not content:
                # Degraded path: the search snippet is all we have.
                if snippet:
                    passages.append({**meta, "text": snippet, "excerpt": _excerpt(snippet)})
                continue

            candidates = self._split_into_passages(content)
            pool_size = top_k * config.RERANK_POOL_MULTIPLIER if self.reranker else top_k
            pool = self._rank_passages(claim, candidates, pool_size)
            ranked = self._rerank_passages(claim, pool, top_k) if self.reranker else pool
            if not ranked and snippet:
                ranked = [{"text": snippet, "excerpt": _excerpt(snippet)}]
            for passage in ranked:
                passages.append({**meta, **passage})

        logger.info("Extracted %d passages from %d sources", len(passages), len(sources))
        return passages

    # ── Fetching ─────────────────────────────────────────────────────────────

    async def _fetch_all(self, urls: list[str]) -> list[str]:
        """
        Fetch every URL concurrently, bounded by a semaphore.

        The bound exists because unbounded fan-out across many sources opens as
        many sockets at once and gets us rate-limited or throttled; a single
        shared ClientSession also reuses connections and the DNS cache.
        """
        semaphore = asyncio.Semaphore(config.FETCH_CONCURRENCY)
        timeout = aiohttp.ClientTimeout(total=config.FETCH_TIMEOUT_S)

        async with aiohttp.ClientSession(timeout=timeout, headers=_HEADERS) as session:

            async def fetch_one(url: str) -> str:
                async with semaphore:
                    return await self._fetch_page(session, url)

            results = await asyncio.gather(
                *(fetch_one(u) for u in urls), return_exceptions=True
            )

        pages: list[str] = []
        for url, result in zip(urls, results):
            if isinstance(result, BaseException):
                logger.debug("Fetch failed for %s: %s", url, result)
                pages.append("")
            else:
                pages.append(result)

        succeeded = sum(1 for p in pages if p)
        logger.info("Fetched %d/%d pages", succeeded, len(urls))
        return pages

    async def _fetch_page(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch and strip one page. Returns "" on any failure."""
        if not url:
            return ""
        try:
            async with session.get(url, allow_redirects=True) as response:
                if response.status != 200:
                    logger.debug("HTTP %s for %s", response.status, url)
                    return ""
                content_type = response.headers.get("Content-Type", "")
                if "html" not in content_type and "text" not in content_type:
                    # PDFs and other binaries would otherwise be decoded into
                    # mojibake and ranked as if they were prose.
                    return ""
                html = await response.text(errors="ignore")
        except asyncio.TimeoutError:
            logger.debug("Timeout fetching %s", url)
            return ""
        except Exception as e:
            logger.debug("Error fetching %s: %s", url, e)
            return ""

        return self._html_to_text(html)

    def _html_to_text(self, html: str) -> str:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for element in soup(_STRIP_TAGS):
                element.decompose()
            text = soup.get_text(separator=" ", strip=True)
            return " ".join(text.split())[: config.MAX_PAGE_CHARS]
        except Exception as e:
            logger.debug("HTML parse failed: %s", e)
            return ""

    # ── Passage construction ─────────────────────────────────────────────────

    def _split_into_passages(self, text: str, max_words: int = 120) -> list[str]:
        """
        Group sentences into passages of roughly max_words.

        120 words rather than the model's 512-token ceiling: a passage should be
        one coherent argument. Stuffing a passage to the token limit dilutes the
        stance signal, because a single contradicting sentence gets averaged
        against several paragraphs of unrelated context.
        """
        sentences = [s.strip() for s in text.replace("! ", ". ").replace("? ", ". ").split(". ")]
        sentences = [s for s in sentences if s]

        passages: list[str] = []
        current: list[str] = []
        current_words = 0

        for sentence in sentences:
            words = len(sentence.split())
            if current and current_words + words > max_words:
                passages.append(". ".join(current).strip() + ".")
                current, current_words = [sentence], words
            else:
                current.append(sentence)
                current_words += words

        if current:
            passages.append(". ".join(current).strip() + ".")

        # Fragments this short are navigation residue, not evidence.
        return [p for p in passages if len(p.split()) >= 10]

    def _rank_passages(self, claim: str, passages: list[str], top_k: int) -> list[dict]:
        if not passages:
            return []
        try:
            claim_embedding = self.embedder.encode(claim, convert_to_tensor=True)
            passage_embeddings = self.embedder.encode(
                passages, convert_to_tensor=True, show_progress_bar=False, batch_size=32
            )
            similarities = util.cos_sim(claim_embedding, passage_embeddings)[0].tolist()
            k = min(top_k, len(passages))
            order = sorted(range(len(passages)), key=lambda i: similarities[i], reverse=True)
            selected = order[:k]

            # Reserve up to VERDICT_SLOTS for passages carrying verdict
            # language that didn't make the pure-similarity cut but still
            # clear a minimum relevance floor -- see _has_verdict_language.
            selected_set = set(selected)
            verdict_extra = [
                i for i in order
                if i not in selected_set
                and similarities[i] >= config.VERDICT_MIN_RELEVANCE
                and _has_verdict_language(passages[i])
            ][: config.VERDICT_SLOTS]
            for i in verdict_extra:
                weakest = min(selected, key=lambda j: similarities[j])
                selected[selected.index(weakest)] = i
            # Keep the pool sorted by relevance -- a verdict-language swap-in
            # is still ranked by its own score, not appended out of order.
            selected.sort(key=lambda i: similarities[i], reverse=True)

            return [
                {
                    "text": passages[i],
                    "excerpt": _excerpt(passages[i]),
                    "relevance": float(similarities[i]),
                }
                for i in selected
            ]
        except Exception as e:
            # Ranking failure should cost relevance, not the whole source.
            logger.error("Passage ranking failed: %s", e)
            return [{"text": p, "excerpt": _excerpt(p)} for p in passages[:top_k]]

    def _rerank_passages(self, claim: str, candidates: list[dict], top_k: int) -> list[dict]:
        """
        Rerank a bi-encoder candidate pool with a cross-encoder.

        A cross-encoder scores (claim, passage) jointly through one transformer
        pass, rather than comparing independently-encoded embeddings -- more
        accurate at judging whether a specific passage actually addresses the
        claim, but too slow to run over every passage from every source. It
        only ever sees `candidates`, the bi-encoder's own pre-filtered pool,
        never the raw passage list.
        """
        if not candidates:
            return []
        try:
            pairs = [(claim, c["text"]) for c in candidates]
            scores = self.reranker.predict(pairs)
            order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
            return [
                {**candidates[i], "rerank_score": float(scores[i])}
                for i in order[:top_k]
            ]
        except Exception as e:
            # Same degradation contract as _rank_passages: a broken reranker
            # costs precision, not the whole source.
            logger.error("Reranking failed: %s", e)
            return candidates[:top_k]


def _excerpt(text: str, limit: int = 280) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."
