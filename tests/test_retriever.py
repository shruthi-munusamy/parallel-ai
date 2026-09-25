"""
Retriever tests.

Network is never touched: _fetch_all is stubbed so passage construction,
semantic ranking and the snippet-fallback path are all tested deterministically.
"""

import asyncio

import pytest

from backend.models.retriever import EvidenceRetriever, _excerpt


@pytest.fixture(scope="module")
def retriever() -> EvidenceRetriever:
    return EvidenceRetriever()


pytestmark = pytest.mark.slow


ARTICLE = (
    "Researchers examined the relationship between coffee consumption and cognitive "
    "decline over a ten year period. The cohort included four thousand adults aged "
    "sixty five and older from three regions. Participants in the highest quartile of "
    "consumption showed a markedly lower incidence of dementia than non drinkers. "
    "Separately, the building's architecture was renovated in 1974 by a local firm. "
    "The cafeteria on the third floor serves lunch between eleven and two each weekday. "
    "The authors caution that observational designs cannot establish causation here."
)


class TestPassageSplitting:
    def test_splits_into_bounded_passages(self, retriever):
        text = ". ".join(f"This is sentence number {i} in a long document" for i in range(60))
        passages = retriever._split_into_passages(text, max_words=40)
        assert len(passages) > 1
        for p in passages:
            # Allows one sentence of overshoot: a sentence is never split apart.
            assert len(p.split()) <= 60

    def test_drops_navigation_fragments(self, retriever):
        text = "Home. About. Contact. This is a genuine sentence of article body text that carries real meaning."
        passages = retriever._split_into_passages(text)
        assert all(len(p.split()) >= 10 for p in passages)

    def test_empty_text(self, retriever):
        assert retriever._split_into_passages("") == []


class TestSemanticRanking:
    def test_ranks_relevant_passage_first(self, retriever):
        """
        The point of the embedding stage: relevance by meaning, not keywords.
        The cafeteria and architecture sentences must rank below the study.
        """
        passages = [
            "The cafeteria on the third floor serves lunch between eleven and two each weekday.",
            "Participants in the highest quartile of coffee consumption showed a markedly lower incidence of dementia.",
            "The building's architecture was renovated in 1974 by a local firm.",
        ]
        ranked = retriever._rank_passages("Coffee reduces dementia risk", passages, top_k=1)
        assert len(ranked) == 1
        assert "quartile" in ranked[0]["text"]

    def test_matches_paraphrase_without_lexical_overlap(self, retriever):
        """'intermittent fasting' must match 'time-restricted eating'."""
        passages = [
            "The mitochondrion is an organelle found in most eukaryotic cells.",
            "Time restricted eating protocols limit food intake to a narrow daily window.",
        ]
        ranked = retriever._rank_passages("Does intermittent fasting help?", passages, top_k=1)
        assert "Time restricted eating" in ranked[0]["text"]

    def test_top_k_respected_and_scored(self, retriever):
        passages = [f"Passage number {i} containing several words of content here." for i in range(10)]
        ranked = retriever._rank_passages("a claim under test", passages, top_k=3)
        assert len(ranked) == 3
        assert all("relevance" in r for r in ranked)
        scores = [r["relevance"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_larger_than_corpus(self, retriever):
        ranked = retriever._rank_passages("claim", ["Only one passage of text exists here."], top_k=10)
        assert len(ranked) == 1

    def test_empty_passages(self, retriever):
        assert retriever._rank_passages("claim", [], top_k=3) == []


class TestReranking:
    def test_ranks_relevant_passage_first(self, retriever):
        candidates = [
            {"text": "The cafeteria on the third floor serves lunch between eleven and two.", "excerpt": ""},
            {"text": "Participants in the highest quartile of coffee consumption showed a markedly lower incidence of dementia.", "excerpt": ""},
            {"text": "The building's architecture was renovated in 1974 by a local firm.", "excerpt": ""},
        ]
        reranked = retriever._rerank_passages("Coffee reduces dementia risk", candidates, top_k=1)
        assert len(reranked) == 1
        assert "quartile" in reranked[0]["text"]

    def test_top_k_respected_and_scored(self, retriever):
        candidates = [
            {"text": f"Passage number {i} containing several words of content here.", "excerpt": ""}
            for i in range(10)
        ]
        reranked = retriever._rerank_passages("a claim under test", candidates, top_k=3)
        assert len(reranked) == 3
        assert all("rerank_score" in r for r in reranked)
        scores = [r["rerank_score"] for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_preserves_original_candidate_fields(self, retriever):
        candidates = [{"text": "Some passage text here.", "excerpt": "Some passage...", "relevance": 0.5}]
        reranked = retriever._rerank_passages("claim", candidates, top_k=1)
        assert reranked[0]["excerpt"] == "Some passage..."
        assert reranked[0]["relevance"] == 0.5

    def test_empty_candidates(self, retriever):
        assert retriever._rerank_passages("claim", [], top_k=3) == []

    def test_top_k_larger_than_pool(self, retriever):
        candidates = [{"text": "Only one candidate passage here.", "excerpt": ""}]
        reranked = retriever._rerank_passages("claim", candidates, top_k=10)
        assert len(reranked) == 1

    def test_reranker_failure_degrades_to_pool_order(self, retriever, monkeypatch):
        """A broken reranker should cost precision, not the whole source."""
        def broken_predict(pairs):
            raise RuntimeError("model error")

        monkeypatch.setattr(retriever.reranker, "predict", broken_predict)
        candidates = [{"text": f"Passage {i}.", "excerpt": ""} for i in range(5)]
        reranked = retriever._rerank_passages("claim", candidates, top_k=2)
        assert reranked == candidates[:2]

    async def test_disabled_reranker_is_skipped_in_extract_evidence(self, retriever, monkeypatch):
        """A None reranker (RERANK_ENABLED=False, or a failed load) must not be called."""
        monkeypatch.setattr(retriever, "reranker", None)

        async def fake_fetch_all(urls):
            return [ARTICLE for _ in urls]

        monkeypatch.setattr(retriever, "_fetch_all", fake_fetch_all)
        sources = [{"url": "https://example.com/a", "title": "Study", "snippet": "s"}]
        passages = await retriever.extract_evidence("Coffee reduces dementia risk", sources, top_k=2)
        assert 1 <= len(passages) <= 2
        assert all("rerank_score" not in p for p in passages)


class TestExtractEvidence:
    async def test_end_to_end_with_stubbed_fetch(self, retriever, monkeypatch):
        async def fake_fetch_all(urls):
            return [ARTICLE for _ in urls]

        monkeypatch.setattr(retriever, "_fetch_all", fake_fetch_all)
        sources = [{"url": "https://example.com/a", "title": "Study", "snippet": "s", "date": "2024-01-01"}]
        passages = await retriever.extract_evidence("Coffee reduces dementia risk", sources, top_k=2)

        assert 1 <= len(passages) <= 2
        for p in passages:
            assert p["source_url"] == "https://example.com/a"
            assert p["source_title"] == "Study"
            assert p["date"] == "2024-01-01"
            assert p["text"] and p["excerpt"]

    async def test_falls_back_to_snippet_when_fetch_fails(self, retriever, monkeypatch):
        """A dead URL must cost relevance, not the whole source."""
        async def fake_fetch_all(urls):
            return ["" for _ in urls]

        monkeypatch.setattr(retriever, "_fetch_all", fake_fetch_all)
        sources = [{"url": "https://dead.example/x", "title": "T", "snippet": "A useful snippet of evidence."}]
        passages = await retriever.extract_evidence("some claim", sources)

        assert len(passages) == 1
        assert passages[0]["text"] == "A useful snippet of evidence."

    async def test_source_with_no_content_at_all_is_dropped(self, retriever, monkeypatch):
        async def fake_fetch_all(urls):
            return ["" for _ in urls]

        monkeypatch.setattr(retriever, "_fetch_all", fake_fetch_all)
        sources = [{"url": "https://dead.example/x", "title": "T", "snippet": ""}]
        assert await retriever.extract_evidence("some claim", sources) == []

    async def test_mock_mode_skips_network(self, retriever, monkeypatch):
        """fetch_pages=False must not open a socket; it ranks over snippets."""
        called = False

        async def fake_fetch_all(urls):
            nonlocal called
            called = True
            return ["" for _ in urls]

        monkeypatch.setattr(retriever, "_fetch_all", fake_fetch_all)
        sources = [{"url": "https://mock.test/a", "title": "T", "snippet": "Mock snippet body."}]
        passages = await retriever.extract_evidence("claim", sources, fetch_pages=False)

        assert called is False
        assert len(passages) == 1

    async def test_empty_sources(self, retriever):
        assert await retriever.extract_evidence("claim", []) == []

    async def test_partial_fetch_failure_is_isolated(self, retriever, monkeypatch):
        async def fake_fetch_all(urls):
            return [ARTICLE, ""]

        monkeypatch.setattr(retriever, "_fetch_all", fake_fetch_all)
        sources = [
            {"url": "https://good.example/a", "title": "Good", "snippet": "g"},
            {"url": "https://bad.example/b", "title": "Bad", "snippet": "fallback snippet text"},
        ]
        passages = await retriever.extract_evidence("Coffee and dementia", sources, top_k=1)
        urls = {p["source_url"] for p in passages}
        assert urls == {"https://good.example/a", "https://bad.example/b"}


class TestHtmlExtraction:
    def test_strips_scripts_and_chrome(self, retriever):
        html = """
        <html><head><style>body{color:red}</style></head>
        <body>
          <nav>Home About Contact</nav>
          <script>console.log('tracking pixel')</script>
          <p>The actual article body text lives here.</p>
          <footer>Copyright 2024</footer>
        </body></html>
        """
        text = retriever._html_to_text(html)
        assert "actual article body text" in text
        assert "tracking pixel" not in text
        assert "Copyright" not in text
        assert "color:red" not in text

    def test_malformed_html_does_not_raise(self, retriever):
        assert isinstance(retriever._html_to_text("<div><p>unclosed <b>tags"), str)

    def test_truncates_to_max_chars(self, retriever, monkeypatch):
        from backend.config import config

        monkeypatch.setattr(config, "MAX_PAGE_CHARS", 50)
        text = retriever._html_to_text("<p>" + "word " * 500 + "</p>")
        assert len(text) <= 50


class TestFetchConcurrency:
    async def test_fetches_run_concurrently(self, retriever, monkeypatch):
        """
        Regression guard on the dominant latency term. Ten sources at a 0.2s
        delay must finish in well under the 2.0s a sequential loop would take.
        """
        async def slow_fetch(session, url):
            await asyncio.sleep(0.2)
            return f"content for {url}"

        monkeypatch.setattr(retriever, "_fetch_page", slow_fetch)
        urls = [f"https://example.com/{i}" for i in range(10)]

        loop = asyncio.get_running_loop()
        start = loop.time()
        pages = await retriever._fetch_all(urls)
        elapsed = loop.time() - start

        assert len(pages) == 10
        assert all(pages)
        assert elapsed < 1.0, f"fetches appear serialized: {elapsed:.2f}s for 10 x 0.2s"

    async def test_one_failure_does_not_abort_the_batch(self, retriever, monkeypatch):
        async def flaky_fetch(session, url):
            if "bad" in url:
                raise RuntimeError("connection reset")
            return "ok"

        monkeypatch.setattr(retriever, "_fetch_page", flaky_fetch)
        pages = await retriever._fetch_all(["https://good.test/1", "https://bad.test/2"])
        assert pages == ["ok", ""]


class TestExcerpt:
    def test_short_text_unchanged(self):
        assert _excerpt("Short text.") == "Short text."

    def test_long_text_truncated_on_word_boundary(self):
        result = _excerpt("word " * 200, limit=50)
        assert len(result) <= 54
        assert result.endswith("...")
