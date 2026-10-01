"""
Tests for credibility scoring, caching, search and query decomposition.

None of these load model weights, so this module runs in well under a second.
"""

import asyncio

import pytest

from backend.models.credibility_scorer import SCORES, CredibilityScorer
from backend.models.query_decomposer import QueryDecomposer, _parse_json_array
from backend.services.cache_service import CacheService
from backend.services.mock_corpus import search_mock_corpus
from backend.services.search_service import SearchService


class TestCredibilityScorer:
    async def test_tier_assignment(self, credibility_scorer, test_data):
        for case in test_data["credibility_cases"]:
            score = await credibility_scorer.score({"source_url": case["url"]})
            assert score == SCORES[case["tier"]], (
                f"{case['url']} scored {score}, expected tier {case['tier']}"
            )

    async def test_ordering_across_tiers(self, credibility_scorer):
        async def s(url):
            return await credibility_scorer.score({"source_url": url})

        assert await s("https://pubmed.ncbi.nlm.nih.gov/1") > await s("https://www.reuters.com/a")
        assert await s("https://www.reuters.com/a") > await s("https://en.wikipedia.org/wiki/A")
        assert await s("https://en.wikipedia.org/wiki/A") > await s("https://www.reddit.com/r/a")

    async def test_subdomain_inherits_parent_tier(self, credibility_scorer):
        """
        Exact-match only would drop these to the unknown default, silently
        under-weighting some of the most credible sources in the corpus.
        """
        assert await credibility_scorer.score(
            {"source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/"}
        ) == SCORES["high"]
        assert await credibility_scorer.score(
            {"source_url": "https://blogs.nature.com/post"}
        ) == SCORES["high"]

    async def test_explicit_tier_beats_tld_heuristic(self, credibility_scorer):
        """reddit.com is low despite no TLD rule; wikipedia.org must not be promoted by .org."""
        assert await credibility_scorer.score(
            {"source_url": "https://en.wikipedia.org/wiki/X"}
        ) == SCORES["neutral"]

    async def test_tld_fallback(self, credibility_scorer):
        assert await credibility_scorer.score(
            {"source_url": "https://www.ca.gov/page"}
        ) == SCORES["high"]
        assert await credibility_scorer.score(
            {"source_url": "https://www.berkeley.edu/page"}
        ) == SCORES["good"]

    async def test_www_and_port_stripped(self, credibility_scorer):
        assert await credibility_scorer.score(
            {"source_url": "https://www.nature.com:443/articles/x"}
        ) == SCORES["high"]

    @pytest.mark.parametrize("url", ["", "not a url", "://broken", None])
    async def test_malformed_urls_return_default(self, credibility_scorer, url):
        score = await credibility_scorer.score({"source_url": url} if url is not None else {})
        assert score == SCORES["unknown"]

    async def test_score_all_annotates_in_place(self, credibility_scorer):
        passages = [{"source_url": "https://www.cdc.gov/a"}, {"source_url": "https://reddit.com/b"}]
        await credibility_scorer.score_all(passages)
        assert passages[0]["credibility_score"] == SCORES["high"]
        assert passages[1]["credibility_score"] == SCORES["low"]


class TestCred1Integration:
    """
    CRED-1 (CC BY 4.0, github.com/aloth/cred-1) extends the low-credibility
    end that the hand-curated LOW_CREDIBILITY tier never covered: actual
    fake/conspiracy/unreliable domains, not just social/UGC platforms.
    """

    async def test_known_misinformation_domain_scores_below_unknown_default(self, credibility_scorer):
        """
        The regression this whole integration exists to fix: before CRED-1,
        a known fake-news domain fell through every tier to the 0.40
        "unknown" default -- scoring *above* Reddit's hand-curated 0.25.
        """
        score = await credibility_scorer.score({"source_url": "https://www.infowars.com/article"})
        assert score < SCORES["unknown"]
        assert score < SCORES["low"]

    async def test_cred1_subdomain_inherits_parent_score(self, credibility_scorer):
        flat = await credibility_scorer.score({"source_url": "https://infowars.com/a"})
        sub = await credibility_scorer.score({"source_url": "https://forums.infowars.com/a"})
        assert flat == sub

    async def test_hand_curated_tier_still_wins_over_cred1(self, credibility_scorer, monkeypatch):
        """
        If a domain somehow appeared in both, the hand-curated tiers (checked
        first in score()) must take precedence -- they are specifically
        vetted for this project, CRED-1 is a broader supplementary signal.
        """
        monkeypatch.setitem(credibility_scorer._cred1, "nature.com", 0.05)
        score = await credibility_scorer.score({"source_url": "https://nature.com/article"})
        assert score == SCORES["high"]

    async def test_domain_not_in_cred1_or_tiers_falls_back_to_tld_or_unknown(self, credibility_scorer):
        score = await credibility_scorer.score({"source_url": "https://some-random-blog-xyz123.net/post"})
        assert score == SCORES["unknown"]

    def test_dataset_loaded_with_real_domain_count(self, credibility_scorer):
        """Loose sanity bound, not an exact count -- the upstream dataset is versioned and updates."""
        assert len(credibility_scorer._cred1) > 1000


class TestRspIntegration:
    """
    WP:RSP (CC BY-SA 4.0, en.wikipedia.org/wiki/Wikipedia:Reliable_sources/
    Perennial_sources) extends *both* ends -- unlike CRED-1, which only
    covers known-bad domains, RSP is a general community reliability
    judgment and includes genuinely reliable sources CRED-1 structurally
    cannot rate.
    """

    async def test_rsp_only_domain_scores_above_unknown_default(self, credibility_scorer):
        """
        buzzfeednews.com is in neither the hand-curated tiers nor CRED-1 --
        without RSP it would fall to the 0.40 unknown default despite being
        a generally-reliable source by community consensus.
        """
        score = await credibility_scorer.score({"source_url": "https://www.buzzfeednews.com/article"})
        assert score > SCORES["unknown"]

    async def test_rsp_only_domain_scores_below_unknown_default_when_deprecated(self, credibility_scorer):
        """baike.baidu.com is RSP-deprecated and absent from CRED-1 and the hand tiers."""
        score = await credibility_scorer.score({"source_url": "https://baike.baidu.com/item/x"})
        assert score < SCORES["unknown"]

    async def test_cred1_wins_over_rsp_when_both_present(self, credibility_scorer):
        """
        breitbart.com is rated by both datasets with different scores --
        CRED-1 (checked first in score()) must take precedence as the more
        targeted misinformation-specific signal.
        """
        score = await credibility_scorer.score({"source_url": "https://www.breitbart.com/a"})
        assert score == credibility_scorer._cred1["breitbart.com"]

    async def test_rsp_subdomain_inherits_parent_score(self, credibility_scorer):
        flat = await credibility_scorer.score({"source_url": "https://buzzfeednews.com/a"})
        sub = await credibility_scorer.score({"source_url": "https://m.buzzfeednews.com/a"})
        assert flat == sub

    def test_dataset_loaded_with_real_domain_count(self, credibility_scorer):
        """Loose sanity bound -- the live page is edited continuously, so an exact count would be brittle."""
        assert len(credibility_scorer._rsp) > 500


class TestCacheService:
    async def test_roundtrip(self):
        cache = CacheService()
        await cache.set("A claim about coffee", {"verdict": "SUPPORTED"})
        assert (await cache.get("A claim about coffee"))["verdict"] == "SUPPORTED"

    async def test_miss_returns_none(self):
        assert await CacheService().get("never stored") is None

    async def test_key_normalizes_case_and_whitespace(self):
        cache = CacheService()
        await cache.set("Coffee Reduces  Risk", {"v": 1})
        assert await cache.get("  coffee reduces risk ") == {"v": 1}

    async def test_ttl_expiry(self):
        cache = CacheService(ttl_s=0.05)
        await cache.set("claim", {"v": 1})
        assert await cache.get("claim") is not None
        await asyncio.sleep(0.08)
        assert await cache.get("claim") is None

    async def test_lru_eviction_bounds_memory(self):
        cache = CacheService(max_entries=3)
        for i in range(5):
            await cache.set(f"claim {i}", {"i": i})
        stats = await cache.get_stats()
        assert stats["entries"] == 3
        assert stats["evictions"] == 2
        # The two oldest are gone, the newest survive.
        assert await cache.get("claim 0") is None
        assert await cache.get("claim 4") is not None

    async def test_read_refreshes_lru_position(self):
        cache = CacheService(max_entries=2)
        await cache.set("a claim", {"v": "a"})
        await cache.set("b claim", {"v": "b"})
        await cache.get("a claim")      # 'a' is now the most recent
        await cache.set("c claim", {"v": "c"})
        assert await cache.get("a claim") is not None
        assert await cache.get("b claim") is None

    async def test_stats_track_hit_rate(self):
        cache = CacheService()
        await cache.set("a claim", {"v": 1})
        await cache.get("a claim")
        await cache.get("missing claim")
        stats = await cache.get_stats()
        assert stats["hits"] == 1 and stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    async def test_clear(self):
        cache = CacheService()
        await cache.set("a claim", {"v": 1})
        await cache.clear()
        assert (await cache.get_stats())["entries"] == 0


class TestMockCorpus:
    def test_returns_relevant_documents(self):
        results = search_mock_corpus("vaccines autism", 10)
        assert results
        assert all("url" in r and "snippet" in r for r in results)
        assert any("autism" in r["snippet"].lower() for r in results)

    def test_returns_both_sides(self):
        """A one-sided corpus would make every mock verdict unanimous."""
        results = search_mock_corpus("vaccines autism", 10)
        text = " ".join(r["snippet"].lower() for r in results)
        assert "no link" in text
        assert "cause autism" in text

    def test_unrelated_query_returns_nothing(self):
        """Padding with filler would let a confident verdict rest on noise."""
        assert search_mock_corpus("xylophone mongolian throat singing", 10) == []

    def test_respects_max_results(self):
        assert len(search_mock_corpus("climate warming", 2)) <= 2

    def test_empty_query(self):
        assert search_mock_corpus("", 10) == []


class TestSearchService:
    async def test_falls_back_to_mock_without_key(self):
        service = SearchService(api_key="")
        assert service.is_mock
        results = await service.search("vaccines autism", 5)
        assert results

    async def test_search_many_deduplicates_by_url(self):
        service = SearchService(api_key="")
        # Overlapping queries hit the same corpus documents.
        results = await service.search_many(
            ["vaccines autism", "autism vaccines link", "vaccine safety autism"], 10
        )
        urls = [r["url"] for r in results]
        assert len(urls) == len(set(urls))

    async def test_search_many_empty(self):
        assert await SearchService(api_key="").search_many([], 5) == []

    async def test_provider_failure_returns_empty_not_raise(self, monkeypatch):
        """One bad sub-query must not abort the verification."""
        service = SearchService(api_key="")
        service.is_mock = False

        class BoomClient:
            def search(self, **kwargs):
                raise RuntimeError("provider 503")

        service.client = BoomClient()
        assert await service.search("anything", 5) == []

    async def test_search_many_isolates_failing_subquery(self, monkeypatch):
        service = SearchService(api_key="")
        original = service.search

        async def flaky(query, max_results=10):
            if query == "bad":
                raise RuntimeError("boom")
            return await original(query, max_results)

        monkeypatch.setattr(service, "search", flaky)
        results = await service.search_many(["bad", "vaccines autism"], 5)
        assert results  # the good sub-query still contributed


class TestQueryDecomposer:
    async def test_rule_based_fallback_without_key(self):
        decomposer = QueryDecomposer(api_key="")
        assert decomposer.use_llm is False
        queries = await decomposer.decompose("Coffee reduces Alzheimer's risk")
        assert len(queries) > 1
        assert queries[0] == "Coffee reduces Alzheimer's risk"

    async def test_fallback_includes_opposing_framings(self):
        """Confirmation-only expansion would bias the whole evidence set."""
        queries = await QueryDecomposer(api_key="").decompose("Coffee reduces Alzheimer's risk")
        joined = " ".join(queries).lower()
        assert "against" in joined
        assert "supporting" in joined

    async def test_llm_failure_falls_back(self, monkeypatch):
        decomposer = QueryDecomposer(api_key="")
        decomposer.use_llm = True

        class BoomClient:
            class messages:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("API down")

        decomposer.client = BoomClient()
        queries = await decomposer.decompose("A claim under test")
        assert queries[0] == "A claim under test"
        assert len(queries) > 1

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ('["a", "b"]', ["a", "b"]),
            ('```json\n["a", "b"]\n```', ["a", "b"]),
            ('```\n["a"]\n```', ["a"]),
            ('Here you go:\n["a", "b"]\nHope that helps', ["a", "b"]),
            ('["a", 3, "b", ""]', ["a", "b"]),
            ("not json at all", []),
            # An array wrapped in an object still yields its array: the
            # bracket-scan salvages it rather than dropping to the fallback.
            ('{"queries": ["a"]}', ["a"]),
        ],
    )
    def test_json_array_parsing(self, raw, expected):
        """
        Models wrap JSON in prose or fences often enough that strict parsing
        would drop us to the weaker fallback for no real reason.
        """
        assert _parse_json_array(raw) == expected
