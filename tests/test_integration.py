"""
Integration tests: the full pipeline through the HTTP API.

Search is always stubbed to the offline corpus, so these run with no API keys
and no network. Model weights are real - the point is to prove the stages
compose correctly, and a mocked stance detector would prove nothing.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def client():
    """
    TestClient as a context manager so FastAPI's lifespan runs and the models
    actually load. Without the `with`, `pipeline` stays empty and every request
    returns 503.
    """
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_reports_ready(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["models_loaded"] is True
        assert body["stance_checkpoint"]

    def test_health_declares_degraded_modes(self, client):
        """
        A mock-backed run must be self-identifying. Silently serving fake
        evidence that looks live is the worst failure this system could have.
        """
        body = client.get("/health").json()
        assert "live_search" in body
        assert "llm_decomposition" in body


class TestVerifyContract:
    def test_returns_complete_schema(self, client):
        response = client.post("/verify", json={"claim": "Vaccines cause autism", "max_sources": 5})
        assert response.status_code == 200
        body = response.json()

        expected = {
            "claim", "verdict", "confidence", "uncertainty", "evidence",
            "consensus_summary", "processing_time_ms", "sources_analyzed",
            "passages_analyzed", "support_ratio", "refute_ratio",
            "neutral_ratio", "mock_search", "cached",
        }
        assert expected <= set(body)

        assert body["verdict"] in {"SUPPORTED", "REFUTED", "MIXED", "INSUFFICIENT"}
        assert 0.0 <= body["confidence"] <= 1.0
        assert 0.0 <= body["uncertainty"] <= 1.0
        ratios = body["support_ratio"] + body["refute_ratio"] + body["neutral_ratio"]
        assert abs(ratios - 1.0) < 1e-6

    def test_evidence_items_are_well_formed(self, client):
        body = client.post("/verify", json={"claim": "Vaccines cause autism", "max_sources": 5}).json()
        assert body["evidence"], "expected at least one evidence item from the mock corpus"

        for item in body["evidence"]:
            assert item["source_url"].startswith("http")
            assert item["stance"] in {"SUPPORTS", "REFUTES", "NEUTRAL"}
            assert 0.0 <= item["stance_confidence"] <= 1.0
            assert 0.0 <= item["credibility_score"] <= 1.0
            assert item["excerpt"]

    def test_unrelated_evidence_excluded_from_response(self, client):
        body = client.post("/verify", json={"claim": "Vaccines cause autism", "max_sources": 5}).json()
        assert all(item["stance"] != "UNRELATED" for item in body["evidence"])

    def test_evidence_sorted_by_weight(self, client):
        """Ordering must match what actually drove the verdict."""
        body = client.post("/verify", json={"claim": "Climate change is caused by humans", "max_sources": 6}).json()
        weights = [e["stance_confidence"] * e["credibility_score"] for e in body["evidence"]]
        assert weights == sorted(weights, reverse=True)


class TestVerdictCorrectness:
    def test_refuted_claim(self, client):
        """
        End-to-end behavioural check: the corpus contains CDC, JAMA and Lancet
        retraction sources against one low-credibility blog. Credibility
        weighting must land on REFUTED.
        """
        body = client.post("/verify", json={"claim": "Vaccines cause autism", "max_sources": 8}).json()
        assert body["verdict"] == "REFUTED"
        assert body["refute_ratio"] > body["support_ratio"]

    def test_no_sources_yields_insufficient(self, client):
        """A claim outside the mock corpus must not produce a confident verdict."""
        body = client.post(
            "/verify",
            json={"claim": "The xylophone was invented by Mongolian throat singers", "max_sources": 5},
        ).json()
        assert body["verdict"] == "INSUFFICIENT"
        assert body["confidence"] == 0.0
        assert body["evidence"] == []


class TestValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {"claim": "short"},                          # below min_length
            {"claim": "          "},                     # whitespace only
            {"claim": "x" * 501},                        # above max_length
            {},                                          # missing field
            {"claim": "A valid length claim", "max_sources": 0},    # below ge
            {"claim": "A valid length claim", "max_sources": 100},  # above le
        ],
    )
    def test_invalid_payloads_rejected(self, client, payload):
        assert client.post("/verify", json=payload).status_code == 422

    def test_whitespace_is_normalized(self, client):
        body = client.post("/verify", json={"claim": "  Vaccines   cause    autism  "}).json()
        assert body["claim"] == "Vaccines cause autism"


class TestCaching:
    def test_second_call_is_cached_and_faster(self, client):
        claim = "Intermittent fasting improves metabolic health outcomes"
        client.post("/cache/clear")

        first = client.post("/verify", json={"claim": claim, "max_sources": 5}).json()
        second = client.post("/verify", json={"claim": claim, "max_sources": 5}).json()

        assert first["cached"] is False
        assert second["cached"] is True
        assert second["verdict"] == first["verdict"]
        # The cached path skips every model call, so it must be dramatically faster.
        assert second["processing_time_ms"] < max(first["processing_time_ms"], 10)

    def test_cache_stats_endpoint(self, client):
        client.post("/cache/clear")
        claim = "Coffee consumption reduces the risk of Alzheimer's disease"
        client.post("/verify", json={"claim": claim, "max_sources": 5})
        client.post("/verify", json={"claim": claim, "max_sources": 5})

        stats = client.get("/cache/stats").json()
        assert stats["entries"] >= 1
        assert stats["hits"] >= 1
        assert 0.0 <= stats["hit_rate"] <= 1.0

    def test_cache_clear(self, client):
        client.post("/verify", json={"claim": "Vaccines cause autism", "max_sources": 5})
        assert client.post("/cache/clear").status_code == 200
        assert client.get("/cache/stats").json()["entries"] == 0


class TestResilience:
    def test_search_failure_degrades_to_insufficient(self, client, monkeypatch):
        """A total search outage yields a clean verdict, not a 500."""
        from backend.main import pipeline

        async def dead_search(queries, max_results=10):
            return []

        monkeypatch.setattr(pipeline["search"], "search_many", dead_search)
        client.post("/cache/clear")

        response = client.post("/verify", json={"claim": "Some claim that needs sources", "max_sources": 5})
        assert response.status_code == 200
        body = response.json()
        assert body["verdict"] == "INSUFFICIENT"
        assert body["sources_analyzed"] == 0

    def test_stance_model_failure_degrades(self, client, monkeypatch):
        """A model outage must not take the API down."""
        from backend.main import pipeline

        def boom(*args, **kwargs):
            raise RuntimeError("simulated CUDA OOM")

        monkeypatch.setattr(pipeline["stance"], "_predict_probs", boom)
        client.post("/cache/clear")

        response = client.post("/verify", json={"claim": "Vaccines cause autism", "max_sources": 5})
        assert response.status_code == 200
        assert response.json()["verdict"] == "INSUFFICIENT"

    def test_unexpected_pipeline_error_returns_500_not_crash(self, client, monkeypatch):
        from backend.main import pipeline

        async def boom(*args, **kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(pipeline["retriever"], "extract_evidence", boom)
        client.post("/cache/clear")

        response = client.post("/verify", json={"claim": "Vaccines cause autism", "max_sources": 5})
        assert response.status_code == 500
        assert "detail" in response.json()


class TestPerformance:
    def test_latency_within_budget(self, client):
        """
        Guards the concurrency work. Mock mode skips network fetching, so this
        measures decomposition + stance + synthesis on CPU.
        """
        client.post("/cache/clear")
        body = client.post(
            "/verify", json={"claim": "Climate change is caused by human activity", "max_sources": 8}
        ).json()
        assert body["processing_time_ms"] < 10_000, (
            f"pipeline took {body['processing_time_ms']}ms"
        )
