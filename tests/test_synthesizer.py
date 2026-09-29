"""
Synthesizer tests - the aggregation logic, with no model in the loop.

The worked example from the module docstring is asserted directly, because that
example is the system's central design claim: credibility-weighted aggregation
reaches a different (and better) answer than averaging stance confidence.
"""

from datetime import datetime, timezone

import pytest

from backend.models.synthesizer import TEMPORAL_HALF_LIFE_YEARS, TEMPORAL_MIN_WEIGHT, recency_weight
from tests.conftest import make_passage

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class TestCredibilityWeighting:
    async def test_credible_minority_outweighs_weak_majority(self, synthesizer):
        """
        The worked example. A PubMed source refuting at 0.60 confidence must beat
        a blog supporting at 0.85, because 0.95*0.60 > 0.35*0.85. Plain confidence
        averaging would return SUPPORTED here.
        """
        evidence = [
            make_passage("REFUTES", confidence=0.60, credibility=0.95, url="https://pubmed.ncbi.nlm.nih.gov/1"),
            make_passage("SUPPORTS", confidence=0.85, credibility=0.35, url="https://blog.example/1"),
        ]
        result = await synthesizer.synthesize("test claim", evidence)
        assert result["verdict"] == "REFUTED"
        assert result["refute_ratio"] > result["support_ratio"]

    async def test_equal_credibility_falls_back_to_confidence(self, synthesizer):
        evidence = [
            make_passage("SUPPORTS", confidence=0.90, credibility=0.70),
            make_passage("REFUTES", confidence=0.40, credibility=0.70),
        ]
        result = await synthesizer.synthesize("test claim", evidence)
        assert result["verdict"] == "SUPPORTED"

    async def test_many_weak_sources_do_not_beat_one_strong_one(self, synthesizer):
        evidence = [make_passage("SUPPORTS", 0.55, 0.25) for _ in range(3)]
        evidence.append(make_passage("REFUTES", 0.95, 0.92))
        result = await synthesizer.synthesize("test claim", evidence)
        # 3 * (0.55*0.25) = 0.4125 vs 0.95*0.92 = 0.874
        assert result["verdict"] == "REFUTED"


class TestRatios:
    async def test_ratios_sum_to_one(self, synthesizer):
        """
        Regression test. An earlier version divided only by support+refute and
        derived neutral as the remainder, making neutral_ratio identically zero
        and leaving the UI's neutral segment permanently empty.
        """
        evidence = [
            make_passage("SUPPORTS", 0.9, 0.9),
            make_passage("REFUTES", 0.8, 0.8),
            make_passage("NEUTRAL", 0.95, 0.9),
        ]
        result = await synthesizer.synthesize("test claim", evidence)
        total = result["support_ratio"] + result["refute_ratio"] + result["neutral_ratio"]
        assert abs(total - 1.0) < 1e-6

    async def test_neutral_evidence_is_represented(self, synthesizer):
        evidence = [
            make_passage("NEUTRAL", 0.95, 0.9),
            make_passage("NEUTRAL", 0.90, 0.9),
            make_passage("SUPPORTS", 0.85, 0.9),
        ]
        result = await synthesizer.synthesize("test claim", evidence)
        assert result["neutral_ratio"] > 0.5

    async def test_unrelated_passages_excluded(self, synthesizer):
        """UNRELATED carries no information and must not dilute the ratios."""
        base = [make_passage("SUPPORTS", 0.9, 0.9)]
        with_noise = base + [make_passage("UNRELATED", 0.2, 0.9) for _ in range(5)]
        a = await synthesizer.synthesize("test claim", base)
        b = await synthesizer.synthesize("test claim", with_noise)
        assert a["support_ratio"] == b["support_ratio"]
        assert a["verdict"] == b["verdict"]


class TestVerdicts:
    async def test_unanimous_support(self, synthesizer):
        evidence = [make_passage("SUPPORTS", 0.9, 0.9) for _ in range(3)]
        result = await synthesizer.synthesize("test claim", evidence)
        assert result["verdict"] == "SUPPORTED"

    async def test_genuine_disagreement_is_mixed(self, synthesizer):
        evidence = [
            make_passage("SUPPORTS", 0.85, 0.90),
            make_passage("REFUTES", 0.85, 0.90),
        ]
        result = await synthesizer.synthesize("test claim", evidence)
        assert result["verdict"] == "MIXED"

    async def test_mostly_neutral_corpus_is_insufficient(self, synthesizer):
        """
        A lone weak directional vote inside an overwhelmingly neutral corpus is
        not enough to call a verdict, however lopsided that single vote is.
        """
        evidence = [make_passage("NEUTRAL", 0.95, 0.9) for _ in range(8)]
        evidence.append(make_passage("SUPPORTS", 0.45, 0.3))
        result = await synthesizer.synthesize("test claim", evidence)
        assert result["verdict"] == "INSUFFICIENT"
        assert result["confidence"] == 0.0

    async def test_empty_evidence(self, synthesizer):
        result = await synthesizer.synthesize("test claim", [])
        assert result["verdict"] == "INSUFFICIENT"
        assert result["confidence"] == 0.0
        assert result["neutral_ratio"] == 1.0

    async def test_all_unrelated_evidence(self, synthesizer):
        evidence = [make_passage("UNRELATED", 0.2, 0.9) for _ in range(3)]
        result = await synthesizer.synthesize("test claim", evidence)
        assert result["verdict"] == "INSUFFICIENT"

    async def test_zero_mass_evidence(self, synthesizer):
        """All-zero confidence must not divide by zero."""
        evidence = [make_passage("SUPPORTS", 0.0, 0.0)]
        result = await synthesizer.synthesize("test claim", evidence)
        assert result["verdict"] == "INSUFFICIENT"


class TestRecencyWeight:
    def test_missing_date_scores_full_weight(self):
        """Silence about publication date is not evidence of staleness."""
        assert recency_weight(None) == 1.0
        assert recency_weight("") == 1.0

    def test_unparseable_date_scores_full_weight(self):
        assert recency_weight("not a date", now=NOW) == 1.0

    def test_todays_date_scores_full_weight(self):
        assert recency_weight("2026-01-01", now=NOW) == pytest.approx(1.0)

    def test_future_date_scores_full_weight_not_a_bonus(self):
        assert recency_weight("2030-01-01", now=NOW) == 1.0

    def test_half_life_lands_at_one_half(self):
        n_years_ago = NOW.replace(year=NOW.year - int(TEMPORAL_HALF_LIFE_YEARS))
        weight = recency_weight(n_years_ago.date().isoformat(), now=NOW)
        assert weight == pytest.approx(0.5, abs=0.02)

    def test_very_old_date_floors_rather_than_vanishing(self):
        ancient = "1990-01-01"
        assert recency_weight(ancient, now=NOW) == pytest.approx(TEMPORAL_MIN_WEIGHT)

    def test_monotonically_decreasing_with_age(self):
        recent = recency_weight("2025-06-01", now=NOW)
        older = recency_weight("2020-01-01", now=NOW)
        assert recent > older


class TestTemporalWeightingInSynthesis:
    async def test_recent_source_outweighs_stale_source_of_equal_strength(self, synthesizer):
        """
        Same confidence, same credibility, same stance strength on both sides --
        only publication date differs. Recency alone must be able to swing it,
        the same way credibility alone already can.
        """
        evidence = [
            make_passage("REFUTES", confidence=0.85, credibility=0.8, date="2026-01-01"),
            make_passage("SUPPORTS", confidence=0.85, credibility=0.8, date="1990-01-01"),
        ]
        result = await synthesizer.synthesize("test claim", evidence)
        assert result["verdict"] == "REFUTED"
        assert result["refute_ratio"] > result["support_ratio"]

    async def test_missing_dates_do_not_change_existing_behaviour(self, synthesizer):
        """Regression guard: every pre-existing test omits `date`, and must still pass unaffected."""
        evidence = [
            make_passage("REFUTES", confidence=0.60, credibility=0.95, url="https://pubmed.ncbi.nlm.nih.gov/1"),
            make_passage("SUPPORTS", confidence=0.85, credibility=0.35, url="https://blog.example/1"),
        ]
        result = await synthesizer.synthesize("test claim", evidence)
        assert result["verdict"] == "REFUTED"


class TestUncertaintyPropagation:
    async def test_high_uncertainty_lowers_confidence(self, synthesizer):
        confident = [make_passage("SUPPORTS", 0.9, 0.9, uncertainty=0.0) for _ in range(3)]
        unstable = [make_passage("SUPPORTS", 0.9, 0.9, uncertainty=0.8) for _ in range(3)]

        a = await synthesizer.synthesize("test claim", confident)
        b = await synthesizer.synthesize("test claim", unstable)

        assert a["verdict"] == b["verdict"] == "SUPPORTED"
        assert b["confidence"] < a["confidence"]

    async def test_uncertainty_discount_is_bounded(self, synthesizer):
        """Even total model instability must not erase a real evidence majority."""
        evidence = [make_passage("SUPPORTS", 1.0, 1.0, uncertainty=1.0) for _ in range(3)]
        result = await synthesizer.synthesize("test claim", evidence)
        # The discount caps at half, so confidence floors at 0.5 (modulo the
        # epsilon guarding the division).
        assert result["confidence"] == pytest.approx(0.5, abs=1e-6)

    async def test_confidence_and_ratios_in_range(self, synthesizer):
        evidence = [
            make_passage("SUPPORTS", 0.7, 0.8),
            make_passage("REFUTES", 0.6, 0.5),
            make_passage("NEUTRAL", 0.9, 0.6),
        ]
        result = await synthesizer.synthesize("test claim", evidence)
        for key in ("confidence", "uncertainty", "support_ratio", "refute_ratio", "neutral_ratio"):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"
