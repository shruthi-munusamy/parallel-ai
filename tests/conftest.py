"""
Shared fixtures.

Test strategy: the transformer-backed components are session-scoped because
loading weights costs seconds and dominates the suite otherwise. Anything that
touches the network (search, page fetching) is stubbed, so the suite is
deterministic and runs offline with no API keys.
"""

import json
import pathlib
import sys

import pytest

# Allows `import backend...` when pytest is run from the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from backend.models.credibility_scorer import CredibilityScorer
from backend.models.stance_detector import StanceDetector
from backend.models.synthesizer import EvidenceSynthesizer

TEST_DATA = pathlib.Path(__file__).parent / "test_data.json"


@pytest.fixture(scope="session")
def test_data() -> dict:
    with open(TEST_DATA) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def stance_detector() -> StanceDetector:
    """Loads real model weights once; reused across the whole session."""
    return StanceDetector()


@pytest.fixture
def credibility_scorer() -> CredibilityScorer:
    return CredibilityScorer()


@pytest.fixture
def synthesizer() -> EvidenceSynthesizer:
    return EvidenceSynthesizer()


def make_passage(
    stance: str = "SUPPORTS",
    confidence: float = 0.9,
    credibility: float = 0.9,
    uncertainty: float = 0.05,
    url: str = "https://example.com/a",
    date: str | None = None,
) -> dict:
    """Builds a synthesized passage so synthesizer tests need no model."""
    return {
        "source_url": url,
        "source_title": "Test source",
        "text": "Test passage body.",
        "excerpt": "Test passage body.",
        "stance": stance,
        "stance_confidence": confidence,
        "credibility_score": credibility,
        "uncertainty": uncertainty,
        "date": date,
    }
