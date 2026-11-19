"""
Credibility-weighted evidence aggregation.

Credibility is a multiplier on evidence weight rather than a filter, so a
credible source disagreeing with a claim outweighs a weak one agreeing with it.
Per passage i:

    w_i = stance_confidence_i * credibility_i * recency_i

    support_mass = sum(w_i : stance_i == SUPPORTS)   (likewise refute, neutral)

For example, PubMed (cred 0.95) REFUTES at 0.60 gives w = 0.570, and a blog
(cred 0.35) SUPPORTS at 0.85 gives w = 0.298: averaging confidence would pick
SUPPORTS, weighting picks REFUTES.

The reported *_ratio fields are shares of total mass including neutral (they
sum to 1). The verdict is decided by support vs refute alone, since neutral
evidence abstains. MC dropout uncertainty discounts the reported confidence.
"""

import logging
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

# Share of the contested (support + refute) mass required for a directional call.
STRONG_MAJORITY = 0.70
# Below this, neither side has a meaningful edge.
WEAK_MAJORITY = 0.55
# Both sides above this share means genuine disagreement, not a near-tie.
CONTESTED_FLOOR = 0.35
# If contested mass is below this share of total, the corpus is mostly neutral
# and there is not enough directional evidence to call it either way.
MIN_CONTESTED_SHARE = 0.20

# Years for a source's temporal weight to halve.
TEMPORAL_HALF_LIFE_YEARS = 5.0
# Floor on the decay: age alone does not make a stable fact wrong.
TEMPORAL_MIN_WEIGHT = 0.3

EPSILON = 1e-9


def recency_weight(date_str: str | None, now: datetime | None = None) -> float:
    """Half-life decay by source age. Missing or unparseable dates score 1.0."""
    if not date_str:
        return 1.0
    try:
        published = datetime.fromisoformat(date_str.strip())
    except (ValueError, TypeError):
        return 1.0

    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)

    age_years = (now - published).days / 365.25
    if age_years <= 0:
        return 1.0  # future or same-day dates: no decay, not extra credit

    decay = 0.5 ** (age_years / TEMPORAL_HALF_LIFE_YEARS)
    return max(decay, TEMPORAL_MIN_WEIGHT)


def _empty_result(summary: str) -> dict:
    return {
        "verdict": "INSUFFICIENT",
        "confidence": 0.0,
        "uncertainty": 1.0,
        "summary": summary,
        "support_ratio": 0.0,
        "refute_ratio": 0.0,
        "neutral_ratio": 1.0,
    }


class EvidenceSynthesizer:
    async def synthesize(self, claim: str, evidence: list[dict]) -> dict:
        if not evidence:
            return _empty_result("No evidence found for this claim.")

        # UNRELATED passages were below the stance detector's confidence floor;
        # they carry no information and must not dilute the ratios.
        relevant = [e for e in evidence if e.get("stance") != "UNRELATED"]
        if not relevant:
            return _empty_result("Retrieved sources do not address this claim.")

        def mass(stance: str) -> float:
            return sum(
                float(e.get("stance_confidence", 0.0))
                * float(e.get("credibility_score", 0.5))
                * recency_weight(e.get("date"))
                for e in relevant
                if e.get("stance") == stance
            )

        support_mass = mass("SUPPORTS")
        refute_mass = mass("REFUTES")
        neutral_mass = mass("NEUTRAL")

        total_mass = support_mass + refute_mass + neutral_mass
        if total_mass <= EPSILON:
            return _empty_result("Evidence carried no usable stance signal.")

        # Reported composition of the evidence set; sums to 1.
        support_ratio = support_mass / total_mass
        refute_ratio = refute_mass / total_mass
        neutral_ratio = neutral_mass / total_mass

        # The verdict contest: neutral mass abstains rather than voting.
        contested_mass = support_mass + refute_mass
        contested_share = contested_mass / total_mass
        support_share = support_mass / (contested_mass + EPSILON)
        refute_share = refute_mass / (contested_mass + EPSILON)

        verdict = self._decide(contested_share, support_share, refute_share)
        # Confidence is the winning side's share of the contest, so a 9-1 split
        # reads as high confidence even if half the corpus was neutral.
        raw_confidence = float(max(support_share, refute_share))

        # ── Uncertainty propagation ──────────────────────────────────────────
        directional = [
            float(e.get("uncertainty", 0.5))
            for e in relevant
            if e.get("stance") in ("SUPPORTS", "REFUTES")
        ]
        mean_uncertainty = float(np.mean(directional)) if directional else 0.5

        if verdict == "INSUFFICIENT":
            confidence = 0.0
        else:
            # Discount by at most half: high model instability halves a verdict's
            # reported confidence but never erases a real evidence majority.
            confidence = raw_confidence * (1.0 - 0.5 * mean_uncertainty)

        n_support = sum(1 for e in relevant if e.get("stance") == "SUPPORTS")
        n_refute = sum(1 for e in relevant if e.get("stance") == "REFUTES")
        n_neutral = sum(1 for e in relevant if e.get("stance") == "NEUTRAL")

        logger.info(
            "Synthesis: %s | masses S=%.3f R=%.3f N=%.3f | conf=%.3f unc=%.3f",
            verdict, support_mass, refute_mass, neutral_mass, confidence, mean_uncertainty,
        )

        return {
            "verdict": verdict,
            "confidence": confidence,
            "uncertainty": mean_uncertainty,
            "summary": self._summarize(
                verdict, n_support, n_refute, n_neutral, raw_confidence, mean_uncertainty
            ),
            "support_ratio": support_ratio,
            "refute_ratio": refute_ratio,
            "neutral_ratio": neutral_ratio,
        }

    @staticmethod
    def _decide(contested_share: float, support_share: float, refute_share: float) -> str:
        # Checked first: a corpus that is overwhelmingly neutral cannot support
        # a directional verdict no matter how lopsided the few votes are.
        if contested_share < MIN_CONTESTED_SHARE:
            return "INSUFFICIENT"
        if support_share >= STRONG_MAJORITY:
            return "SUPPORTED"
        if refute_share >= STRONG_MAJORITY:
            return "REFUTED"
        if support_share >= CONTESTED_FLOOR and refute_share >= CONTESTED_FLOOR:
            return "MIXED"
        if max(support_share, refute_share) < WEAK_MAJORITY:
            return "INSUFFICIENT"
        return "SUPPORTED" if support_share > refute_share else "REFUTED"

    @staticmethod
    def _summarize(
        verdict: str,
        n_support: int,
        n_refute: int,
        n_neutral: int,
        confidence: float,
        uncertainty: float,
    ) -> str:
        counts = (
            f"{n_support} passage(s) support, {n_refute} refute, {n_neutral} neutral"
        )
        explanations = {
            "SUPPORTED": "Credibility-weighted evidence favours the claim.",
            "REFUTED": "Credibility-weighted evidence contradicts the claim.",
            "MIXED": "Credible sources disagree substantively on this claim.",
            "INSUFFICIENT": "Retrieved evidence is too weak or too neutral to decide.",
        }
        return (
            f"{explanations.get(verdict, '')} {counts}. "
            f"Weighted agreement {int(confidence * 100)}%, "
            f"model uncertainty {int(uncertainty * 100)}%."
        )
