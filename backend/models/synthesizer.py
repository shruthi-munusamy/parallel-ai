"""
synthesizer.py - Credibility-weighted evidence aggregation.

The contribution:
  Most NLI-based fact-checking pipelines treat source credibility as a post-hoc
  filter ("discard anything below threshold X"). That throws away signal and is
  brittle at the boundary. Here credibility is a multiplier on evidence weight
  during aggregation, so a highly credible source disagreeing with a claim
  outweighs a weak source agreeing with it, without either being discarded.

Formally, per passage i:
    w_i = stance_confidence_i * credibility_i

    support_mass = sum(w_i : stance_i == SUPPORTS)
    refute_mass  = sum(w_i : stance_i == REFUTES)
    neutral_mass = sum(w_i : stance_i == NEUTRAL)

Worked example of why this differs from averaging confidence:
    PubMed  (cred 0.95) says REFUTES   at confidence 0.60 -> w = 0.570
    Blog    (cred 0.35) says SUPPORTS  at confidence 0.85 -> w = 0.298
  Plain averaging picks SUPPORTS (0.85 > 0.60); weighting picks REFUTES.

Two different ratios, deliberately kept separate:
  - The reported *_ratio fields are shares of total evidence mass including
    neutral, so they sum to 1 and describe what the corpus looks like.
  - The verdict is decided by the support-vs-refute contest alone, since neutral
    evidence is not a vote for either side.
  An earlier version divided only by (support + refute) and then derived
  neutral_ratio as the remainder, which made it identically zero and left the
  UI's neutral segment permanently empty.

Uncertainty is propagated, not discarded: MC dropout spread from the stance
detector discounts the reported confidence, so "many sources agree but the model
is unstable" does not read the same as genuine consensus.
"""

import logging

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

EPSILON = 1e-9


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
                float(e.get("stance_confidence", 0.0)) * float(e.get("credibility_score", 0.5))
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
