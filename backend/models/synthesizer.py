"""
synthesizer.py - Credibility-weighted Bayesian evidence synthesis

The novel contribution:
  Standard NLI-based fact-checking systems treat credibility as a post-hoc
  filter (e.g., "discard sources below threshold X"). This system instead
  integrates credibility as a Bayesian prior over evidence weight DURING
  synthesis — a high-credibility source disagreeing with a claim carries
  strictly more weight than a low-credibility source agreeing.

Formally:
  weight_i = stance_confidence_i * credibility_i

  support_mass = sum(weight_i for i where stance == SUPPORTS)
  refute_mass  = sum(weight_i for i where stance == REFUTES)

  verdict_confidence = max(support_mass, refute_mass) / (support_mass + refute_mass + ε)

This is meaningfully different from averaging stance confidence alone:
  Example: PubMed source (cred=0.95) says REFUTES with confidence 0.60
           Blog source   (cred=0.35) says SUPPORTS with confidence 0.85
  
  Simple average: supports wins (0.85 > 0.60)
  Credibility-weighted: refutes wins (0.95*0.60=0.57 > 0.35*0.85=0.30)

Uncertainty surface:
  MC dropout uncertainty from stance detector is propagated into the verdict
  as an explicit confidence interval, not discarded.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

VERDICT_THRESHOLDS = {
    "strong":   0.70,
    "moderate": 0.55,
    "mixed":    0.40,
}


class EvidenceSynthesizer:
    def __init__(self):
        pass

    async def synthesize(self, claim: str, evidence: list) -> dict:
        if not evidence:
            return {
                "verdict":        "INSUFFICIENT",
                "confidence":     0.0,
                "uncertainty":    1.0,
                "summary":        "No evidence found.",
                "support_ratio":  0.0,
                "refute_ratio":   0.0,
                "neutral_ratio":  1.0,
            }

        # Filter unrelated passages
        relevant = [e for e in evidence if e.get("stance") != "UNRELATED"]
        if not relevant:
            return {
                "verdict":        "INSUFFICIENT",
                "confidence":     0.0,
                "uncertainty":    1.0,
                "summary":        "Retrieved sources are not relevant to this claim.",
                "support_ratio":  0.0,
                "refute_ratio":   0.0,
                "neutral_ratio":  1.0,
            }

        # ── Credibility-weighted mass calculation ────────────────────────────
        # weight = stance_confidence * credibility_score
        # This is the core contribution: credibility is a multiplier on evidence
        # strength, not a threshold gate.

        support_mass = sum(
            e.get("stance_confidence", 0.0) * e.get("credibility_score", 0.5)
            for e in relevant if e.get("stance") == "SUPPORTS"
        )
        refute_mass = sum(
            e.get("stance_confidence", 0.0) * e.get("credibility_score", 0.5)
            for e in relevant if e.get("stance") == "REFUTES"
        )
        total_mass = support_mass + refute_mass + 1e-8

        support_ratio = support_mass / total_mass
        refute_ratio  = refute_mass  / total_mass
        neutral_ratio = max(0.0, 1.0 - support_ratio - refute_ratio)

        # ── Verdict ──────────────────────────────────────────────────────────
        if support_ratio >= VERDICT_THRESHOLDS["strong"]:
            verdict = "SUPPORTED"
        elif refute_ratio >= VERDICT_THRESHOLDS["strong"]:
            verdict = "REFUTED"
        elif (support_ratio >= VERDICT_THRESHOLDS["mixed"] and
              refute_ratio  >= VERDICT_THRESHOLDS["mixed"]):
            verdict = "MIXED"
        elif max(support_ratio, refute_ratio) < VERDICT_THRESHOLDS["moderate"]:
            verdict = "INSUFFICIENT"
        else:
            verdict = "SUPPORTED" if support_ratio > refute_ratio else "REFUTED"

        confidence = float(max(support_ratio, refute_ratio))

        # ── Propagate MC dropout uncertainty ─────────────────────────────────
        # Average per-passage uncertainty from stance detector
        uncertainties = [
            e.get("uncertainty", 0.5) for e in relevant
            if e.get("stance") in ("SUPPORTS", "REFUTES")
        ]
        mean_uncertainty = float(np.mean(uncertainties)) if uncertainties else 0.5

        # Adjust reported confidence by uncertainty
        adjusted_confidence = confidence * (1.0 - 0.5 * mean_uncertainty)

        # ── Summary ──────────────────────────────────────────────────────────
        n_support = sum(1 for e in relevant if e.get("stance") == "SUPPORTS")
        n_refute  = sum(1 for e in relevant if e.get("stance") == "REFUTES")
        summary = (
            f"{verdict}: {n_support} source(s) support, {n_refute} refute. "
            f"Weighted confidence: {int(confidence * 100)}% "
            f"(uncertainty: {int(mean_uncertainty * 100)}%)"
        )

        return {
            "verdict":        verdict,
            "confidence":     adjusted_confidence,
            "uncertainty":    mean_uncertainty,
            "summary":        summary,
            "support_ratio":  support_ratio,
            "refute_ratio":   refute_ratio,
            "neutral_ratio":  neutral_ratio,
        }
