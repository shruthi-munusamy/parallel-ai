"""
stance_detector.py - DeBERTa-v3 stance detection with uncertainty quantification

Two modes:
  - Fine-tuned (production): loads from local checkpoint after FEVER+VitaminC training
  - Zero-shot fallback: uses cross-encoder/nli-deberta-v3-base if no checkpoint found

Uncertainty quantification via Monte Carlo dropout:
  At inference, run T forward passes with dropout ENABLED (model.train() mode).
  The variance across passes gives a calibrated confidence interval on the verdict.
  This is meaningful signal: high variance = model is uncertain, surface to user.
"""

import os
import numpy as np
import torch
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# Path to fine-tuned checkpoint (produced by train_stance.py)
FINETUNED_CHECKPOINT = "./checkpoints/stance_deberta"

# Fallback: strong zero-shot NLI model (already fine-tuned on MNLI/FEVER)
FALLBACK_MODEL = "cross-encoder/nli-deberta-v3-base"

# MC dropout: number of stochastic forward passes
MC_SAMPLES = 10


class StanceDetector:
    def __init__(self):
        checkpoint = (
            FINETUNED_CHECKPOINT
            if os.path.exists(FINETUNED_CHECKPOINT)
            else FALLBACK_MODEL
        )
        logger.info(f"Loading stance detector from: {checkpoint}")

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.label_map = {0: "REFUTES", 1: "SUPPORTS", 2: "NEUTRAL"}
        logger.info(f"✓ Stance detector ready on {self.device}")

    def _forward(self, premise: str, hypothesis: str) -> np.ndarray:
        """Single forward pass → softmax probabilities [3]"""
        inputs = self.tokenizer(
            premise[:500],
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        return torch.softmax(logits, dim=-1)[0].cpu().numpy()

    def _mc_predict(self, premise: str, hypothesis: str):
        """
        Monte Carlo dropout inference.

        Why: Standard softmax confidence is poorly calibrated — a model can be
        confident and wrong. MC dropout approximates Bayesian inference by
        sampling T predictions with dropout active, giving us both a mean
        prediction and an epistemic uncertainty estimate.

        Returns:
          mean_probs: averaged softmax over T passes
          std:        standard deviation (uncertainty signal)
        """
        # Enable dropout for stochastic passes
        self.model.train()

        samples = []
        inputs = self.tokenizer(
            premise[:500],
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            for _ in range(MC_SAMPLES):
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                samples.append(probs)

        # Return to eval mode
        self.model.eval()

        samples = np.array(samples)          # [T, 3]
        mean_probs = samples.mean(axis=0)    # [3]
        std = samples.std(axis=0).mean()     # scalar uncertainty

        return mean_probs, float(std)

    async def predict(self, claim: str, evidence: str) -> dict:
        """
        Predict stance of evidence toward claim.

        NLI framing:
          premise    = evidence passage (what the source says)
          hypothesis = claim (what we're checking)

        Labels: SUPPORTS (entailment), NEUTRAL, REFUTES (contradiction)
        """
        try:
            mean_probs, uncertainty = self._mc_predict(evidence, claim)

            pred_idx = int(mean_probs.argmax())
            confidence = float(mean_probs[pred_idx])
            stance = self.label_map[pred_idx]

            # Low confidence → downgrade to UNRELATED
            if confidence < 0.40:
                stance = "UNRELATED"

            return {
                "stance":            stance,
                "stance_confidence": confidence,
                "uncertainty":       uncertainty,   # surfaced in API response
            }

        except Exception as e:
            logger.error(f"Stance prediction error: {e}")
            return {"stance": "NEUTRAL", "stance_confidence": 0.0, "uncertainty": 1.0}

    async def predict_batch(self, claim: str, passages: list) -> list:
        for p in passages:
            text = p.get("excerpt") or p.get("text", "")
            pred = await self.predict(claim, text)
            p.update(pred)
        return passages
