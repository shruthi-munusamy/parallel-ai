"""
stance_detector.py - DeBERTa-v3 stance detection with uncertainty quantification.

This is the intelligence layer of the system; everything around it is plumbing.
The task is Natural Language Inference: given a premise (an evidence passage)
and a hypothesis (the claim), decide entailment / contradiction / neutral, which
maps onto SUPPORTS / REFUTES / NEUTRAL.

Two modes:
  - Fine-tuned: loads from a local checkpoint produced by train_stance.py.
  - Zero-shot fallback: cross-encoder/nli-deberta-v3-base, already trained on
    MNLI, used when no checkpoint exists.

Label mapping is read from the model's own config.id2label rather than
hardcoded. The two checkpoints order their labels differently -- the fallback is
(contradiction, entailment, neutral) while train_stance.py emits
(REFUTES, NOT ENOUGH INFO, SUPPORTS) -- so a hardcoded map silently inverts
SUPPORTS and NEUTRAL the moment a fine-tuned checkpoint is dropped in. That
failure is invisible: the API keeps returning confident, wrong verdicts.

Uncertainty via Monte Carlo dropout:
  Softmax confidence is poorly calibrated -- a model can be confidently wrong.
  Running T forward passes with dropout left enabled approximates sampling from
  the posterior over model weights; the spread across those samples is an
  epistemic uncertainty estimate. High spread means the model does not know,
  which is different from the model predicting NEUTRAL, and the two are worth
  distinguishing to a user reading a fact-check.
"""

import asyncio
import logging
import os
import re

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from backend.config import config

logger = logging.getLogger(__name__)

# Canonical stance vocabulary used by the rest of the pipeline.
SUPPORTS, REFUTES, NEUTRAL, UNRELATED = "SUPPORTS", "REFUTES", "NEUTRAL", "UNRELATED"

# Maps the varied label strings checkpoints use onto that vocabulary.
_LABEL_ALIASES = {
    "entailment": SUPPORTS,
    "supports": SUPPORTS,
    "support": SUPPORTS,
    "contradiction": REFUTES,
    "refutes": REFUTES,
    "refute": REFUTES,
    "neutral": NEUTRAL,
    "not enough info": NEUTRAL,
    "not_enough_info": NEUTRAL,
    "nei": NEUTRAL,
}


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
# Fragments below this length are headings, captions or navigation residue --
# never enough context for an entailment decision.
_MIN_SENTENCE_WORDS = 5


def _split_sentences(text: str, limit: int) -> list[str]:
    """Split a passage into scoreable sentences, longest-first up to `limit`."""
    if not text:
        return []
    candidates = [s.strip() for s in _SENTENCE_BOUNDARY.split(text)]
    sentences = [s for s in candidates if len(s.split()) >= _MIN_SENTENCE_WORDS]

    if len(sentences) <= limit:
        return sentences
    # Keep the longest sentences but restore document order: length is a decent
    # proxy for substance, while order keeps the surfaced excerpt readable.
    keep = sorted(sentences, key=lambda s: len(s.split()), reverse=True)[:limit]
    return [s for s in sentences if s in keep]


class StanceDetector:
    def __init__(self, checkpoint: str | None = None):
        checkpoint = checkpoint or (
            config.STANCE_CHECKPOINT
            if os.path.isdir(config.STANCE_CHECKPOINT)
            else config.STANCE_FALLBACK_MODEL
        )
        self.checkpoint = checkpoint
        self.is_finetuned = checkpoint == config.STANCE_CHECKPOINT
        logger.info("Loading stance detector: %s", checkpoint)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.label_map = self._build_label_map()
        self.mc_samples = max(1, config.MC_SAMPLES)
        # See config.STANCE_TEMPERATURE: 1.0 until calibrate_temperature.py has
        # been run against this specific checkpoint.
        self.temperature = config.STANCE_TEMPERATURE

        logger.info(
            "Stance detector ready on %s | labels=%s | MC samples=%d | temperature=%.3f",
            self.device, self.label_map, self.mc_samples, self.temperature,
        )

    def _build_label_map(self) -> dict[int, str]:
        """Derive index -> canonical stance from the checkpoint's own config."""
        raw = getattr(self.model.config, "id2label", None) or {}
        mapping: dict[int, str] = {}
        for index, label in raw.items():
            canonical = _LABEL_ALIASES.get(str(label).strip().lower())
            if canonical:
                mapping[int(index)] = canonical

        expected = self.model.config.num_labels
        if len(mapping) != expected:
            # An unrecognized label set means we cannot trust any prediction, so
            # fail loudly at startup instead of guessing an ordering.
            raise ValueError(
                f"Cannot map labels from checkpoint {self.checkpoint!r}: "
                f"id2label={raw}. Expected {expected} labels resolvable to "
                f"{{SUPPORTS, REFUTES, NEUTRAL}}."
            )
        return mapping

    # ── Inference ────────────────────────────────────────────────────────────

    def _set_dropout_training(self, enabled: bool) -> None:
        """
        Toggle dropout while leaving the rest of the model in eval mode.

        model.train() would be the shorter way, but it also switches any
        normalization layer that tracks running statistics into training mode.
        Toggling only nn.Dropout keeps MC sampling from mutating model state.
        """
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train(enabled)

    @torch.no_grad()
    def _predict_probs(self, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Run MC dropout over a batch of (premise, hypothesis) pairs.

        Returns (mean_probs [B, C], uncertainty [B]).

        The T passes run over the whole batch rather than per pair: one padded
        forward pass over B pairs costs far less than B separate passes, and the
        stance stage runs T x (passages) times, so this constant factor
        dominates end-to-end latency.
        """
        if not pairs:
            return np.zeros((0, 3)), np.zeros((0,))

        inputs = self.tokenizer(
            [premise[: config.MAX_PREMISE_CHARS] for premise, _ in pairs],
            [hypothesis for _, hypothesis in pairs],
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        samples = []
        if self.mc_samples > 1:
            self._set_dropout_training(True)
        try:
            # fork_rng isolates and restores global RNG state, so seeding the
            # dropout masks here is reproducible without perturbing any other
            # torch consumer in the process. The estimate stays a valid MC
            # average -- it is over a fixed sample of masks rather than a fresh
            # one, which is the tradeoff that buys deterministic verdicts.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.MC_SEED)
                for _ in range(self.mc_samples):
                    logits = self.model(**inputs).logits
                    scaled = logits / self.temperature
                    samples.append(torch.softmax(scaled, dim=-1).cpu().numpy())
        finally:
            # Restored even if a pass raises, so one failure cannot leave the
            # shared model stochastic for every later request.
            self._set_dropout_training(False)

        stacked = np.stack(samples)                      # [T, B, C]
        mean_probs = stacked.mean(axis=0)                # [B, C]
        # Mean per-class standard deviation across passes. With T=1 this is 0,
        # which is honest: a single deterministic pass measures no spread.
        uncertainty = stacked.std(axis=0).mean(axis=-1)  # [B]
        return mean_probs, uncertainty

    def _to_result(self, probs: np.ndarray, uncertainty: float) -> dict:
        index = int(probs.argmax())
        confidence = float(probs[index])
        stance = self.label_map[index]

        # A low-confidence prediction means the passage does not speak to the
        # claim. Marking it UNRELATED excludes it from the verdict entirely,
        # rather than letting weak noise accumulate into a confident result.
        if confidence < config.UNRELATED_THRESHOLD:
            stance = UNRELATED

        return {
            "stance": stance,
            "stance_confidence": confidence,
            "uncertainty": float(uncertainty),
            "raw_scores": {
                self.label_map[i].lower(): float(probs[i]) for i in range(len(probs))
            },
        }

    async def predict(self, claim: str, evidence: str) -> dict:
        """Single-pair wrapper. NLI premise = evidence, hypothesis = claim."""
        results = await self.predict_pairs(claim, [evidence])
        return results[0]

    async def predict_pairs(self, claim: str, evidences: list[str]) -> list[dict]:
        """Predict stance for many evidence strings against one claim."""
        if not evidences:
            return []

        def run() -> list[dict]:
            results: list[dict] = []
            batch_size = max(1, config.STANCE_BATCH_SIZE)
            for start in range(0, len(evidences), batch_size):
                chunk = evidences[start : start + batch_size]
                probs, uncertainty = self._predict_probs([(e, claim) for e in chunk])
                results.extend(
                    self._to_result(probs[i], uncertainty[i]) for i in range(len(chunk))
                )
            return results

        try:
            # Torch inference is CPU-bound and releases the GIL, so off-loading
            # it keeps the event loop responsive to other requests.
            return await asyncio.to_thread(run)
        except Exception as e:
            # Graceful degradation: a model failure yields NEUTRAL at maximum
            # uncertainty, which the synthesizer treats as no signal.
            logger.error("Stance prediction failed: %s", e, exc_info=True)
            return [
                {"stance": NEUTRAL, "stance_confidence": 0.0, "uncertainty": 1.0, "raw_scores": {}}
                for _ in evidences
            ]

    async def predict_batch(self, claim: str, passages: list[dict]) -> list[dict]:
        """Annotate passage dicts in place with stance fields."""
        if not passages:
            return []

        texts = [p.get("text") or p.get("excerpt") or "" for p in passages]

        if not config.SENTENCE_LEVEL_STANCE:
            predictions = await self.predict_pairs(claim, texts)
            for passage, prediction in zip(passages, predictions):
                passage.update(prediction)
            return passages

        # Sentence-level path. Flatten every sentence from every passage into one
        # batch so the whole stage is a single set of padded forward passes,
        # then fold the results back per passage.
        sentences_per_passage = [
            _split_sentences(text, config.MAX_SENTENCES_PER_PASSAGE) or [text]
            for text in texts
        ]
        flat = [s for group in sentences_per_passage for s in group]
        flat_predictions = await self.predict_pairs(claim, flat)

        cursor = 0
        for passage, group in zip(passages, sentences_per_passage):
            group_predictions = flat_predictions[cursor : cursor + len(group)]
            cursor += len(group)
            passage.update(self._aggregate_sentences(group, group_predictions))

        return passages

    @staticmethod
    def _aggregate_sentences(sentences: list[str], predictions: list[dict]) -> dict:
        """
        Fold sentence-level predictions into one passage-level stance.

        Rule: the most confident directional sentence wins; a passage is NEUTRAL
        only when no sentence takes a side.

        Why max rather than mean: NLI entailment over a multi-sentence premise
        requires the *whole* premise to entail the hypothesis, which is the wrong
        semantics for fact-checking. A source refutes a claim if any sentence in
        it refutes the claim, even when the surrounding sentences are background.
        Averaging would let three neutral sentences of context bury the one
        sentence that actually carries the evidence.
        """
        if not predictions:
            return {"stance": NEUTRAL, "stance_confidence": 0.0, "uncertainty": 1.0, "raw_scores": {}}

        directional = [
            (prediction, sentence)
            for prediction, sentence in zip(predictions, sentences)
            if prediction["stance"] in (SUPPORTS, REFUTES)
        ]

        if directional:
            best, sentence = max(directional, key=lambda pair: pair[0]["stance_confidence"])
        else:
            # No side taken. Report the most confident neutral read, and keep its
            # sentence so the UI still points at what was actually scored.
            index = max(range(len(predictions)), key=lambda i: predictions[i]["stance_confidence"])
            best, sentence = predictions[index], sentences[index]

        # The deciding sentence is surfaced so a reader can see exactly which
        # text produced the stance, instead of a paragraph they must re-scan.
        return {**best, "stance_sentence": sentence}
