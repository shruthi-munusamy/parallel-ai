"""
DeBERTa-v3 NLI stance detection with MC dropout uncertainty.

Premise = evidence, hypothesis = claim; entailment / contradiction / neutral
map to SUPPORTS / REFUTES / NEUTRAL. Uses the fine-tuned checkpoint from
train_stance.py when present, else a zero-shot MNLI model.

Labels are read from the checkpoint's id2label rather than hardcoded: the two
checkpoints order them differently, and a fixed map would silently swap
SUPPORTS and NEUTRAL.

Uncertainty is the spread across T forward passes with dropout enabled.
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

# A sentence that attributes the claim to someone ("a blog claims X") scores as
# entailment on lexical overlap alone (McCoy et al. 2019), so a fact-check's
# restatement of the claim it refutes reads as support. An echo needs both an
# attribution marker and high overlap, so independent sentences that merely
# share vocabulary are kept. See ARCHITECTURE.md §12.
_CLAIM_ATTRIBUTION = re.compile(
    r"\b(says?|said|saying|claims?|claimed|alleges?|alleged|stated|states)\b"
    r"|[“”\"]",
    re.IGNORECASE,
)
_ECHO_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "that", "this", "it", "as",
    "at", "by", "with", "from", "has", "have", "had", "will", "would",
    "its", "their", "his", "her", "he", "she", "they", "we", "you", "i",
    "sen", "rep", "mr", "mrs", "ms", "dr",
}
_ECHO_OVERLAP_THRESHOLD = 0.5


def _is_claim_echo(sentence: str, claim: str) -> bool:
    if not _CLAIM_ATTRIBUTION.search(sentence):
        return False
    claim_words = set(re.findall(r"[a-z0-9]+", claim.lower())) - _ECHO_STOPWORDS
    if not claim_words:
        return False
    sentence_words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
    overlap = len(claim_words & sentence_words) / len(claim_words)
    return overlap >= _ECHO_OVERLAP_THRESHOLD


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
        """Toggle only nn.Dropout; model.train() would also flip normalization layers."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train(enabled)

    @torch.no_grad()
    def _predict_probs(self, pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        """
        Run MC dropout over a batch of (premise, hypothesis) pairs.

        Returns (mean_probs [B, C], uncertainty [B]).
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
            # fork_rng seeds the dropout masks without touching global RNG state.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config.MC_SEED)
                for _ in range(self.mc_samples):
                    logits = self.model(**inputs).logits
                    scaled = logits / self.temperature
                    samples.append(torch.softmax(scaled, dim=-1).cpu().numpy())
        finally:
            self._set_dropout_training(False)

        stacked = np.stack(samples)                      # [T, B, C]
        mean_probs = stacked.mean(axis=0)                # [B, C]
        # Mean per-class std across passes; 0 when T=1.
        uncertainty = stacked.std(axis=0).mean(axis=-1)  # [B]
        return mean_probs, uncertainty

    def _to_result(self, probs: np.ndarray, uncertainty: float) -> dict:
        index = int(probs.argmax())
        confidence = float(probs[index])
        stance = self.label_map[index]

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
            return await asyncio.to_thread(run)
        except Exception as e:
            # A model failure yields NEUTRAL at maximum uncertainty.
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

        # Score every sentence of every passage in one batch, then regroup.
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
            passage.update(self._aggregate_sentences(claim, group, group_predictions))

        return passages

    @staticmethod
    def _aggregate_sentences(claim: str, sentences: list[str], predictions: list[dict]) -> dict:
        """
        Fold sentence predictions into one passage stance.

        The most confident directional sentence wins (max, not mean, so
        background sentences cannot bury the one that carries the evidence).
        Claim echoes are excluded first, unless every sentence is an echo.
        """
        if not predictions:
            return {"stance": NEUTRAL, "stance_confidence": 0.0, "uncertainty": 1.0, "raw_scores": {}}

        pairs = list(zip(predictions, sentences))
        non_echo = [(p, s) for p, s in pairs if not _is_claim_echo(s, claim)]
        pool = non_echo or pairs

        directional = [(p, s) for p, s in pool if p["stance"] in (SUPPORTS, REFUTES)]

        if directional:
            best, sentence = max(directional, key=lambda pair: pair[0]["stance_confidence"])
        else:
            best, sentence = max(pool, key=lambda pair: pair[0]["stance_confidence"])

        return {**best, "stance_sentence": sentence}
