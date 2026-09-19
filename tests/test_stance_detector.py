"""
Stance detector tests.

These assert behaviour that would break the product, not implementation detail:
label mapping must match the checkpoint, uncertainty must be produced and
bounded, batching must not change per-item results, and failures must degrade
to NEUTRAL rather than raise.
"""

import numpy as np
import pytest

from backend.models.stance_detector import StanceDetector, _LABEL_ALIASES


pytestmark = pytest.mark.slow


class TestLabelMapping:
    def test_label_map_covers_all_classes(self, stance_detector):
        """
        The single highest-consequence invariant in the system. A wrong mapping
        inverts SUPPORTS and REFUTES while every verdict still looks confident,
        so nothing downstream can detect it.
        """
        assert set(stance_detector.label_map.values()) == {"SUPPORTS", "REFUTES", "NEUTRAL"}
        assert len(stance_detector.label_map) == stance_detector.model.config.num_labels

    def test_label_map_derived_from_checkpoint_not_hardcoded(self, stance_detector):
        """Each index must agree with the checkpoint's own id2label."""
        id2label = stance_detector.model.config.id2label
        for index, stance in stance_detector.label_map.items():
            assert _LABEL_ALIASES[str(id2label[index]).lower()] == stance

    def test_unknown_label_set_raises(self, monkeypatch):
        """An unmappable checkpoint must fail loudly at load, not guess."""
        detector = StanceDetector.__new__(StanceDetector)

        class FakeConfig:
            id2label = {0: "label_0", 1: "label_1", 2: "label_2"}
            num_labels = 3

        class FakeModel:
            config = FakeConfig()

        detector.model = FakeModel()
        detector.checkpoint = "fake/checkpoint"
        with pytest.raises(ValueError, match="Cannot map labels"):
            detector._build_label_map()


class TestPredictions:
    async def test_known_nli_pairs(self, stance_detector, test_data):
        """
        Accuracy check on curated pairs. Asserted as an aggregate rather than
        per-case: the zero-shot fallback is not expected to be perfect, and a
        per-case assert would make this a change detector for model updates.
        """
        pairs = test_data["nli_pairs"]
        # Each pair carries its own claim, so these cannot share a batch.
        results = [
            await stance_detector.predict(p["claim"], p["evidence"]) for p in pairs
        ]
        correct = sum(
            1 for p, r in zip(pairs, results) if r["stance"] == p["expected_stance"]
        )
        assert correct >= len(pairs) - 1, (
            "Expected at most one miss on curated NLI pairs, got "
            f"{len(pairs) - correct}: "
            + str([(p["claim"], p["expected_stance"], r["stance"]) for p, r in zip(pairs, results)])
        )

    async def test_direct_contradiction_is_refuted(self, stance_detector):
        result = await stance_detector.predict(
            "The Earth is flat",
            "The Earth is an oblate spheroid, as confirmed by satellite measurements "
            "and direct observation; it is not flat.",
        )
        assert result["stance"] == "REFUTES"
        assert result["stance_confidence"] > 0.5

    async def test_output_shape_and_bounds(self, stance_detector):
        result = await stance_detector.predict("A test claim about coffee", "Coffee is a beverage.")
        assert set(result) >= {"stance", "stance_confidence", "uncertainty", "raw_scores"}
        assert 0.0 <= result["stance_confidence"] <= 1.0
        assert 0.0 <= result["uncertainty"] <= 1.0
        assert result["stance"] in {"SUPPORTS", "REFUTES", "NEUTRAL", "UNRELATED"}
        assert pytest.approx(sum(result["raw_scores"].values()), abs=1e-4) == 1.0


class TestTemperatureScaling:
    async def test_temperature_above_one_softens_confidence_without_flipping_stance(
        self, stance_detector, monkeypatch
    ):
        """
        Calibration (config.STANCE_TEMPERATURE) must move confidence toward 0.5
        without changing which class wins -- that is the whole point of
        temperature scaling versus retraining.
        """
        claim, evidence = (
            "The Earth is flat",
            "The Earth is an oblate spheroid, as confirmed by satellite measurements "
            "and direct observation; it is not flat.",
        )
        monkeypatch.setattr(stance_detector, "temperature", 1.0)
        baseline = await stance_detector.predict(claim, evidence)

        monkeypatch.setattr(stance_detector, "temperature", 4.0)
        softened = await stance_detector.predict(claim, evidence)

        assert softened["stance"] == baseline["stance"]
        assert softened["stance_confidence"] < baseline["stance_confidence"]

    async def test_empty_input_list_returns_empty(self, stance_detector):
        assert await stance_detector.predict_pairs("claim", []) == []
        assert await stance_detector.predict_batch("claim", []) == []

    async def test_predict_batch_annotates_passages(self, stance_detector):
        passages = [
            {"text": "Coffee is widely consumed worldwide."},
            {"text": "Coffee has never been consumed by anyone."},
        ]
        out = await stance_detector.predict_batch("Coffee is a popular drink", passages)
        assert out is passages  # annotated in place
        for p in passages:
            assert "stance" in p and "uncertainty" in p


class TestUncertainty:
    async def test_mc_dropout_produces_nonzero_spread(self, stance_detector):
        """
        With T > 1, dropout must actually be active. A zero spread everywhere
        would mean the sampling is a no-op and the uncertainty column is a lie.
        """
        if stance_detector.mc_samples <= 1:
            pytest.skip("MC sampling disabled via MC_SAMPLES=1")

        texts = [
            "Coffee consumption has been linked to reduced cognitive decline in cohort studies.",
            "There is no evidence that coffee affects cognition in any way.",
            "The chemical composition of coffee includes caffeine and chlorogenic acids.",
        ]
        results = await stance_detector.predict_pairs("Coffee protects the brain", texts)
        assert any(r["uncertainty"] > 0 for r in results)

    async def test_single_sample_has_zero_uncertainty(self, stance_detector, monkeypatch):
        """T=1 measures no spread, and must report 0 rather than a fabricated value."""
        monkeypatch.setattr(stance_detector, "mc_samples", 1)
        result = await stance_detector.predict("A claim about physics", "Physics studies matter.")
        assert result["uncertainty"] == pytest.approx(0.0, abs=1e-9)

    async def test_mc_sampling_is_reproducible(self, stance_detector):
        """
        Seeded dropout masks. Unseeded, the same claim can flip verdicts between
        runs whenever evidence sits near a decision threshold.
        """
        if stance_detector.mc_samples <= 1:
            pytest.skip("MC sampling disabled via MC_SAMPLES=1")

        text = "Coffee consumption was associated with reduced cognitive decline in the cohort."
        first = await stance_detector.predict("Coffee protects the brain", text)
        second = await stance_detector.predict("Coffee protects the brain", text)

        assert first["stance"] == second["stance"]
        assert first["stance_confidence"] == pytest.approx(second["stance_confidence"], abs=1e-9)
        assert first["uncertainty"] == pytest.approx(second["uncertainty"], abs=1e-9)

    async def test_mc_sampling_does_not_disturb_global_rng(self, stance_detector):
        """fork_rng must leave any other torch consumer's stream untouched."""
        import torch

        torch.manual_seed(1234)
        expected = torch.rand(3)

        torch.manual_seed(1234)
        await stance_detector.predict("A claim under test", "Some evidence text here.")
        actual = torch.rand(3)

        assert torch.allclose(expected, actual)

    async def test_dropout_disabled_after_prediction(self, stance_detector):
        """
        The model is a shared singleton. If MC sampling leaves dropout on, every
        later request becomes silently stochastic.
        """
        await stance_detector.predict("A claim", "Some evidence text.")
        import torch.nn as nn

        assert not any(
            m.training for m in stance_detector.model.modules() if isinstance(m, nn.Dropout)
        )


class TestBatching:
    async def test_batching_matches_individual_prediction(self, stance_detector, monkeypatch):
        """
        Batched and one-at-a-time inference must agree. Padding is the usual
        culprit when they diverge: without a correct attention mask, short
        sequences attend to pad tokens and get different logits.
        """
        monkeypatch.setattr(stance_detector, "mc_samples", 1)  # deterministic
        texts = [
            "Short text.",
            "A considerably longer passage of evidence that will require substantially "
            "more tokens than the first one, forcing the tokenizer to pad the shorter "
            "sequence up to this length within the same batch.",
        ]
        claim = "This is a claim under test"

        batched = await stance_detector.predict_pairs(claim, texts)
        individual = [await stance_detector.predict(claim, t) for t in texts]

        for b, i in zip(batched, individual):
            assert b["stance"] == i["stance"]
            assert b["stance_confidence"] == pytest.approx(i["stance_confidence"], abs=1e-4)

    async def test_respects_batch_size_chunking(self, stance_detector, monkeypatch):
        """More inputs than the batch size must still produce one result each."""
        monkeypatch.setattr(stance_detector, "mc_samples", 1)
        from backend.config import config

        monkeypatch.setattr(config, "STANCE_BATCH_SIZE", 2)
        texts = ["Evidence passage number %d." % i for i in range(5)]
        results = await stance_detector.predict_pairs("A claim under test", texts)
        assert len(results) == 5


class TestSentenceLevelScoring:
    """
    The mitigation for MNLI domain shift: the checkpoints are trained on
    single-sentence premises, so a multi-sentence web passage collapses to
    NEUTRAL at near-1.0 confidence even when one sentence plainly decides it.
    """

    def test_split_sentences_filters_fragments(self):
        from backend.models.stance_detector import _split_sentences

        text = "Home. Menu. This is a full sentence carrying real evidential content here."
        assert _split_sentences(text, 6) == [
            "This is a full sentence carrying real evidential content here."
        ]

    def test_split_sentences_respects_limit_and_order(self):
        from backend.models.stance_detector import _split_sentences

        text = " ".join(
            f"Sentence number {i} contains at least five words." for i in range(10)
        )
        result = _split_sentences(text, 3)
        assert len(result) <= 3
        # Surfaced excerpts must stay readable, so document order is preserved.
        assert result == [s for s in _split_sentences(text, 99) if s in result]

    def test_split_sentences_empty(self):
        from backend.models.stance_detector import _split_sentences

        assert _split_sentences("", 6) == []

    async def test_multi_sentence_passage_keeps_directional_stance(self, stance_detector):
        """
        The regression this whole path exists for. Scored whole, this passage
        returns NEUTRAL at 0.89; scored per sentence it returns REFUTES.
        """
        passages = [
            {
                "text": (
                    "Multiple large-scale studies covering more than 1.2 million "
                    "children have found no link between vaccines and autism spectrum "
                    "disorder. The 1998 study that first proposed such a link was "
                    "retracted after its data were found to have been falsified."
                )
            }
        ]
        await stance_detector.predict_batch("Vaccines cause autism", passages)
        assert passages[0]["stance"] == "REFUTES"

    async def test_deciding_sentence_is_surfaced(self, stance_detector):
        """Explainability: the user must see which sentence produced the label."""
        passages = [
            {
                "text": (
                    "The conference was held in Geneva last spring. Aspirin reduces "
                    "fever by inhibiting prostaglandin synthesis in the hypothalamus."
                )
            }
        ]
        await stance_detector.predict_batch("Aspirin reduces fever", passages)
        sentence = passages[0].get("stance_sentence")
        assert sentence
        assert "prostaglandin" in sentence  # not the Geneva filler

    async def test_passage_level_mode_still_works(self, stance_detector, monkeypatch):
        from backend.config import config

        monkeypatch.setattr(config, "SENTENCE_LEVEL_STANCE", False)
        passages = [{"text": "The Earth orbits the Sun once per year."}]
        await stance_detector.predict_batch("The Earth orbits the Sun", passages)
        assert passages[0]["stance"] in {"SUPPORTS", "REFUTES", "NEUTRAL", "UNRELATED"}
        assert "stance_sentence" not in passages[0]

    def test_aggregation_prefers_confident_directional_sentence(self, stance_detector):
        """Max over directional, not mean: context must not bury the evidence."""
        sentences = ["Background context sentence.", "The deciding sentence.", "More context."]
        predictions = [
            {"stance": "NEUTRAL", "stance_confidence": 0.99, "uncertainty": 0.01},
            {"stance": "REFUTES", "stance_confidence": 0.71, "uncertainty": 0.05},
            {"stance": "NEUTRAL", "stance_confidence": 0.98, "uncertainty": 0.02},
        ]
        result = stance_detector._aggregate_sentences(sentences, predictions)
        assert result["stance"] == "REFUTES"
        assert result["stance_sentence"] == "The deciding sentence."

    def test_aggregation_picks_strongest_of_competing_sides(self, stance_detector):
        sentences = ["a", "b"]
        predictions = [
            {"stance": "SUPPORTS", "stance_confidence": 0.62, "uncertainty": 0.1},
            {"stance": "REFUTES", "stance_confidence": 0.93, "uncertainty": 0.1},
        ]
        assert stance_detector._aggregate_sentences(sentences, predictions)["stance"] == "REFUTES"

    def test_aggregation_falls_back_to_neutral(self, stance_detector):
        sentences = ["a", "b"]
        predictions = [
            {"stance": "NEUTRAL", "stance_confidence": 0.80, "uncertainty": 0.1},
            {"stance": "NEUTRAL", "stance_confidence": 0.95, "uncertainty": 0.1},
        ]
        result = stance_detector._aggregate_sentences(sentences, predictions)
        assert result["stance"] == "NEUTRAL"
        assert result["stance_confidence"] == 0.95

    def test_aggregation_handles_empty(self, stance_detector):
        result = stance_detector._aggregate_sentences([], [])
        assert result["stance"] == "NEUTRAL"
        assert result["uncertainty"] == 1.0


class TestDegradation:
    async def test_model_failure_returns_neutral(self, stance_detector, monkeypatch):
        """A broken model must not 500 the API; it yields no-signal NEUTRAL."""

        def boom(*args, **kwargs):
            raise RuntimeError("simulated inference failure")

        monkeypatch.setattr(stance_detector, "_predict_probs", boom)
        results = await stance_detector.predict_pairs("A claim", ["a", "b"])
        assert len(results) == 2
        assert all(r["stance"] == "NEUTRAL" for r in results)
        assert all(r["uncertainty"] == 1.0 for r in results)
        assert all(r["stance_confidence"] == 0.0 for r in results)

    async def test_low_confidence_downgraded_to_unrelated(self, stance_detector, monkeypatch):
        from backend.config import config

        # Force every prediction under the floor.
        monkeypatch.setattr(config, "UNRELATED_THRESHOLD", 0.99)
        result = await stance_detector.predict("A claim", "Some unrelated evidence.")
        assert result["stance"] == "UNRELATED"

    def test_to_result_handles_flat_distribution(self, stance_detector, monkeypatch):
        from backend.config import config

        monkeypatch.setattr(config, "UNRELATED_THRESHOLD", 0.40)
        # Uniform over 3 classes = 0.333, below the 0.40 floor.
        result = stance_detector._to_result(np.array([1 / 3, 1 / 3, 1 / 3]), 0.2)
        assert result["stance"] == "UNRELATED"
