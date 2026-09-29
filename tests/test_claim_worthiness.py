"""
Claim worthiness tests.

Pure regex, no model weights -- this module runs in well under a second.
False positives (rejecting a real factual claim) are the expensive mistake,
so the "still checkable" cases below are as important as the rejections.
"""

from backend.models.claim_worthiness import ClaimWorthinessChecker


class TestCheckableClaims:
    def setup_method(self):
        self.checker = ClaimWorthinessChecker()

    def test_ordinary_factual_claim_passes(self):
        result = self.checker.check("Vaccines cause autism")
        assert result["checkable"] is True
        assert result["category"] is None
        assert result["reason"] is None

    def test_objective_superlative_with_implied_measure_passes(self):
        """'best-selling' has an objective measure (sales figures) behind it."""
        result = self.checker.check("The iPhone is the best-selling smartphone of 2023")
        assert result["checkable"] is True

    def test_scheduled_future_event_passes(self):
        """A scheduled fact, not a speculative forecast -- 'will' alone must not trip prediction."""
        result = self.checker.check("The meeting will start at 9am tomorrow")
        assert result["checkable"] is True

    def test_measurement_superlative_passes(self):
        result = self.checker.check("Mount Everest is the tallest mountain on Earth")
        assert result["checkable"] is True


class TestQuestions:
    def setup_method(self):
        self.checker = ClaimWorthinessChecker()

    def test_question_is_rejected(self):
        result = self.checker.check("Do vaccines cause autism?")
        assert result["checkable"] is False
        assert result["category"] == "question"

    def test_question_mark_mid_sentence_does_not_trip_it(self):
        """Only a *trailing* question mark signals a question, not one embedded in a quote."""
        result = self.checker.check('The book "Who Moved My Cheese?" topped bestseller lists')
        assert result["checkable"] is True


class TestOpinions:
    def setup_method(self):
        self.checker = ClaimWorthinessChecker()

    def test_first_person_opinion_is_rejected(self):
        result = self.checker.check("I think remote work is more productive than office work")
        assert result["checkable"] is False
        assert result["category"] == "opinion"

    def test_prescriptive_should_is_rejected(self):
        result = self.checker.check("We should ban all plastic straws immediately")
        assert result["checkable"] is False
        assert result["category"] == "opinion"


class TestSubjectiveJudgments:
    def setup_method(self):
        self.checker = ClaimWorthinessChecker()

    def test_aesthetic_superlative_is_rejected(self):
        result = self.checker.check("Paris is the most beautiful city in the world")
        assert result["checkable"] is False
        assert result["category"] == "subjective_judgment"

    def test_taste_comparison_is_rejected(self):
        result = self.checker.check("Pineapple pizza tastes better than pepperoni pizza")
        assert result["checkable"] is False
        assert result["category"] == "subjective_judgment"


class TestPredictions:
    def setup_method(self):
        self.checker = ClaimWorthinessChecker()

    def test_speculative_future_claim_is_rejected(self):
        result = self.checker.check("AI will replace all programmers by 2030")
        assert result["checkable"] is False
        assert result["category"] == "prediction"

    def test_forecast_language_is_rejected(self):
        result = self.checker.check("The economy is expected to enter a recession next year")
        assert result["checkable"] is False
        assert result["category"] == "prediction"


class TestCategoryPriority:
    def setup_method(self):
        self.checker = ClaimWorthinessChecker()

    def test_question_wins_over_opinion_when_both_present(self):
        """Categories are checked in order; a match returns immediately rather than combining."""
        result = self.checker.check("Don't you think we should ban plastic straws?")
        assert result["checkable"] is False
        assert result["category"] == "question"
