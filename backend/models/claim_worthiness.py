"""
Reject non-factual input (questions, opinions, subjective judgments,
predictions) before it reaches the pipeline, which would otherwise return a
confident-looking verdict for it.

Rule-based so it needs no API key. The patterns are deliberately conservative:
rejecting a real claim costs the user an answer, while letting an opinion
through just yields weak evidence.
"""

import re

_QUESTION = re.compile(r"\?\s*$")

_OPINION = re.compile(
    r"\b(i think|i believe|i feel|in my opinion|imo|"
    r"we (?:should|must)|you should|ought to)\b",
    re.IGNORECASE,
)

# Only judgments with no objective measure. (?!-) lets compound modifiers like
# "best-selling" through, since sales figures can be checked.
_SUBJECTIVE_JUDGMENT = re.compile(
    r"\b(is|are|was|were) (the )?(best|worst|greatest|most beautiful|"
    r"most delicious|tastiest|prettiest|ugliest)\b(?!-)|"
    r"\btastes? (better|worse) than\b",
    re.IGNORECASE,
)

# Forecasting verbs only, not any future tense: "the meeting will start at 9am"
# is a scheduled fact.
_PREDICTION = re.compile(
    r"\b(will (?:become|replace|surpass|overtake|dominate|disappear|end)|"
    r"is (?:expected|predicted|forecast|projected|likely) to|"
    r"is going to (?:become|replace|surpass|overtake|dominate))\b",
    re.IGNORECASE,
)

_CATEGORIES = [
    ("question", _QUESTION, "This reads as a question, not a factual claim to verify."),
    ("opinion", _OPINION, "This reads as a personal opinion or recommendation, not a checkable factual claim."),
    ("subjective_judgment", _SUBJECTIVE_JUDGMENT, "This is a subjective aesthetic/preference judgment with no objective measure to check it against."),
    ("prediction", _PREDICTION, "This reads as a speculative prediction about the future rather than a claim about a present or past state of the world."),
]


class ClaimWorthinessChecker:
    def check(self, claim: str) -> dict:
        """Returns {"checkable", "category", "reason"} for the first matching category."""
        for category, pattern, reason in _CATEGORIES:
            if pattern.search(claim):
                return {"checkable": False, "category": category, "reason": reason}
        return {"checkable": True, "category": None, "reason": None}
