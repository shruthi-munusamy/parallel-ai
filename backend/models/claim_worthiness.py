"""
claim_worthiness.py - Reject non-factual input before it enters the pipeline.

Why this exists:
  Everything downstream assumes the input is a falsifiable factual claim.
  "Pizza is the best food" or "AI will replace all programmers by 2030" will
  still produce a confident-looking SUPPORTED/REFUTED verdict -- the pipeline
  has no way to represent "this was never a checkable claim to begin with."
  That is a worse failure than INSUFFICIENT: it looks like the system did the
  work and found evidence, when the category error happened before any
  evidence was even searched for.

Why rule-based rather than an LLM call, unlike query decomposition:
  An LLM call here would be the more accurate classifier, and the existing
  QueryDecomposer's Anthropic call could eventually absorb this (one round
  trip instead of two). But that path requires a funded API key, which this
  project does not currently have, and opinion/prediction/question framing is
  syntactically marked often enough that regex catches the obvious cases
  cheaply and with zero external dependency. Like the credibility tier list,
  this is a deliberately simple stand-in for a component whose interface
  (claim in, checkable/not + reason out) will not need to change if it is
  later replaced by something smarter.

False negatives (letting a non-factual claim through) cost nothing the
pipeline doesn't already handle -- it just returns weak/INSUFFICIENT
evidence. False positives (rejecting a real factual claim) cost the user
an answer entirely. The heuristics below are deliberately conservative:
only the clear cases are rejected, everything ambiguous passes through.
"""

import re

# A trailing "?" is the strongest, least ambiguous signal available: a
# question is a request for information, not an assertion to verify.
_QUESTION = re.compile(r"\?\s*$")

# First-person and prescriptive framing marks a value judgment, not a
# claim about the world that evidence could support or refute.
_OPINION = re.compile(
    r"\b(i think|i believe|i feel|in my opinion|imo|"
    r"we (?:should|must)|you should|ought to)\b",
    re.IGNORECASE,
)

# Comparative/superlative aesthetic or preference judgments. Deliberately
# narrow: "X is the best-selling Y" or "X is the largest Y" are factual
# (verifiable against sales figures or measurements), so only judgments with
# no implied objective measure are matched -- the (?!-) excludes compound
# modifiers like "best-selling" or "worst-performing", which carry one.
_SUBJECTIVE_JUDGMENT = re.compile(
    r"\b(is|are|was|were) (the )?(best|worst|greatest|most beautiful|"
    r"most delicious|tastiest|prettiest|ugliest)\b(?!-)|"
    r"\btastes? (better|worse) than\b",
    re.IGNORECASE,
)

# Speculative future framing. Deliberately narrow to modal/forecasting verbs
# rather than any future tense, so "the meeting will start at 9am tomorrow"
# (a scheduled fact) is not caught alongside "AI will replace programmers."
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
        """
        Returns {"checkable": bool, "category": str | None, "reason": str | None}.

        Runs the categories in order and returns the first match; a claim
        rarely trips more than one, and the first is as good an explanation
        as any combination would be.
        """
        for category, pattern, reason in _CATEGORIES:
            if pattern.search(claim):
                return {"checkable": False, "category": category, "reason": reason}
        return {"checkable": True, "category": None, "reason": None}
