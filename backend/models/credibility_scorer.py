"""
credibility_scorer.py - Source credibility as a Bayesian prior on evidence weight.

Design decision: rule-based rather than learned, because
  1. no labeled credibility dataset exists at the scale this would need,
  2. domain authority is stable and well understood, and
  3. the ML capacity in this system belongs in the stance detector -- credibility
     is a prior, not the prediction.

Production would replace the tier lists with NewsGuard ratings or a classifier
trained on Media Bias/Fact Check and Ad Fontes labels. The interface would not
change: a source dict in, a scalar in [0, 1] out.

The score is consumed by the synthesizer as a multiplier on stance confidence,
so it must be a graded weight rather than a pass/fail gate -- a low-credibility
source still contributes, just less.
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Tier 1: peer-reviewed literature, government and intergovernmental bodies.
HIGH_CREDIBILITY = {
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "nature.com", "science.org", "sciencemag.org",
    "nejm.org", "thelancet.com", "jamanetwork.com", "bmj.com",
    "frontiersin.org", "plos.org", "cell.com",
    "cdc.gov", "nih.gov", "who.int", "fda.gov", "epa.gov",
    "nasa.gov", "noaa.gov", "nist.gov", "census.gov", "ipcc.ch",
    "cochrane.org", "cochranelibrary.com",
}

# Tier 2: reputable journalism, major clinical institutions, universities.
GOOD_CREDIBILITY = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "wsj.com", "economist.com", "ft.com", "npr.org", "pbs.org",
    "mayoclinic.org", "clevelandclinic.org", "hopkinsmedicine.org",
    "britannica.com",
    "stanford.edu", "mit.edu", "harvard.edu", "ox.ac.uk", "cam.ac.uk",
}

# Tier 3: general interest, science journalism, crowd-edited reference.
NEUTRAL_CREDIBILITY = {
    "wikipedia.org", "healthline.com", "webmd.com", "medicalnewstoday.com",
    "medium.com", "forbes.com", "time.com", "theatlantic.com",
    "vox.com", "slate.com", "salon.com", "newsweek.com",
    "sciencedaily.com", "livescience.com", "newscientist.com",
    "arxiv.org", "ssrn.com", "biorxiv.org", "medrxiv.org",
}

# Tier 4: user-generated or aggregated content with no editorial review.
LOW_CREDIBILITY = {
    "quora.com", "reddit.com", "answers.yahoo.com", "ask.com",
    "facebook.com", "x.com", "twitter.com", "tiktok.com",
    "pinterest.com", "substack.com", "blogspot.com", "wordpress.com",
}

SCORES = {
    "high": 0.92,
    "good": 0.75,
    "neutral": 0.55,
    "low": 0.25,
    "unknown": 0.40,
}

# Preprint servers carry a deliberate caveat: arXiv/bioRxiv/medRxiv are indexed
# as tier 3 above rather than tier 1 precisely because they are not peer
# reviewed, even though their content is often academic.

_TLD_SCORES = (
    (".gov", SCORES["high"]),
    (".mil", SCORES["high"]),
    (".edu", SCORES["good"]),
    (".ac.uk", SCORES["good"]),
    (".org", 0.60),
)


class CredibilityScorer:
    def __init__(self):
        logger.info("Credibility scorer ready")

    @staticmethod
    def _hostname(url: str) -> str:
        try:
            netloc = urlparse(url).netloc.lower()
        except Exception:
            return ""
        # Strip port and a leading www. so "www.nature.com:443" matches "nature.com".
        host = netloc.split("@")[-1].split(":")[0]
        return host[4:] if host.startswith("www.") else host

    @staticmethod
    def _matches(host: str, domains: set[str]) -> bool:
        """
        True when host equals a listed domain or is a subdomain of one.

        Suffix matching rather than exact equality so that "blogs.nature.com"
        and "pmc.ncbi.nlm.nih.gov" inherit their parent's tier instead of
        silently dropping to the unknown default.
        """
        return any(host == d or host.endswith("." + d) for d in domains)

    async def score(self, passage: dict) -> float:
        url = passage.get("source_url", "") or ""
        host = self._hostname(url)
        if not host:
            return SCORES["unknown"]

        # Ordered most to least specific: an explicit tier always wins over the
        # TLD heuristic, so a low-credibility .org is not promoted by its suffix.
        if self._matches(host, HIGH_CREDIBILITY):
            return SCORES["high"]
        if self._matches(host, GOOD_CREDIBILITY):
            return SCORES["good"]
        if self._matches(host, NEUTRAL_CREDIBILITY):
            return SCORES["neutral"]
        if self._matches(host, LOW_CREDIBILITY):
            return SCORES["low"]

        for suffix, score in _TLD_SCORES:
            if host.endswith(suffix):
                return score

        return SCORES["unknown"]

    async def score_all(self, passages: list[dict]) -> list[dict]:
        """Annotate passages in place with credibility_score."""
        for passage in passages:
            passage["credibility_score"] = await self.score(passage)
        return passages
