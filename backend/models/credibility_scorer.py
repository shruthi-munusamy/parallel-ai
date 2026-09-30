"""
credibility_scorer.py - Source credibility as a Bayesian prior on evidence weight.

Design decision: hand-curated tiers for the positive end, a real open dataset
for the negative end. NewsGuard itself is a paid partnership, not something an
open-source project can depend on, and the obvious open alternative (CRED-1;
see data/cred1_compact.json and CRED1_ATTRIBUTION below) turns out to cover
almost no reliable sources by its own documentation (n=8) -- it is built from
OpenSources.co and Iffy.news, both of which exist specifically to flag
unreliable/fake/conspiracy domains, not to rate good ones. So the positive
tiers (HIGH/GOOD/NEUTRAL_CREDIBILITY below) stay hand-curated, and CRED-1 only
extends the *low* end, which is exactly where a hand-typed 12-domain list
(LOW_CREDIBILITY, all social/UGC platforms) previously had no coverage at all
for actual misinformation domains -- a site like infowars.com fell through
every tier to the 0.40 "unknown" default, scoring *above* Reddit's 0.25.

The score is consumed by the synthesizer as a multiplier on stance confidence,
so it must be a graded weight rather than a pass/fail gate -- a low-credibility
source still contributes, just less.
"""

import functools
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

CRED1_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cred1_compact.json"
CRED1_ATTRIBUTION = (
    "Domain credibility data from CRED-1 by Alexander Loth (CC BY 4.0, "
    "https://github.com/aloth/cred-1), built from OpenSources.co (CC BY 4.0) "
    "by Melissa Zimdars et al. and the Iffy.news Index (MIT)."
)

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


@functools.lru_cache(maxsize=1)
def _load_cred1() -> dict[str, float]:
    """
    domain -> credibility_score (0.0-1.0, CRED-1's own composite), loaded once
    and shared across every CredibilityScorer instance. A missing/corrupt file
    degrades to an empty mapping rather than failing scorer construction --
    this is an enrichment of the low end, not something the pipeline depends
    on to run.
    """
    try:
        raw = json.loads(CRED1_PATH.read_text())
        return {domain: float(entry["s"]) for domain, entry in raw.items()}
    except Exception as e:
        logger.warning("CRED-1 dataset failed to load from %s: %s", CRED1_PATH, e)
        return {}


class CredibilityScorer:
    def __init__(self):
        self._cred1 = _load_cred1()
        logger.info("Credibility scorer ready (CRED-1: %d domains)", len(self._cred1))

    def _cred1_score(self, host: str) -> float | None:
        """
        Exact host, then progressively shorter parent domains, so a subdomain
        of a flagged domain (e.g. "mirror.infowars.com") inherits its score
        instead of silently missing it. Stops one label short of a bare TLD.
        """
        parts = host.split(".")
        for i in range(len(parts) - 1):
            candidate = ".".join(parts[i:])
            if candidate in self._cred1:
                return self._cred1[candidate]
        return None

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

        cred1_score = self._cred1_score(host)
        if cred1_score is not None:
            return cred1_score

        for suffix, score in _TLD_SCORES:
            if host.endswith(suffix):
                return score

        return SCORES["unknown"]

    async def score_all(self, passages: list[dict]) -> list[dict]:
        """Annotate passages in place with credibility_score."""
        for passage in passages:
            passage["credibility_score"] = await self.score(passage)
        return passages
