"""
credibility_scorer.py - Source credibility as a Bayesian prior on evidence weight.

Design decision: hand-curated tiers for the small set of sources this project
specifically vetted, two open datasets layered underneath for everything else.
NewsGuard itself is a paid partnership, not something an open-source project
can depend on. CRED-1 (data/cred1_compact.json) was added first (2026-09-30)
to extend the *low* end -- it is built from OpenSources.co and Iffy.news, both
of which exist specifically to flag unreliable/fake/conspiracy domains, so it
covers almost no reliable sources by its own documentation (n=8). Checked
after CRED-1: Wikipedia's Reliable Sources/Perennial Sources list (WP:RSP;
data/rsp_compact.json), which -- unlike CRED-1 -- rates *both* ends, since it
is a community consensus process for judging source reliability in general,
not a watchdog list for misinformation specifically. It adds 621 domains CRED-1
and the hand-curated tiers don't cover (171 of them "generally reliable"),
extending the positive end that CRED-1 structurally cannot. See
RSP_ATTRIBUTION below and _load_rsp()'s docstring for how the raw wikitext was
parsed and how WP:RSP's five-way status maps to a score. CRED-1 is checked
first because it is a narrower, higher-precision list purpose-built for known
bad actors; RSP is broader (frequently-discussed sources in general) and comes
second so a domain in both keeps CRED-1's more targeted call.

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

RSP_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "rsp_compact.json"
RSP_ATTRIBUTION = (
    "Domain reliability data derived from Wikipedia's Reliable sources/"
    "Perennial sources (WP:RSP), https://en.wikipedia.org/wiki/"
    "Wikipedia:Reliable_sources/Perennial_sources -- text available under "
    "CC BY-SA 4.0."
)

# WP:RSP's five-way consensus status, collapsed to a single score. "No
# consensus" sources score the same as the hand-curated NEUTRAL tier;
# "deprecated" and "blacklisted" are scored below plain "generally
# unreliable" since both represent a stronger, formal community judgment
# against the source (deprecation bars its use nearly outright; blacklisting
# is enforced by MediaWiki's spam filter) rather than ordinary unreliability.
RSP_STATUS_SCORES = {
    "generally_reliable": 0.75,
    "no_consensus": 0.55,
    "generally_unreliable": 0.25,
    "deprecated": 0.15,
    "blacklisted": 0.05,
}

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


@functools.lru_cache(maxsize=1)
def _load_rsp() -> dict[str, float]:
    """
    domain -> credibility_score, derived from WP:RSP's five-way consensus
    status (RSP_STATUS_SCORES) rather than any score of CRED-1's own. Built
    by parsing the raw wikitext of the list's 8 alphabetical subpages (each
    row's "s-<status>" row class plus its {{WP:RSPUSES|...}} domain list) --
    not scraped HTML, since the rendered page loses the structured status
    markers. See parse_rsp.py in the dev scratchpad (not checked in) for the
    one-off extraction script; this loader only reads the resulting compact
    JSON. Same degrade-to-empty behavior as _load_cred1 for the same reason.
    """
    try:
        raw = json.loads(RSP_PATH.read_text())
        return {domain: float(entry["s"]) for domain, entry in raw.items()}
    except Exception as e:
        logger.warning("WP:RSP dataset failed to load from %s: %s", RSP_PATH, e)
        return {}


def _lookup_by_suffix(mapping: dict[str, float], host: str) -> float | None:
    """
    Exact host, then progressively shorter parent domains, so a subdomain of
    a listed domain (e.g. "mirror.infowars.com") inherits its score instead
    of silently missing it. Stops one label short of a bare TLD. Shared by
    both open datasets (CRED-1, WP:RSP) since both are flat domain->score
    maps needing the same subdomain-inheritance rule.
    """
    parts = host.split(".")
    for i in range(len(parts) - 1):
        candidate = ".".join(parts[i:])
        if candidate in mapping:
            return mapping[candidate]
    return None


class CredibilityScorer:
    def __init__(self):
        self._cred1 = _load_cred1()
        self._rsp = _load_rsp()
        logger.info(
            "Credibility scorer ready (CRED-1: %d domains, WP:RSP: %d domains)",
            len(self._cred1), len(self._rsp),
        )

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

        cred1_score = _lookup_by_suffix(self._cred1, host)
        if cred1_score is not None:
            return cred1_score

        rsp_score = _lookup_by_suffix(self._rsp, host)
        if rsp_score is not None:
            return rsp_score

        for suffix, score in _TLD_SCORES:
            if host.endswith(suffix):
                return score

        return SCORES["unknown"]

    async def score_all(self, passages: list[dict]) -> list[dict]:
        """Annotate passages in place with credibility_score."""
        for passage in passages:
            passage["credibility_score"] = await self.score(passage)
        return passages
