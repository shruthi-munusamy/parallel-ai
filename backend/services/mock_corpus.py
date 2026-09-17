"""
mock_corpus.py - Offline stand-in for live web search.

Kept deliberately realistic rather than lorem-ipsum: snippets carry genuine
stance-bearing language (entailment and contradiction), and domains span the
credibility tiers, so the stance detector and the credibility-weighted
synthesizer are exercised for real when no API key is present. A corpus of
bland placeholder text would make every passage NEUTRAL and hide bugs in the
two stages that matter most.

Retrieval is deliberate keyword overlap, not embeddings. The semantic layer
belongs to EvidenceRetriever; duplicating it here would mask retrieval bugs by
making the mock look smarter than the real path.
"""

import re

# Each topic supplies both supporting and refuting sources across credibility
# tiers, so mock-backed verdicts are contested rather than unanimous.
_CORPUS: list[dict] = [
    # ── Coffee / Alzheimer's ────────────────────────────────────────────────
    {
        "url": "https://pubmed.ncbi.nlm.nih.gov/mock/22201278",
        "title": "Caffeine intake and cognitive decline: a prospective cohort",
        "snippet": (
            "In a 10-year prospective cohort of 4,197 adults, participants in the highest "
            "quartile of coffee consumption showed a 27% lower incidence of Alzheimer's "
            "disease than non-drinkers. The protective association persisted after "
            "adjustment for age, education, smoking and APOE4 status."
        ),
        "date": "2023-04-11",
        "keywords": ["coffee", "caffeine", "alzheimer", "dementia", "cognitive", "brain"],
    },
    {
        "url": "https://www.nature.com/articles/mock-neuro-2022-118",
        "title": "Re-examining the coffee-neuroprotection hypothesis",
        "snippet": (
            "Pooled analysis across 19 cohorts found no statistically significant "
            "relationship between habitual coffee intake and Alzheimer's risk once "
            "reverse causation was accounted for. Earlier protective estimates appear "
            "driven by preclinical patients reducing intake before diagnosis."
        ),
        "date": "2022-09-30",
        "keywords": ["coffee", "caffeine", "alzheimer", "dementia", "neuroprotection"],
    },
    {
        "url": "https://www.mayoclinic.org/mock/coffee-and-memory",
        "title": "Coffee and memory: what the evidence actually shows",
        "snippet": (
            "Observational studies suggest moderate coffee drinkers may have somewhat "
            "lower rates of cognitive decline, but no randomized trial has demonstrated "
            "that drinking coffee prevents Alzheimer's disease."
        ),
        "date": "2024-01-22",
        "keywords": ["coffee", "memory", "alzheimer", "cognitive", "caffeine"],
    },
    {
        "url": "https://www.healthblogdaily.mock/coffee-cures-brain-fog",
        "title": "Why coffee is the ultimate brain protector",
        "snippet": (
            "Coffee absolutely prevents Alzheimer's disease. Drinking six cups a day "
            "will keep your brain sharp forever, and the science is completely settled "
            "on this point."
        ),
        "date": "2024-06-02",
        "keywords": ["coffee", "brain", "alzheimer", "caffeine"],
    },
    # ── Vaccines / autism ───────────────────────────────────────────────────
    {
        "url": "https://www.cdc.gov/mock/vaccine-safety/autism.html",
        "title": "Vaccines do not cause autism",
        "snippet": (
            "Multiple large-scale studies covering more than 1.2 million children have "
            "found no link between vaccines and autism spectrum disorder. The 1998 study "
            "that first proposed such a link was retracted after its data were found to "
            "have been falsified."
        ),
        "date": "2024-02-14",
        "keywords": ["vaccine", "vaccines", "autism", "mmr", "immunization", "thimerosal"],
    },
    {
        "url": "https://www.thelancet.com/mock/retraction-notice-1998",
        "title": "Retraction: Ileal-lymphoid-nodular hyperplasia and pervasive development disorder",
        "snippet": (
            "Following the findings of the General Medical Council, it has become clear "
            "that several elements of the 1998 paper are incorrect. We therefore fully "
            "retract this paper from the published record. No causal link between the MMR "
            "vaccine and autism was established."
        ),
        "date": "2010-02-02",
        "keywords": ["vaccine", "mmr", "autism", "retraction", "wakefield"],
    },
    {
        "url": "https://jamanetwork.com/mock/cohort-657461",
        "title": "MMR vaccination and autism in a nationwide cohort",
        "snippet": (
            "Among 657,461 children followed for a mean of 8.6 years, MMR vaccination was "
            "not associated with increased risk of autism, and did not trigger autism in "
            "susceptible children or cluster of cases after vaccination."
        ),
        "date": "2019-03-05",
        "keywords": ["vaccine", "mmr", "autism", "cohort", "immunization"],
    },
    {
        "url": "https://www.naturalhealthtruth.mock/vaccine-dangers",
        "title": "The vaccine-autism connection they won't tell you about",
        "snippet": (
            "Vaccines cause autism in genetically vulnerable children, and the preservatives "
            "used in childhood immunization schedules are responsible for the rise in autism "
            "diagnoses over the past three decades."
        ),
        "date": "2023-11-19",
        "keywords": ["vaccine", "vaccines", "autism", "immunization", "thimerosal"],
    },
    # ── Climate change ──────────────────────────────────────────────────────
    {
        "url": "https://www.nasa.gov/mock/climate/evidence",
        "title": "Scientific consensus: Earth's climate is warming",
        "snippet": (
            "Multiple independent lines of evidence show that the observed warming since "
            "the mid-20th century is caused primarily by human emission of greenhouse "
            "gases. More than 97% of actively publishing climate scientists agree."
        ),
        "date": "2025-01-08",
        "keywords": ["climate", "warming", "greenhouse", "emissions", "human", "carbon"],
    },
    {
        "url": "https://www.science.org/mock/attribution-2024",
        "title": "Attribution of observed warming to anthropogenic forcing",
        "snippet": (
            "Detection and attribution analyses indicate that anthropogenic forcing "
            "accounts for essentially all of the observed global mean surface warming "
            "since 1950, with natural forcings contributing negligibly."
        ),
        "date": "2024-07-17",
        "keywords": ["climate", "warming", "anthropogenic", "attribution", "carbon", "human"],
    },
    {
        "url": "https://www.reuters.com/mock/climate-record-temps",
        "title": "Global temperatures hit record high for second straight year",
        "snippet": (
            "Scientists attribute the record temperatures to continued greenhouse gas "
            "emissions from human activity, compounded by a strong El Nino cycle."
        ),
        "date": "2025-01-10",
        "keywords": ["climate", "temperature", "warming", "emissions", "global"],
    },
    {
        "url": "https://www.climatereality-check.mock/natural-cycles",
        "title": "Climate has always changed naturally",
        "snippet": (
            "Current warming is part of a natural cycle and is not caused by human "
            "activity. Solar variation and ocean oscillations explain the observed "
            "temperature record without invoking carbon dioxide."
        ),
        "date": "2024-03-21",
        "keywords": ["climate", "warming", "natural", "solar", "cycle", "carbon"],
    },
    # ── Intermittent fasting ────────────────────────────────────────────────
    {
        "url": "https://www.nejm.org/mock/fasting-metabolic-2023",
        "title": "Effects of time-restricted eating on weight and metabolic risk",
        "snippet": (
            "In a randomized trial of 139 patients with obesity, time-restricted eating "
            "produced weight loss of 8.0 kg versus 6.3 kg for daily calorie restriction, "
            "a difference that was not statistically significant. Metabolic risk markers "
            "improved similarly in both groups."
        ),
        "date": "2023-04-20",
        "keywords": ["fasting", "intermittent", "time-restricted", "weight", "metabolic", "eating", "diet"],
    },
    {
        "url": "https://pubmed.ncbi.nlm.nih.gov/mock/fasting-insulin",
        "title": "Intermittent fasting improves insulin sensitivity",
        "snippet": (
            "Participants assigned to an alternate-day fasting protocol showed a "
            "significant improvement in insulin sensitivity and a reduction in fasting "
            "glucose relative to controls over 12 weeks."
        ),
        "date": "2022-11-04",
        "keywords": ["fasting", "intermittent", "insulin", "glucose", "metabolic", "diet"],
    },
    {
        "url": "https://www.healthline.com/mock/intermittent-fasting-guide",
        "title": "Intermittent fasting: benefits and drawbacks",
        "snippet": (
            "Evidence suggests intermittent fasting can support weight loss, though most "
            "trials find it no more effective than conventional calorie restriction when "
            "total intake is matched."
        ),
        "date": "2024-05-30",
        "keywords": ["fasting", "intermittent", "weight", "diet", "eating"],
    },
    # ── 5G / health ─────────────────────────────────────────────────────────
    {
        "url": "https://www.who.int/mock/5g-public-health",
        "title": "5G mobile networks and health",
        "snippet": (
            "To date, and after much research, no adverse health effect has been causally "
            "linked with exposure to wireless technologies. Radiofrequency exposure from 5G "
            "base stations remains far below international safety limits."
        ),
        "date": "2024-08-01",
        "keywords": ["5g", "wireless", "radiation", "health", "radiofrequency", "cancer"],
    },
    {
        "url": "https://www.truthsignal.mock/5g-radiation-danger",
        "title": "5G radiation is making people sick",
        "snippet": (
            "5G towers emit dangerous radiation that causes cancer and a range of chronic "
            "illnesses. Communities near new installations report a sharp rise in symptoms."
        ),
        "date": "2023-07-12",
        "keywords": ["5g", "radiation", "cancer", "towers", "health", "wireless"],
    },
    # ── Aspirin / fever (counterfactual negation case) ──────────────────────
    {
        "url": "https://pubmed.ncbi.nlm.nih.gov/mock/aspirin-antipyretic",
        "title": "Antipyretic efficacy of aspirin in febrile adults",
        "snippet": (
            "Aspirin reduces fever by inhibiting prostaglandin synthesis in the "
            "hypothalamus. In controlled trials, a single 650 mg dose lowered core "
            "temperature significantly more than placebo within two hours."
        ),
        "date": "2021-06-18",
        "keywords": ["aspirin", "fever", "antipyretic", "temperature", "salicylate"],
    },
    {
        "url": "https://www.mayoclinic.org/mock/aspirin-uses",
        "title": "Aspirin: uses and side effects",
        "snippet": (
            "Aspirin is used to reduce fever and relieve mild to moderate pain. It "
            "belongs to a class of drugs known as nonsteroidal anti-inflammatory "
            "drugs, which lower fever as well as inflammation."
        ),
        "date": "2024-03-07",
        "keywords": ["aspirin", "fever", "antipyretic", "pain", "nsaid"],
    },
    # ── Exercise / cardiovascular health ────────────────────────────────────
    {
        "url": "https://www.nejm.org/mock/exercise-cardiovascular",
        "title": "Physical activity and cardiovascular outcomes",
        "snippet": (
            "Regular moderate-intensity exercise lowers the risk of coronary heart "
            "disease, reduces resting blood pressure and improves lipid profiles. "
            "Pooled trial data show a 20 to 30 percent reduction in cardiovascular "
            "mortality among consistently active adults."
        ),
        "date": "2023-09-12",
        "keywords": ["exercise", "cardiovascular", "heart", "physical", "activity", "health"],
    },
    {
        "url": "https://www.who.int/mock/physical-activity-guidelines",
        "title": "Physical activity guidelines",
        "snippet": (
            "Adults should undertake at least 150 minutes of moderate-intensity "
            "aerobic activity per week. Doing so improves cardiovascular health, "
            "reduces all-cause mortality and lowers the incidence of hypertension."
        ),
        "date": "2024-06-26",
        "keywords": ["exercise", "physical", "activity", "cardiovascular", "heart", "health"],
    },
    # ── Eight glasses of water (popular myth) ───────────────────────────────
    {
        "url": "https://www.bmj.com/mock/water-intake-myth",
        "title": "Medical myths: we should drink eight glasses of water a day",
        "snippet": (
            "There is no scientific evidence supporting the recommendation that "
            "everyone must drink eight glasses of water daily. The claim appears to "
            "originate from a 1945 guideline that also noted most of that water is "
            "obtained from prepared foods."
        ),
        "date": "2022-12-20",
        "keywords": ["water", "hydration", "glasses", "drink", "intake", "daily"],
    },
    {
        "url": "https://www.mayoclinic.org/mock/water-how-much",
        "title": "Water: how much should you drink every day?",
        "snippet": (
            "Daily fluid needs vary with body size, activity level, climate and "
            "health status. No single figure applies to everyone, and a fixed eight "
            "glasses per day is not a medical requirement."
        ),
        "date": "2024-04-15",
        "keywords": ["water", "hydration", "drink", "glasses", "daily", "intake"],
    },
    # ── Eiffel Tower (simple factual control) ───────────────────────────────
    {
        "url": "https://en.wikipedia.org/wiki/mock/Eiffel_Tower",
        "title": "Eiffel Tower",
        "snippet": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
            "in Paris, France. It is named after the engineer Gustave Eiffel, whose "
            "company designed and built the tower from 1887 to 1889."
        ),
        "date": "2025-03-04",
        "keywords": ["eiffel", "tower", "paris", "france", "landmark", "located"],
    },
    {
        "url": "https://www.britannica.com/mock/eiffel-tower",
        "title": "Eiffel Tower | History and facts",
        "snippet": (
            "The Eiffel Tower stands in Paris, on the Left Bank of the Seine. Erected "
            "for the 1889 Exposition Universelle, it remains the most recognisable "
            "landmark of the French capital."
        ),
        "date": "2024-11-30",
        "keywords": ["eiffel", "tower", "paris", "france", "located", "city"],
    },
    # ── Generic reference fillers ───────────────────────────────────────────
    {
        "url": "https://en.wikipedia.org/wiki/mock/Scientific_method",
        "title": "Scientific method",
        "snippet": (
            "The scientific method is an empirical method of acquiring knowledge involving "
            "careful observation, formulation of hypotheses, and testing of deductions "
            "drawn from those hypotheses through experiment."
        ),
        "date": "2025-02-01",
        "keywords": ["science", "method", "research", "evidence", "study", "hypothesis"],
    },
    {
        "url": "https://www.britannica.com/mock/evidence",
        "title": "Evidence and inference",
        "snippet": (
            "Evaluating a factual claim requires weighing the quality of the sources "
            "reporting it, the methodology behind any study cited, and whether independent "
            "replications reach the same conclusion."
        ),
        "date": "2024-10-09",
        "keywords": ["evidence", "claim", "fact", "study", "research", "source"],
    },
]

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "does", "do",
    "did", "of", "for", "to", "in", "on", "and", "or", "that", "this", "it",
    "with", "as", "at", "by", "from", "can", "will", "about", "evidence",
    "against", "fact", "check", "studies", "study", "scientific", "expert",
    "opinion",
}


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


def search_mock_corpus(query: str, max_results: int = 10) -> list[dict]:
    """
    Rank corpus documents by keyword overlap with the query.

    Documents with zero overlap are dropped rather than padded in: returning
    unrelated filler would let a mock-backed run produce a confident verdict
    from evidence that has nothing to do with the claim.
    """
    tokens = _tokenize(query)
    if not tokens:
        return []

    scored = []
    for doc in _CORPUS:
        overlap = len(tokens & set(doc["keywords"]))
        # Title/snippet overlap counts for less than a curated keyword hit.
        text_overlap = len(tokens & _tokenize(doc["title"] + " " + doc["snippet"]))
        score = overlap * 2 + text_overlap
        if overlap:
            scored.append((score, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [
        {
            "url": doc["url"],
            "title": doc["title"],
            "snippet": doc["snippet"],
            "date": doc["date"],
        }
        for _, doc in scored[:max_results]
    ]
