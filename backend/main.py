"""
main.py - FastAPI server for Neural Claim Verification

Pipeline:
  1. QueryDecomposer   → LLM-generated sub-queries
  2. SearchService     → web retrieval
  3. EvidenceRetriever → semantic passage ranking
  4. StanceDetector    → DeBERTa NLI with MC dropout uncertainty
  5. CredibilityScorer → domain authority heuristic
  6. EvidenceSynthesizer → credibility-weighted Bayesian verdict
"""

import time
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.models.query_decomposer  import QueryDecomposer
from backend.models.retriever         import EvidenceRetriever
from backend.models.stance_detector   import StanceDetector
from backend.models.credibility_scorer import CredibilityScorer
from backend.models.synthesizer       import EvidenceSynthesizer
from backend.services.search_service  import SearchService
from backend.services.cache_service   import CacheService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model singletons (loaded once at startup) ─────────────────────────────────
pipeline = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    pipeline["decomposer"]  = QueryDecomposer()
    pipeline["retriever"]   = EvidenceRetriever()
    pipeline["stance"]      = StanceDetector()
    pipeline["credibility"] = CredibilityScorer()
    pipeline["synthesizer"] = EvidenceSynthesizer()
    pipeline["search"]      = SearchService()
    pipeline["cache"]       = CacheService()
    logger.info("✓ All models loaded")
    yield
    pipeline.clear()


app = FastAPI(
    title="Neural Claim Verification API",
    description="Fact-checking via DeBERTa stance detection + credibility-weighted synthesis",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ClaimRequest(BaseModel):
    claim:       str           = Field(..., min_length=10, max_length=500)
    max_sources: Optional[int] = Field(default=10, ge=5, le=20)


class Evidence(BaseModel):
    source_url:         str
    source_title:       str
    excerpt:            str
    stance:             str
    stance_confidence:  float
    uncertainty:        float
    credibility_score:  float
    publication_date:   Optional[str] = None


class VerificationResponse(BaseModel):
    claim:              str
    verdict:            str
    confidence:         float
    uncertainty:        float
    evidence:           List[Evidence]
    consensus_summary:  str
    processing_time_ms: int
    sources_analyzed:   int
    support_ratio:      float
    refute_ratio:       float
    neutral_ratio:      float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status":    "healthy",
        "timestamp": datetime.now().isoformat(),
        "version":   "2.0.0",
    }


@app.post("/verify", response_model=VerificationResponse)
async def verify_claim(request: ClaimRequest):
    start = time.time()
    logger.info(f"Verifying: {request.claim}")

    # Cache check
    cached = await pipeline["cache"].get(request.claim)
    if cached:
        logger.info("Cache hit")
        return cached

    try:
        # 1. Decompose claim into sub-queries
        queries = await pipeline["decomposer"].decompose(request.claim)
        logger.info(f"Generated {len(queries)} queries")

        # 2. Search
        all_sources = []
        seen_urls = set()
        for query in queries:
            results = await pipeline["search"].search(query, max_results=5)
            for r in results:
                if r["url"] not in seen_urls:
                    all_sources.append(r)
                    seen_urls.add(r["url"])
            if len(all_sources) >= request.max_sources:
                break

        logger.info(f"Retrieved {len(all_sources)} unique sources")

        # 3. Extract and rank evidence passages
        passages = await pipeline["retriever"].extract_evidence(
            request.claim,
            all_sources[:request.max_sources],
            top_k=3
        )

        # 4. Stance detection (DeBERTa + MC dropout)
        passages = await pipeline["stance"].predict_batch(request.claim, passages)

        # 5. Credibility scoring
        for p in passages:
            p["credibility_score"] = await pipeline["credibility"].score(p)

        # 6. Credibility-weighted synthesis
        synthesis = await pipeline["synthesizer"].synthesize(request.claim, passages)

        # Build evidence list
        evidence_list = [
            Evidence(
                source_url=        p.get("source_url", ""),
                source_title=      p.get("source_title", "Unknown"),
                excerpt=           p.get("excerpt", "")[:300],
                stance=            p.get("stance", "NEUTRAL"),
                stance_confidence= p.get("stance_confidence", 0.0),
                uncertainty=       p.get("uncertainty", 0.5),
                credibility_score= p.get("credibility_score", 0.4),
                publication_date=  p.get("date"),
            )
            for p in passages
            if p.get("stance") != "UNRELATED"
        ]

        response = VerificationResponse(
            claim=              request.claim,
            verdict=            synthesis["verdict"],
            confidence=         synthesis["confidence"],
            uncertainty=        synthesis["uncertainty"],
            evidence=           evidence_list,
            consensus_summary=  synthesis["summary"],
            processing_time_ms= int((time.time() - start) * 1000),
            sources_analyzed=   len(passages),
            support_ratio=      synthesis["support_ratio"],
            refute_ratio=       synthesis["refute_ratio"],
            neutral_ratio=      synthesis["neutral_ratio"],
        )

        await pipeline["cache"].set(request.claim, response.model_dump())
        return response

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
