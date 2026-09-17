"""
main.py - FastAPI server for neural claim verification.

Pipeline, in order:
  1. QueryDecomposer    -> LLM-generated sub-queries (LLM, or rule-based fallback)
  2. SearchService      -> concurrent web search over those queries
  3. EvidenceRetriever  -> concurrent page fetch, semantic passage ranking
  4. StanceDetector     -> DeBERTa NLI with MC-dropout uncertainty
  5. CredibilityScorer  -> domain authority prior
  6. EvidenceSynthesizer-> credibility-weighted verdict

Every stage degrades rather than failing: missing API keys fall back to a mock
corpus and rule-based decomposition, failed fetches fall back to search
snippets, and a model failure yields NEUTRAL at full uncertainty. The only
uncaught path is a genuine bug, which surfaces as a 500.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from backend.config import config
from backend.utils.logging_config import configure_logging, set_request_id
from backend.models.query_decomposer import QueryDecomposer
from backend.models.retriever import EvidenceRetriever
from backend.models.stance_detector import StanceDetector
from backend.models.credibility_scorer import CredibilityScorer
from backend.models.synthesizer import EvidenceSynthesizer
from backend.services.search_service import SearchService
from backend.services.cache_service import CacheService

configure_logging(config.LOG_LEVEL)
logger = logging.getLogger(__name__)

VERSION = "2.1.0"

# Loaded once at startup and shared across requests. The transformer weights are
# ~750MB combined; loading them per request would dominate every other cost in
# the system by two orders of magnitude.
pipeline: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    logger.info("Loading models (first run downloads weights, this is slow)...")

    # Constructors are synchronous and CPU-bound; off-loading them lets the
    # event loop start serving /health while weights load.
    def build() -> dict[str, Any]:
        return {
            "decomposer": QueryDecomposer(),
            "retriever": EvidenceRetriever(),
            "stance": StanceDetector(),
            "credibility": CredibilityScorer(),
            "synthesizer": EvidenceSynthesizer(),
            "search": SearchService(),
            "cache": CacheService(),
        }

    pipeline.update(await asyncio.to_thread(build))
    logger.info("Pipeline ready in %.1fs", time.perf_counter() - started)

    yield

    pipeline.clear()
    logger.info("Pipeline shut down")


app = FastAPI(
    title="Neural Claim Verification API",
    description=(
        "Fact-checking via DeBERTa NLI stance detection and "
        "credibility-weighted evidence synthesis."
    ),
    version=VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """Tag every request so interleaved pipeline logs stay separable."""
    set_request_id(uuid.uuid4().hex[:8])
    return await call_next(request)


# ── Schemas ──────────────────────────────────────────────────────────────────

Verdict = Literal["SUPPORTED", "REFUTED", "MIXED", "INSUFFICIENT"]
Stance = Literal["SUPPORTS", "REFUTES", "NEUTRAL", "UNRELATED"]


class ClaimRequest(BaseModel):
    claim: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Factual statement to verify.",
        examples=["Coffee consumption reduces the risk of Alzheimer's disease"],
    )
    max_sources: int = Field(default=10, ge=1, le=20)

    @field_validator("claim")
    @classmethod
    def claim_must_have_content(cls, v: str) -> str:
        # min_length alone accepts 10 spaces, which reaches the pipeline and
        # produces a confident INSUFFICIENT instead of a 422.
        cleaned = " ".join(v.split())
        if len(cleaned) < 10:
            raise ValueError("claim must contain at least 10 non-whitespace characters")
        return cleaned


class Evidence(BaseModel):
    source_url: str
    source_title: str
    excerpt: str
    stance: Stance
    stance_confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    credibility_score: float = Field(ge=0.0, le=1.0)
    publication_date: str | None = None
    # The specific sentence the stance was computed from, so a reader can see
    # what drove the label instead of re-scanning the whole excerpt.
    stance_sentence: str | None = None


class VerificationResponse(BaseModel):
    claim: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence]
    consensus_summary: str
    processing_time_ms: int
    sources_analyzed: int
    passages_analyzed: int
    support_ratio: float
    refute_ratio: float
    neutral_ratio: float
    # True when results came from the offline corpus rather than live search, so
    # a demo run is never mistaken for a real one.
    mock_search: bool = False
    cached: bool = False


# ── Core pipeline ────────────────────────────────────────────────────────────


async def run_pipeline(claim: str, max_sources: int) -> dict:
    """Stages 1-6. Returns the response payload minus timing metadata."""
    stage_start = time.perf_counter()

    # 1. Decompose
    queries = await pipeline["decomposer"].decompose(claim)
    logger.info("Stage 1 decompose: %d queries in %.2fs", len(queries), time.perf_counter() - stage_start)

    # 2. Search (sub-queries run concurrently and are merged by URL)
    stage_start = time.perf_counter()
    search: SearchService = pipeline["search"]
    sources = await search.search_many(queries, max_results=max(5, max_sources // 2))
    sources = sources[:max_sources]
    logger.info("Stage 2 search: %d unique sources in %.2fs", len(sources), time.perf_counter() - stage_start)

    if not sources:
        return {
            "verdict": "INSUFFICIENT",
            "confidence": 0.0,
            "uncertainty": 1.0,
            "evidence": [],
            "consensus_summary": "No sources were retrieved for this claim.",
            "sources_analyzed": 0,
            "passages_analyzed": 0,
            "support_ratio": 0.0,
            "refute_ratio": 0.0,
            "neutral_ratio": 1.0,
            "mock_search": search.is_mock,
        }

    # 3. Retrieve passages
    stage_start = time.perf_counter()
    passages = await pipeline["retriever"].extract_evidence(
        claim, sources, fetch_pages=not search.is_mock
    )
    logger.info("Stage 3 retrieve: %d passages in %.2fs", len(passages), time.perf_counter() - stage_start)

    # 4 & 5. Stance detection and credibility scoring.
    # Independent of one another: stance reads passage text, credibility reads
    # the URL, so they overlap instead of running back to back.
    stage_start = time.perf_counter()
    passages, _ = await asyncio.gather(
        pipeline["stance"].predict_batch(claim, passages),
        pipeline["credibility"].score_all(passages),
    )
    logger.info("Stage 4+5 stance/credibility: %.2fs", time.perf_counter() - stage_start)

    # 6. Synthesize
    synthesis = await pipeline["synthesizer"].synthesize(claim, passages)

    evidence = [
        Evidence(
            source_url=p.get("source_url", ""),
            source_title=p.get("source_title", "Untitled"),
            excerpt=p.get("excerpt", ""),
            stance=p.get("stance", "NEUTRAL"),
            stance_confidence=p.get("stance_confidence", 0.0),
            uncertainty=p.get("uncertainty", 0.5),
            credibility_score=p.get("credibility_score", 0.4),
            publication_date=p.get("date"),
            stance_sentence=p.get("stance_sentence"),
        )
        for p in passages
        if p.get("stance") != "UNRELATED"
    ]
    # Strongest evidence first: weight is exactly the quantity the synthesizer
    # aggregated, so the ordering matches what actually drove the verdict.
    evidence.sort(key=lambda e: e.stance_confidence * e.credibility_score, reverse=True)

    return {
        "verdict": synthesis["verdict"],
        "confidence": synthesis["confidence"],
        "uncertainty": synthesis["uncertainty"],
        "evidence": evidence,
        "consensus_summary": synthesis["summary"],
        "sources_analyzed": len({p.get("source_url") for p in passages}),
        "passages_analyzed": len(passages),
        "support_ratio": synthesis["support_ratio"],
        "refute_ratio": synthesis["refute_ratio"],
        "neutral_ratio": synthesis["neutral_ratio"],
        "mock_search": search.is_mock,
    }


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
async def health_check():
    """Readiness plus a view of which degraded modes are active."""
    ready = bool(pipeline)
    stance = pipeline.get("stance")
    search = pipeline.get("search")
    return {
        "status": "healthy" if ready else "loading",
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_loaded": ready,
        "stance_checkpoint": getattr(stance, "checkpoint", None),
        "device": getattr(stance, "device", None),
        "live_search": bool(search) and not search.is_mock,
        "llm_decomposition": getattr(pipeline.get("decomposer"), "use_llm", False),
    }


@app.post("/verify", response_model=VerificationResponse)
async def verify_claim(request: ClaimRequest):
    if not pipeline:
        # 503 rather than 500: the client should retry once weights finish loading.
        raise HTTPException(status_code=503, detail="Models are still loading, retry shortly.")

    start = time.perf_counter()
    logger.info("Verifying: %r (max_sources=%d)", request.claim, request.max_sources)

    cached = await pipeline["cache"].get(request.claim)
    if cached:
        logger.info("Cache hit")
        # Timing and the cached flag are refreshed so the client sees the cost
        # of *this* call, not the original computation.
        return VerificationResponse(
            **{**cached, "cached": True,
               "processing_time_ms": int((time.perf_counter() - start) * 1000)}
        )

    try:
        result = await run_pipeline(request.claim, request.max_sources)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Verification pipeline error: {e}")

    response = VerificationResponse(
        claim=request.claim,
        processing_time_ms=int((time.perf_counter() - start) * 1000),
        cached=False,
        **result,
    )

    await pipeline["cache"].set(request.claim, response.model_dump())
    logger.info(
        "Verdict %s (confidence %.2f) in %dms",
        response.verdict, response.confidence, response.processing_time_ms,
    )
    return response


@app.get("/cache/stats")
async def cache_stats():
    if not pipeline:
        raise HTTPException(status_code=503, detail="Models are still loading, retry shortly.")
    return await pipeline["cache"].get_stats()


@app.post("/cache/clear")
async def cache_clear():
    if not pipeline:
        raise HTTPException(status_code=503, detail="Models are still loading, retry shortly.")
    await pipeline["cache"].clear()
    return {"status": "cleared"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Last resort: never leak a stack trace to the client."""
    logger.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_config=None,  # keep our formatter rather than uvicorn's
    )
