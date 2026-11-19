"""
Runtime configuration, read from the environment.

Importing this module loads .env, so entrypoints import it before anything
that reads an API key.
"""

import os
from dotenv import load_dotenv

# Must run before any os.getenv() below.
load_dotenv()


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


class Config:
    # ── API keys ──────────────────────────────────────────────────────────────
    # Absence is not fatal: the pipeline degrades to mock search / rule-based
    # decomposition rather than failing at startup.
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "").strip()
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()

    # ── Models ────────────────────────────────────────────────────────────────
    # Checkpoint produced by backend/training/train_stance.py. When absent the
    # stance detector falls back to a checkpoint already fine-tuned for NLI.
    STANCE_CHECKPOINT: str = os.getenv("STANCE_CHECKPOINT", "./checkpoints/stance_deberta")
    STANCE_FALLBACK_MODEL: str = os.getenv("STANCE_FALLBACK_MODEL", "cross-encoder/nli-deberta-v3-base")
    EMBEDDINGS_MODEL: str = os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
    DECOMPOSER_MODEL: str = os.getenv("DECOMPOSER_MODEL", "claude-haiku-4-5-20251001")

    # ── Inference ─────────────────────────────────────────────────────────────
    # MC dropout passes. Latency is linear in this (CPU, 30 passages: T=10 7.2s,
    # T=5 3.7s, T=1 0.6s). 1 disables uncertainty estimation.
    MC_SAMPLES: int = int(os.getenv("MC_SAMPLES", 5))
    STANCE_BATCH_SIZE: int = int(os.getenv("STANCE_BATCH_SIZE", 16))
    # Fixed dropout masks, so the same claim gets the same verdict across runs.
    MC_SEED: int = int(os.getenv("MC_SEED", 42))
    # MNLI-trained checkpoints expect single-sentence premises; whole passages
    # collapse to NEUTRAL. See stance_detector.py.
    SENTENCE_LEVEL_STANCE: bool = _flag("SENTENCE_LEVEL_STANCE", True)
    MAX_SENTENCES_PER_PASSAGE: int = int(os.getenv("MAX_SENTENCES_PER_PASSAGE", 6))
    # Predictions below this confidence become UNRELATED and are excluded.
    UNRELATED_THRESHOLD: float = float(os.getenv("UNRELATED_THRESHOLD", 0.40))
    MAX_PREMISE_CHARS: int = int(os.getenv("MAX_PREMISE_CHARS", 500))
    # Logits are divided by this before softmax; 1.0 is uncalibrated. Set from
    # backend/training/calibrate_temperature.py and refit whenever
    # STANCE_CHECKPOINT changes.
    STANCE_TEMPERATURE: float = float(os.getenv("STANCE_TEMPERATURE", 1.0))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    FETCH_CONCURRENCY: int = int(os.getenv("FETCH_CONCURRENCY", 10))
    FETCH_TIMEOUT_S: float = float(os.getenv("FETCH_TIMEOUT_S", 8.0))
    MAX_PAGE_CHARS: int = int(os.getenv("MAX_PAGE_CHARS", 10000))
    PASSAGES_PER_SOURCE: int = int(os.getenv("PASSAGES_PER_SOURCE", 3))
    # The cross-encoder reranks a pool of RERANK_POOL_MULTIPLIER x
    # PASSAGES_PER_SOURCE bi-encoder candidates per source.
    RERANK_ENABLED: bool = _flag("RERANK_ENABLED", True)
    # Not MiniLM-L-6-v2: it crashes with SIGBUS on Apple Silicon (torch 2.10).
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
    RERANK_POOL_MULTIPLIER: int = int(os.getenv("RERANK_POOL_MULTIPLIER", 3))
    # Pool slots reserved for fact-check verdict sentences, which similarity
    # ranking under-ranks. See retriever._VERDICT_MARKERS.
    VERDICT_SLOTS: int = int(os.getenv("VERDICT_SLOTS", 2))
    VERDICT_MIN_RELEVANCE: float = float(os.getenv("VERDICT_MIN_RELEVANCE", 0.2))

    # ── Cache ─────────────────────────────────────────────────────────────────
    CACHE_TTL_S: float = float(os.getenv("CACHE_TTL_S", 3600))
    CACHE_MAX_ENTRIES: int = int(os.getenv("CACHE_MAX_ENTRIES", 512))

    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = _flag("DEBUG", False)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    CORS_ORIGINS: list[str] = [
        o.strip() for o in os.getenv(
            "CORS_ORIGINS",
            # 3000 is this project's vite port; 5173 is vite's default.
            "http://localhost:3000,http://127.0.0.1:3000,"
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",") if o.strip()
    ]


config = Config()
