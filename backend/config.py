"""
config.py - Single source of truth for runtime configuration.

Importing this module has a side effect by design: it calls load_dotenv() at
import time. Every entrypoint (main.py, demo_script.py, tests) imports config
first, which is what guarantees .env is on the environment before any service
reads an API key. The previous layout defined load_dotenv() here but nothing
imported it, so .env was never loaded and SearchService raised on startup.
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
    # MC dropout passes, and the main latency dial in the whole system: cost is
    # linear in T. Measured on CPU over 30 passages, T=10 costs 7.2s, T=5 costs
    # 3.7s, T=1 costs 0.6s. 5 is the default because it keeps a typical request
    # near the latency budget while still producing a usable uncertainty spread;
    # set to 1 to disable uncertainty estimation entirely.
    MC_SAMPLES: int = int(os.getenv("MC_SAMPLES", 5))
    STANCE_BATCH_SIZE: int = int(os.getenv("STANCE_BATCH_SIZE", 16))
    # Seed for the MC dropout masks. Without it the same claim can return
    # different verdicts across runs whenever the evidence sits near a decision
    # threshold, which makes the system impossible to debug or demo.
    MC_SEED: int = int(os.getenv("MC_SEED", 42))
    # Score stance per sentence rather than per passage. The NLI checkpoints are
    # MNLI-trained, where premises are single sentences; feeding a multi-sentence
    # web passage pushes them to NEUTRAL with near-1.0 confidence even when one
    # sentence plainly entails or contradicts the claim. See stance_detector.py.
    SENTENCE_LEVEL_STANCE: bool = _flag("SENTENCE_LEVEL_STANCE", True)
    # Caps the fan-out, since cost is linear in sentences x MC samples.
    MAX_SENTENCES_PER_PASSAGE: int = int(os.getenv("MAX_SENTENCES_PER_PASSAGE", 6))
    # Below this confidence a prediction is downgraded to UNRELATED and excluded
    # from the verdict rather than counted as weak evidence.
    UNRELATED_THRESHOLD: float = float(os.getenv("UNRELATED_THRESHOLD", 0.40))
    MAX_PREMISE_CHARS: int = int(os.getenv("MAX_PREMISE_CHARS", 500))
    # Softmax temperature (logits are divided by this before softmax). 1.0 is a
    # no-op -- MC dropout gives a relative uncertainty signal, not a calibrated
    # probability, so "confidence": 0.9 does not mean "correct 90% of the time"
    # until this is set. Produced by backend/training/calibrate_temperature.py
    # against whichever checkpoint is currently active; must be recalibrated
    # whenever STANCE_CHECKPOINT changes, since a different model has a
    # different confidence/accuracy relationship.
    STANCE_TEMPERATURE: float = float(os.getenv("STANCE_TEMPERATURE", 1.0))

    # ── Retrieval ─────────────────────────────────────────────────────────────
    FETCH_CONCURRENCY: int = int(os.getenv("FETCH_CONCURRENCY", 10))
    FETCH_TIMEOUT_S: float = float(os.getenv("FETCH_TIMEOUT_S", 8.0))
    MAX_PAGE_CHARS: int = int(os.getenv("MAX_PAGE_CHARS", 10000))
    PASSAGES_PER_SOURCE: int = int(os.getenv("PASSAGES_PER_SOURCE", 3))
    # Bi-encoder cosine similarity is a good first pass and a mediocre final
    # ranking: it embeds claim and passage independently, so it cannot model
    # interactions between them. The cross-encoder reranks a wider bi-encoder
    # candidate pool (RERANK_POOL_MULTIPLIER x PASSAGES_PER_SOURCE, per source)
    # by scoring (claim, passage) jointly, then keeps the final
    # PASSAGES_PER_SOURCE -- more accurate, but too slow to run over every
    # passage from every source, which is why it only sees the pre-filtered
    # pool rather than all candidates.
    RERANK_ENABLED: bool = _flag("RERANK_ENABLED", True)
    # Not ms-marco-MiniLM-L-6-v2, the more commonly cited default: that specific
    # checkpoint crashes with SIGBUS on forward pass on at least one dev
    # machine (Apple Silicon, torch 2.10), reproducible outside this codebase
    # with a bare transformers forward pass. L-12 is confirmed working and is
    # generally rated higher-quality anyway, just marginally slower.
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")
    RERANK_POOL_MULTIPLIER: int = int(os.getenv("RERANK_POOL_MULTIPLIER", 3))
    # Both ranking stages above score passages purely by similarity to the
    # claim, which systematically under-ranks a fact-check's own verdict
    # sentence ("we rate this claim False") against a passage that merely
    # restates the claim -- maximally similar to it by construction. See
    # retriever.py's _has_verdict_language docstring; found via the
    # end-to-end LIAR eval, ARCHITECTURE.md §12.
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
    # Tightened from "*" so credentialed CORS is not paired with a wildcard.
    CORS_ORIGINS: list[str] = [
        o.strip() for o in os.getenv(
            "CORS_ORIGINS",
            # 3000 is this project's vite port (frontend/vite.config.js);
            # 5173 is vite's default, kept for a plain `vite` invocation.
            "http://localhost:3000,http://127.0.0.1:3000,"
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",") if o.strip()
    ]

    @property
    def has_search_key(self) -> bool:
        return bool(self.TAVILY_API_KEY)

    @property
    def has_llm_key(self) -> bool:
        return bool(self.ANTHROPIC_API_KEY)


config = Config()
