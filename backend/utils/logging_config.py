"""
logging_config.py - Structured logging with per-request correlation.

Why a request id: the pipeline fans out across six stages and concurrent HTTP
fetches, so interleaved log lines from two in-flight claims are otherwise
impossible to separate. The id is stored in a ContextVar, which propagates into
asyncio tasks automatically, and injected into every record by a filter.
"""

import logging
import sys
from contextvars import ContextVar

# Empty string rather than None so the format string never renders "None".
_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(value: str) -> None:
    _request_id.set(value)


def get_request_id() -> str:
    return _request_id.get()


class RequestIdFilter(logging.Filter):
    """Injects the current request id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """
    Idempotent root-logger setup. Safe to call from main.py, demo_script.py and
    tests; repeated calls replace handlers instead of stacking duplicates.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    handler.addFilter(RequestIdFilter())

    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # These libraries log a line per HTTP request at INFO, which drowns out the
    # pipeline's own output during a fan-out fetch.
    for noisy in ("httpx", "httpcore", "urllib3", "aiohttp.client", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
