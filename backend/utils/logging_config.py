"""
Logging setup that tags every record with the current request id, so lines from
concurrent requests can be told apart. The id lives in a ContextVar, which
propagates into asyncio tasks.
"""

import logging
import sys
from contextvars import ContextVar

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def set_request_id(value: str) -> None:
    _request_id.set(value)


class RequestIdFilter(logging.Filter):
    """Injects the current request id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger. Idempotent: repeat calls replace handlers."""
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
