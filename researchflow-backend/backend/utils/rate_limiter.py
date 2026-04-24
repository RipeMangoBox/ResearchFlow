"""Async token-bucket rate limiter for external API calls.

Usage:
    limiter = TokenBucketLimiter(rate=1.0, burst=3, name="s2")
    await limiter.acquire()       # blocks until a token is available
    await limiter.acquire(2)      # consume 2 tokens (batch request)
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class TokenBucketLimiter:
    """Async token-bucket rate limiter.

    Parameters:
        rate:  tokens refilled per second (e.g., 1.0 = 1 req/s)
        burst: max tokens the bucket can hold (allows short bursts)
        name:  label for logging
    """

    def __init__(self, rate: float, burst: int, name: str = "default") -> None:
        self.rate = rate
        self.burst = burst
        self.name = name
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_refill = now

    async def acquire(self, tokens: int = 1) -> None:
        """Wait until *tokens* are available, then consume them."""
        async with self._lock:
            self._refill()
            while self._tokens < tokens:
                deficit = tokens - self._tokens
                wait = deficit / self.rate
                logger.debug(
                    "rate_limiter[%s]: waiting %.2fs for %d token(s)",
                    self.name, wait, tokens,
                )
                await asyncio.sleep(wait)
                self._refill()
            self._tokens -= tokens

    async def backoff_429(self, attempt: int = 0, base: float = 3.0) -> None:
        """Exponential backoff on HTTP 429.  Call this when the API returns 429."""
        delay = min(base * (2 ** attempt), 120.0)
        logger.warning(
            "rate_limiter[%s]: 429 backoff attempt=%d, sleeping %.1fs",
            self.name, attempt, delay,
        )
        await asyncio.sleep(delay)
