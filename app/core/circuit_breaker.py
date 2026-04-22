"""Circuit breaker pattern for external service resilience.

Provides automatic protection against cascading failures when external
services (VLM, RAG, Embedding) become unavailable or slow.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes in half-open to close
    timeout_seconds: float = 30.0  # Time before trying half-open
    half_open_timeout: float = 10.0  # Timeout for half-open requests
    excluded_exceptions: tuple = tuple()  # Exceptions that don't count as failures


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state: CircuitState = CircuitState.CLOSED
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open and rejecting requests."""

    def __init__(self, service_name: str, retry_after: float):
        self.service_name = service_name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker is OPEN for {service_name}. Retry after {retry_after:.1f}s"
        )


class CircuitBreaker:
    """Circuit breaker implementation for external services.

    Prevents cascading failures by failing fast when a service is down,
    allowing it time to recover.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._state_lock = asyncio.Lock()
        self._last_state_change = time.time()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        return self._stats

    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        return self._state != CircuitState.OPEN

    def _should_try_close(self) -> bool:
        """Check if circuit should transition from half-open to closed."""
        if self._stats.consecutive_successes >= self.config.success_threshold:
            return True
        # Also close if enough time passed in half-open
        if time.time() - self._last_state_change > self.config.half_open_timeout:
            return True
        return False

    def _should_open(self) -> bool:
        """Check if circuit should transition to open."""
        return self._stats.consecutive_failures >= self.config.failure_threshold

    async def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        async with self._state_lock:
            if self._state == new_state:
                return

            old_state = self._state
            self._state = new_state
            self._last_state_change = time.time()

            logger.info(
                f"Circuit breaker '{self.name}' state change: {old_state.value} -> {new_state.value}"
            )

            # Reset counters on state change
            if new_state == CircuitState.HALF_OPEN:
                self._stats.consecutive_failures = 0
                self._stats.consecutive_successes = 0

    async def call(
        self,
        func: Callable[..., T],
        *args: Any,
        fallback: Optional[Callable[[], T]] = None,
        **kwargs: Any,
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            fallback: Optional fallback function if circuit is open
            **kwargs: Keyword arguments

        Returns:
            Result from func or fallback

        Raises:
            CircuitBreakerOpen: If circuit is open and no fallback provided
        """
        self._stats.total_calls += 1

        # Check if circuit is open
        if self._state == CircuitState.OPEN:
            # Check if it's time to try half-open
            time_since_open = time.time() - self._last_state_change
            if time_since_open >= self.config.timeout_seconds:
                await self._transition_to(CircuitState.HALF_OPEN)
            else:
                # Circuit is open, reject request
                self._stats.rejected_calls += 1
                if fallback:
                    logger.debug(f"Circuit breaker '{self.name}': using fallback")
                    return fallback()
                retry_after = self.config.timeout_seconds - time_since_open
                raise CircuitBreakerOpen(self.name, retry_after)

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success handling
            self._stats.successful_calls += 1
            self._stats.last_success_time = time.time()
            self._stats.consecutive_failures = 0

            if self._state == CircuitState.HALF_OPEN:
                self._stats.consecutive_successes += 1
                if self._should_try_close():
                    await self._transition_to(CircuitState.CLOSED)
                    self._stats.consecutive_successes = 0

            return result

        except Exception as e:
            # Check if this exception should count as a failure
            if self.config.excluded_exceptions and isinstance(e, self.config.excluded_exceptions):
                raise

            # Failure handling
            self._stats.failed_calls += 1
            self._stats.last_failure_time = time.time()
            self._stats.consecutive_failures += 1
            self._stats.consecutive_successes = 0

            logger.warning(
                f"Circuit breaker '{self.name}': call failed ({self._stats.consecutive_failures}/{self.config.failure_threshold})"
            )

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                await self._transition_to(CircuitState.OPEN)
            elif self._should_open():
                await self._transition_to(CircuitState.OPEN)

            # Re-raise the original exception
            raise


# Global circuit breakers for each service
_circuit_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a service."""
    if name not in _circuit_breakers:
        config = CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=30.0,
        )
        _circuit_breakers[name] = CircuitBreaker(name, config)
        logger.info(f"Created circuit breaker for '{name}'")
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict[str, CircuitBreakerStats]:
    """Get stats for all circuit breakers."""
    return {name: cb.stats for name, cb in _circuit_breakers.items()}


async def reset_circuit_breaker(name: str) -> bool:
    """Manually reset a circuit breaker."""
    if name in _circuit_breakers:
        await _circuit_breakers[name]._transition_to(CircuitState.CLOSED)
        _circuit_breakers[name]._stats = CircuitBreakerStats()
        return True
    return False
