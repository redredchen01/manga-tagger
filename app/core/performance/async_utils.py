"""Async utilities for Python 3.12+ performance optimizations."""

import asyncio
from typing import Any, Callable, List, TypeVar

T = TypeVar("T")


async def run_in_executor(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous function in a thread pool executor.

    Use this to offload blocking operations (like CPU-bound tasks or
    synchronous I/O) to avoid blocking the async event loop.

    Args:
        func: Synchronous function to run
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result from the function execution
    """
    loop = asyncio.get_running_loop()

    if kwargs:
        from functools import partial

        func = partial(func, **kwargs)

    return await loop.run_in_executor(None, func, *args)


async def gather_with_limit(limit: int, *tasks: Any) -> List[Any]:
    """Run async tasks with a concurrency limit.

    Uses a semaphore to limit the number of concurrent tasks, preventing
    resource exhaustion when running many async operations in parallel.

    Args:
        limit: Maximum number of concurrent tasks
        *tasks: Coroutines to execute

    Returns:
        List of results from all tasks (in original order)
    """
    semaphore = asyncio.Semaphore(limit)

    async def limited_task(task: asyncio.Task) -> Any:
        async with semaphore:
            return await task

    # Wrap tasks as asyncio.Task objects if they aren't already
    wrapped_tasks = [
        task if isinstance(task, asyncio.Task) else asyncio.create_task(task) for task in tasks
    ]

    return await asyncio.gather(*[limited_task(t) for t in wrapped_tasks])


async def timeout_after(seconds: float, coro: Any) -> Any:
    """Add a timeout to a coroutine.

    Wraps a coroutine with asyncio.wait_for and raises a TimeoutError
    if the operation exceeds the specified duration.

    Args:
        seconds: Timeout in seconds
        coro: Coroutine to execute

    Returns:
        Result from the coroutine

    Raises:
        TimeoutError: If the operation exceeds the timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation timed out after {seconds} seconds")


async def run_with_retry(
    coro: Callable, max_retries: int = 3, backoff_seconds: float = 1.0, *args: Any, **kwargs: Any
) -> Any:
    """Run a coroutine with exponential backoff retry.

    Attempts to execute a coroutine multiple times with increasing delays
    between attempts. Useful for handling transient failures.

    Args:
        coro: Coroutine or async function to execute
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_seconds: Initial backoff delay in seconds (default: 1.0)
        *args: Positional arguments for the coroutine
        **kwargs: Keyword arguments for the coroutine

    Returns:
        Result from the successful execution

    Raises:
        Last exception if all retries fail
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            if asyncio.iscoroutinefunction(coro):
                result = coro(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            elif asyncio.iscoroutine(coro):
                return await coro
            else:
                # It's a regular callable
                return coro(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = backoff_seconds * (2**attempt)
                await asyncio.sleep(delay)

    if last_exception is not None:
        raise last_exception
    else:
        raise RuntimeError("Retry failed with no exception recorded")


class AsyncBatcher:
    """Batches async operations for improved throughput.

    Accumulates async tasks and processes them in batches, reducing
    overhead when many similar operations need to be executed.
    """

    def __init__(self, batch_size: int = 10, max_wait_seconds: float = 1.0):
        """Initialize the async batcher.

        Args:
            batch_size: Maximum number of items per batch
            max_wait_seconds: Maximum time to wait before flushing
        """
        self.batch_size = batch_size
        self.max_wait_seconds = max_wait_seconds
        self._queue: List[asyncio.Future] = []
        self._flush_task: asyncio.Task | None = None

    async def add(self, coro: Callable[[], T]) -> T:
        """Add a coroutine to the batch.

        Args:
            coro: Async function to execute

        Returns:
            Result from the function
        """
        future = asyncio.Future()
        self._queue.append(future)

        # Schedule flush if needed
        if len(self._queue) >= self.batch_size:
            await self._flush()

        return await future

    async def _flush(self) -> None:
        """Flush all pending tasks."""
        if not self._queue:
            return

        # Create batch tasks
        tasks = [asyncio.create_task(self._execute(f)) for f in self._queue]
        await asyncio.gather(*tasks)
        self._queue.clear()

    async def _execute(self, future: asyncio.Future) -> None:
        """Execute a single task and set its result."""
        try:
            result = await future
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
