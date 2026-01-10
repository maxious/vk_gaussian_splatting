"""Unified parallel processing utilities using joblib.

This module provides a consistent interface for parallel operations across
the codebase, with smart backend selection based on workload type.

Key features:
- Smart backend selection (threading for I/O, loky for CPU-bound)
- Generator-based streaming results for memory efficiency
- Consistent configuration across all parallel operations

Example:
    >>> from common.parallel_utils import parallel_map, get_optimal_backend
    >>> # I/O-bound parallel loading (uses threading)
    >>> results = parallel_map(load_image, image_paths)
    >>> # CPU-bound processing (uses loky)
    >>> results = parallel_map(process_frame, frames, backend="loky")
"""

from __future__ import annotations

from typing import Callable, Iterator, TypeVar, Any

import torch
from joblib import Parallel, delayed, parallel_config

T = TypeVar("T")
R = TypeVar("R")


def get_optimal_backend() -> str:
    """Get the optimal joblib backend based on hardware.

    Returns:
        "threading": For I/O-bound work (cv2, file I/O) - GIL is released
        "loky": For CPU-bound work (tensor ops, model inference)
    """
    # Check for GPU availability
    if torch.cuda.is_available():
        return "loky"  # CPU-bound model ops benefit from process isolation

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "loky"  # CPU-bound model ops benefit from process isolation

    # Default for CPU-only
    return "loky"


def get_prefer_mode(backend: str | None = None) -> str:
    """Get the preferred parallel mode based on backend.

    Args:
        backend: joblib backend ("loky", "threading", etc.)

    Returns:
        "threads" or "processes" hint for joblib
    """
    if backend is None:
        backend = get_optimal_backend()

    if backend == "threading":
        return "threads"
    return "processes"


def parallel_map(
    func: Callable[[T], R],
    items: list[T],
    n_jobs: int | None = None,
    backend: str | None = None,
    prefer: str | None = None,
    return_as: str = "list",
    **kwargs,
) -> list[R] | Iterator[R]:
    """Unified parallel map with smart backend selection.

    Args:
        func: Function to apply to each item
        items: Items to process
        n_jobs: Number of parallel workers (None = auto)
        backend: joblib backend (None = auto-detect)
        prefer: "threads" or "processes" (None = auto)
        return_as: "list" for blocking, "generator" for streaming
        **kwargs: Additional joblib.Parallel kwargs

    Returns:
        If return_as="list": List of results
        If return_as="generator": Iterator yielding results as they complete

    Example:
        >>> # Streaming results (memory efficient)
        >>> for result in parallel_map(process, items, return_as="generator"):
        ...     do_something(result)
        >>>
        >>> # Blocking results
        >>> results = parallel_map(process, items)
    """
    if backend is None:
        backend = get_optimal_backend()

    if prefer is None:
        prefer = get_prefer_mode(backend)

    # Auto-detect n_jobs if not specified
    if n_jobs is None:
        import os

        n_jobs = os.cpu_count() or 4

    if return_as == "generator":
        # Generator-based streaming
        with parallel_config(backend=backend, n_jobs=n_jobs, prefer=prefer, **kwargs):
            parallel = Parallel(return_as="generator")
            return parallel(delayed(func)(item) for item in items)
    else:
        # Blocking list
        with parallel_config(backend=backend, n_jobs=n_jobs, prefer=prefer, **kwargs):
            return Parallel(**kwargs)(delayed(func)(item) for item in items)


def parallel_imap(
    func: Callable[[T], R],
    items: list[T],
    n_jobs: int | None = None,
    backend: str | None = None,
) -> Iterator[tuple[int, R]]:
    """Parallel map with index tracking.

    Yields (index, result) tuples as they complete, preserving order.

    Args:
        func: Function to apply to each item
        items: Items to process
        n_jobs: Number of parallel workers
        backend: joblib backend

    Yields:
        (index, result) tuples in order of completion

    Example:
        >>> for idx, result in parallel_imap(process, items):
        ...     print(f"Completed {idx}: {result}")
    """
    if backend is None:
        backend = get_optimal_backend()

    if n_jobs is None:
        import os

        n_jobs = os.cpu_count() or 4

    from concurrent.futures import as_completed, Future

    # Track which item each future corresponds to
    futures: dict[Future, int] = {}

    with parallel_config(backend=backend, n_jobs=n_jobs):
        parallel = Parallel()
        for idx, item in enumerate(items):
            future = delayed(func)(item)
            futures[future] = idx

        # Yield in order of completion, but track original indices
        results: dict[int, R] = {}
        next_idx = 0

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results[idx] = result
                # Yield in order
                while next_idx in results:
                    yield next_idx, results.pop(next_idx)
                    next_idx += 1
            except Exception as e:
                raise RuntimeError(f"Error processing item {idx}") from e


class ParallelPipeline:
    """Reusable parallel pipeline with consistent configuration.

    Example:
        >>> pipeline = ParallelPipeline(n_jobs=4, backend="loky")
        >>> # Use multiple times without re-initializing
        >>> results1 = pipeline.map(func, items1)
        >>> results2 = pipeline.map(func, items2)
    """

    def __init__(
        self,
        n_jobs: int | None = None,
        backend: str | None = None,
        prefer: str | None = None,
    ):
        self.n_jobs = n_jobs
        self.backend = backend or get_optimal_backend()
        self.prefer = prefer or get_prefer_mode(self.backend)

        if self.n_jobs is None:
            import os

            self.n_jobs = os.cpu_count() or 4

    def map(
        self,
        func: Callable[[T], R],
        items: list[T],
        return_as: str = "list",
    ) -> list[R] | Iterator[R]:
        """Apply func to all items in parallel."""
        return parallel_map(
            func,
            items,
            n_jobs=self.n_jobs,
            backend=self.backend,
            prefer=self.prefer,
            return_as=return_as,
        )

    def imap(
        self,
        func: Callable[[T], R],
        items: list[T],
    ) -> Iterator[tuple[int, R]]:
        """Apply func to all items, yielding (index, result) tuples."""
        return parallel_imap(func, items, n_jobs=self.n_jobs, backend=self.backend)
