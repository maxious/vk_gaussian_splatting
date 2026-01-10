"""Generic device worker pool for multi-GPU/XPU processing.

This module provides a generic worker pool pattern that can be used by both:
- Offline batch processing (SHARP, DA3 preprocessing)
- Online streaming (real-time DA3 depth inference)

Key features:
- Auto-detection of available devices (CUDA, XPU, MPS, CPU)
- Process-based isolation for PyTorch multiprocessing
- Persistent worker processes with model loading
- Round-robin or queue-based distribution
- Support for both synchronous and asynchronous interfaces
"""

from __future__ import annotations

import logging
import multiprocessing
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from types import GeneratorType

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Input type
R = TypeVar("R")  # Result type


@dataclass
class DeviceConfig:
    """Configuration for a single device worker."""

    device: str  # e.g., "cuda:0", "xpu:1", "cpu"
    worker_id: int  # Unique worker identifier


class DeviceWorker(ABC, Generic[T, R]):
    """Abstract base class for device-specific workers.

    Subclasses must implement:
    - load_model(): Initialize the model on the device
    - process_item(item: T) -> R: Process a single item
    - Optional: process_batch(items: list[T]) -> list[R]: Process a batch
    """

    def __init__(self, device: str, worker_id: int, **kwargs):
        """Initialize worker with device and worker ID.

        Args:
            device: Device string (e.g., "cuda:0", "xpu:1")
            worker_id: Unique worker identifier
            **kwargs: Additional worker-specific configuration
        """
        self.device = device
        self.worker_id = worker_id
        self.config = kwargs
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the model onto the device.

        This is called once per worker process during initialization.
        """
        pass

    @abstractmethod
    def process_item(self, item: T) -> R:
        """Process a single item.

        Args:
            item: Input item to process

        Returns:
            Processed result
        """
        pass

    def process_batch(self, items: list[T]) -> list[R]:
        """Process a batch of items.

        Default implementation processes items sequentially.
        Subclasses can override for batch optimization.

        Args:
            items: List of input items

        Returns:
            List of processed results
        """
        return [self.process_item(item) for item in items]


# Global worker instance (one per process)
_GLOBAL_WORKER: DeviceWorker | None = None


def _init_worker(
    worker_class: type[DeviceWorker], device: str, worker_id: int, kwargs: dict[str, Any]
) -> None:
    """Initialize global worker instance in subprocess.

    Args:
        worker_class: Worker class to instantiate
        device: Device string
        worker_id: Worker identifier
        kwargs: Additional worker configuration
    """
    global _GLOBAL_WORKER
    logger.info(f"Initializing worker {worker_id} on {device}")
    _GLOBAL_WORKER = worker_class(device, worker_id, **kwargs)
    _GLOBAL_WORKER.load_model()
    logger.info(f"Worker {worker_id} ready on {device}")


def _worker_process_item(item: Any) -> Any:
    """Process a single item using the global worker.

    Args:
        item: Input item

    Returns:
        Processed result
    """
    if _GLOBAL_WORKER is None:
        raise RuntimeError("Worker not initialized!")
    return _GLOBAL_WORKER.process_item(item)


def _worker_process_batch(items: list[Any]) -> list[Any]:
    """Process a batch of items using the global worker.

    Args:
        items: List of input items

    Returns:
        List of processed results
    """
    if _GLOBAL_WORKER is None:
        raise RuntimeError("Worker not initialized!")
    return _GLOBAL_WORKER.process_batch(items)


class DeviceWorkerPool(Generic[T, R]):
    """Worker pool managing multiple device workers.

    Supports both single-item and batch processing with automatic
    distribution across available devices.
    """

    def __init__(
        self,
        worker_class: type[DeviceWorker[T, R]],
        devices: list[str] | None = None,
        device_spec: str = "auto",
        worker_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize worker pool.

        Args:
            worker_class: Worker class to instantiate
            devices: Explicit list of devices, or None to auto-detect
            device_spec: Device specification for auto-detection (default: "auto")
            worker_kwargs: Additional kwargs passed to worker constructor
        """
        self.worker_class = worker_class
        self.devices = devices or self._discover_devices(device_spec)
        self.worker_kwargs = worker_kwargs or {}
        self.executors: dict[str, ProcessPoolExecutor] = {}
        self._next_device_idx = 0

        # PyTorch requires 'spawn' context for CUDA/XPU
        self._mp_context = multiprocessing.get_context("spawn")

        logger.info(f"DeviceWorkerPool initialized with devices: {self.devices}")

    @staticmethod
    def _discover_devices(device_spec: str = "auto") -> list[str]:
        """Discover available GPU devices.

        Args:
            device_spec: Device specification:
                - "auto": Auto-detect available devices
                - "cuda", "xpu", "mps", "cpu": Use specific backend
                - "cuda:0,1" or "xpu:0,1": Use specific device indices

        Returns:
            List of device strings (e.g., ["cuda:0", "cuda:1"])
        """
        import torch

        if ":" in device_spec:
            device_type, device_indices = device_spec.split(":", 1)
            indices = [int(i.strip()) for i in device_indices.split(",")]
            return [f"{device_type}:{i}" for i in indices]

        if device_spec == "auto":
            if torch.cuda.is_available():
                num_devices = torch.cuda.device_count()
                devices = [f"cuda:{i}" for i in range(num_devices)]
                logger.info(f"Auto-detected {num_devices} CUDA device(s): {devices}")
                return devices

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                num_devices = torch.xpu.device_count()
                devices = [f"xpu:{i}" for i in range(num_devices)]
                logger.info(f"Auto-detected {num_devices} XPU device(s): {devices}")
                return devices

            if torch.backends.mps.is_available():
                logger.info("Auto-detected MPS device (Apple Silicon)")
                return ["mps"]

            logger.info("No GPU detected, using CPU")
            return ["cpu"]

        if device_spec == "cuda":
            if torch.cuda.is_available():
                num_devices = torch.cuda.device_count()
                devices = [f"cuda:{i}" for i in range(num_devices)]
                logger.info(f"Using {num_devices} CUDA device(s): {devices}")
                return devices
            logger.warning("CUDA requested but not available, falling back to CPU")
            return ["cpu"]

        if device_spec == "xpu":
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                num_devices = torch.xpu.device_count()
                devices = [f"xpu:{i}" for i in range(num_devices)]
                logger.info(f"Using {num_devices} XPU device(s): {devices}")
                return devices
            logger.warning("XPU requested but not available, falling back to CPU")
            return ["cpu"]

        if device_spec == "mps":
            if torch.backends.mps.is_available():
                logger.info("Using MPS device")
                return ["mps"]
            logger.warning("MPS requested but not available, falling back to CPU")
            return ["cpu"]

        if device_spec == "cpu":
            logger.info("Using CPU")
            return ["cpu"]

        logger.warning(f"Unknown device spec '{device_spec}', falling back to CPU")
        return ["cpu"]

    def _get_executor(self, device: str) -> ProcessPoolExecutor:
        """Get or create executor for a device.

        Args:
            device: Device string

        Returns:
            ProcessPoolExecutor for the device
        """
        if device not in self.executors:
            worker_id = len(self.executors)
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=self._mp_context,
                initializer=_init_worker,
                initargs=(self.worker_class, device, worker_id, self.worker_kwargs),
            )
            self.executors[device] = executor
        return self.executors[device]

    def _select_device(self) -> str:
        """Select next device using round-robin.

        Returns:
            Device string
        """
        device = self.devices[self._next_device_idx]
        self._next_device_idx = (self._next_device_idx + 1) % len(self.devices)
        return device

    def submit(self, item: T) -> tuple[Future[R], str]:
        """Submit a single item for processing.

        Args:
            item: Input item

        Returns:
            Tuple of (Future for result, device string)
        """
        device = self._select_device()
        executor = self._get_executor(device)
        future = executor.submit(_worker_process_item, item)
        return future, device

    def submit_batch(self, items: list[T]) -> tuple[Future[list[R]], str]:
        """Submit a batch of items for processing.

        Args:
            items: List of input items

        Returns:
            Tuple of (Future for results, device string)
        """
        device = self._select_device()
        executor = self._get_executor(device)
        future = executor.submit(_worker_process_batch, items)
        return future, device

    def map(self, items: list[T], batch_size: int | None = None, return_as: str = "list"):
        """Process items across all devices.

        Args:
            items: List of input items
            batch_size: Optional batch size for batched processing
            return_as: "list" for blocking list, "generator" for streaming results

        Returns:
            If return_as="list": List of processed results (order preserved)
            If return_as="generator": Generator yielding results as they complete
        """
        from concurrent.futures import as_completed
        import types

        # Generator return type annotation (Python 3.6 compatible)
        if return_as == "generator":

            def result_generator():
                """Generator yielding results as they complete."""
                if batch_size is None:
                    # Single-item distribution
                    futures = {}
                    for idx, item in enumerate(items):
                        future, device = self.submit(item)
                        futures[future] = (idx, device)

                    completed_results: dict[int, R] = {}
                    next_idx = 0
                    for future in as_completed(futures):
                        idx, device = futures[future]
                        try:
                            result = future.result()
                            completed_results[idx] = result
                            # Yield results in order as they become available
                            while next_idx in completed_results:
                                yield completed_results.pop(next_idx)
                                next_idx += 1
                        except Exception as e:
                            logger.error(f"Error processing item {idx} on {device}: {e}")
                            raise
                else:
                    # Batch distribution
                    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
                    futures = {}
                    for batch_idx, batch in enumerate(batches):
                        future, device = self.submit_batch(batch)
                        futures[future] = (batch_idx, device)

                    completed_batches: dict[int, list[R]] = {}
                    next_batch = 0
                    for future in as_completed(futures):
                        batch_idx, device = futures[future]
                        try:
                            result = future.result()
                            completed_batches[batch_idx] = result
                            # Flatten and yield in order as batches complete
                            while next_batch in completed_batches:
                                for item_result in completed_batches.pop(next_batch):
                                    yield item_result
                                next_batch += 1
                        except Exception as e:
                            logger.error(f"Error processing batch {batch_idx} on {device}: {e}")
                            raise

            return result_generator()

        # Blocking list (original behavior)
        if batch_size is None:
            # Single-item distribution
            futures = {}
            for idx, item in enumerate(items):
                future, device = self.submit(item)
                futures[future] = (idx, device)

            results: list[R | None] = [None] * len(items)
            for future in as_completed(futures):
                idx, device = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing item {idx} on {device}: {e}")
                    raise

            return [r for r in results if r is not None]

        else:
            # Batch distribution
            batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
            futures = {}
            for batch_idx, batch in enumerate(batches):
                future, device = self.submit_batch(batch)
                futures[future] = (batch_idx, device)

            batch_results: list[list[R] | None] = [None] * len(batches)
            for future in as_completed(futures):
                batch_idx, device = futures[future]
                try:
                    batch_results[batch_idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx} on {device}: {e}")
                    raise

            # Flatten results
            flattened = []
            for batch_result in batch_results:
                if batch_result is not None:
                    flattened.extend(batch_result)
            return flattened

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown all executors.

        Args:
            wait: Whether to wait for pending tasks
        """
        for device, executor in self.executors.items():
            logger.info(f"Shutting down executor for {device}")
            executor.shutdown(wait=wait)
        self.executors.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
