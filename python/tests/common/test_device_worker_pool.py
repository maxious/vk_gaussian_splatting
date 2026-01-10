"""Tests for generic device worker pool."""

from __future__ import annotations

import numpy as np
import pytest

from common.device_worker_pool import DeviceWorker, DeviceWorkerPool


class DummyWorker(DeviceWorker[int, int]):
    def __init__(self, device: str, worker_id: int, multiplier: int = 2):
        super().__init__(device, worker_id, multiplier=multiplier)
        self.multiplier = multiplier
        self.model_loaded = False

    def load_model(self) -> None:
        self.model_loaded = True

    def process_item(self, item: int) -> int:
        if not self.model_loaded:
            raise RuntimeError("Model not loaded!")
        return item * self.multiplier

    def process_batch(self, items: list[int]) -> list[int]:
        return [self.process_item(item) for item in items]


class TestDeviceWorkerPool:
    def test_device_discovery_cpu(self):
        devices = DeviceWorkerPool._discover_devices("cpu")
        assert devices == ["cpu"]

    def test_device_discovery_auto_fallback(self):
        devices = DeviceWorkerPool._discover_devices("auto")
        assert len(devices) >= 1
        assert devices[0] in ["cuda:0", "xpu:0", "mps", "cpu"]

    def test_device_spec_explicit(self):
        devices = DeviceWorkerPool._discover_devices("cpu")
        assert devices == ["cpu"]

    def test_worker_pool_creation(self):
        pool = DeviceWorkerPool(
            worker_class=DummyWorker,
            device_spec="cpu",
            worker_kwargs={"multiplier": 3},
        )
        assert pool.devices == ["cpu"]
        assert len(pool.executors) == 0

    def test_single_item_processing(self):
        pool = DeviceWorkerPool(
            worker_class=DummyWorker,
            device_spec="cpu",
            worker_kwargs={"multiplier": 3},
        )

        future, device = pool.submit(5)
        result = future.result()

        assert result == 15
        assert device == "cpu"

        pool.shutdown()

    def test_batch_processing(self):
        pool = DeviceWorkerPool(
            worker_class=DummyWorker,
            device_spec="cpu",
            worker_kwargs={"multiplier": 2},
        )

        items = [1, 2, 3, 4, 5]
        results = pool.map(items)

        assert results == [2, 4, 6, 8, 10]

        pool.shutdown()

    def test_batch_size_processing(self):
        pool = DeviceWorkerPool(
            worker_class=DummyWorker,
            device_spec="cpu",
            worker_kwargs={"multiplier": 2},
        )

        items = list(range(10))
        results = pool.map(items, batch_size=3)

        assert results == [i * 2 for i in range(10)]

        pool.shutdown()

    def test_context_manager(self):
        with DeviceWorkerPool(
            worker_class=DummyWorker,
            device_spec="cpu",
            worker_kwargs={"multiplier": 2},
        ) as pool:
            results = pool.map([1, 2, 3])
            assert results == [2, 4, 6]

    def test_round_robin_distribution(self):
        devices = DeviceWorkerPool._discover_devices("cpu")
        pool = DeviceWorkerPool(
            worker_class=DummyWorker,
            devices=["cpu"] * 2,
            worker_kwargs={"multiplier": 1},
        )

        device_counts = {"cpu": 0}
        for i in range(10):
            future, device = pool.submit(i)
            future.result()

        pool.shutdown()


class ImageProcessingWorker(DeviceWorker[np.ndarray, tuple[np.ndarray, float]]):
    def __init__(self, device: str, worker_id: int):
        super().__init__(device, worker_id)
        self.device_name = device

    def load_model(self) -> None:
        pass

    def process_item(self, item: np.ndarray) -> tuple[np.ndarray, float]:
        processed = item * 2.0
        mean_val = float(np.mean(processed))
        return (processed, mean_val)


class TestImageWorker:
    def test_numpy_array_processing(self):
        pool = DeviceWorkerPool(
            worker_class=ImageProcessingWorker,
            device_spec="cpu",
        )

        image = np.random.rand(64, 64, 3).astype(np.float32)
        future, device = pool.submit(image)
        result_image, mean_val = future.result()

        assert result_image.shape == image.shape
        assert np.allclose(result_image, image * 2.0)
        assert isinstance(mean_val, float)

        pool.shutdown()

    def test_batch_image_processing(self):
        pool = DeviceWorkerPool(
            worker_class=ImageProcessingWorker,
            device_spec="cpu",
        )

        images = [np.random.rand(64, 64, 3).astype(np.float32) for _ in range(5)]
        results = pool.map(images)

        assert len(results) == 5
        for (result_img, mean_val), orig_img in zip(results, images):
            assert result_img.shape == orig_img.shape
            assert np.allclose(result_img, orig_img * 2.0)

        pool.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
