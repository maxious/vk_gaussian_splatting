#!/usr/bin/env python3
"""Test MoGe multi-device worker on a single frame."""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from backend.workers.moge_worker import MoGEDeviceWorker
from common.device_worker_pool import DeviceWorkerPool


def test_single_worker():
    print("Testing single XPU worker...")

    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    worker = MoGEDeviceWorker(device="xpu:0", worker_id=0)
    worker.load_model()

    depth, z_min, z_max, normals = worker.process_item(frame)

    print(f"Single worker result: depth shape={depth.shape}, z_min={z_min:.2f}, z_max={z_max:.2f}")
    if normals is not None:
        print(f"  normals shape={normals.shape}")


def test_worker_pool():
    print("\nTesting dual XPU worker pool...")

    frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(8)]

    pool = DeviceWorkerPool(
        worker_class=MoGEDeviceWorker,
        device_spec="xpu:0,1",
        worker_kwargs={"model_id": "Ruicheng/moge-3-vitl"},
    )

    print(f"Pool ready with devices: {pool.devices}")

    results = list(pool.map(frames, batch_size=4))

    print(f"Processed {len(results)} frames")
    for i, (depth, z_min, z_max, normals) in enumerate(results[:2]):
        print(f"  Frame {i}: depth={depth.shape}, z_min={z_min:.2f}, z_max={z_max:.2f}")

    pool.shutdown()


if __name__ == "__main__":
    device_count = (
        torch.xpu.device_count() if hasattr(torch, "xpu") and torch.xpu.is_available() else 0
    )
    print(f"XPU devices available: {device_count}")

    if device_count == 0:
        print("No XPU devices found, skipping test")
        sys.exit(0)

    test_single_worker()
    test_worker_pool()

    print("\nAll tests passed!")
