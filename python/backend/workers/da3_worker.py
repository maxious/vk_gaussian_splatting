"""Deprecated: DA3DeviceWorker is now an alias for InfiniDepthDeviceWorker."""

import warnings

warnings.warn(
    "DA3DeviceWorker is deprecated. Use InfiniDepthDeviceWorker instead.",
    DeprecationWarning,
    stacklevel=2,
)

from backend.workers.infini_depth_worker import InfiniDepthDeviceWorker

DA3DeviceWorker = InfiniDepthDeviceWorker
