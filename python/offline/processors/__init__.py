"""Processor modules for Gaussian Splatting generation."""

from .base import GaussianProcessor
from .sam3dbody import Sam3DBodyProcessor
from .hybrid import HybridProcessor

__all__ = ["GaussianProcessor", "Sam3DBodyProcessor", "HybridProcessor"]
