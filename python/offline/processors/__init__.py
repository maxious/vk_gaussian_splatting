"""Processor modules for Gaussian Splatting generation."""

from .base import GaussianProcessor
from .hybrid import HybridProcessor
from .sam3dbody import Sam3DBodyProcessor
from .sharp import SharpGaussianProcessor
from .unisharp import UniSHARPGaussianProcessor

__all__ = [
    "GaussianProcessor",
    "HybridProcessor",
    "Sam3DBodyProcessor",
    "SharpGaussianProcessor",
    "UniSHARPGaussianProcessor",
]


def get_diffusion_repair():
    """Lazy import for diffusion repair (requires diffusion-repair extra)."""
    from .diffusion_repair import (
        RepairConfig,
        RepairModel,
        RepairResult,
        create_repair_model,
        repair_gaussian_render,
    )

    return RepairConfig, RepairModel, RepairResult, create_repair_model, repair_gaussian_render
