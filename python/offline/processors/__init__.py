"""Processor modules for Gaussian Splatting generation."""

from .base import GaussianProcessor
from .fastgs import FastGSProcessor
from .sam3dbody import Sam3DBodyProcessor
from .hybrid import HybridProcessor

__all__ = ["GaussianProcessor", "FastGSProcessor", "Sam3DBodyProcessor", "HybridProcessor"]


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
