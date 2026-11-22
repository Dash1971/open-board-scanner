"""Synthetic chess-diagram generator for Open Board Scanner."""

from .positions import CLASSES
from .sample import KINDS, Sample, generate, rectify

__all__ = ["CLASSES", "KINDS", "Sample", "generate", "rectify"]
