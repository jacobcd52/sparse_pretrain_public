"""
Conversion utilities for HookedSparseGPT models.

This module provides wrappers and adapters for using HookedSparseGPT with
circuit_tracer and other interpretability tools.
"""

from .wrapper import SparseGPTReplacementModel

__all__ = ["SparseGPTReplacementModel"]

