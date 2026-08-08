"""Predictor adapters for canonical NK_resume cases."""

from __future__ import annotations

from .base import PredictorAdapter
from .direct import DirectPredictorAdapter
from .fsb import FSBPredictorAdapter

__all__ = ["DirectPredictorAdapter", "FSBPredictorAdapter", "PredictorAdapter"]
