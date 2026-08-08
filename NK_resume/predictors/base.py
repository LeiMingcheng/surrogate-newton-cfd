"""Base predictor adapter contract."""

from __future__ import annotations

from typing import Any, Protocol

from ..schema import ResumeCase


class PredictorAdapter(Protocol):
    """Build canonical cases without exposing model-specific internals."""

    name: str
    predictor_kind: str

    def build_case(self, **fields: Any) -> ResumeCase:
        """Build one canonical case from already prepared fields."""
        ...
