"""Direct surrogate predictor adapter."""

from __future__ import annotations

from typing import Any, Iterable

from ..cases import build_direct_case
from ..exceptions import ContractError
from ..schema import ResumeCase


class DirectPredictorAdapter:
    name = "direct"
    predictor_kind = "direct"

    def build_case(self, **fields: Any) -> ResumeCase:
        return build_direct_case(**fields)

    def collect_cases(self, ordinals: Iterable[int]) -> list[ResumeCase]:
        list(ordinals)
        raise ContractError(
            "DirectPredictorAdapter is a runtime case builder and does not own model inference; "
            "export direct model-side cases with `python -m surrogate.scripts.main "
            "--task nk_resume --ordinals ...`"
        )
