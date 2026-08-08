"""FSB predictor adapter."""

from __future__ import annotations

from typing import Any, Iterable

from ..cases import build_fsb_case
from ..exceptions import ContractError
from ..schema import ResumeCase


class FSBPredictorAdapter:
    name = "fsb"
    predictor_kind = "fsb"

    def build_case(self, **fields: Any) -> ResumeCase:
        return build_fsb_case(**fields)

    def collect_cases(self, ordinals: Iterable[int]) -> list[ResumeCase]:
        list(ordinals)
        raise ContractError(
            "FSBPredictorAdapter is a runtime case builder and does not own model inference; "
            "export FSB model-side cases with `python -m surrogate.scripts.main "
            "--task nk_resume --ordinals ...`"
        )

    def predict_terminal_fields(self, cases: Iterable[ResumeCase]) -> dict[str, object]:
        list(cases)
        raise ContractError(
            "FSB terminal prediction is model-side; use surrogate.inference or the "
            "surrogate nk_resume export CLI"
        )
