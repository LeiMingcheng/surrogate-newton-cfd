"""Pitching-moment convention at the NK/solver boundary."""

from __future__ import annotations

from typing import Any


STANDARD_MOMENT_REFERENCE = (0.25, 0.0)
STANDARD_MOMENT_SIGN_CONVENTION = "nose_up_positive"
ADFLOW_CMZ_SIGN_CONVENTION = "right_hand_positive_z"


def right_hand_cmz_to_standard_cm(value: Any) -> Any:
    return -value


def adflow_cmz_to_standard_cm(value: Any) -> Any:
    return right_hand_cmz_to_standard_cm(value)


__all__ = [
    "ADFLOW_CMZ_SIGN_CONVENTION",
    "STANDARD_MOMENT_REFERENCE",
    "STANDARD_MOMENT_SIGN_CONVENTION",
    "adflow_cmz_to_standard_cm",
    "right_hand_cmz_to_standard_cm",
]
