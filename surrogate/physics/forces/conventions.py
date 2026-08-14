"""Shared pitching-moment convention for public aerodynamic coefficients."""

from __future__ import annotations

from typing import Any


STANDARD_MOMENT_REFERENCE = (0.25, 0.0)
STANDARD_MOMENT_SIGN_CONVENTION = "nose_up_positive"
ADFLOW_CMZ_SIGN_CONVENTION = "right_hand_positive_z"


def right_hand_cmz_to_standard_cm(value: Any) -> Any:
    """Convert a right-hand-positive z moment to nose-up-positive ``Cm``."""

    return -value


def adflow_cmz_to_standard_cm(value: Any) -> Any:
    """Convert native ADflow ``cmz`` to the public pitching-moment sign."""

    return right_hand_cmz_to_standard_cm(value)


__all__ = [
    "ADFLOW_CMZ_SIGN_CONVENTION",
    "STANDARD_MOMENT_REFERENCE",
    "STANDARD_MOMENT_SIGN_CONVENTION",
    "adflow_cmz_to_standard_cm",
    "right_hand_cmz_to_standard_cm",
]
