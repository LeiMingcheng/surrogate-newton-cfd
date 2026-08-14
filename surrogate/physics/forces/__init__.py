"""Aerodynamic force and coefficient utilities."""

from surrogate.physics.forces.coefficients import ForceCoefficientsCalculator, compute_force_coefficients
from surrogate.physics.forces.conventions import (
    ADFLOW_CMZ_SIGN_CONVENTION,
    STANDARD_MOMENT_REFERENCE,
    STANDARD_MOMENT_SIGN_CONVENTION,
    adflow_cmz_to_standard_cm,
    right_hand_cmz_to_standard_cm,
)
from surrogate.physics.forces.torch_coefficients import (
    compute_cdp_torch,
    compute_force_components_ogrid_torch,
    compute_force_coefficients_ogrid_torch,
    extract_wall_cp_ogrid_torch,
)

__all__ = [
    "ADFLOW_CMZ_SIGN_CONVENTION",
    "ForceCoefficientsCalculator",
    "STANDARD_MOMENT_REFERENCE",
    "STANDARD_MOMENT_SIGN_CONVENTION",
    "adflow_cmz_to_standard_cm",
    "compute_cdp_torch",
    "compute_force_coefficients",
    "compute_force_components_ogrid_torch",
    "compute_force_coefficients_ogrid_torch",
    "extract_wall_cp_ogrid_torch",
    "right_hand_cmz_to_standard_cm",
]
