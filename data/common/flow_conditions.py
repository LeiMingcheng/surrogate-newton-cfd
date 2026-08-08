#!/usr/bin/env python
"""
Shared reference flow-condition utilities for dataset generation and packing.
"""

from __future__ import annotations

import math


REFERENCE_T_INF = 300.0
REFERENCE_GAMMA = 1.4
REFERENCE_GAS_CONSTANT = 287.05
REFERENCE_P_INF = 101325.0
REFERENCE_MU0 = 1.716e-5
REFERENCE_T0 = 273.15
REFERENCE_SUTHERLAND = 110.4
REFERENCE_CHORD = 1.0


def sutherland_viscosity(
    temperature: float,
    *,
    mu0: float = REFERENCE_MU0,
    t0: float = REFERENCE_T0,
    sutherland: float = REFERENCE_SUTHERLAND,
) -> float:
    temp = float(temperature)
    return float(mu0 * (temp / t0) ** 1.5 * (t0 + sutherland) / (temp + sutherland))


def coupled_reynolds_from_mach(
    mach: float,
    *,
    chord: float = REFERENCE_CHORD,
    temperature: float = REFERENCE_T_INF,
    pressure: float = REFERENCE_P_INF,
    gamma: float = REFERENCE_GAMMA,
    gas_constant: float = REFERENCE_GAS_CONSTANT,
) -> float:
    """
    Convert Mach to Reynolds number under the shared sea-level reference state.
    """
    mach_value = float(mach)
    chord_value = float(chord)
    temp = float(temperature)
    p_inf = float(pressure)
    a_inf = math.sqrt(float(gamma) * float(gas_constant) * temp)
    rho_inf = p_inf / (float(gas_constant) * temp)
    mu_inf = sutherland_viscosity(temp)
    return float(rho_inf * (mach_value * a_inf) * chord_value / mu_inf)
