from __future__ import annotations

import unittest

from NK_resume.metrics.moment import (
    STANDARD_MOMENT_REFERENCE as NK_MOMENT_REFERENCE,
    adflow_cmz_to_standard_cm,
)
from NK_resume.metrics.force import normalize_force_coefficients
from surrogate.physics.forces.conventions import (
    STANDARD_MOMENT_REFERENCE,
    STANDARD_MOMENT_SIGN_CONVENTION,
    right_hand_cmz_to_standard_cm,
)
from surrogate.serving.aoa import AoASolverConfig


class ForceConventionTests(unittest.TestCase):
    def test_public_moment_contract_is_quarter_chord_nose_up_positive(self) -> None:
        self.assertEqual(STANDARD_MOMENT_REFERENCE, (0.25, 0.0))
        self.assertEqual(NK_MOMENT_REFERENCE, STANDARD_MOMENT_REFERENCE)
        self.assertEqual(STANDARD_MOMENT_SIGN_CONVENTION, "nose_up_positive")
        self.assertEqual(AoASolverConfig().moment_center, STANDARD_MOMENT_REFERENCE)

    def test_native_right_hand_cmz_is_negated_at_public_boundary(self) -> None:
        self.assertAlmostEqual(right_hand_cmz_to_standard_cm(0.17), -0.17)
        self.assertAlmostEqual(adflow_cmz_to_standard_cm(-0.08), 0.08)
        self.assertEqual(normalize_force_coefficients({"cmz": 0.17}), {"cm": -0.17})
        self.assertEqual(normalize_force_coefficients({"cm": -0.17}), {"cm": -0.17})


if __name__ == "__main__":
    unittest.main()
