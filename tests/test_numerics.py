import math
import unittest

import numpy as np

from relex.core import _integrate_tabulated_curve, _integrate_trajectory, run_relex


class NumericalIntegrationTests(unittest.TestCase):
    def test_complex_trajectory(self) -> None:
        def derivative(x, y, _ctx):
            return 1j * y

        result = _integrate_trajectory(
            np.array([1.0 + 0.0j]),
            0.0,
            math.pi / 2.0,
            1.0e-9,
            0.1,
            derivative,
            None,
        )

        np.testing.assert_allclose(result, np.array([1j]), rtol=1.0e-8, atol=1.0e-8)

    def test_tabulated_linear_curve(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 4.0])
        y = 3.0 * x + 2.0
        lower = 0.25
        upper = 3.5
        expected = (1.5 * upper**2 + 2.0 * upper) - (1.5 * lower**2 + 2.0 * lower)

        result = _integrate_tabulated_curve(x, y, lower, upper)

        self.assertAlmostEqual(result, expected, places=11)


class FortranRegressionTests(unittest.TestCase):
    def test_sulfur_42_on_gold_cross_section(self) -> None:
        inputs = {
            "system": {
                "ap": 42.0,
                "zp": 16.0,
                "at": 197.0,
                "zt": 79.0,
                "eca": 20.0,
                "iw": 0,
                "iout": 0,
            },
            "integration": {"nb": 30, "accur": 0.001, "bmin": 14.0, "itot": 0},
            "options": {"iopw": 0, "iopnuc": 0},
            "grid": {"ngrid": 200},
            "states": [
                {"i": 1, "ex": 0.0, "spin": 0.0},
                {"i": 2, "ex": 0.890, "spin": 2.0},
            ],
            "matrix_elements": [
                {
                    "i": 1,
                    "j": 2,
                    "mate1": 0.0,
                    "mate2": math.sqrt(397.0),
                    "matm1": 0.0,
                }
            ],
        }

        result = run_relex(inputs)

        self.assertAlmostEqual(float(result.cross_sections[2]), 266.9, delta=0.1)


if __name__ == "__main__":
    unittest.main()
