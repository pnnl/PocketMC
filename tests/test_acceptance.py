from __future__ import annotations

from pathlib import Path
import math
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from gcmc_port.moves import acceptance_probability


class FixedRandom:
    def __init__(self, value: float) -> None:
        self.value = value

    def random(self) -> float:
        return self.value


class AcceptanceProbabilityTests(unittest.TestCase):
    def test_configured_gas_constant_controls_beta(self) -> None:
        common = {
            "de": 1.0,
            "temperature": 1.0,
            "move": 2,
            "veff": 1.0,
            "v0": 1.0,
            "nins": 1,
            "rng": FixedRandom(0.1),
        }

        self.assertEqual(acceptance_probability(**common, gas_constant=0.5), 2)
        self.assertEqual(acceptance_probability(**common, gas_constant=0.25), 0)

    def test_insertion_and_deletion_prefactors_are_reciprocal(self) -> None:
        temperature = 300.0
        gas_constant = 0.008314
        veff = 0.8
        v0 = 0.04
        nins = 2
        pref_insert = (veff / v0) / (nins + 1)
        pref_delete_reverse = (nins + 1) * (v0 / veff)

        self.assertAlmostEqual(math.log(pref_insert) + math.log(pref_delete_reverse), 0.0)
        self.assertEqual(
            acceptance_probability(
                de=-100.0,
                temperature=temperature,
                move=1,
                veff=veff,
                v0=v0,
                nins=nins,
                rng=FixedRandom(0.5),
                gas_constant=gas_constant,
            ),
            2,
        )

    def test_invalid_thermodynamic_inputs_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "temperature must be positive"):
            acceptance_probability(
                de=0.0,
                temperature=0.0,
                move=1,
                veff=1.0,
                v0=1.0,
                nins=0,
                rng=FixedRandom(0.5),
            )
        with self.assertRaisesRegex(ValueError, "veff must be positive"):
            acceptance_probability(
                de=0.0,
                temperature=300.0,
                move=1,
                veff=0.0,
                v0=1.0,
                nins=0,
                rng=FixedRandom(0.5),
            )


if __name__ == "__main__":
    unittest.main()
