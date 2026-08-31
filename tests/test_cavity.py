from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gcmc_port.cavity import build_cavity_from_structure, parse_residue_spec, _matching_atoms
from gcmc_port.gro import coordinates_center, parse_gro


class BuildCavityAutoTests(unittest.TestCase):
    def test_auto_mode_prefers_pocket_near_excluded_residue(self) -> None:
        gro_path = ROOT / "assets" / "example" / "init.gro"
        structure = parse_gro(gro_path)
        excluded_atoms = _matching_atoms(structure, [parse_residue_spec("800ATC")])
        excluded_center = np.asarray(coordinates_center(excluded_atoms), dtype=float)

        with tempfile.TemporaryDirectory() as tmpdir:
            outprefix = Path(tmpdir) / "cavity"
            build_cavity_from_structure(
                gro_path,
                outprefix=outprefix,
                mode="auto",
                dx=0.075,
                probe_radius=0.10,
                search_radius=0.9,
                exclude_residues=["800ATC"],
                candidate_limit=3,
            )

            summary_lines = outprefix.with_name("cavity_candidates.tsv").read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(summary_lines), 2)

            first_fields = summary_lines[1].split("\t")
            first_seed = np.asarray([float(first_fields[3]), float(first_fields[4]), float(first_fields[5])], dtype=float)
            self.assertLess(float(np.linalg.norm(first_seed - excluded_center)), 0.75)
            self.assertTrue(first_fields[10].strip())


if __name__ == "__main__":
    unittest.main()
