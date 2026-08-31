from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from gcmc_port.gro import (
    Atom,
    GroStructure,
    count_water_molecules,
    last_residue_atoms,
    parse_atom_line,
    parse_gro,
    write_gro,
)
from gcmc_port.helpers import remove_waters_near_centroid


class GroFormattingTests(unittest.TestCase):
    def test_large_serials_wrap_without_shifting_coordinate_columns(self) -> None:
        line = Atom(47_925, "NA", "NA", 1_072_061, 20.630, 20.745, 3.984).line()

        self.assertEqual(len(line), 44)
        self.assertEqual(line[0:5], "47925")
        self.assertEqual(line[5:10], "NA   ")
        self.assertEqual(line[10:15], "   NA")
        self.assertEqual(line[15:20], "72061")
        self.assertEqual(line[20:28], "  20.630")
        self.assertAlmostEqual(parse_atom_line(line).x, 20.630)

    def test_residue_and_atom_serials_use_gromacs_modulo_100000(self) -> None:
        self.assertEqual(Atom(99_999, "SOL", "OW", 99_999, 0.0, 0.0, 0.0).line()[0:20], "99999SOL     OW99999")
        self.assertEqual(Atom(100_000, "SOL", "OW", 100_000, 0.0, 0.0, 0.0).line()[0:20], "    0SOL     OW    0")
        self.assertEqual(Atom(100_001, "SOL", "OW", 100_001, 0.0, 0.0, 0.0).line()[0:20], "    1SOL     OW    1")

    def test_parse_gro_recovers_line_order_atom_indices_and_unwraps_residues(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large.gro"
            path.write_text(
                "large\n"
                "3\n"
                + Atom(99_999, "SOL", "OW", 99_999, 0.1, 0.2, 0.3).line()
                + "\n"
                + Atom(100_000, "SOL", "OW", 100_000, 0.4, 0.5, 0.6).line()
                + "\n"
                + Atom(100_001, "SOL", "OW", 100_001, 0.7, 0.8, 0.9).line()
                + "\n2.0 2.0 2.0\n",
                encoding="utf-8",
            )

            structure = parse_gro(path)

        self.assertEqual([atom.atomnr for atom in structure.atoms], [1, 2, 3])
        self.assertEqual([atom.resid for atom in structure.atoms], [99_999, 100_000, 100_001])

    def test_parse_variable_precision_coordinates(self) -> None:
        prefix = f"{1:5d}{'SOL':<5}{'OW':>5}{1:5d}"
        line = prefix + f"{0.12345:10.5f}{-1.23456:10.5f}{2.34567:10.5f}"

        atom = parse_atom_line(line)

        self.assertAlmostEqual(atom.x, 0.12345)
        self.assertAlmostEqual(atom.y, -1.23456)
        self.assertAlmostEqual(atom.z, 2.34567)

    def test_atom_count_line_is_not_wrapped(self) -> None:
        class LargeAtomList(list[Atom]):
            def __len__(self) -> int:
                return 1_072_061

            def __iter__(self):
                return iter(())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "count.gro"
            structure = GroStructure("large", LargeAtomList(), "1.0 1.0 1.0")
            write_gro(path, structure)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines[1].strip(), "1072061")


class LargeGroResidueTests(unittest.TestCase):
    def test_water_count_and_last_residue_use_contiguous_occurrences(self) -> None:
        structure = GroStructure(
            "wrapped",
            [
                Atom(99_999, "SOL", "OW", 1, 0.0, 0.0, 0.0),
                Atom(100_000, "SOL", "OW", 2, 1.0, 0.0, 0.0),
                Atom(99_999, "SOL", "OW", 3, 2.0, 0.0, 0.0),
            ],
            "3.0 3.0 3.0",
        )

        self.assertEqual(count_water_molecules(structure), 3)
        self.assertEqual([atom.atomnr for atom in last_residue_atoms(structure)], [3])

    def test_sphere_water_removal_preserves_topology_atom_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.gro"
            output_path = Path(tmpdir) / "output.gro"
            write_gro(
                input_path,
                GroStructure(
                    "ordered",
                    [
                        Atom(1, "PRO", "CA", 1, 1.0, 1.0, 1.0),
                        Atom(2, "SOL", "OW", 2, 0.0, 0.0, 0.0),
                        Atom(3, "SOL", "OW", 3, 2.0, 2.0, 2.0),
                        Atom(4, "NA", "NA", 4, 2.5, 2.5, 2.5),
                    ],
                    "3.0 3.0 3.0",
                ),
            )

            remove_waters_near_centroid(input_path, output_path, (0.0, 0.0, 0.0), 0.2)
            output = parse_gro(output_path)

        self.assertEqual([(atom.resname, atom.resid) for atom in output.atoms], [("PRO", 1), ("SOL", 3), ("NA", 4)])


if __name__ == "__main__":
    unittest.main()
