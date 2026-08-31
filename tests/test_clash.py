from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gcmc_port.clash import (
    ClashDetail,
    atom_radius_nm,
    describe_clash,
    find_heavy_atom_clash,
    infer_element,
    pair_clash_cutoff,
    residue_clash_in_structure,
)
from gcmc_port.config import (
    AnchorConfig,
    CavityConfig,
    Config,
    ExecutionConfig,
    LoopConfig,
    PathsConfig,
    SimulationConfig,
    SlurmConfig,
)
from gcmc_port.gro import Atom, GroStructure, write_gro
from gcmc_port.moves import propose_insertion, propose_rotation, propose_translation
from gcmc_port.workflow import GCMCWorkflow


class FixedRandom:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def random(self) -> float:
        if not self._values:
            return 0.5
        return self._values.pop(0)

    def randrange(self, stop: int) -> int:
        return 0 if stop > 0 else 0


def _write_structure(
    path: Path,
    atoms: list[Atom],
    *,
    title: str = "test",
    box_line: str = "2.00000 2.00000 2.00000",
) -> None:
    write_gro(path, GroStructure(title=title, atoms=atoms, box_line=box_line))


def _write_topology(path: Path) -> None:
    path.write_text("[ molecules ]\nSOL 1\n", encoding="utf-8")


def _dummy_paths(root: Path) -> PathsConfig:
    dummy = root / "dummy"
    dummy.mkdir(exist_ok=True)
    ff_dir = dummy / "amber14sb_parmbsc1.ff"
    ff_dir.mkdir(exist_ok=True)
    for name in (
        "topol.top",
        "chk.mdp",
        "steep.mdp",
        "em.mdp",
        "init.gro",
        "COM.gro",
        "WAT.itp",
        "residuetypes.dat",
    ):
        (dummy / name).write_text("", encoding="utf-8")
    return PathsConfig(
        project_root=dummy,
        work_root=dummy,
        forcefield_dir=ff_dir,
        residue_types=dummy / "residuetypes.dat",
        topology=dummy / "topol.top",
        water_itp=dummy / "WAT.itp",
        chk_mdp=dummy / "chk.mdp",
        steep_mdp=dummy / "steep.mdp",
        em_mdp=dummy / "em.mdp",
        init_gro=dummy / "init.gro",
        gas_gro=dummy / "COM.gro",
    )


class ClashHelperTests(unittest.TestCase):
    def test_element_inference_and_radius_reuses_existing_rules(self) -> None:
        self.assertEqual(infer_element("OW"), "O")
        self.assertEqual(infer_element("HW1"), "H")
        self.assertEqual(infer_element("CL1"), "CL")
        self.assertEqual(infer_element("BRA"), "BR")
        self.assertEqual(infer_element("CA"), "C")
        self.assertAlmostEqual(atom_radius_nm("OW"), 0.152)
        self.assertAlmostEqual(atom_radius_nm("CL1"), 0.175)

    def test_pair_cutoff_uses_element_radii_with_rvdw_floor(self) -> None:
        oxygen = Atom(1, "SOL", "OW", 1, 0.0, 0.0, 0.0)
        carbon = Atom(2, "PRO", "CA", 2, 0.0, 0.0, 0.0)
        fluorine = Atom(3, "LIG", "F1", 3, 0.0, 0.0, 0.0)

        self.assertAlmostEqual(pair_clash_cutoff(oxygen, carbon, 0.20), 0.85 * (0.152 + 0.170))
        self.assertAlmostEqual(pair_clash_cutoff(oxygen, fluorine, 0.30), 0.30)

    def test_heavy_atom_clash_ignores_hydrogen_only_contacts(self) -> None:
        moved = [Atom(2, "SOL", "HW1", 2, 0.10, 0.10, 0.10)]
        others = [Atom(1, "PRO", "CA", 1, 0.11, 0.10, 0.10)]

        self.assertIsNone(find_heavy_atom_clash(moved, others, 0.20))


class MoveClashTests(unittest.TestCase):
    def test_insertion_rejects_existing_cavity_water_even_when_nonwater_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            previous_gro = tmp / "previous.gro"
            current_gro = tmp / "current.gro"
            previous_top = tmp / "previous.top"
            current_top = tmp / "current.top"
            gas_gro = tmp / "gas.gro"
            xyz_path = tmp / "trials.xyz"

            _write_structure(
                previous_gro,
                [
                    Atom(1, "PRO", "CA", 1, 1.20, 1.20, 1.20),
                    Atom(2, "SOL", "OW", 2, 0.25, 0.25, 0.25),
                    Atom(2, "SOL", "HW1", 3, 0.25, 0.35, 0.25),
                    Atom(2, "SOL", "HW2", 4, 0.35, 0.25, 0.25),
                ],
            )
            _write_structure(gas_gro, [Atom(1, "SOL", "OW", 1, 0.00, 0.00, 0.00)])
            _write_topology(previous_top)

            import numpy as np

            from gcmc_port.cavity import VoxelMask

            mask = VoxelMask(
                points=np.asarray([[0.50, 0.25, 0.25]], dtype=float),
                dx=0.05,
                reference_point=(0.50, 0.25, 0.25),
                effective_volume=0.05 ** 3,
            )

            with self.assertRaisesRegex(RuntimeError, "heavy-atom clash"):
                propose_insertion(
                    previous_gro,
                    current_gro,
                    previous_top,
                    current_top,
                    rvdw=0.20,
                    gas_name="SOL",
                    gas_gro=gas_gro,
                    rmax=0.6,
                    xyz_path=xyz_path,
                    out_dir=tmp,
                    mask_model=mask,
                    mask_dx=0.0,
                    rng=FixedRandom([0.0, 0.0, 0.0]),
                )

    def test_translation_rejects_moved_water_that_creates_heavy_atom_clash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            previous_gro = tmp / "previous.gro"
            current_gro = tmp / "current.gro"
            previous_top = tmp / "previous.top"
            current_top = tmp / "current.top"

            _write_structure(
                previous_gro,
                [
                    Atom(1, "PRO", "CA", 1, 0.20, 0.50, 0.50),
                    Atom(2, "SOL", "OW", 2, 0.50, 0.50, 0.50),
                ],
            )
            _write_topology(previous_top)

            with self.assertRaisesRegex(RuntimeError, "heavy-atom clash"):
                propose_translation(
                    previous_gro,
                    current_gro,
                    previous_top,
                    current_top,
                    nmol=1,
                    gas_name="SOL",
                    orig_atom_count=1,
                    rvdw=0.20,
                    delta=0.05,
                    rng=FixedRandom([0.0, 0.5, 0.5]),
                )

    def test_rotation_rejects_moved_inserted_residue_that_creates_heavy_atom_clash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            previous_gro = tmp / "previous.gro"
            current_gro = tmp / "current.gro"
            previous_top = tmp / "previous.top"
            current_top = tmp / "current.top"

            _write_structure(
                previous_gro,
                [
                    Atom(1, "PRO", "CA", 1, 0.85, 0.17, 0.50),
                    Atom(2, "LIG", "C1", 2, 0.80, 0.50, 0.50),
                    Atom(2, "LIG", "C2", 3, 0.90, 0.50, 0.50),
                ],
            )
            _write_topology(previous_top)

            with self.assertRaisesRegex(RuntimeError, "heavy-atom clash"):
                propose_rotation(
                    previous_gro,
                    current_gro,
                    previous_top,
                    current_top,
                    nmol=1,
                    gas_name="LIG",
                    orig_atom_count=1,
                    rvdw=0.20,
                    rng=FixedRandom([1.0, 0.0, 0.125]),
                )


class WorkflowClashTests(unittest.TestCase):
    def test_post_em_clash_helper_detects_remaining_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            confout = tmp / "confout.gro"
            _write_structure(
                confout,
                [
                    Atom(1, "PRO", "CA", 1, 0.20, 0.50, 0.50),
                    Atom(2, "SOL", "OW", 2, 0.45, 0.50, 0.50),
                ],
            )

            workflow = GCMCWorkflow(
                Config(
                    config_path=tmp / "config.toml",
                    paths=_dummy_paths(tmp),
                    execution=ExecutionConfig(),
                    anchor=AnchorConfig(),
                    simulation=SimulationConfig(rvdw=0.20),
                    cavity=CavityConfig(),
                    loop=LoopConfig(),
                    slurm=SlurmConfig(),
                )
            )

            clash = workflow._post_em_clash(confout, 2, type_move="I")

            self.assertIsNotNone(clash)
            self.assertLess(clash.distance, clash.cutoff)
            self.assertIn("SOL2:OW", describe_clash(clash))

    def test_post_em_clash_helper_ignores_deletion_moves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            confout = tmp / "confout.gro"
            _write_structure(
                confout,
                [
                    Atom(1, "PRO", "CA", 1, 0.20, 0.50, 0.50),
                    Atom(2, "SOL", "OW", 2, 0.45, 0.50, 0.50),
                ],
            )

            workflow = GCMCWorkflow(
                Config(
                    config_path=tmp / "config.toml",
                    paths=_dummy_paths(tmp),
                    execution=ExecutionConfig(),
                    anchor=AnchorConfig(),
                    simulation=SimulationConfig(rvdw=0.20),
                    cavity=CavityConfig(),
                    loop=LoopConfig(),
                    slurm=SlurmConfig(),
                )
            )

            self.assertIsNone(workflow._post_em_clash(confout, 2, type_move="D"))

    def test_residue_clash_in_structure_reports_pair(self) -> None:
        structure = GroStructure(
            title="clash",
            atoms=[
                Atom(1, "PRO", "CA", 1, 0.20, 0.50, 0.50),
                Atom(2, "SOL", "OW", 2, 0.45, 0.50, 0.50),
            ],
            box_line="2.00000 2.00000 2.00000",
        )

        clash = residue_clash_in_structure(structure, 2, 0.20)

        self.assertIsInstance(clash, ClashDetail)
        self.assertEqual(clash.moved_atom.atomname, "OW")
        self.assertEqual(clash.other_atom.atomname, "CA")


if __name__ == "__main__":
    unittest.main()
