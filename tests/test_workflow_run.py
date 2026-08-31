from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gcmc_port.config import (
    AnchorConfig,
    CavityBuildConfig,
    CavityConfig,
    Config,
    ExecutionConfig,
    LoopConfig,
    PathsConfig,
    SimulationConfig,
    SlurmConfig,
)
from gcmc_port.gro import Atom, GroStructure, write_gro
from gcmc_port.topology import ensure_molecule_include
from gcmc_port.workflow import GCMCWorkflow, WorkflowContext


def _dummy_config(root: Path, *, replica_dirs: list[str]) -> Config:
    return Config(
        config_path=root / "config.toml",
        paths=PathsConfig(
            project_root=root,
            work_root=root,
            forcefield_dir=root,
            residue_types=root / "residuetypes.dat",
            topology=root / "topol.top",
            water_itp=root / "WAT.itp",
            chk_mdp=root / "chk.mdp",
            steep_mdp=root / "steep.mdp",
            em_mdp=root / "em.mdp",
            init_gro=root / "init.gro",
            gas_gro=root / "COM.gro",
        ),
        execution=ExecutionConfig(),
        anchor=AnchorConfig(),
        simulation=SimulationConfig(),
        cavity=CavityConfig(),
        loop=LoopConfig(replica_dirs=replica_dirs, replica_count=len(replica_dirs)),
        slurm=SlurmConfig(),
        cavity_build=CavityBuildConfig(),
    )


class WorkflowRunBehaviorTests(unittest.TestCase):
    def test_select_centering_candidate_prefers_lower_boundary_fraction(self) -> None:
        chosen_name, chosen_fraction = GCMCWorkflow._select_centering_candidate(0.18, 0.16)

        self.assertEqual(chosen_name, "fallback")
        self.assertAlmostEqual(chosen_fraction, 0.16)

    def test_run_continues_to_next_case_for_multi_run_batches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow = GCMCWorkflow(_dummy_config(root, replica_dirs=["00", "01"]))

            with (
                patch.object(workflow, "_prepare_runtime_inputs"),
                patch.object(workflow, "_validate_inputs"),
                patch.object(workflow, "_run_single", side_effect=[RuntimeError("centering failed"), None]) as mocked_run,
            ):
                with self.assertRaisesRegex(RuntimeError, "1 GCMC run.*failed"):
                    workflow.run()

            failure_log = root / "00" / "workflow.log"
            failure_exists = failure_log.exists()
            failure_text = failure_log.read_text(encoding="utf-8")

        self.assertEqual(mocked_run.call_count, 2)
        self.assertTrue(failure_exists)
        self.assertIn("Run failed and will be skipped", failure_text)
        self.assertIn("centering failed", failure_text)

    def test_run_still_raises_for_single_case_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            workflow = GCMCWorkflow(_dummy_config(root, replica_dirs=["run"]))

            with (
                patch.object(workflow, "_prepare_runtime_inputs"),
                patch.object(workflow, "_validate_inputs"),
                patch.object(workflow, "_run_single", side_effect=RuntimeError("single failure")),
            ):
                with self.assertRaisesRegex(RuntimeError, "single failure"):
                    workflow.run()

    def test_centering_atoms_include_solute_before_water_and_exclude_inserted_gas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            residue_types = root / "residuetypes.dat"
            residue_types.write_text("ALA Protein\nSOL Water\nNA Ion\n", encoding="utf-8")
            gro_path = root / "center.gro"
            write_gro(
                gro_path,
                GroStructure(
                    title="center",
                    atoms=[
                        Atom(1, "ALA", "N", 1, 0.0, 0.0, 0.0),
                        Atom(1, "ALA", "CA", 2, 0.1, 0.0, 0.0),
                        Atom(2, "LIG", "C1", 3, 0.2, 0.0, 0.0),
                        Atom(10, "SOL", "OW", 4, 0.3, 0.0, 0.0),
                        Atom(10, "SOL", "HW1", 5, 0.4, 0.0, 0.0),
                        Atom(20, "NA", "NA", 6, 0.5, 0.0, 0.0),
                        Atom(30, "COM", "CJ", 7, 0.6, 0.0, 0.0),
                    ],
                    box_line="1.0 1.0 1.0",
                ),
            )
            workflow = GCMCWorkflow(_dummy_config(root, replica_dirs=["run"]))

            atom_numbers = workflow._centering_atom_numbers(gro_path, "COM")

        self.assertEqual(atom_numbers, [1, 2, 3])

    def test_sphere_membership_uses_same_water_oxygen_support_as_insertion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gro_path = root / "sphere.gro"
            write_gro(
                gro_path,
                GroStructure(
                    title="sphere",
                    atoms=[
                        Atom(1, "ALA", "CA", 1, 0.0, 0.0, 0.0),
                        Atom(2, "WAT", "OW", 2, 0.4, 0.0, 0.0),
                        Atom(2, "WAT", "HW1", 3, 0.45, 0.0, 0.0),
                        Atom(3, "WAT", "OW", 4, 0.7, 0.0, 0.0),
                        Atom(3, "WAT", "HW1", 5, 0.75, 0.0, 0.0),
                    ],
                    box_line="2.0 2.0 2.0",
                ),
            )
            config = _dummy_config(root, replica_dirs=["run"])
            config.anchor = AnchorConfig(
                anchor="1ALA",
                resid=1,
                resname="ALA",
                residues=["1ALA"],
                reference_mode="atoms",
                center_atoms=["CA"],
            )
            config.simulation.rmax = 0.6
            workflow = GCMCWorkflow(config)
            ctx = WorkflowContext(
                run_dir=root,
                log_file=root / "mc.log",
                progress_file=root / "workflow.log",
                positions_file=root / "trials.xyz",
                mask_trajectory_file=root / "mask.gro",
                previous_top=root / "previous.top",
                current_top=root / "current.top",
                previous_gro=gro_path,
                current_gro=root / "current.gro",
                gas_name="WAT",
                na_gas=2,
                orig_atom_count=1,
            )

            residue_ids = workflow._inserted_residue_ids_in_sphere(gro_path, ctx)

        self.assertEqual(residue_ids, [2])

    def test_ensure_molecule_include_inserts_before_system_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = Path(tmpdir) / "topol.top"
            top_path.write_text(
                "\n".join(
                    [
                        '#include "./amber14sb_parmbsc1.ff/forcefield.itp"',
                        "",
                        "[ moleculetype ]",
                        "Protein 3",
                        "",
                        "[ system ]",
                        "demo",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            ensure_molecule_include(top_path, "COM.itp", "COM_atomtypes.itp")
            ensure_molecule_include(top_path, "COM.itp", "COM_atomtypes.itp")
            text = top_path.read_text(encoding="utf-8")

        self.assertEqual(text.count('#include "COM.itp"'), 1)
        self.assertEqual(text.count('#include "COM_atomtypes.itp"'), 1)
        self.assertLess(text.index('#include "COM_atomtypes.itp"'), text.index("[ moleculetype ]"))
        self.assertLess(text.index('#include "COM.itp"'), text.index("[ system ]"))

    def test_molecule_atomtypes_include_skips_when_topology_forcefield_defines_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            forcefield_dir = root / "amber.ff"
            forcefield_dir.mkdir()
            (forcefield_dir / "forcefield.itp").write_text('#include "ffnonbonded.itp"\n', encoding="utf-8")
            (forcefield_dir / "ffnonbonded.itp").write_text(
                "\n".join(
                    [
                        "[ atomtypes ]",
                        "CMT 6 12.01 0.0000 A 3.39967e-01 3.59824e-01",
                        "MMO 0 0.00 0.0000 A 0.00000e+00 0.00000e+00",
                        "OMT 8 15.9994 0.0000 A 2.95992e-01 8.78640e-01",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            top_path = root / "topol.top"
            top_path.write_text('#include "./amber.ff/forcefield.itp"\n[ system ]\ndemo\n', encoding="utf-8")
            (root / "COM.itp").write_text("[ moleculetype ]\nCOM 1\n", encoding="utf-8")
            (root / "COM_atomtypes.itp").write_text(
                "[ atomtypes ]\nCMT 6 12.01 0.0000 A 1 1\nMMO 0 0.00 0.0000 A 0 0\nOMT 8 15.9994 0.0000 A 1 1\n",
                encoding="utf-8",
            )
            config = _dummy_config(root, replica_dirs=["run"])
            config.paths.forcefield_dir = forcefield_dir
            config.paths.topology = top_path
            config.paths.water_itp = root / "COM.itp"
            workflow = GCMCWorkflow(config)

            include_name = workflow._molecule_atomtypes_include_name()

        self.assertIsNone(include_name)

    def test_molecule_atomtypes_include_used_when_topology_forcefield_lacks_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            forcefield_dir = root / "amber.ff"
            forcefield_dir.mkdir()
            (forcefield_dir / "forcefield.itp").write_text('#include "ffnonbonded.itp"\n', encoding="utf-8")
            (forcefield_dir / "ffnonbonded.itp").write_text("[ atomtypes ]\nCT 6 12.01 0.0000 A 1 1\n", encoding="utf-8")
            top_path = root / "topol.top"
            top_path.write_text('#include "./amber.ff/forcefield.itp"\n[ system ]\ndemo\n', encoding="utf-8")
            (root / "COM.itp").write_text("[ moleculetype ]\nCOM 1\n", encoding="utf-8")
            (root / "COM_atomtypes.itp").write_text(
                "[ atomtypes ]\nCMT 6 12.01 0.0000 A 1 1\nMMO 0 0.00 0.0000 A 0 0\nOMT 8 15.9994 0.0000 A 1 1\n",
                encoding="utf-8",
            )
            config = _dummy_config(root, replica_dirs=["run"])
            config.paths.forcefield_dir = forcefield_dir
            config.paths.topology = top_path
            config.paths.water_itp = root / "COM.itp"
            workflow = GCMCWorkflow(config)

            include_name = workflow._molecule_atomtypes_include_name()

        self.assertEqual(include_name, "COM_atomtypes.itp")

    def test_copy_inputs_copies_user_topology_includes_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            forcefield_dir = root / "amber03.ff"
            forcefield_dir.mkdir()
            (forcefield_dir / "forcefield.itp").write_text('#include "ffnonbonded.itp"\n', encoding="utf-8")
            (forcefield_dir / "ffnonbonded.itp").write_text("[ atomtypes ]\nCT 6 12.01 0.0000 A 1 1\n", encoding="utf-8")
            (root / "topol.top").write_text(
                "\n".join(
                    [
                        "; Include forcefield parameters",
                        '#include "./amber03.ff/forcefield.itp"',
                        '#include "./acs.itp"',
                        '#include "./nested/codh.itp"',
                        "",
                        "[ system ]",
                        "demo",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "acs.itp").write_text('#include "acs_posre.itp"\n[ moleculetype ]\nACS 1\n', encoding="utf-8")
            (root / "acs_posre.itp").write_text("[ position_restraints ]\n", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "codh.itp").write_text('#include "../shared/cofactor.itp"\n[ moleculetype ]\nCODH 3\n', encoding="utf-8")
            (root / "shared").mkdir()
            (root / "shared" / "cofactor.itp").write_text("[ moleculetype ]\nCOF 1\n", encoding="utf-8")
            for name in ("residuetypes.dat", "WAT.itp", "chk.mdp", "steep.mdp", "em.mdp", "init.gro", "COM.gro"):
                (root / name).write_text("placeholder\n", encoding="utf-8")
            config = _dummy_config(root, replica_dirs=["run"])
            config.paths.forcefield_dir = forcefield_dir
            workflow = GCMCWorkflow(config)
            run_dir = root / "run"
            run_dir.mkdir()

            workflow._copy_inputs(run_dir)

            self.assertTrue((run_dir / "topol.top").exists())
            self.assertTrue((run_dir / "amber03.ff" / "forcefield.itp").exists())
            self.assertTrue((run_dir / "acs.itp").exists())
            self.assertTrue((run_dir / "acs_posre.itp").exists())
            self.assertTrue((run_dir / "nested" / "codh.itp").exists())
            self.assertTrue((run_dir / "shared" / "cofactor.itp").exists())


if __name__ == "__main__":
    unittest.main()
