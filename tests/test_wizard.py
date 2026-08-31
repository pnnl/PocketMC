from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gcmc_port.assets import default_asset_path
from gcmc_port.cli import main
from gcmc_port.config import load_config
from gcmc_port.gro import Atom, GroStructure, write_gro
from gcmc_port.workflow import GCMCWorkflow
from gcmc_port.wizard import _load_residue_types, _prompt_anchor_residues, build_wizard_case


def _write_demo_case(case_dir: Path) -> tuple[Path, Path]:
    case_dir.mkdir(parents=True, exist_ok=True)
    gro_path = case_dir / "init.gro"
    top_path = case_dir / "topol.top"
    structure = GroStructure(
        title="demo",
        atoms=[
            Atom(1, "ALA", "N", 1, 0.10, 0.10, 0.10),
            Atom(1, "ALA", "CA", 2, 0.14, 0.10, 0.10),
            Atom(800, "ATC", "C2", 3, 0.50, 0.50, 0.50),
            Atom(800, "ATC", "C4", 4, 0.54, 0.50, 0.50),
            Atom(800, "ATC", "C7", 5, 0.50, 0.54, 0.50),
            Atom(800, "ATC", "O1", 6, 0.52, 0.52, 0.54),
            Atom(700, "OPA", "C1", 7, 0.80, 0.80, 0.80),
            Atom(700, "OPA", "C2", 8, 0.84, 0.80, 0.80),
            Atom(700, "OPA", "O1", 9, 0.80, 0.84, 0.80),
            Atom(20, "NA", "NA", 10, 1.10, 1.10, 1.10),
            Atom(21, "CL", "CL", 11, 1.20, 1.20, 1.20),
            Atom(30, "SOL", "OW", 12, 1.40, 1.40, 1.40),
            Atom(30, "SOL", "HW1", 13, 1.42, 1.40, 1.40),
            Atom(30, "SOL", "HW2", 14, 1.40, 1.42, 1.40),
        ],
        box_line="2.00000 2.00000 2.00000",
    )
    write_gro(gro_path, structure)
    top_path.write_text("[ system ]\ndemo\n", encoding="utf-8")
    return gro_path, top_path


class WizardTests(unittest.TestCase):
    def test_prompt_anchor_residues_prefers_non_protein_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gro_path, _ = _write_demo_case(Path(tmpdir))
            residue_types = _load_residue_types(default_asset_path("residuetypes.dat"))
            with patch("builtins.input", side_effect=["1"]):
                selected = _prompt_anchor_residues(gro_path, residue_types)

        self.assertEqual([residue.token for residue in selected], ["800ATC"])

    def test_build_wizard_case_writes_defaults_when_advanced_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            _write_demo_case(case_dir)
            original_cwd = Path.cwd()
            os.chdir(case_dir)
            try:
                answers = [""] * 15
                with patch("builtins.input", side_effect=answers):
                    artifacts = build_wizard_case()
            finally:
                os.chdir(original_cwd)

            config = load_config(artifacts.config_path)
            local_exists = artifacts.local_script_path.exists()
            generic_exists = artifacts.generic_sbatch_path.exists()
            tahoma_only_exists = artifacts.tahoma_only_sbatch_path.exists()

        self.assertEqual(config.anchor.residues, ["800ATC"])
        self.assertEqual(config.anchor.reference_mode, "atoms")
        self.assertEqual(config.anchor.center_atoms, ["C2", "C4", "C7"])
        self.assertAlmostEqual(config.simulation.rmax, 0.6)
        self.assertEqual(config.execution.gmx_cmd, "gmx_mpi")
        self.assertEqual(config.loop.replica_dirs, ["run"])
        self.assertEqual(config.simulation.target_nmol, 0)
        self.assertFalse(config.cavity_build.enabled)
        self.assertTrue(local_exists)
        self.assertTrue(generic_exists)
        self.assertTrue(tahoma_only_exists)
        self.assertEqual(artifacts.local_script_path.parent, case_dir)
        self.assertEqual(artifacts.generic_sbatch_path.parent, case_dir)
        self.assertEqual(artifacts.tahoma_only_sbatch_path.parent, case_dir)
        self.assertEqual(artifacts.local_script_path.name, "run_gcmc.sh")
        self.assertEqual(artifacts.generic_sbatch_path.name, "run_gcmc.sbatch")
        self.assertEqual(artifacts.tahoma_only_sbatch_path.name, "run_gcmc_tahoma_only.sbatch")

    def test_build_wizard_case_supports_manual_paths_and_multi_residue_com(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            case_dir = tmp_path / "case"
            gro_path, top_path = _write_demo_case(case_dir)
            original_cwd = Path.cwd()
            os.chdir(tmp_path)
            try:
                answers = [
                    str(gro_path),
                    str(top_path),
                    "",
                    "",
                    "1,2",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
                with patch("builtins.input", side_effect=answers):
                    artifacts = build_wizard_case()
            finally:
                os.chdir(original_cwd)

            config = load_config(artifacts.config_path)

        self.assertEqual(config.anchor.residues, ["800ATC", "700OPA"])
        self.assertEqual(config.anchor.reference_mode, "com")
        self.assertEqual(config.config_path.parent, case_dir)
        self.assertEqual(artifacts.generic_sbatch_path.parent, case_dir)

    def test_build_wizard_case_mask_advanced_values_and_threaded_gmx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            _write_demo_case(case_dir)
            original_cwd = Path.cwd()
            os.chdir(case_dir)
            try:
                answers = [
                    "",
                    "",
                    "",
                    "2",
                    "1",
                    "1",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "y",
                    "0.009",
                    "0.1",
                    "1.8",
                    "0.04",
                    "9000000",
                    "12",
                    "0.01",
                    "0.08",
                    "0.11",
                    "1.2",
                    "0.55",
                    "25",
                    "2",
                ]
                with patch("builtins.input", side_effect=answers):
                    artifacts = build_wizard_case()
            finally:
                os.chdir(original_cwd)

            config = load_config(artifacts.config_path)

        self.assertEqual(config.cavity.mode, "mask")
        self.assertTrue(config.cavity_build.enabled)
        self.assertAlmostEqual(config.simulation.rvdw, 0.1)
        self.assertAlmostEqual(config.cavity_build.dx, 0.08)
        self.assertAlmostEqual(config.cavity_build.probe_radius, 0.11)
        self.assertEqual(config.execution.gmx_cmd, "gmx")
        self.assertEqual(config.execution.launcher_single, "")
        self.assertEqual(config.execution.mdrun_multi_args, ["-ntmpi", "1", "-ntomp", "{cores}"])
        self.assertEqual(config.cavity.mask_file.name, "cavity_mask.dat")

    def test_main_rejects_write_config_without_interactive(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["--write-config", "custom.toml"])
        self.assertEqual(ctx.exception.code, 2)

    def test_prepare_runtime_inputs_auto_builds_mask_from_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            gro_path, top_path = _write_demo_case(case_dir)
            config_path = case_dir / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'project_root = "."',
                        'work_root = "."',
                        f'topology = "{top_path.name}"',
                        f'init_gro = "{gro_path.name}"',
                        "",
                        "[anchor]",
                        'residues = ["800ATC"]',
                        'reference_mode = "atoms"',
                        'center_atoms = ["C2", "C4", "C7"]',
                        "",
                        "[cavity]",
                        'mode = "mask"',
                        "",
                        "[cavity_build]",
                        "enabled = true",
                        'mode = "seeded"',
                        'output_prefix = "cavity"',
                        'exclude_residues = ["800ATC"]',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            workflow = GCMCWorkflow(load_config(config_path))

            def fake_build_cavity_from_structure(gro_path_arg, **kwargs):
                outprefix = Path(kwargs["outprefix"])
                outprefix.with_name(f"{outprefix.name}_mask.dat").write_text("0.1 0.1 0.1\n", encoding="utf-8")
                outprefix.with_suffix(".meta.json").write_text("{}", encoding="utf-8")
                return [
                    outprefix.with_name(f"{outprefix.name}_mask.dat"),
                    outprefix.with_suffix(".meta.json"),
                ]

            with patch("gcmc_port.workflow.build_cavity_from_structure", side_effect=fake_build_cavity_from_structure) as mocked:
                workflow._prepare_runtime_inputs()
                mask_exists = workflow.config.cavity.mask_file.exists()
                meta_exists = workflow.config.cavity.mask_meta.exists()

        self.assertEqual(mocked.call_count, 1)
        self.assertTrue(mask_exists)
        self.assertTrue(meta_exists)

    def test_build_wizard_case_can_select_co_insert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            _write_demo_case(case_dir)
            original_cwd = Path.cwd()
            os.chdir(case_dir)
            try:
                answers = [
                    "",
                    "",
                    "2",
                    "2",
                    "1",
                    "1",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
                with patch("builtins.input", side_effect=answers):
                    artifacts = build_wizard_case()
            finally:
                os.chdir(original_cwd)

            config = load_config(artifacts.config_path)

        self.assertEqual(config.paths.water_itp, default_asset_path("co/COM.itp"))
        self.assertEqual(config.paths.gas_gro, default_asset_path("co/COM.gro"))
        self.assertEqual(config.cavity.mode, "mask")
        self.assertTrue(config.cavity_build.enabled)
        self.assertAlmostEqual(config.simulation.mu0, -58.9323290)


if __name__ == "__main__":
    unittest.main()
