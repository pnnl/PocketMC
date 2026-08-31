from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gcmc_port.analysis.cli import build_parser
from gcmc_port.analysis.aggregate import write_aggregate
from gcmc_port.analysis.cache import cache_directory, write_analysis_cache
from gcmc_port.analysis.config import load_analysis_config, validate_analysis_config
from gcmc_port.analysis.density import build_density, legacy_gaussian_density
from gcmc_port.analysis.events import build_visits
from gcmc_port.analysis.grand_alignment import (
    GRAND_ALIGNMENT_SCHEMA_VERSION,
    _copy_grand_plot_script,
    _effective_grand_style,
    _nice_upper_and_step,
    _square_plane_limits,
    default_fixed_substrates,
    discover_completed_analysis_roots,
    discover_grand_alignment_outputs,
    grand_align_analysis_roots,
    repair_grand_alignment_output,
)
from gcmc_port.analysis.mc_reader import read_mc_dataset
from gcmc_port.analysis.md_reader import read_md_dataset
from gcmc_port.analysis.models import (
    AnalysisConfig,
    ANALYSIS_CACHE_VERSION,
    AnalysisOptions,
    CavitySpec,
    DatasetSpec,
    FrameRecord,
    MCMove,
    MoleculeFrame,
    MoleculeSpec,
    OutputOptions,
    RunResult,
    SubstrateSpec,
    config_for_dataset,
    expand_tasks,
)
from gcmc_port.analysis.plot_editor import (
    PlotTarget,
    discover_plot_targets,
    load_plot_style,
    render_plot_targets,
    save_plot_style,
    stale_analysis_runs,
    stale_md_analysis_runs,
    stale_pose_hydration_runs,
)
from gcmc_port.analysis.plotting import _plt, _style_with_saved_overrides
from gcmc_port.analysis.runner import _copy_plot_script, _fingerprint, run_analysis
from gcmc_port.analysis.tables import write_run_tables
from gcmc_port.analysis.vmd import _trimmed_trace_points, write_vmd_session
from gcmc_port.analysis.wizard import (
    ExistingAnalysis,
    _edit_completed_plots,
    _recompute_stale_analysis,
    _recompute_stale_md_analysis,
    run_wizard,
)
from gcmc_port.gro import Atom, GroStructure, parse_gro, write_gro
from gcmc_port.moves import write_trajectory


def molecule(uid: str, inside: bool, point: tuple[float, float, float] = (0.1, 0.0, 0.0)) -> MoleculeFrame:
    return MoleculeFrame(uid, 10, "WAT", point, inside, "1ALA", 0.25)


def dataset(root: Path, kind: str = "md", run_id: str = "run") -> DatasetSpec:
    trajectory = root / "trajectory.gro"
    topology = root / "topology.gro"
    return DatasetSpec(run_id, kind, root, topology if kind == "md" else None, trajectory)


def config(root: Path, kind: str, data: tuple[DatasetSpec, ...], tasks: tuple[str, ...]) -> AnalysisConfig:
    return AnalysisConfig(
        root / "analyses.toml",
        kind,
        data,
        MoleculeSpec.from_values(preset="water"),
        CavitySpec(mode="sphere", anchor="800ATC", radius_nm=0.6),
        AnalysisOptions(tasks=tasks, density_bin_a=1.0, density_sigma_a=1.0),
        OutputOptions(root / "results"),
    )


def _mask_test_atom(
    serial: int,
    name: str,
    resname: str,
    resid: int,
    point: tuple[float, float, float],
    element: str,
) -> str:
    record = "ATOM  " if resname == "ALA" else "HETATM"
    x, y, z = point
    return (
        f"{record}{serial:5d} {name:^4s} {resname:>3s} A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
    )


def _write_rotated_mask_reference_fixture(source: Path, reference: Path, trajectory: Path) -> None:
    atoms = [
        (1, "N", "ALA", 1, (0.0, 0.0, 0.0), "N"),
        (2, "CA", "ALA", 1, (1.5, 0.0, 0.0), "C"),
        (3, "C", "ALA", 1, (1.5, 1.5, 0.0), "C"),
        (4, "O", "ALA", 1, (0.0, 1.5, 0.0), "O"),
        (5, "C1", "LIG", 10, (4.0, 4.0, 4.0), "C"),
        (6, "C2", "LIG", 10, (5.0, 4.0, 4.0), "C"),
        (7, "C3", "LIG", 10, (4.0, 5.0, 4.0), "C"),
        (8, "OW", "WAT", 20, (4.5, 4.5, 5.0), "O"),
    ]

    def fitted(point: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = point
        return -y + 30.0, x + 20.0, z + 10.0

    crystal = "CRYST1   80.000   80.000   80.000  90.00  90.00  90.00 P 1           1"
    source.write_text(
        "\n".join(
            [
                crystal,
                *[
                    _mask_test_atom(serial, name, resname, resid, point, element)
                    for serial, name, resname, resid, point, element in atoms[:-1]
                ],
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    fitted_atoms = [
        _mask_test_atom(serial, name, resname, resid, fitted(point), element)
        for serial, name, resname, resid, point, element in atoms
    ]
    reference.write_text("\n".join([crystal, *fitted_atoms, "END"]) + "\n", encoding="utf-8")
    trajectory.write_text(
        "\n".join(
            [
                crystal,
                "MODEL        1",
                *fitted_atoms,
                "ENDMDL",
                "MODEL        2",
                *fitted_atoms,
                "ENDMDL",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_periodic_custom_alignment_fixture(reference: Path, trajectory: Path) -> tuple[float, float, float]:
    """Write a rotated frame whose reference fit atoms use another box image."""
    atoms = [
        (1, "N", "ALA", 1, np.asarray((49.0, 50.0, 50.0)), "N"),
        (2, "CA", "ALA", 1, np.asarray((51.0, 50.0, 50.0)), "C"),
        (3, "C", "ALA", 1, np.asarray((50.0, 52.0, 50.0)), "C"),
        (4, "O", "ALA", 1, np.asarray((50.0, 50.0, 52.0)), "O"),
        (5, "C1", "LIG", 10, np.asarray((50.0, 51.0, 50.0)), "C"),
        (6, "C2", "LIG", 10, np.asarray((50.8, 51.0, 50.0)), "C"),
        (7, "C3", "LIG", 10, np.asarray((50.0, 51.8, 50.0)), "C"),
        (8, "OW", "WAT", 20, np.asarray((50.2, 51.1, 50.3)), "O"),
    ]
    target_center = np.mean([item[4] for item in atoms[:4]], axis=0)
    mobile_center = np.asarray((30.0, 30.0, 30.0))
    angle = np.deg2rad(45.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    def mobile(point: np.ndarray) -> tuple[float, float, float]:
        value = (point - target_center) @ rotation.T + mobile_center
        return tuple(float(item) for item in value)

    crystal = "CRYST1  100.000  100.000  100.000  90.00  90.00  90.00 P 1           1"
    reference_rows = []
    mobile_rows = []
    for serial, name, resname, resid, point, element in atoms:
        stored = point + (np.asarray((100.0, 0.0, 0.0)) if resname == "ALA" else 0.0)
        reference_rows.append(
            _mask_test_atom(serial, name, resname, resid, tuple(float(item) for item in stored), element)
        )
        mobile_rows.append(_mask_test_atom(serial, name, resname, resid, mobile(point), element))
    reference.write_text("\n".join([crystal, *reference_rows, "END"]) + "\n", encoding="utf-8")
    trajectory.write_text(
        "\n".join(
            [
                crystal,
                "MODEL        1",
                *mobile_rows,
                "ENDMDL",
                "MODEL        2",
                *mobile_rows,
                "ENDMDL",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    water = atoms[-1][4] / 10.0
    return tuple(float(item) for item in water)


class AnalysisModelTests(unittest.TestCase):
    def test_task_dependencies_and_mc_scientific_boundary(self) -> None:
        self.assertEqual(expand_tasks("md", ["paths"]), ("lifetime", "paths"))
        self.assertEqual(expand_tasks("pocketmc", ["density"]), ("density", "mc-states"))
        mixed = expand_tasks("mixed", ["plots"])
        self.assertIn("lifetime", mixed)
        self.assertIn("mc-states", mixed)
        with self.assertRaisesRegex(ValueError, "do not support physical lifetime"):
            expand_tasks("pocketmc", ["lifetime"])

    def test_molecule_presets_and_custom_validation(self) -> None:
        self.assertEqual(MoleculeSpec.from_values(preset="co").point_mode, "cog")
        self.assertIn("OH2", MoleculeSpec.from_values(preset="water").atom_names)
        with self.assertRaisesRegex(ValueError, "resnames"):
            MoleculeSpec.from_values(preset="custom", resnames=[])

    def test_cli_exposes_batch_overrides(self) -> None:
        args = build_parser().parse_args(
            ["run", "-c", "a.toml", "--tasks", "density,vmd", "--runs", "r1", "--force", "--jobs", "3"]
        )
        self.assertEqual(args.jobs, 3)
        self.assertTrue(args.force)
        self.assertEqual(args.tasks, ["density,vmd"])


class EventTests(unittest.TestCase):
    def test_gap_healing_and_boundary_censoring(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            data = dataset(Path(raw))
            states = [True, False, True, False, False, True, True]
            frames = [FrameRecord(i, i * 500.0, (molecule("w1", value),), int(value)) for i, value in enumerate(states)]
            result = RunResult(data, frames)
            visits = build_visits(result, gap_ps=500.0)
            self.assertEqual(len(visits), 2)
            self.assertTrue(visits[0].left_censored)
            self.assertFalse(visits[0].right_censored)
            self.assertEqual(visits[1].event_type, "reentry")
            self.assertTrue(visits[1].right_censored)


class DensityAndOutputTests(unittest.TestCase):
    def test_saved_plot_backend_is_headless_even_in_an_interactive_terminal(self) -> None:
        import matplotlib

        with patch.object(matplotlib, "get_backend", return_value="QtAgg"), patch.object(matplotlib, "use") as use:
            _plt()
        use.assert_called_once_with("Agg", force=True)

    def test_xyz2cube_legacy_gaussian_numeric_regression(self) -> None:
        axes = (np.asarray([-0.5, 0.5]),) * 3
        rho = legacy_gaussian_density(np.asarray([[0.0, 0.0, 0.0]]), axes, sigma_a=1.0, cutoff_sigma=1.0)
        self.assertTrue(np.allclose(rho, np.exp(-0.375)))

    def test_cache_fingerprint_changes_with_input_or_settings(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = dataset(root)
            data.topology.write_text("topology", encoding="utf-8")
            data.trajectory.write_text("frame-one", encoding="utf-8")
            cfg = config(root, "md", (data,), ("lifetime",))
            first = _fingerprint(cfg, data)
            data.trajectory.write_text("frame-one-and-two", encoding="utf-8")
            second = _fingerprint(cfg, data)
            changed_settings = _fingerprint(replace(cfg, analysis=replace(cfg.analysis, stride=2)), data)
            self.assertNotEqual(first, second)
            self.assertNotEqual(second, changed_settings)

    def test_generic_md_mask_uses_source_frame_alignment_for_occupancy_density_and_cache(self) -> None:
        try:
            import MDAnalysis  # noqa: F401
        except Exception as exc:
            self.skipTest(f"MDAnalysis unavailable: {exc}")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "mask_source.pdb"
            reference = root / "reference.pdb"
            trajectory = root / "trajectory.pdb"
            _write_rotated_mask_reference_fixture(source, reference, trajectory)
            mask = root / "cavity_mask.dat"
            mask.write_text("0.450000 0.450000 0.500000\n", encoding="utf-8")
            meta = root / "cavity.meta.json"
            meta.write_text(
                json.dumps(
                    {
                        "dx": 0.1,
                        "reference_point": [0.45, 0.45, 0.5],
                        "effective_volume": 0.001,
                        "source_gro": str(source),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            substrate_selection = "resid 10 and resname LIG"
            data = DatasetSpec(
                run_id="mask-fit",
                kind="md",
                run_dir=root,
                topology=reference,
                trajectory=trajectory,
                reference=reference,
                substrate_selection=substrate_selection,
            )
            cfg = AnalysisConfig(
                root / "analyses.toml",
                "md",
                (data,),
                MoleculeSpec.from_values(preset="water"),
                CavitySpec(
                    mode="mask",
                    mask=mask,
                    meta=meta,
                    anchor="10LIG",
                    protein_selection="protein",
                ),
                AnalysisOptions(
                    tasks=("density", "lifetime"),
                    density_bin_a=1.0,
                    density_sigma_a=1.0,
                    density_cutoff_sigma=2.0,
                ),
                OutputOptions(root / "results"),
                substrate=SubstrateSpec(enabled=True, selection=substrate_selection),
            )

            result = read_md_dataset(cfg, data)

            self.assertEqual([frame.occupancy for frame in result.frames], [1, 1])
            self.assertEqual(result.metadata["mask_alignment"]["method"], "mask source-structure to analysis-reference Kabsch fit")
            self.assertEqual(Path(result.metadata["mask_alignment"]["source"]), source)
            build_density(cfg, result, root / "density")
            with (
                np.load(root / "density" / "density_maps.npz") as density,
                np.load(root / "density" / "substrate_overlay.npz") as overlay,
            ):
                rho = np.asarray(density["rho"], dtype=float)
                axes = [np.asarray(density[f"{name}_A"], dtype=float) for name in "xyz"]
                grids = np.meshgrid(*axes, indexing="ij")
                density_center = np.asarray([(rho * grid).sum() / rho.sum() for grid in grids])
                substrate_center = np.asarray(overlay["positions_A"], dtype=float).mean(axis=0)
            self.assertGreater(float(rho.sum()), 0.0)
            self.assertLess(float(np.linalg.norm(density_center - substrate_center)), 3.0)

            first_fingerprint = _fingerprint(cfg, data)
            source_stat = source.stat()
            os.utime(
                source,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000_000),
            )
            self.assertNotEqual(_fingerprint(cfg, data), first_fingerprint)

    def test_mask_custom_alignment_coimages_whole_fit_before_rotated_kabsch(self) -> None:
        try:
            import MDAnalysis  # noqa: F401
        except Exception as exc:
            self.skipTest(f"MDAnalysis unavailable: {exc}")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = root / "reference.pdb"
            trajectory = root / "trajectory.pdb"
            expected_water_nm = _write_periodic_custom_alignment_fixture(reference, trajectory)
            mask = root / "cavity_mask.dat"
            mask.write_text(" ".join(f"{value:.6f}" for value in expected_water_nm) + "\n", encoding="utf-8")
            meta = root / "cavity.meta.json"
            meta.write_text(
                json.dumps(
                    {
                        "dx": 0.1,
                        "reference_point": list(expected_water_nm),
                        "effective_volume": 0.001,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            substrate_selection = "resid 10 and resname LIG"
            data = DatasetSpec(
                run_id="periodic-custom-fit",
                kind="md",
                run_dir=root,
                topology=reference,
                trajectory=trajectory,
                reference=reference,
                substrate_selection=substrate_selection,
            )
            cfg = AnalysisConfig(
                root / "analyses.toml",
                "md",
                (data,),
                MoleculeSpec.from_values(preset="water"),
                CavitySpec(
                    mode="mask",
                    mask=mask,
                    meta=meta,
                    anchor="10LIG",
                    protein_selection="protein",
                    align_selection="protein and backbone",
                ),
                AnalysisOptions(tasks=("lifetime",)),
                OutputOptions(root / "results"),
                substrate=SubstrateSpec(enabled=True, selection=substrate_selection),
            )

            result = read_md_dataset(cfg, data)

            self.assertEqual([frame.occupancy for frame in result.frames], [1, 1])
            self.assertIn("custom selection", result.metadata["alignment"]["description"])
            for frame in result.frames:
                self.assertTrue(
                    np.allclose(frame.molecules[0].point_nm, expected_water_nm, atol=2.0e-4),
                    msg=f"aligned water was {frame.molecules[0].point_nm}",
                )

    def test_density_integrals_and_projection_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = dataset(root)
            frames = [
                FrameRecord(0, 0.0, (molecule("w1", True, (0.0, 0.0, 0.0)),), 1),
                FrameRecord(1, 1.0, (molecule("w1", True, (0.1, 0.0, 0.0)),), 1),
            ]
            result = RunResult(data, frames)
            cfg = config(root, "md", (data,), ("density", "lifetime"))
            build_density(cfg, result, root / "density")
            with np.load(root / "density" / "density_maps.npz") as payload:
                probability = payload["rho_probability"]
                occupancy = payload["rho_occupancy"]
                xy = payload["xy_projection"]
            self.assertAlmostEqual(float(probability.sum()), 1.0, places=8)
            self.assertAlmostEqual(float(occupancy.sum()), 1.0, places=8)
            self.assertTrue(np.allclose(xy, occupancy.sum(axis=2).T))
            tcl = write_vmd_session(cfg, result, root)[0].read_text(encoding="utf-8")
            self.assertIn("# HPD 90%", tcl)
            self.assertIn("# HPD 30%", tcl)

    def test_empty_inside_selection_does_not_fall_back_to_all_waters(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = dataset(root)
            frames = [
                FrameRecord(0, 0.0, (molecule("w1", False, (5.0, 5.0, 5.0)),), 0),
                FrameRecord(1, 1.0, (molecule("w1", False, (6.0, 6.0, 6.0)),), 0),
            ]
            result = RunResult(data, frames, metadata={"cavity_center_nm": (0.0, 0.0, 0.0)})
            cfg = config(root, "md", (data,), ("density", "lifetime"))

            build_density(cfg, result, root / "density")

            with np.load(root / "density" / "density_maps.npz") as payload:
                self.assertEqual(float(np.asarray(payload["rho_probability"]).sum()), 0.0)
                self.assertEqual(float(np.asarray(payload["rho_occupancy"]).sum()), 0.0)
                self.assertLess(float(np.asarray(payload["x_A"]).max()), 50.0)
            with np.load(root / "density" / "cavity_overlay.npz") as cavity:
                self.assertEqual(str(cavity["mode"]), "sphere")
                np.testing.assert_allclose(cavity["center_A"], np.zeros(3))
                self.assertAlmostEqual(float(cavity["radius_A"]), 6.0)
            metadata = json.loads((root / "density" / "density_maps.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["point_count"], 0)
            self.assertTrue(metadata["empty_inside_selection"])

    def test_empty_repair_clears_old_event_path_overlay_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            table_dir = root / "tables"
            plot_dir = root / "plots"
            density_dir = root / "density"
            plot_dir.mkdir()
            density_dir.mkdir()
            for name in ("lifetime_distribution.png", "residence_timeline.png", "path_transition_matrix.png"):
                (plot_dir / name).write_bytes(b"old")
            (table_dir / "events.tsv").parent.mkdir(parents=True, exist_ok=True)
            (table_dir / "events.tsv").write_text("molecule_uid\tlifetime_ps\nold\t10\n", encoding="utf-8")
            (table_dir / "paths.tsv").write_text("molecule_uid\tsample_index\tlabel\nold\t0\tBulk\n", encoding="utf-8")
            (density_dir / "substrate_overlay.npz").write_bytes(b"old")
            data = dataset(root)
            result = RunResult(
                data,
                [FrameRecord(0, 0.0, (), 0)],
                metadata={"cavity_center_nm": (0.0, 0.0, 0.0)},
            )
            cfg = config(root, "md", (data,), ("density", "lifetime", "paths", "plots"))
            write_run_tables(result, table_dir)
            build_density(cfg, result, density_dir)
            from gcmc_port.analysis.plotting import render_result_plots

            render_result_plots(root, load_plot_style(root))
            self.assertEqual((table_dir / "events.tsv").read_text(encoding="utf-8").count("\n"), 1)
            self.assertEqual((table_dir / "paths.tsv").read_text(encoding="utf-8").count("\n"), 1)
            self.assertFalse((density_dir / "substrate_overlay.npz").exists())
            for name in ("lifetime_distribution.png", "residence_timeline.png", "path_transition_matrix.png"):
                self.assertFalse((plot_dir / name).exists())

    def test_mc_tables_and_editable_plot_script(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = dataset(root, "pocketmc")
            result = RunResult(
                data,
                [FrameRecord(1, 1.0, (molecule("w1", True),), 1, -10.0, 3, "I")],
                mc_moves=[MCMove("run", 3, 0, "I", True, -10.0, -2.0, 0)],
            )
            outputs = write_run_tables(result, root / "tables")
            names = {path.name for path in outputs}
            self.assertIn("mc_states.tsv", names)
            self.assertIn("mc_acceptance_by_move.tsv", names)
            generated = _copy_plot_script(root, reset=False)
            generated.write_text(generated.read_text(encoding="utf-8") + "# USER MARKER\n", encoding="utf-8")
            _copy_plot_script(root, reset=False)
            self.assertIn("USER MARKER", generated.read_text(encoding="utf-8"))
            _copy_plot_script(root, reset=True)
            self.assertNotIn("USER MARKER", generated.read_text(encoding="utf-8"))
            refreshed = generated.read_text(encoding="utf-8")
            self.assertIn('"density_3d_opacity"', refreshed)
            self.assertIn('"substrate_atom_brightness_3d"', refreshed)
            self.assertIn('"substrate_atom_color_3d"', refreshed)
            self.assertIn('"substrate_draw_on_top_3d"', refreshed)

    def test_plot_only_density_regeneration_uses_saved_npz(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "results"
            run = root / "case-one"
            (run / "tables").mkdir(parents=True)
            (run / "tables" / "summary.json").write_text('{"kind": "md"}\n', encoding="utf-8")
            density = run / "density"
            density.mkdir()
            axes = np.asarray([-1.0, 0.0, 1.0])
            rho = np.zeros((3, 3, 3), dtype=float)
            rho[1, 1, 1] = 1.0
            np.savez_compressed(
                density / "density_maps.npz",
                rho=rho,
                x_A=axes,
                y_A=axes,
                z_A=axes,
                xy_projection=rho.sum(axis=2).T,
                xz_projection=rho.sum(axis=1).T,
                yz_projection=rho.sum(axis=0).T,
            )
            np.savez_compressed(
                density / "substrate_overlay.npz",
                positions_A=np.asarray([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.8, 0.0]]),
                elements=np.asarray(["C", "O", "P"]),
                resnames=np.asarray(["ATC", "ATC", "OPP"]),
                resids=np.asarray([1, 1, 2]),
            )
            targets = discover_plot_targets(root)
            selected = [target for target in targets if target.key in {"case-one:density-2d", "case-one:density-3d"}]
            self.assertEqual(len(selected), 2)
            style = load_plot_style(root)
            style.update(
                {
                    "dpi": 40,
                    "density_3d_isosurface_levels_percent": [20.0],
                    "density_3d_opacity": 0.14,
                    "density_3d_max_faces": 5000,
                    "substrate_atom_color_3d": "#00E5FF",
                    "substrate_atom_brightness_3d": 1.25,
                    "substrate_atom_opacity_3d": 0.95,
                    "substrate_atom_depthshade_3d": False,
                    "substrate_draw_on_top_3d": True,
                }
            )
            save_plot_style(root, style)
            with (
                patch("gcmc_port.analysis.plotting._draw_substrate_2d") as draw_substrate_2d,
                patch("gcmc_port.analysis.plotting._density_isosurfaces") as density_isosurfaces,
            ):
                outputs = render_plot_targets(root, style, selected)
            self.assertEqual(draw_substrate_2d.call_count, 3)
            for call in draw_substrate_2d.call_args_list:
                overlay = call.args[1]
                self.assertIsNotNone(overlay)
                np.testing.assert_allclose(overlay["positions_A"], [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.8, 0.0]])
            density_isosurfaces.assert_called_once()
            overlay_3d = density_isosurfaces.call_args.args[6]
            self.assertIsNotNone(overlay_3d)
            np.testing.assert_array_equal(overlay_3d["resnames"], ["ATC", "ATC", "OPP"])
            self.assertEqual(len(outputs), 4)
            self.assertTrue((run / "plots" / "density_xy.png").exists())
            self.assertTrue((run / "plots" / "density_3d.png").exists())

    def test_generated_python_style_can_override_interactive_json(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "plot_style.json").write_text('{"density_3d_opacity": 0.6}\n', encoding="utf-8")
            interactive = _style_with_saved_overrides(root, {"density_3d_opacity": 0.2})
            direct_python = _style_with_saved_overrides(
                root,
                {"density_3d_opacity": 0.2, "_ignore_saved_plot_style": True},
            )
            self.assertEqual(interactive["density_3d_opacity"], 0.6)
            self.assertEqual(direct_python["density_3d_opacity"], 0.2)
            self.assertNotIn("_ignore_saved_plot_style", direct_python)

    def test_old_pose_density_is_flagged_as_requiring_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "results"
            pose_root = root / "case-one" / "poses"
            density = pose_root / "cluster_01" / "pocket-frame" / "density_maps.npz"
            density.parent.mkdir(parents=True)
            density.write_bytes(b"saved-grid-placeholder")
            (pose_root / "pose_manifest.json").write_text(
                '{"schema_version": 1, "status": "complete"}\n',
                encoding="utf-8",
            )
            self.assertEqual(stale_pose_hydration_runs(root), ["case-one"])

    def test_interactive_replot_repairs_stale_pose_density_then_continues(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "results"
            output.mkdir()
            config_path = root / "analyses.toml"
            config_path.write_text("[output]\nroot = \"results\"\n", encoding="utf-8")
            manifest = output / "analysis_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            plot = output / "case-one" / "poses" / "cluster_01" / "plots" / "pocket-frame_xy.png"
            target = PlotTarget(
                "case-one:pose-density-2d",
                "case-one: cluster 2D hydration heatmaps",
                "1 plot",
                (plot,),
                output / "case-one",
            )
            existing = ExistingAnalysis(
                manifest=manifest,
                config_path=config_path,
                output_root=output,
                failures=(),
                modified_ns=manifest.stat().st_mtime_ns,
                status="complete",
            )
            with (
                patch("gcmc_port.analysis.wizard.discover_plot_targets", return_value=[target]),
                patch("gcmc_port.analysis.wizard._select_numbers", side_effect=[[1], [9]]),
                patch("gcmc_port.analysis.wizard.stale_pose_hydration_runs", return_value=["case-one"]),
                patch("gcmc_port.analysis.wizard._yes_no", return_value=True),
                patch("gcmc_port.analysis.wizard._recompute_stale_pose_hydration") as repair,
                patch("gcmc_port.analysis.wizard.load_plot_style", return_value={}),
                patch("gcmc_port.analysis.wizard.save_plot_style", return_value=output / "plot_style.json"),
                patch("gcmc_port.analysis.wizard.render_plot_targets", return_value=[plot]) as render,
            ):
                _edit_completed_plots(existing)
            repair.assert_called_once_with(existing)
            render.assert_called_once()

    def test_minimal_md_repair_handles_overwrite_false_and_updates_manifest_version(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            config_path = root / "analyses.toml"
            config_path.write_text(
                """
[input]
kind = "md"
run_id = "case-one"
topology = "topology.gro"
trajectory = "trajectory.xtc"

[molecule]
preset = "water"

[cavity]
mode = "sphere"
anchor = "800ATC"

[analysis]
tasks = ["lifetime"]

[output]
root = "results"
overwrite = false
cache = true
""".strip()
                + "\n",
                encoding="utf-8",
            )
            output_root = root / "results"
            run_root = output_root / "case-one"
            (run_root / "tables").mkdir(parents=True)
            (run_root / "tables" / "summary.json").write_text('{"kind": "md"}\n', encoding="utf-8")
            (run_root / "analysis_manifest.json").write_text(
                json.dumps({"kind": "md", "analysis_cache_version": 0, "fingerprint": "old"}) + "\n",
                encoding="utf-8",
            )
            root_manifest = output_root / "analysis_manifest.json"
            root_manifest.write_text(
                json.dumps({"config": str(config_path), "analysis_cache_version": 0}) + "\n",
                encoding="utf-8",
            )
            existing = ExistingAnalysis(
                manifest=root_manifest,
                config_path=config_path,
                output_root=output_root,
                failures=(),
                modified_ns=root_manifest.stat().st_mtime_ns,
                status="complete",
            )
            loaded = load_analysis_config(config_path)
            synthetic = RunResult(
                loaded.datasets[0],
                [FrameRecord(0, 0.0, (molecule("w1", True),), 1)],
            )
            with patch("gcmc_port.analysis.runner._load_or_analyze", return_value=synthetic):
                _recompute_stale_md_analysis(existing)
            case_manifest = json.loads((run_root / "analysis_manifest.json").read_text(encoding="utf-8"))
            repaired_root = json.loads(root_manifest.read_text(encoding="utf-8"))
            self.assertEqual(case_manifest["analysis_cache_version"], ANALYSIS_CACHE_VERSION)
            self.assertEqual(repaired_root["analysis_cache_version"], ANALYSIS_CACHE_VERSION)

    def test_minimal_case_repair_recomputes_stale_pocketmc_without_pose_work(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = dataset(root, kind="pocketmc", run_id="mc")
            cfg = config(root, "pocketmc", (data,), ("density",))
            output_root = cfg.output.root
            output_root.mkdir(parents=True)
            root_manifest = output_root / "analysis_manifest.json"
            root_manifest.write_text("{}\n", encoding="utf-8")
            existing = ExistingAnalysis(
                manifest=root_manifest,
                config_path=cfg.config_path,
                output_root=output_root,
                failures=(),
                modified_ns=root_manifest.stat().st_mtime_ns,
                status="complete",
            )
            repaired = RunResult(data, [FrameRecord(1, 1.0, (), 0)])
            with (
                patch("gcmc_port.analysis.wizard.load_analysis_config", return_value=cfg),
                patch("gcmc_port.analysis.wizard.stale_analysis_runs", side_effect=[[data.run_id], []]),
                patch("gcmc_port.analysis.runner.run_dataset", return_value=repaired) as run,
                patch("gcmc_port.analysis.runner._copy_plot_script", return_value=output_root / "plot_results.py"),
                patch("gcmc_port.analysis.aggregate.write_aggregate") as aggregate,
                patch("builtins.print") as printed,
            ):
                _recompute_stale_analysis(existing)
            run.assert_called_once()
            self.assertTrue(run.call_args.args[0].output.overwrite)
            self.assertEqual(run.call_args.args[1], data)
            self.assertEqual(run.call_args.kwargs["force"], False)
            self.assertIsNone(run.call_args.kwargs["pose_stage"])
            aggregate.assert_called_once()
            output = " ".join(str(call.args[0]) for call in printed.call_args_list if call.args)
            self.assertIn("PocketMC", output)
            self.assertNotIn("lifetime", output.lower())

    def test_minimal_md_repair_reuses_only_current_generic_caches_for_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            stale_data = dataset(root, run_id="stale")
            cached_data = dataset(root, run_id="cached")
            cfg = config(root, "md", (stale_data, cached_data), ("lifetime",))
            output_root = cfg.output.root
            output_root.mkdir(parents=True)
            root_manifest = output_root / "analysis_manifest.json"
            root_manifest.write_text("{}\n", encoding="utf-8")
            existing = ExistingAnalysis(
                manifest=root_manifest,
                config_path=cfg.config_path,
                output_root=output_root,
                failures=(),
                modified_ns=root_manifest.stat().st_mtime_ns,
                status="complete",
            )
            repaired = RunResult(stale_data, [FrameRecord(0, 0.0, (), 0)])
            cached = RunResult(cached_data, [FrameRecord(0, 0.0, (), 0)])
            cache_run_dir = output_root / cached_data.run_id
            cache_run_dir.mkdir(parents=True)
            current_fingerprint = _fingerprint(config_for_dataset(cfg, cached_data), cached_data)
            variants = (
                (
                    "old schema",
                    {"version": ANALYSIS_CACHE_VERSION - 1, "fingerprint": current_fingerprint, "result": cached},
                    False,
                ),
                (
                    "stale fingerprint",
                    {"version": ANALYSIS_CACHE_VERSION, "fingerprint": "stale", "result": cached},
                    False,
                ),
                (
                    "current cache",
                    {"version": ANALYSIS_CACHE_VERSION, "fingerprint": current_fingerprint, "result": cached},
                    True,
                ),
            )
            for label, payload, should_aggregate in variants:
                with self.subTest(label=label):
                    write_analysis_cache(
                        cache_run_dir,
                        payload["result"],
                        analysis_version=payload["version"],
                        fingerprint=payload["fingerprint"],
                    )
                    with (
                        patch("gcmc_port.analysis.wizard.load_analysis_config", return_value=cfg),
                        patch(
                            "gcmc_port.analysis.wizard.stale_analysis_runs",
                            side_effect=[[stale_data.run_id], []],
                        ),
                        patch("gcmc_port.analysis.runner.run_dataset", return_value=repaired),
                        patch("gcmc_port.analysis.runner._copy_plot_script", return_value=output_root / "plot_results.py"),
                        patch("gcmc_port.analysis.aggregate.write_aggregate") as aggregate,
                    ):
                        _recompute_stale_md_analysis(existing)
                    if should_aggregate:
                        aggregate.assert_called_once()
                        aggregated = aggregate.call_args.args[0]
                        self.assertEqual({item.dataset.run_id for item in aggregated}, {"stale", "cached"})
                    else:
                        aggregate.assert_not_called()

    def test_replot_staleness_checks_current_input_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "topology.gro").write_text("topology", encoding="utf-8")
            trajectory = root / "trajectory.xtc"
            trajectory.write_text("frame one", encoding="utf-8")
            config_path = root / "analyses.toml"
            config_path.write_text(
                """
[input]
kind = "md"
run_id = "case-one"
topology = "topology.gro"
trajectory = "trajectory.xtc"
[molecule]
preset = "water"
[cavity]
mode = "sphere"
anchor = "800ATC"
[analysis]
tasks = ["lifetime"]
[output]
root = "results"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            cfg = load_analysis_config(config_path)
            output = cfg.output.root
            run = output / "case-one"
            (run / "tables").mkdir(parents=True)
            (run / "tables" / "summary.json").write_text('{"kind": "md"}\n', encoding="utf-8")
            fingerprint = _fingerprint(cfg, cfg.datasets[0])
            (run / "analysis_manifest.json").write_text(
                json.dumps(
                    {
                        "kind": "md",
                        "analysis_cache_version": ANALYSIS_CACHE_VERSION,
                        "fingerprint": fingerprint,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output.mkdir(exist_ok=True)
            (output / "analysis_manifest.json").write_text(
                json.dumps({"config": str(config_path)}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(stale_md_analysis_runs(output), [])
            trajectory.write_text("frame one and two", encoding="utf-8")
            self.assertEqual(stale_md_analysis_runs(output), ["case-one"])

    def test_replot_staleness_includes_pocketmc_and_blocks_plot_only_render(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            run = output / "mc"
            (run / "tables").mkdir(parents=True)
            (run / "tables" / "summary.json").write_text('{"kind": "pocketmc"}\n', encoding="utf-8")
            (run / "analysis_manifest.json").write_text(
                json.dumps({"analysis_cache_version": ANALYSIS_CACHE_VERSION - 1, "fingerprint": "old"}) + "\n",
                encoding="utf-8",
            )
            (output / "analysis_manifest.json").write_text("{}\n", encoding="utf-8")
            target = PlotTarget(
                "mc:density-2d",
                "mc: 2D density heatmaps",
                "3 plots",
                (run / "plots" / "density_xy.png",),
                run,
            )
            self.assertEqual(stale_analysis_runs(output), ["mc"])
            self.assertEqual(stale_md_analysis_runs(output), [])
            with self.assertRaisesRegex(RuntimeError, "mc"):
                render_plot_targets(output, load_plot_style(output), [target])


class PocketMCTrajectoryTests(unittest.TestCase):
    def _write_structure(self, path: Path) -> None:
        write_gro(
            path,
            GroStructure(
                "state",
                [
                    Atom(800, "ATC", "C2", 1, 0.0, 0.0, 0.0),
                    Atom(800, "ATC", "C4", 2, 0.0, 0.1, 0.0),
                    Atom(800, "ATC", "C7", 3, 0.0, 0.0, 0.1),
                    Atom(1, "ALA", "CA", 4, 0.4, 0.0, 0.0),
                    Atom(2, "SOL", "OW", 5, 2.0, 2.0, 2.0),
                    Atom(10, "WAT", "OW", 6, 0.1, 0.0, 0.0),
                    Atom(10, "WAT", "HW1", 7, 0.11, 0.0, 0.0),
                    Atom(10, "WAT", "HW2", 8, 0.1, 0.01, 0.0),
                ],
                "3.0 3.0 3.0",
            ),
        )

    def test_generic_padding_and_sidecar_identity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state.gro"
            gas = root / "gas.gro"
            trajectory = root / "trajectory.gro"
            sidecar = root / "trajectory.meta.jsonl"
            self._write_structure(state)
            structure = parse_gro(state)
            structure.atoms.extend(
                [Atom(20, "LIG", "C1", 9, 0.1, 0.0, 0.0), Atom(20, "LIG", "O1", 10, 0.2, 0.0, 0.0)]
            )
            write_gro(state, structure)
            write_gro(
                gas,
                GroStructure("gas", [Atom(1, "LIG", "C1", 1, 0, 0, 0), Atom(1, "LIG", "O1", 2, 0.1, 0, 0)], "1 1 1"),
            )
            write_trajectory(
                state,
                -12.5,
                1,
                1,
                3,
                trajectory_path=trajectory,
                gas_gro=gas,
                trial=7,
                move="I",
                active_resids=[20],
                provenance={20: "insert:t007"},
                trajectory_meta_path=sidecar,
            )
            parsed = parse_gro(trajectory)
            self.assertEqual(len(parsed.atoms), 14)  # 10 real + 2 atoms x 2 dummy molecules
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(record["template_atom_count"], 2)
            self.assertEqual(record["dummy_atom_count"], 4)
            self.assertEqual(record["molecules"][0]["uid"], "insert:t007")

    def test_mc_mask_trajectory_is_matched_per_accepted_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trajectory = root / "trajectory.gro"
            mask_trajectory = root / "cavity_trajectory.gro"

            def gro_frame(title: str, atom: Atom) -> list[str]:
                return [title, "    1", atom.line(), "   4.00000   4.00000   4.00000"]

            trajectory.write_text(
                "\n".join(
                    gro_frame("1 -1.0", Atom(10, "WAT", "OW", 1, 1.0, 0.0, 0.0))
                    + gro_frame("2 -2.0", Atom(10, "WAT", "OW", 1, 2.0, 0.0, 0.0))
                )
                + "\n",
                encoding="utf-8",
            )
            mask_trajectory.write_text(
                "\n".join(
                    gro_frame("0 start", Atom(1, "CAV", "HE", 1, 0.0, 0.0, 0.0))
                    + gro_frame("1 trial=1", Atom(1, "CAV", "HE", 1, 1.0, 0.0, 0.0))
                    + gro_frame("2 trial=2", Atom(1, "CAV", "HE", 1, 2.0, 0.0, 0.0))
                )
                + "\n",
                encoding="utf-8",
            )
            mask = root / "cavity_mask.dat"
            mask.write_text("0.0 0.0 0.0\n", encoding="utf-8")
            meta = root / "cavity.meta.json"
            meta.write_text('{"dx": 0.1, "reference_point": [0.0, 0.0, 0.0]}\n', encoding="utf-8")
            data = DatasetSpec("mc", "pocketmc", root, None, trajectory)
            cfg = config(root, "pocketmc", (data,), ("mc-states",))
            cfg = replace(
                cfg,
                cavity=CavitySpec(
                    mode="mask",
                    mask=mask,
                    meta=meta,
                    mask_trajectory=mask_trajectory,
                ),
            )
            result = read_mc_dataset(cfg, data)
            self.assertEqual([frame.occupancy for frame in result.frames], [1, 1])
            self.assertEqual(result.metadata["coordinate_frame"], "pocketmc-first-accepted-mask")
            np.testing.assert_allclose(
                [frame.molecules[0].point_nm for frame in result.frames],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                atol=1.0e-6,
            )
            density_dir = root / "density"
            build_density(cfg, result, density_dir)
            with np.load(density_dir / "density_maps.npz") as density:
                occupancy = np.asarray(density["rho_occupancy"], dtype=float)
                x_axis = np.asarray(density["x_A"], dtype=float)
                x_profile = occupancy.sum(axis=(1, 2))
                density_center_x = float(np.sum(x_axis * x_profile) / np.sum(x_profile))
            self.assertAlmostEqual(density_center_x, 10.0, delta=0.6)
            vmd_outputs = write_vmd_session(cfg, result, root)
            session = next(path for path in vmd_outputs if path.name == "session.vmd.tcl")
            session_text = session.read_text(encoding="utf-8")
            self.assertEqual(session_text.count("sphere {10.0000 0.0000 0.0000}"), 2)
            self.assertIn("set script_dir [file dirname [file normalize [info script]]]", session_text)
            self.assertNotIn(root.resolve().as_posix(), session_text)

    def test_new_sidecar_excludes_bulk_and_dummy_molecules(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            state = root / "state.gro"
            trajectory = root / "trajectory.gro"
            sidecar = root / "trajectory.meta.jsonl"
            gas = root / "gas.gro"
            self._write_structure(state)
            write_gro(gas, GroStructure("water", [Atom(1, "WAT", "OW", 1, 0, 0, 0)], "1 1 1"))
            write_trajectory(
                state,
                -5.0,
                1,
                1,
                2,
                trajectory_path=trajectory,
                gas_gro=gas,
                trial=4,
                move="I",
                active_resids=[10],
                provenance={10: "insert:t004"},
                trajectory_meta_path=sidecar,
            )
            data = DatasetSpec("mc", "pocketmc", root, None, trajectory, trajectory_meta=sidecar)
            cfg = config(root, "pocketmc", (data,), ("mc-states",))
            result = read_mc_dataset(cfg, data)
            self.assertEqual(result.frames[0].occupancy, 1)
            self.assertEqual([item.uid for item in result.frames[0].molecules], ["insert:t004"])
            self.assertNotIn("legacy_identity_inferred", " ".join(result.warnings))
            full_cfg = replace(cfg, analysis=replace(cfg.analysis, tasks=("density", "mc-states", "plots", "vmd")))
            completed, failures = run_analysis(full_cfg, force=True)
            self.assertEqual(len(completed), 1)
            self.assertFalse(failures)
            run_root = full_cfg.output.root / "mc"
            self.assertTrue((run_root / "tables" / "mc_states.tsv").exists())
            self.assertTrue((run_root / "plots" / "occupancy_temporal.png").exists())
            self.assertTrue((run_root / "density" / "density.cube").exists())
            self.assertTrue((run_root / "vmd" / "session.vmd.tcl").exists())

    def test_legacy_water_padding_is_removed_and_identity_warning_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trajectory = root / "trajectory.gro"
            self._write_structure(trajectory)
            structure = parse_gro(trajectory)
            structure.title = "1 -5.000000"
            structure.atoms.extend(
                [
                    Atom(1, "WAT", "OW", 1, 2.0, 2.0, 2.0),
                    Atom(1, "WAT", "HW1", 2, 2.0, 2.0, 2.0),
                    Atom(1, "WAT", "HW2", 3, 2.0, 2.0, 2.0),
                ]
            )
            write_gro(trajectory, structure)
            data = DatasetSpec("legacy", "pocketmc", root, None, trajectory)
            cfg = config(root, "pocketmc", (data,), ("mc-states",))
            result = read_mc_dataset(cfg, data)
            self.assertEqual(result.frames[0].occupancy, 1)
            self.assertEqual(len(result.frames[0].molecules), 1)
            self.assertIn("legacy_identity_inferred", " ".join(result.warnings))

    def test_legacy_reused_residue_id_gets_a_new_generation_uid(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            occupied = root / "occupied.gro"
            empty = root / "empty.gro"
            trajectory = root / "trajectory.gro"
            self._write_structure(occupied)
            empty_structure = parse_gro(occupied)
            empty_structure.atoms = [atom for atom in empty_structure.atoms if atom.resname != "WAT"]
            write_gro(empty, empty_structure)
            for state, source, count in ((1, occupied, 1), (2, empty, 0), (3, occupied, 1)):
                write_trajectory(source, -float(state), state, count, 0, trajectory_path=trajectory)
            data = DatasetSpec("legacy", "pocketmc", root, None, trajectory)
            cfg = config(root, "pocketmc", (data,), ("mc-states",))
            result = read_mc_dataset(cfg, data)
            self.assertEqual(result.frames[0].molecules[0].uid, "10WAT@0")
            self.assertFalse(result.frames[1].molecules)
            self.assertEqual(result.frames[2].molecules[0].uid, "10WAT@1")

    def test_mc_vmd_never_draws_transport_lines(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = dataset(root, "pocketmc")
            data.trajectory.write_text("placeholder", encoding="utf-8")
            result = RunResult(data, [FrameRecord(1, 1.0, (molecule("w1", True),), 1)])
            cfg = config(root, "pocketmc", (data,), ("mc-states", "vmd"))
            tcl = write_vmd_session(cfg, result, root / "result")[0].read_text(encoding="utf-8")
            self.assertNotIn("graphics $base_mol line", tcl)
            self.assertIn("not physical paths", tcl)

    def test_md_vmd_trace_categories_keep_entry_exit_resident_colors(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = dataset(root, "md")
            entry0, entry1 = molecule("entry", False), molecule("entry", True)
            exit0, exit1 = molecule("exit", True), molecule("exit", False)
            resident0, resident1 = molecule("resident", True), molecule("resident", True)
            result = RunResult(
                data,
                [
                    FrameRecord(0, 0.0, (entry0, exit0, resident0), 2),
                    FrameRecord(1, 1.0, (entry1, exit1, resident1), 2),
                ],
            )
            cfg = config(root, "md", (data,), ("lifetime", "vmd"))
            tcl = write_vmd_session(cfg, result, root / "result")[0].read_text(encoding="utf-8")
            self.assertIn("# trace entry category=entry", tcl)
            self.assertIn("# trace exit category=exit", tcl)
            self.assertIn("# trace resident category=resident", tcl)
            self.assertIn("set ::pocketmc_trace_colors($trace_label) red", tcl)
            self.assertIn("set ::pocketmc_trace_colors($trace_label) blue", tcl)
            self.assertIn("set ::pocketmc_trace_colors($trace_label) gray", tcl)

    def test_md_vmd_prefers_coordinate_reference_over_tpr_topology(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            topology = root / "md.tpr"
            trajectory = root / "md.xtc"
            reference = root / "previous.gro"
            for path in (topology, trajectory, reference):
                path.write_text("placeholder", encoding="utf-8")
            data = DatasetSpec("md-one", "md", root, topology, trajectory, reference=reference)
            result = RunResult(data, [FrameRecord(0, 0.0, (molecule("w1", True),), 1)])
            cfg = config(root, "md", (data,), ("vmd",))

            tcl = write_vmd_session(cfg, result, root / "result")[0].read_text(encoding="utf-8")

            self.assertIn("set structure_path [file normalize [file join $script_dir {../../previous.gro}]]", tcl)
            self.assertNotIn(reference.as_posix(), tcl)
            self.assertNotIn(topology.as_posix(), tcl)
            self.assertIn("proc pocketmc_load_session", tcl)
            self.assertIn("PocketMC VMD session could not be loaded", tcl)
            self.assertIn("without PBC molecule reconstruction", tcl)
            self.assertIn("Frame 0 and the static density, trace, cavity, and substrate overlays", tcl)

    def test_vmd_trace_trimming_prefers_surface_window_and_preserves_graphics_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = dataset(root, "md")
            rows = []
            for index in range(6):
                item = MoleculeFrame(
                    "water", 10, "WAT", (0.1 * index, 0.0, 0.0), index == 5,
                    "1ALA", 0.5 if index == 1 else 2.0,
                )
                rows.append(FrameRecord(index, float(index * 1000), (item,), int(item.inside)))
            result = RunResult(data, rows)
            trimmed = _trimmed_trace_points(result, "water", 1000.0)
            self.assertEqual([round(item.point_nm[0], 2) for item in trimmed], [0.0, 0.1, 0.2])
            cfg = config(root, "md", (data,), ("lifetime", "vmd"))
            tcl = write_vmd_session(cfg, result, root / "result")[0].read_text(encoding="utf-8")
            self.assertIn("proc trace_show", tcl)
            self.assertIn("proc trace_hide", tcl)
            self.assertNotIn("graphics $::base_mol delete all", tcl)
            self.assertIn("catch {array unset ::pocketmc_trace_commands}", tcl)
            self.assertIn("graphics $::base_mol material Opaque", tcl)

    def test_vmd_uses_analysis_image_substrate_and_single_point_trace(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = dataset(root, "md")
            tracked = MoleculeFrame("water", 10, "WAT", (1.0, 2.0, 3.0), True, "1ALA", 0.2)
            result = RunResult(
                data,
                [FrameRecord(0, 0.0, (tracked,), 1)],
                metadata={
                    "substrate_overlay": {
                        "positions_A": [[10.0, 20.0, 30.0], [11.4, 20.0, 30.0]],
                        "atom_names": ["C1", "O1"],
                        "elements": ["C", "O"],
                        "resnames": ["LIG", "LIG"],
                        "resids": [10, 10],
                        "atom_indices_0based": [4, 5],
                    }
                },
            )
            cfg = config(root, "md", (data,), ("lifetime", "vmd"))
            outputs = write_vmd_session(cfg, result, root / "result")
            tcl = outputs[0].read_text(encoding="utf-8")
            substrate_path = root / "result" / "vmd" / "substrate_overlay.pdb"
            self.assertIn(substrate_path, outputs)
            self.assertIn("10.000  20.000  30.000", substrate_path.read_text(encoding="utf-8"))
            self.assertIn("mol new $substrate_path", tcl)
            self.assertIn("[list sphere {10.0000 20.0000 30.0000}", tcl)


@unittest.skipUnless(importlib.util.find_spec("MDAnalysis"), "MDAnalysis is an analysis dependency")
class SyntheticMDIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write_pdb(root: Path) -> Path:
        pdb = root / "trajectory.pdb"
        lines: list[str] = []
        for model, water_x in ((1, 3.0), (2, 10.0)):
            lines.append(f"MODEL     {model:4d}")
            atoms = [
                (1, "CA", "ALA", 1, 0.0, 0.0, 0.0, "C"),
                (2, "C", "ALA", 1, 1.0, 0.0, 0.0, "C"),
                (3, "O", "ALA", 1, 0.0, 1.0, 0.0, "O"),
                (4, "C2", "ATC", 800, 0.0, 0.0, 0.0, "C"),
                (5, "C4", "ATC", 800, 0.0, 0.0, 1.0, "C"),
                (6, "C7", "ATC", 800, 0.0, 1.0, 0.0, "C"),
                (7, "O", "HOH", 10, water_x, 0.0, 0.0, "O"),
            ]
            for serial, name, resname, resid, x, y, z, element in atoms:
                record = "ATOM  " if resname == "ALA" else "HETATM"
                lines.append(
                    f"{record}{serial:5d} {name:^4s} {resname:>3s} A{resid:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
                )
            lines.append("ENDMDL")
        lines.append("END")
        pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return pdb

    def test_two_frame_pdb_occupancy(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pdb = self._write_pdb(root)
            data = DatasetSpec("md", "md", root, pdb, pdb)
            cfg = config(root, "md", (data,), ("lifetime",))
            result = read_md_dataset(cfg, data)
            self.assertEqual([frame.occupancy for frame in result.frames], [1, 0])

    def test_full_md_pipeline_generates_plots_density_vmd_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pdb = self._write_pdb(root)
            data = DatasetSpec("md", "md", root, pdb, pdb)
            cfg = config(root, "md", (data,), ("density", "lifetime", "paths", "plots", "vmd"))
            results, failures = run_analysis(cfg, force=True)
            self.assertFalse(failures)
            self.assertEqual(len(results), 1)
            run_root = cfg.output.root / "md"
            self.assertTrue((cfg.output.root / "plot_results.py").exists())
            self.assertTrue((run_root / "plots" / "occupancy_temporal.png").exists())
            self.assertTrue((run_root / "density" / "density.cube").exists())
            self.assertTrue((run_root / "vmd" / "session.vmd.tcl").exists())
            manifest = json.loads((run_root / "analysis_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")

    def test_canonical_and_homolog_residue_labels_are_retained(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            pdb = self._write_pdb(root)
            mapping = root / "mapping.pdb"
            mapping.write_text(
                "".join(
                    [
                        f"ATOM  {1:5d} {'CA':^4s} {'ALA':>3s} A{101:4d}    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n",
                        f"ATOM  {2:5d} {'C':^4s} {'ALA':>3s} A{101:4d}    {1.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n",
                        "END\n",
                    ]
                ),
                encoding="utf-8",
            )
            data = DatasetSpec("md", "md", root, pdb, pdb)
            cfg = config(root, "md", (data,), ("lifetime",))
            cfg = replace(
                cfg,
                analysis=replace(cfg.analysis, canonical_source=mapping, homolog_source=mapping, canonical_chain="A", homolog_chain="A"),
            )
            result = read_md_dataset(cfg, data)
            tracked = result.frames[0].molecules[0]
            self.assertEqual(tracked.nearest_residue, "101ALA")
            self.assertEqual(tracked.nearest_residue_sim, "1ALA")
            self.assertEqual(tracked.nearest_residue_homolog, "101ALA")


class GrandAlignmentTests(unittest.TestCase):
    @staticmethod
    def _write_saved_analysis(
        root: Path,
        overlay_positions: np.ndarray,
        density_center: np.ndarray,
        axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> Path:
        density_dir = root / "replica-01" / "density"
        density_dir.mkdir(parents=True)
        grids = np.meshgrid(*axes, indexing="ij")
        rho = np.exp(
            -sum((grid - density_center[index]) ** 2 for index, grid in enumerate(grids)) / 1.2
        )
        bin_a = float(np.median(np.diff(axes[0])))
        np.savez_compressed(
            density_dir / "density_maps.npz",
            rho=rho,
            rho_probability=rho / (rho.sum() * bin_a ** 3),
            x_A=axes[0], y_A=axes[1], z_A=axes[2], bin_A=np.asarray(bin_a),
            xy_projection=rho.sum(axis=2).T * bin_a,
            xz_projection=rho.sum(axis=1).T * bin_a,
            yz_projection=rho.sum(axis=0).T * bin_a,
        )
        np.savez_compressed(
            density_dir / "substrate_overlay.npz",
            positions_A=overlay_positions,
            atom_names=np.asarray(["C1", "C2", "C3", "O1"]),
            elements=np.asarray(["C", "C", "C", "O"]),
            resnames=np.asarray(["OPP", "OPP", "OPP", "OPP"]),
            resids=np.asarray([700, 700, 700, 700]),
            coordinate_frame=np.asarray("analysis-reference"),
        )
        (root / "analysis_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "completed_runs": ["replica-01"],
                    "failures": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return density_dir / "density_maps.npz"

    @staticmethod
    def _write_cavity_cache(
        root: Path,
        metadata: dict[str, object],
        cavity: dict[str, object],
    ) -> None:
        run_root = root / "replica-01"
        cached_dataset = DatasetSpec(
            run_id="replica-01",
            kind="md",
            run_dir=run_root,
            topology=None,
            trajectory=run_root / "trajectory.xtc",
        )
        write_analysis_cache(
            run_root,
            RunResult(cached_dataset, [], metadata=metadata),
            analysis_version=ANALYSIS_CACHE_VERSION,
            fingerprint="grand-alignment-test",
        )
        (run_root / "analysis_manifest.json").write_text(
            json.dumps({"status": "complete", "settings": {"cavity": cavity}}) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _replace_cache_with_vmd_mask(root: Path, points_a: np.ndarray) -> None:
        run_root = root / "replica-01"
        shutil.rmtree(cache_directory(run_root))
        vmd = run_root / "vmd"
        vmd.mkdir()
        lines = ["REMARK canonical cavity mask"]
        for serial, (x, y, z) in enumerate(points_a, start=1):
            lines.append(
                f"HETATM{serial:5d} HE   CAV M{serial:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          He"
            )
        lines.append("END")
        (vmd / "cavity_mask_points.pdb").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_nice_color_scale_and_square_panel_limits(self) -> None:
        self.assertEqual(_nice_upper_and_step(0.37), (0.4, 0.05))
        self.assertEqual(_nice_upper_and_step(0.08), (0.08, 0.01))
        self.assertEqual(_nice_upper_and_step(0.12), (0.12, 0.02))
        self.assertEqual(_square_plane_limits([0.0, 4.0, 1.0, 3.0]), [0.0, 4.0, 0.0, 4.0])

    def test_mask_cavity_boundary_is_saved_in_grand_aligned_frame(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target_overlay = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
            target_mask = np.asarray(
                [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
                dtype=float,
            )
            axes = tuple(np.arange(-5.0, 8.0, 1.0) for _ in range(3))
            first_root = root / "1" / "analysis-results"
            self._write_saved_analysis(first_root, target_overlay, np.asarray([0.5, 0.5, 0.5]), axes)
            self._write_cavity_cache(
                first_root,
                {"cavity_mask_points_nm": (target_mask / 10.0).tolist()},
                {"mode": "mask"},
            )
            self._replace_cache_with_vmd_mask(first_root, target_mask)

            rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            shift = np.asarray([5.0, -2.0, 3.0])
            second_root = root / "2" / "analysis-results"
            mobile_overlay = target_overlay @ rotation + shift
            mobile_mask = target_mask @ rotation + shift
            self._write_saved_analysis(
                second_root,
                mobile_overlay,
                np.asarray([0.5, 0.5, 0.5]) @ rotation + shift,
                axes,
            )
            self._write_cavity_cache(
                second_root,
                {"cavity_mask_points_nm": (mobile_mask / 10.0).tolist()},
                {"mode": "mask"},
            )
            self._replace_cache_with_vmd_mask(second_root, mobile_mask)

            result = grand_align_analysis_roots(
                discover_completed_analysis_roots(root),
                root / "grand-aligned",
                fixed_substrates=("OPP",),
                render_plots=True,
            )

            self.assertEqual(len(result.aligned_maps), 2)
            self.assertEqual(len(result.plots), 8)
            self.assertTrue(all(path.is_file() for path in result.plots))
            for aligned_map in result.aligned_maps:
                with np.load(aligned_map.parent / "cavity_overlay.npz") as cavity:
                    self.assertEqual(str(cavity["mode"]), "mask")
                    np.testing.assert_allclose(cavity["points_A"], target_mask, atol=1.0e-6)
                    self.assertEqual(str(cavity["coordinate_frame"]), "grand-aligned")

    def test_sphere_cavity_is_transformed_and_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target_overlay = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
            target_center = np.asarray([0.4, 0.3, 0.8])
            rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            transforms = (
                (np.eye(3), np.zeros(3)),
                (rotation, np.asarray([4.0, -2.0, 2.0])),
            )
            axes = tuple(np.arange(-5.0, 8.0, 1.0) for _ in range(3))
            for serial, (mobile_rotation, shift) in enumerate(transforms, start=1):
                analysis_root = root / str(serial) / "analysis-results"
                mobile_overlay = target_overlay @ mobile_rotation + shift
                mobile_center = target_center @ mobile_rotation + shift
                self._write_saved_analysis(analysis_root, mobile_overlay, mobile_center, axes)
                self._write_cavity_cache(
                    analysis_root,
                    {"cavity_center_nm": (mobile_center / 10.0).tolist()},
                    {"mode": "sphere", "radius_nm": 0.6},
                )
                run_root = analysis_root / "replica-01"
                shutil.rmtree(cache_directory(run_root))
                vmd = run_root / "vmd"
                vmd.mkdir()
                (vmd / "session.vmd.tcl").write_text(
                    "graphics $base_mol sphere {"
                    + " ".join(f"{value:.4f}" for value in mobile_center)
                    + "} radius 6.0000 resolution 30\n",
                    encoding="utf-8",
                )

            result = grand_align_analysis_roots(
                discover_completed_analysis_roots(root),
                root / "grand-aligned",
                fixed_substrates=("OPP",),
                render_plots=True,
            )

            self.assertEqual(len(result.plots), 8)
            for aligned_map in result.aligned_maps:
                with np.load(aligned_map.parent / "cavity_overlay.npz") as cavity:
                    self.assertEqual(str(cavity["mode"]), "sphere")
                    np.testing.assert_allclose(cavity["center_A"], target_center, atol=1.0e-6)
                    self.assertAlmostEqual(float(cavity["radius_A"]), 6.0)

    def test_aggregate_only_analysis_uses_cached_substrate_frame_for_cavity(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target_overlay = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
            target_mask = np.asarray(
                [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
                dtype=float,
            )
            rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            transforms = ((np.eye(3), np.zeros(3)), (rotation, np.asarray([5.0, -2.0, 3.0])))
            axes = tuple(np.arange(-6.0, 9.0, 1.0) for _ in range(3))
            grids = np.meshgrid(*axes, indexing="ij")
            for serial, (mobile_rotation, shift) in enumerate(transforms, start=1):
                analysis_root = root / str(serial) / "analysis-results"
                cluster = analysis_root / "aggregate" / "pose-groups" / "group" / "cluster_01"
                cluster.mkdir(parents=True)
                mobile_overlay = target_overlay @ mobile_rotation + shift
                mobile_mask = target_mask @ mobile_rotation + shift
                density_center = np.asarray([0.5, 0.5, 0.5]) @ mobile_rotation + shift
                rho = np.exp(-sum((grid - density_center[index]) ** 2 for index, grid in enumerate(grids)))
                np.savez_compressed(
                    cluster / "CO2.mean_density.npz",
                    rho=rho,
                    x_A=axes[0], y_A=axes[1], z_A=axes[2], bin_A=np.asarray(1.0),
                )
                overlay_payload = {
                    "positions_A": mobile_overlay,
                    "atom_names": np.asarray(["C1", "C2", "C3", "O1"]),
                    "elements": np.asarray(["C", "C", "C", "O"]),
                    "resnames": np.asarray(["OPP", "OPP", "OPP", "OPP"]),
                    "resids": np.asarray([700, 700, 700, 700]),
                }
                np.savez_compressed(cluster / "substrate_overlay.npz", **overlay_payload)
                run_root = analysis_root / "replica-01"
                run_root.mkdir()
                cached_dataset = DatasetSpec(
                    run_id="replica-01",
                    kind="md",
                    run_dir=run_root,
                    topology=None,
                    trajectory=run_root / "trajectory.xtc",
                )
                write_analysis_cache(
                    run_root,
                    RunResult(
                        cached_dataset,
                        [],
                        metadata={
                            "cavity_mask_points_nm": (mobile_mask / 10.0).tolist(),
                            "substrate_overlay": {
                                key: value.tolist() for key, value in overlay_payload.items()
                            },
                        },
                    ),
                    analysis_version=ANALYSIS_CACHE_VERSION,
                    fingerprint="grand-alignment-test",
                )
                (run_root / "analysis_manifest.json").write_text(
                    json.dumps({"status": "complete", "settings": {"cavity": {"mode": "mask"}}})
                    + "\n",
                    encoding="utf-8",
                )
                (analysis_root / "analysis_manifest.json").write_text(
                    json.dumps({"status": "complete", "completed_runs": ["replica-01"]}) + "\n",
                    encoding="utf-8",
                )

            result = grand_align_analysis_roots(
                discover_completed_analysis_roots(root),
                root / "grand-aligned",
                fixed_substrates=("OPP",),
                render_plots=True,
            )

            self.assertEqual(len(result.aligned_maps), 2)
            self.assertEqual(len(result.plots), 4)
            for aligned_map in result.aligned_maps:
                with np.load(aligned_map.parent / "cavity_overlay.npz") as cavity:
                    np.testing.assert_allclose(cavity["points_A"], target_mask, atol=1.0e-6)

    def test_nested_discovery_and_saved_npz_grand_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target_overlay = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
            target_center = np.asarray([0.5, 0.5, 1.0])
            target_axes = tuple(np.arange(-5.0, 6.0, 1.0) for _ in range(3))
            first_source = self._write_saved_analysis(
                root / "1" / "analysis-results", target_overlay, target_center, target_axes
            )

            rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            shift = np.asarray([7.0, -4.0, 2.0])
            mobile_overlay = target_overlay @ rotation + shift
            mobile_center = target_center @ rotation + shift
            mobile_axes = (
                np.arange(2.0, 13.0, 1.0),
                np.arange(-9.0, 2.0, 1.0),
                np.arange(-3.0, 8.0, 1.0),
            )
            second_source = self._write_saved_analysis(
                root / "2" / "analysis-results", mobile_overlay, mobile_center, mobile_axes
            )
            source_bytes = {first_source: first_source.read_bytes(), second_source: second_source.read_bytes()}

            analyses = discover_completed_analysis_roots(root)
            self.assertEqual(len(analyses), 2)
            self.assertEqual(default_fixed_substrates(analyses), ("OPP",))
            result = grand_align_analysis_roots(
                analyses,
                root / "grand-aligned",
                fixed_substrates=("OPP",),
                reference_root=analyses[0],
                spacing_a=1.0,
                render_plots=True,
            )

            self.assertEqual(len(result.aligned_maps), 2)
            self.assertTrue(result.manifest.exists())
            self.assertFalse(result.skipped_maps)
            self.assertEqual(len(result.plots), 8)
            self.assertTrue(all(path.exists() for path in result.plots))
            self.assertTrue((result.output_root / "plot_grand_aligned.py").exists())
            generated_script = (result.output_root / "plot_grand_aligned.py").read_text(encoding="utf-8")
            self.assertNotIn(str(SRC.resolve()), generated_script)
            self.assertIn("_GENERATED_PACKAGE_SOURCE = None", generated_script)
            self.assertIn("__GCMC_PORT_GENERATED_SOURCE_BEGIN__", generated_script)
            self.assertEqual(source_bytes, {path: path.read_bytes() for path in source_bytes})
            grand_manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                grand_manifest["schema_version"], GRAND_ALIGNMENT_SCHEMA_VERSION
            )
            generic_limits = grand_manifest["display_bounds_A"]["generic"]
            self.assertLess(generic_limits[1] - generic_limits[0], 6.0)
            self.assertLess(generic_limits[3] - generic_limits[2], 6.0)
            self.assertLess(generic_limits[5] - generic_limits[4], 6.0)
            aligned_centers = []
            aligned_overlays = []
            saved_axes = []
            for path in result.aligned_maps:
                with np.load(path) as data:
                    rho = np.asarray(data["rho"], dtype=float)
                    axes = [np.asarray(data[f"{name}_A"], dtype=float) for name in "xyz"]
                    grids = np.meshgrid(*axes, indexing="ij")
                    aligned_centers.append(
                        np.asarray([(rho * grid).sum() / rho.sum() for grid in grids])
                    )
                    saved_axes.append(axes)
                with np.load(path.parent / "substrate_overlay.npz") as overlay:
                    aligned_overlays.append(np.asarray(overlay["positions_A"], dtype=float))
            for first_axis, second_axis in zip(saved_axes[0], saved_axes[1]):
                np.testing.assert_allclose(first_axis, second_axis)
            np.testing.assert_allclose(aligned_overlays[0], aligned_overlays[1], atol=1.0e-6)
            np.testing.assert_allclose(aligned_centers[0], aligned_centers[1], atol=0.25)
            np.testing.assert_allclose(aligned_centers[0], target_center, atol=0.25)

            script_path = result.output_root / "plot_grand_aligned.py"
            spec = importlib.util.spec_from_file_location("test_grand_plot_script", script_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.render(result.output_root)

    def test_existing_grand_plot_script_gets_portable_import_without_losing_style(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            script = root / "plot_grand_aligned.py"
            script.write_text(
                "\n".join(
                    (
                        "from pathlib import Path",
                        "import sys",
                        'STYLE = {\"density_figure_size\": (14.0, 10.0)}',
                        "# USER CUSTOMIZATION",
                        "# Source-checkout fallback; installed users resolve the package normally.",
                        "from gcmc_port.analysis.grand_alignment import replot_grand_alignment",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            _copy_grand_plot_script(root, reset=False)

            upgraded = script.read_text(encoding="utf-8")
            self.assertNotIn(str(SRC.resolve()), upgraded)
            self.assertIn("_GENERATED_PACKAGE_SOURCE = None", upgraded)
            self.assertIn("# USER CUSTOMIZATION", upgraded)
            self.assertIn("(14.0, 10.0)", upgraded)

    def test_old_editable_grand_script_receives_new_cavity_controls(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            script = root / "plot_grand_aligned.py"
            script.write_text(
                "\n".join(
                    (
                        "from pathlib import Path",
                        "import sys",
                        "STYLE = {",
                        '    "density_cmap": "magma",',
                        '    "grand_shared_color_scale": True,',
                        "}",
                        "# Source-checkout fallback; installed users resolve the package normally.",
                        "from gcmc_port.analysis.grand_alignment import replot_grand_alignment",
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            _copy_grand_plot_script(root, reset=False)

            upgraded = script.read_text(encoding="utf-8")
            self.assertIn('"density_cmap": "magma"', upgraded)
            self.assertIn("'cavity_boundary_color': '#00E5FF'", upgraded)
            self.assertIn("'cavity_boundary_alpha_3d': 0.68", upgraded)
            compile(upgraded, str(script), "exec")

    def test_auto_content_zooms_each_map_instead_of_using_group_union(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            axes = tuple(np.arange(-5.0, 36.0, 1.0) for _ in range(3))
            first = self._write_saved_analysis(
                root / "first",
                np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                ),
                np.asarray([0.5, 0.5, 0.5]),
                axes,
            )
            second = self._write_saved_analysis(
                root / "second",
                np.asarray(
                    [[30.0, 30.0, 30.0], [31.0, 30.0, 30.0], [30.0, 31.0, 30.0], [30.0, 30.0, 31.0]]
                ),
                np.asarray([30.5, 30.5, 30.5]),
                axes,
            )
            effective, shared = _effective_grand_style(
                [(first, "generic"), (second, "generic")],
                {
                    "grand_axis_mode": "auto-content",
                    "grand_axis_scope": "per-map",
                    "grand_axis_padding_fraction": 0.0,
                    "grand_axis_padding_A": 1.0,
                    "density_3d_isosurface_levels_percent": [8.0, 25.0, 50.0],
                    "substrate_overlay": True,
                    "substrate_show_hydrogens": False,
                },
            )

            by_map = effective["_grand_axis_limits_by_map"]
            first_limits = by_map[str(first.resolve())]
            second_limits = by_map[str(second.resolve())]
            self.assertGreater(shared["generic"][1] - shared["generic"][0], 25.0)
            self.assertLess(first_limits[1] - first_limits[0], 8.0)
            self.assertLess(second_limits[1] - second_limits[0], 8.0)
            self.assertLess(first_limits[1], second_limits[0])

    def test_cli_exposes_noninteractive_grand_alignment(self) -> None:
        args = build_parser().parse_args(
            ["grand-align", "first", "second", "--substrate", "OPP,ATC", "--no-plots"]
        )
        self.assertEqual(args.command, "grand-align")
        self.assertEqual(args.substrate, ["OPP,ATC"])
        self.assertTrue(args.no_plots)
        replot_args = build_parser().parse_args(["grand-replot", "grand-aligned", "--repair"])
        self.assertEqual(replot_args.command, "grand-replot")
        self.assertTrue(replot_args.repair)

    def test_discovery_wizard_offers_and_runs_grand_alignment_locally(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            overlay = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
            axes = tuple(np.arange(-2.0, 3.0, 1.0) for _ in range(3))
            self._write_saved_analysis(root / "1" / "analysis-results", overlay, np.zeros(3), axes)
            self._write_saved_analysis(root / "2" / "analysis-results", overlay, np.zeros(3), axes)
            fake = SimpleNamespace(
                aligned_maps=(root / "grand-aligned" / "map-1.npz",),
                plots=(),
                skipped_maps=(),
                warnings=(),
                output_root=root / "grand-aligned",
                manifest=root / "grand-aligned" / "grand_alignment_manifest.json",
            )
            original_cwd = Path.cwd()
            os.chdir(root)
            try:
                with patch("builtins.input", side_effect=[""] * 10), patch(
                    "gcmc_port.analysis.wizard.grand_align_analysis_roots", return_value=fake
                ) as align:
                    result = run_wizard(discover_first=True)
            finally:
                os.chdir(original_cwd)
            self.assertIsNone(result)
            self.assertEqual(align.call_args.kwargs["fixed_substrates"], ["OPP"])
            self.assertEqual(align.call_args.args[1], root / "grand-aligned")

    def test_discovery_wizard_auto_detects_old_grand_output_and_offers_repair(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            grand = root / "grand-aligned"
            grand.mkdir()
            manifest = grand / "grand_alignment_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "aligned_maps": [{"output": str(grand / "map.npz"), "map_kind": "generic"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            original_cwd = Path.cwd()
            os.chdir(root)
            try:
                with patch("builtins.input", side_effect=[""]), patch(
                    "gcmc_port.analysis.wizard.repair_grand_alignment_output", return_value=[]
                ) as repair:
                    result = run_wizard(discover_first=True)
            finally:
                os.chdir(original_cwd)
            self.assertIsNone(result)
            repair.assert_called_once_with(grand.resolve())

    def test_aggregate_mean_and_difference_maps_are_aligned_and_rendered(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            axes = tuple(np.arange(-2.0, 3.0, 1.0) for _ in range(3))
            grids = np.meshgrid(*axes, indexing="ij")
            mean = np.exp(-sum(grid * grid for grid in grids))
            difference = (grids[0] - grids[1]) * mean
            overlay_positions = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
            for serial in (1, 2):
                analysis_root = root / str(serial) / "analysis-results"
                cluster = analysis_root / "aggregate" / "pose-groups" / "group" / "cluster_01"
                cluster.mkdir(parents=True)
                np.savez_compressed(
                    cluster / "system.mean_density.npz",
                    rho=mean,
                    rho_ci95_low=mean * 0.8,
                    rho_ci95_high=mean * 1.2,
                    x_A=axes[0], y_A=axes[1], z_A=axes[2], bin_A=np.asarray(1.0),
                )
                np.savez_compressed(
                    cluster / "difference.B-minus-A.npz",
                    rho_difference=difference,
                    rho_difference_ci95_low=difference - 0.1,
                    rho_difference_ci95_high=difference + 0.1,
                    x_A=axes[0], y_A=axes[1], z_A=axes[2], bin_A=np.asarray(1.0),
                )
                np.savez_compressed(
                    cluster / "substrate_overlay.npz",
                    positions_A=overlay_positions,
                    atom_names=np.asarray(["C1", "C2", "C3", "O1"]),
                    elements=np.asarray(["C", "C", "C", "O"]),
                    resnames=np.asarray(["OPP", "OPP", "OPP", "OPP"]),
                    resids=np.zeros(4, dtype=int),
                    coordinate_frame=np.asarray("common-cluster-representative"),
                )
                (analysis_root / "analysis_manifest.json").write_text(
                    json.dumps({"status": "complete", "completed_runs": ["replica-01"], "failures": []}) + "\n",
                    encoding="utf-8",
                )
            analyses = discover_completed_analysis_roots(root)
            result = grand_align_analysis_roots(
                analyses,
                root / "grand-aligned",
                fixed_substrates=("OPP",),
                render_plots=True,
            )
            self.assertEqual(len(result.aligned_maps), 4)
            self.assertEqual(len(result.plots), 6)
            self.assertTrue(all(path.exists() for path in result.plots))
            self.assertFalse(result.skipped_maps)
            manifest_payload = json.loads(result.manifest.read_text(encoding="utf-8"))
            self.assertIn("aggregate-mean", manifest_payload["display_bounds_A"])
            self.assertIn("difference", manifest_payload["display_bounds_A"])

    def test_old_grand_output_is_detected_and_plot_only_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            overlay = np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )
            axes = tuple(np.arange(-3.0, 4.0, 1.0) for _ in range(3))
            sources = []
            for serial in (1, 2):
                sources.append(
                    self._write_saved_analysis(
                        root / str(serial) / "analysis-results", overlay, np.zeros(3), axes
                    )
                )
            analyses = discover_completed_analysis_roots(root)
            result = grand_align_analysis_roots(
                analyses,
                root / "grand-aligned",
                fixed_substrates=("OPP",),
                render_plots=False,
            )
            aligned_bytes = {path: path.read_bytes() for path in result.aligned_maps}
            payload = json.loads(result.manifest.read_text(encoding="utf-8"))
            payload["schema_version"] = 1
            payload.pop("display_bounds_A", None)
            result.manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            (result.output_root / "plot_grand_aligned.py").unlink()

            discovered = discover_grand_alignment_outputs(root)
            self.assertEqual(len(discovered), 1)
            self.assertTrue(discovered[0].stale)
            plots = repair_grand_alignment_output(discovered[0].root)
            self.assertTrue(plots)
            self.assertTrue((result.output_root / "plot_grand_aligned.py").exists())
            repaired = json.loads(result.manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                repaired["schema_version"], GRAND_ALIGNMENT_SCHEMA_VERSION
            )
            self.assertIn("display_bounds_A", repaired)
            self.assertEqual(aligned_bytes, {path: path.read_bytes() for path in aligned_bytes})


class BatchFailureTests(unittest.TestCase):
    def test_partial_batch_failure_is_manifested_and_other_run_survives(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            good = dataset(root, "pocketmc", "good")
            bad = replace(good, run_id="bad")
            good.trajectory.write_text("input", encoding="utf-8")
            cfg = config(root, "pocketmc", (good, bad), ("mc-states",))
            success = RunResult(good, [FrameRecord(1, 1.0, (), 0)])

            def fake_run(_cfg: AnalysisConfig, item: DatasetSpec, **_kwargs: object) -> RunResult:
                if item.run_id == "bad":
                    raise RuntimeError("synthetic failure")
                return success

            with patch("gcmc_port.analysis.runner.run_dataset", side_effect=fake_run), patch(
                "gcmc_port.analysis.runner.write_aggregate", return_value=[]
            ):
                results, failures = run_analysis(cfg)
            self.assertEqual([item.dataset.run_id for item in results], ["good"])
            self.assertEqual(failures[0]["run_id"], "bad")
            failed_manifest = json.loads((cfg.output.root / "bad" / "analysis_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(failed_manifest["status"], "failed")

    def test_md_batch_writes_common_time_mean_and_ci(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first_data = replace(dataset(root, "md", "replica-00"), replica="00", sweep="300")
            second_data = replace(dataset(root, "md", "replica-01"), replica="01", sweep="300")
            first = RunResult(first_data, [FrameRecord(i, i * 1000.0, (), value) for i, value in enumerate((1, 2, 3))])
            second = RunResult(second_data, [FrameRecord(i, i * 1000.0, (), value) for i, value in enumerate((2, 2, 4))])
            outputs = write_aggregate([first, second], root / "results")
            names = {path.name for path in outputs}
            self.assertIn("md_common_time_occupancy.tsv", names)
            self.assertIn("md_common_time_occupancy.png", names)
            header = (root / "results" / "aggregate" / "run_summary.tsv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("replica", header)
            self.assertIn("sweep", header)


class AnalysisConfigTests(unittest.TestCase):
    def test_relative_paths_and_task_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "run").mkdir()
            (root / "run" / "trajectory.gro").write_text("x", encoding="utf-8")
            path = root / "analyses.toml"
            path.write_text(
                """
[input]
kind = "pocketmc"
run_id = "mc-one"
trajectory = "run/trajectory.gro"

[molecule]
preset = "co"

[cavity]
mode = "sphere"
anchor = "800ATC"

[analysis]
tasks = ["density", "vmd"]

[output]
root = "results"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            loaded = load_analysis_config(path)
            self.assertEqual(loaded.datasets[0].trajectory, (root / "run" / "trajectory.gro").resolve())
            self.assertEqual(loaded.output.root, (root / "results").resolve())
            self.assertEqual(loaded.analysis.tasks, ("density", "mc-states", "vmd"))

    def test_wizard_multiple_selection_writes_round_trip_toml(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw) / "analyses.toml"
            answers = [
                "md", "single", "sample", "md.tpr", "md.xtc", "", "water", "sphere",
                "800ATC", "C2,C4,C7", "0.6", "lifetime,density,vmd", "results",
            ]
            with patch("builtins.input", side_effect=answers):
                path, loaded = run_wizard(destination=destination)
            self.assertEqual(path, destination.resolve())
            self.assertEqual(loaded.analysis.tasks, ("density", "lifetime", "vmd"))
            self.assertIn('[analysis]', path.read_text(encoding="utf-8"))

    def test_mask_validation_does_not_require_a_sphere_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name in ("md.tpr", "md.xtc"):
                (root / name).write_text("x", encoding="utf-8")
            (root / "cavity_mask.dat").write_text("0.0 0.0 0.0\n", encoding="utf-8")
            (root / "cavity.meta.json").write_text(
                '{"dx": 0.1, "reference_point": [0.0, 0.0, 0.0]}\n', encoding="utf-8"
            )
            path = root / "analyses.toml"
            path.write_text(
                "\n".join(
                    [
                        "[input]", 'kind = "md"', 'topology = "md.tpr"', 'trajectory = "md.xtc"', "",
                        "[molecule]", 'preset = "water"', "", "[cavity]", 'mode = "mask"',
                        'mask = "cavity_mask.dat"', 'meta = "cavity.meta.json"',
                        'anchor = "800ATC"', "", "[analysis]", 'tasks = ["lifetime"]', "",
                        "[output]", 'root = "results"',
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            validate_analysis_config(load_analysis_config(path), check_files=True)


if __name__ == "__main__":
    unittest.main()
