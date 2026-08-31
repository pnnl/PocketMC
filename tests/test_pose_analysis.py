from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import replace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gcmc_port.analysis.anchors import select_mda_anchor
from gcmc_port.analysis.config import load_analysis_config
from gcmc_port.analysis.cavity_setup import prepare_analysis_cavities
from gcmc_port.analysis.discovery import DiscoveredCase, discover_cases
from gcmc_port.analysis.geometry import _cell_vectors, apply_inverse_transform, apply_transform, kabsch_transform, minimum_image
from gcmc_port.analysis.pose import (
    _alignment_indices_and_reference,
    _canonicalize_pose_mask,
    _deposit_with_growth,
    _regrid_density_groups,
    _ring_pucker,
    run_pose_stage,
    uniform_frame_indices,
)
from gcmc_port.cavity import load_voxel_mask
from gcmc_port.analysis.runner import run_analysis
from gcmc_port.analysis.slurm import render_analysis_launchers, render_analysis_sbatch
from gcmc_port.analysis.wizard import (
    ExistingAnalysis,
    _recompute_stale_pose_hydration,
    _resume_existing_analysis,
    run_wizard,
)


def _pdb_atom(serial: int, name: str, resname: str, resid: int, x: float, y: float, z: float, element: str) -> str:
    record = "ATOM  " if resname == "ALA" else "HETATM"
    return (
        f"{record}{serial:5d} {name:^4s} {resname:>3s} A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
    )


def _write_pose_trajectory(path: Path, frames: int = 6) -> None:
    lines = ["CRYST1   40.000   40.000   40.000  90.00  90.00  90.00 P 1           1"]
    for frame in range(frames):
        shift = 0.0 if frame < frames // 2 else 4.0
        lines.extend(
            [
                f"MODEL     {frame + 1:4d}",
                _pdb_atom(1, "N", "ALA", 1, 0.0, 0.0, 0.0, "N"),
                _pdb_atom(2, "CA", "ALA", 1, 1.5, 0.0, 0.0, "C"),
                _pdb_atom(3, "C", "ALA", 1, 1.5, 1.5, 0.0, "C"),
                _pdb_atom(4, "O", "ALA", 1, 0.0, 1.5, 0.0, "O"),
                _pdb_atom(5, "C1", "LIG", 10, 4.0 + shift, 4.0, 4.0, "C"),
                _pdb_atom(6, "C2", "LIG", 10, 5.0 + shift, 4.0, 4.0, "C"),
                _pdb_atom(7, "C3", "LIG", 10, 4.0 + shift, 5.0, 4.0, "C"),
                _pdb_atom(8, "OW", "WAT", 20, 4.5 + shift, 4.5, 5.0, "O"),
                "ENDMDL",
            ]
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_periodic_pose_trajectory(path: Path, frames: int = 2) -> None:
    crystal = "CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1"
    lines = [crystal]
    for frame in range(frames):
        lines.extend(
            [
                f"MODEL     {frame + 1:4d}",
                crystal,
                _pdb_atom(1, "N", "ALA", 1, 7.0, 7.0, 7.0, "N"),
                _pdb_atom(2, "CA", "ALA", 1, 8.5, 7.0, 7.0, "C"),
                _pdb_atom(3, "C", "ALA", 1, 8.5, 8.5, 7.0, "C"),
                _pdb_atom(4, "O", "ALA", 1, 7.0, 8.5, 7.0, "O"),
                _pdb_atom(5, "C1", "LIG", 10, 1.0, 10.0, 10.0, "C"),
                _pdb_atom(6, "C2", "LIG", 10, 2.0, 10.0, 10.0, "C"),
                _pdb_atom(7, "C3", "LIG", 10, 1.0, 11.0, 10.0, "C"),
                # This is 1.5 A from C1 through the periodic x boundary.
                _pdb_atom(8, "OW", "WAT", 20, 19.5, 10.0, 10.0, "O"),
                "ENDMDL",
            ]
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mask_reference_pair(source: Path, trajectory: Path, frames: int = 2) -> None:
    source_points = {
        "N": (0.0, 0.0, 0.0), "CA": (1.5, 0.0, 0.0),
        "C": (1.5, 1.5, 0.0), "O": (0.0, 1.5, 0.0),
    }
    ligand_points = [(4.0, 4.0, 4.0), (5.0, 4.0, 4.0), (4.0, 5.0, 4.0)]

    def fitted(point: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = point
        return -y + 30.0, x + 20.0, z + 10.0

    source.write_text(
        "\n".join(
            [
                *[
                    _pdb_atom(index, name, "ALA", 1, *point, name[0])
                    for index, (name, point) in enumerate(source_points.items(), start=1)
                ],
                *[
                    _pdb_atom(index, f"C{index - 4}", "LIG", 10, *point, "C")
                    for index, point in enumerate(ligand_points, start=5)
                ],
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    lines: list[str] = []
    for frame in range(frames):
        lines.extend(
            [
                f"MODEL     {frame + 1:4d}",
                *[
                    _pdb_atom(index, name, "ALA", 1, *fitted(point), name[0])
                    for index, (name, point) in enumerate(source_points.items(), start=1)
                ],
                *[
                    _pdb_atom(index, f"C{index - 4}", "LIG", 10, *fitted(point), "C")
                    for index, point in enumerate(ligand_points, start=5)
                ],
                _pdb_atom(8, "OW", "WAT", 20, *fitted((4.5, 4.5, 5.0)), "O"),
                "ENDMDL",
            ]
        )
    lines.append("END")
    trajectory.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PoseSamplingTests(unittest.TestCase):
    def test_substrate_histogram_grows_instead_of_dropping_transformed_points(self) -> None:
        histogram = np.zeros((2, 2, 2), dtype=float)
        grown, low = _deposit_with_growth(
            histogram,
            np.asarray([-2.2, 4.1, 0.5]),
            np.zeros(3),
            1.0,
            margin_a=2.0,
        )
        self.assertGreater(grown.shape[0], histogram.shape[0])
        self.assertGreater(grown.shape[1], histogram.shape[1])
        self.assertLess(low[0], 0.0)
        self.assertEqual(float(grown.sum()), 1.0)

    def test_pose_hydration_reimages_water_across_periodic_boundary(self) -> None:
        try:
            import MDAnalysis  # noqa: F401
        except Exception as exc:
            self.skipTest(f"MDAnalysis unavailable: {exc}")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trajectory = root / "periodic.pdb"
            _write_periodic_pose_trajectory(trajectory)
            config_path = root / "periodic.toml"
            config_path.write_text(
                """
[input]
kind = "md"
run_id = "periodic"
topology = "periodic.pdb"
trajectory = "periodic.pdb"
reference = "periodic.pdb"

[molecule]
preset = "water"

[cavity]
mode = "sphere"
anchor = "10LIG"
anchor_atoms = ["C1", "C2", "C3"]
radius_nm = 0.3
protein_selection = "protein"

[substrate]
enabled = true
selection = "resid 10 and resname LIG"

[pose]
clusters = 1
reference = "periodic.pdb"
pocket_selection = "protein and backbone"
reference_pocket_selection = "protein and backbone"

[pose.sampling]
max_frames_per_trajectory = 2
write_trajectory = false

[analysis]
tasks = ["pose-clusters", "pose-hydration", "compare-hydration"]
density_bin_a = 1.0
density_sigma_a = 1.0
density_cutoff_sigma = 2.0

[output]
root = "results"
cache = true
overwrite = true
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_analysis_config(config_path)
            result = run_pose_stage(config, "all", force=True)
            self.assertFalse(result.failures, result.failures)
            cluster = config.output.root / "periodic" / "poses" / "cluster_01"
            with (
                np.load(cluster / "pocket-frame" / "density_maps.npz") as pocket,
                np.load(cluster / "substrate-frame" / "density_maps.npz") as substrate,
            ):
                pocket_integral = float(np.asarray(pocket["rho_conditional"]).sum() * float(pocket["bin_A"]) ** 3)
                substrate_integral = float(np.asarray(substrate["rho_conditional"]).sum() * float(substrate["bin_A"]) ** 3)
            self.assertGreater(pocket_integral, 0.9)
            self.assertAlmostEqual(substrate_integral, pocket_integral, places=8)

            pose_manifest = cluster.parent / "pose_manifest.json"
            legacy_payload = json.loads(pose_manifest.read_text(encoding="utf-8"))
            legacy_payload.pop("pose_hydration_cache_version", None)
            legacy_payload["fingerprint"] = "legacy-coordinate-frame-cache"
            pose_manifest.write_text(json.dumps(legacy_payload), encoding="utf-8")
            root_manifest = config.output.root / "analysis_manifest.json"
            root_manifest.write_text("{}\n", encoding="utf-8")
            existing = ExistingAnalysis(
                manifest=root_manifest,
                config_path=config_path,
                output_root=config.output.root,
                failures=(),
                modified_ns=root_manifest.stat().st_mtime_ns,
                status="complete",
            )
            _recompute_stale_pose_hydration(existing)
            repaired_payload = json.loads(pose_manifest.read_text(encoding="utf-8"))
            self.assertGreaterEqual(int(repaired_payload["pose_hydration_cache_version"]), 4)

    def test_pose_mask_is_fitted_from_its_source_structure_before_water_selection(self) -> None:
        try:
            import MDAnalysis  # noqa: F401
        except Exception as exc:
            self.skipTest(f"MDAnalysis unavailable: {exc}")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "mask_source.pdb"
            trajectory = root / "trajectory.pdb"
            _write_mask_reference_pair(source, trajectory)
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
                        "exclude_residues": ["10LIG"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "mask-fit.toml"
            config_path.write_text(
                """
[input]
kind = "md"
run_id = "mask-fit"
topology = "trajectory.pdb"
trajectory = "trajectory.pdb"
reference = "trajectory.pdb"

[molecule]
preset = "water"

[cavity]
mode = "mask"
mask = "cavity_mask.dat"
meta = "cavity.meta.json"
anchor = "10LIG"
protein_selection = "protein"

[substrate]
enabled = true
selection = "resid 10 and resname LIG"

[pose]
clusters = 1
reference = "trajectory.pdb"
pocket_selection = "protein and backbone"
reference_pocket_selection = "protein and backbone"

[pose.sampling]
max_frames_per_trajectory = 2
write_trajectory = false

[analysis]
tasks = ["pose-clusters", "pose-hydration", "compare-hydration"]
density_bin_a = 1.0
density_sigma_a = 1.0
density_cutoff_sigma = 2.0

[output]
root = "results"
cache = true
overwrite = true
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_analysis_config(config_path)
            result = run_pose_stage(config, "all", force=True)
            self.assertFalse(result.failures, result.failures)
            cluster = config.output.root / "mask-fit" / "poses" / "cluster_01"
            with (
                np.load(cluster / "pocket-frame" / "density_maps.npz") as density,
                np.load(cluster / "pocket-frame" / "substrate_overlay.npz") as overlay,
            ):
                rho = np.asarray(density["rho_conditional"], dtype=float)
                axes = [np.asarray(density[f"{name}_A"], dtype=float) for name in "xyz"]
                grids = np.meshgrid(*axes, indexing="ij")
                density_center = np.asarray([(rho * grid).sum() / rho.sum() for grid in grids])
                substrate_center = np.asarray(overlay["positions_A"], dtype=float).mean(axis=0)
            self.assertGreater(float(rho.sum()), 0.0)
            self.assertLess(float(np.linalg.norm(density_center - substrate_center)), 3.0)
            payload = json.loads((cluster.parent / "pose_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["mask_alignment"]["method"], "mask source-structure pocket Kabsch fit")
            self.assertEqual(Path(payload["mask_alignment"]["source"]), source)
            first_fingerprint = str(payload["fingerprint"])
            source_stat = source.stat()
            os.utime(
                source,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000_000),
            )
            refreshed = run_pose_stage(config, "hydrate", force=False)
            self.assertFalse(refreshed.failures, refreshed.failures)
            payload = json.loads((cluster.parent / "pose_manifest.json").read_text(encoding="utf-8"))
            self.assertNotEqual(str(payload["fingerprint"]), first_fingerprint)
            aggregate = config.output.root / "aggregate" / "pose-groups" / "default" / "cluster_01"
            with (
                np.load(aggregate / "mask-fit.mean_density.npz") as density,
                np.load(aggregate / "substrate_overlay.npz") as overlay,
            ):
                rho = np.asarray(density["rho"], dtype=float)
                axes = [np.asarray(density[f"{name}_A"], dtype=float) for name in "xyz"]
                grids = np.meshgrid(*axes, indexing="ij")
                density_center = np.asarray([(rho * grid).sum() / rho.sum() for grid in grids])
                substrate_center = np.asarray(overlay["positions_A"], dtype=float).mean(axis=0)
            self.assertLess(float(np.linalg.norm(density_center - substrate_center)), 3.0)

    def test_explicit_mask_trajectory_frame_takes_precedence_over_static_source_mask(self) -> None:
        try:
            import MDAnalysis as mda
        except Exception as exc:
            self.skipTest(f"MDAnalysis unavailable: {exc}")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "mask_source.pdb"
            trajectory = root / "trajectory.pdb"
            _write_mask_reference_pair(source, trajectory)
            mask_path = root / "cavity_mask.dat"
            mask_path.write_text("9.0 9.0 9.0\n", encoding="utf-8")
            meta_path = root / "cavity.meta.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "dx": 0.1,
                        "reference_point": [9.0, 9.0, 9.0],
                        "effective_volume": 0.001,
                        "source_gro": str(source),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            mask_trajectory = root / "cavity_trajectory.gro"
            mask_trajectory.write_text(
                "runtime cavity frame\n"
                "    1\n"
                "    1CAV     HE    1   2.550   2.450   1.500\n"
                "   4.00000   4.00000   4.00000\n",
                encoding="utf-8",
            )
            config_path = root / "explicit-mask-frame.toml"
            config_path.write_text(
                """
[input]
kind = "md"
run_id = "mask-frame"
topology = "trajectory.pdb"
trajectory = "trajectory.pdb"
reference = "trajectory.pdb"

[molecule]
preset = "water"

[cavity]
mode = "mask"
mask = "cavity_mask.dat"
meta = "cavity.meta.json"
mask_trajectory = "cavity_trajectory.gro"
protein_selection = "protein"

[substrate]
enabled = true
selection = "resid 10 and resname LIG"

[pose]
clusters = 1
reference = "trajectory.pdb"
pocket_selection = "protein and backbone"
reference_pocket_selection = "protein and backbone"

[analysis]
tasks = ["pose-clusters", "pose-hydration"]

[output]
root = "results"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_analysis_config(config_path)
            dataset = config.datasets[0]
            universe = mda.Universe(str(trajectory), str(trajectory))
            substrate = universe.select_atoms("resid 10 and resname LIG")
            targets = np.asarray(substrate.positions, dtype=float).reshape(1, substrate.n_atoms, 3)
            mask = load_voxel_mask(mask_path, meta_path)
            aligned, metadata = _canonicalize_pose_mask(config, dataset, universe, mask, targets)
            self.assertEqual(metadata["method"], "explicit first mask-trajectory frame pocket Kabsch fit")
            self.assertTrue(np.allclose(np.asarray(aligned.reference_point) * 10.0, [25.5, 24.5, 15.0]))
            close = getattr(universe.trajectory, "close", None)
            if close is not None:
                close()

    def test_aligned_density_grids_with_different_shapes_are_regridded_before_aggregation(self) -> None:
        first = np.ones((2, 2, 2), dtype=float)
        second = np.ones((3, 2, 2), dtype=float) * 2.0
        entries = {
            "A": [(first, (np.asarray([0.5, 1.5]), np.asarray([0.5, 1.5]), np.asarray([0.5, 1.5])))],
            "B": [(second, (np.asarray([-0.5, 0.5, 1.5]), np.asarray([0.5, 1.5]), np.asarray([0.5, 1.5])))],
        }
        means, replicas, axes = _regrid_density_groups(entries, 1.0)
        self.assertEqual(means["A"].shape, means["B"].shape)
        self.assertEqual(replicas["A"].shape[1:], replicas["B"].shape[1:])
        self.assertEqual(len(axes[0]), 3)
        self.assertAlmostEqual(float(means["A"].sum()), float(first.sum()))
        self.assertAlmostEqual(float(means["B"].sum()), float(second.sum()))

    def test_inverse_transform_recovers_mobile_coordinates(self) -> None:
        mobile = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        target = np.asarray([[2.0, 3.0, 1.0], [2.0, 4.0, 1.0], [1.0, 3.0, 1.0]])
        transform = kabsch_transform(mobile, target)
        aligned = apply_transform(mobile, transform)
        self.assertTrue(np.allclose(apply_inverse_transform(aligned, transform), mobile))

    def test_minimum_image_uses_full_triclinic_cell(self) -> None:
        dimensions = np.asarray([10.0, 10.0, 10.0, 90.0, 90.0, 60.0])
        cell = _cell_vectors(dimensions)
        assert cell is not None
        anchor = np.asarray([2.0, 3.0, 4.0])
        point = anchor + np.asarray([0.49, 0.48, 0.0]) @ cell
        # Fractional component-wise rounding would retain a much longer 8.40 A
        # displacement.  The nearest Cartesian image crosses one a-vector.
        expected = anchor + np.asarray([-0.51, 0.48, 0.0]) @ cell
        self.assertTrue(np.allclose(minimum_image(point, anchor, dimensions), expected))

    def test_uniform_sampling_is_exact_and_includes_endpoints(self) -> None:
        selected = uniform_frame_indices(range(100), 7)
        self.assertEqual(len(selected), 7)
        self.assertEqual(int(selected[0]), 0)
        self.assertEqual(int(selected[-1]), 99)
        self.assertEqual(len(np.unique(selected)), 7)
        self.assertEqual(uniform_frame_indices(range(4), 10).tolist(), [0, 1, 2, 3])
        self.assertEqual(uniform_frame_indices(range(4), 0).tolist(), [0, 1, 2, 3])

    def test_six_member_ring_pucker_descriptor_is_finite(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
        ring = np.column_stack([np.cos(angles), np.sin(angles), 0.2 * ((-1.0) ** np.arange(6))])
        q, theta, phi = _ring_pucker(ring)
        self.assertGreater(q, 0.0)
        self.assertTrue(np.isfinite([q, theta, phi]).all())

    def test_mixed_case_config_and_pose_pipeline(self) -> None:
        try:
            import MDAnalysis  # noqa: F401
        except Exception as exc:
            self.skipTest(f"MDAnalysis unavailable: {exc}")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trajectory = root / "physical.pdb"
            _write_pose_trajectory(trajectory)
            config_path = root / "analyses.toml"
            config_path.write_text(
                f"""
[[case]]
id = "md-one"
kind = "md"
system_id = "protein-a"
comparison_group = "homologs"
run_dir = "."
topology = "physical.pdb"
trajectory = "physical.pdb"
reference = "physical.pdb"
pocketmc_status = "not_detected"

[[case]]
id = "mc-one"
kind = "pocketmc"
system_id = "protein-a"
run_dir = "."
trajectory = "trajectory.gro"
pocketmc_status = "confirmed"
pocketmc_evidence = ["mc.log", "trajectory.gro"]

[molecule]
preset = "water"

[cavity]
mode = "sphere"
anchor = "10LIG"
anchor_atoms = ["C1", "C2", "C3"]
radius_nm = 0.5
protein_selection = "protein"

[substrate]
enabled = true
selection = "resid 10 and resname LIG"

[pose]
clusters = 2
reference = "physical.pdb"
pocket_selection = "protein and backbone"
reference_pocket_selection = "protein and backbone"
seed = 11
restarts = 3

[pose.sampling]
max_frames_per_trajectory = 3
strategy = "uniform"
write_trajectory = true

[analysis]
tasks = ["pose-clusters", "pose-hydration", "compare-hydration", "plots"]
density_bin_a = 1.0
density_sigma_a = 1.0
density_cutoff_sigma = 2.0

[output]
root = "results"
cache = true
overwrite = true
""".strip()
                + "\n",
                encoding="utf-8",
            )
            # Only config loading is exercised for the MC case; pose stages deliberately use the MD subset.
            (root / "trajectory.gro").write_text("placeholder\n0\n1 1 1\n", encoding="utf-8")
            config = load_analysis_config(config_path)
            self.assertEqual(config.kind, "mixed")
            self.assertEqual(config.pose.max_frames_per_trajectory, 3)
            md_only = replace(config, kind="md", datasets=(config.datasets[0],))
            completed, failures = run_analysis(md_only, force=True)
            self.assertEqual(len(completed), 1)
            self.assertFalse(failures, failures)
            sampled = config.output.root / "md-one" / "poses" / "cluster_training_frames.tsv"
            self.assertTrue(sampled.exists())
            self.assertEqual(len(sampled.read_text(encoding="utf-8").splitlines()) - 1, 3)
            assignments = config.output.root / "md-one" / "poses" / "pose_assignments.tsv"
            self.assertEqual(len(assignments.read_text(encoding="utf-8").splitlines()) - 1, 6)
            sampled_xtc = config.output.root / "md-one" / "poses" / "cluster_training.xtc"
            self.assertTrue(sampled_xtc.exists())
            sampled_universe = MDAnalysis.Universe(str(trajectory), str(sampled_xtc))
            self.assertEqual(len(sampled_universe.trajectory), 3)
            sampled_universe.trajectory.close()
            self.assertTrue((config.output.root / "md-one" / "poses" / "cluster_01" / "hydration_sites.pdb").exists())
            self.assertTrue((config.output.root / "md-one" / "poses" / "cluster_01" / "cluster_training.xtc").exists())
            session = (config.output.root / "md-one" / "poses" / "cluster_01" / "cluster_session.vmd.tcl").read_text(encoding="utf-8")
            self.assertIn("# HPD 90%", session)
            self.assertTrue((config.output.root / "md-one" / "poses" / "plots" / "pose_pca.png").exists())

    def test_sequence_mapped_homolog_alignment_recovers_common_frame(self) -> None:
        try:
            import MDAnalysis as mda
        except Exception as exc:
            self.skipTest(f"MDAnalysis unavailable: {exc}")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reference = root / "reference.pdb"
            mobile = root / "mobile.pdb"
            reference_points = [(0, 0, 0), (1.5, 0, 0), (1.5, 1.5, 0), (0, 1.5, 0)]
            ligand_points = [(4, 4, 4), (5, 4, 4), (4, 5, 4)]
            reference.write_text(
                "\n".join(
                    [
                        *[_pdb_atom(index + 1, name, "ALA", 1, *point, name[0]) for index, (name, point) in enumerate(zip(("N", "CA", "C", "O"), reference_points))],
                        *[_pdb_atom(index + 5, f"C{index + 1}", "LIG", 10, *point, "C") for index, point in enumerate(ligand_points)],
                        "END",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def transform(point: tuple[float, float, float]) -> tuple[float, float, float]:
                x, y, z = point
                return -y + 12.0, x + 7.0, z + 3.0

            gly_points = [(-4, 0, 0), (-3, 0, 0), (-3, 1, 0), (-4, 1, 0)]
            mobile.write_text(
                "\n".join(
                    [
                        *[_pdb_atom(index + 1, name, "GLY", 1, *transform(point), name[0]) for index, (name, point) in enumerate(zip(("N", "CA", "C", "O"), gly_points))],
                        *[_pdb_atom(index + 5, name, "ALA", 101, *transform(point), name[0]) for index, (name, point) in enumerate(zip(("N", "CA", "C", "O"), reference_points))],
                        *[_pdb_atom(index + 9, f"C{index + 1}", "LIG", 110, *transform(point), "C") for index, point in enumerate(ligand_points)],
                        "END",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "alignment.toml"
            config_path.write_text(
                """
[input]
kind = "md"
run_id = "homolog"
topology = "mobile.pdb"
trajectory = "mobile.pdb"

[molecule]
preset = "water"

[cavity]
mode = "sphere"
anchor = "110LIG"

[substrate]
enabled = true
selection = "resid 110 and resname LIG"

[pose]
clusters = 1
reference = "reference.pdb"
pocket_selection = "protein and backbone"
reference_pocket_selection = "protein and backbone"

[analysis]
tasks = ["pose-clusters"]

[output]
root = "results"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_analysis_config(config_path)
            universe = mda.Universe(str(mobile))
            mobile_indices, target, metadata = _alignment_indices_and_reference(config, config.datasets[0], universe)
            fitted = apply_transform(universe.atoms.positions, kabsch_transform(universe.atoms.positions[mobile_indices], target))
            ligand = universe.select_atoms("resid 110 and resname LIG")
            self.assertTrue(np.allclose(fitted[ligand.indices], np.asarray(ligand_points), atol=1e-3))
            self.assertIn("sequence mapping", metadata["method"])
            universe.trajectory.close()


class DiscoveryAndSlurmTests(unittest.TestCase):
    def test_discovery_reports_pocketmc_derived_md_and_ignores_internal_trr(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            case = root / "case"
            case.mkdir()
            for name in ("mc.log", "trajectory.gro", "trajectory.meta.jsonl", "md.tpr", "production.xtc", "traj.trr"):
                (case / name).write_text("x", encoding="utf-8")
            found = discover_cases(root, max_depth=2)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].pocketmc_status, "confirmed")
            self.assertEqual(found[0].md_status, "confirmed")
            self.assertTrue(found[0].pocketmc_derived_md)
            self.assertEqual(found[0].trajectory.name, "production.xtc")

    def test_discovery_inherits_pocketmc_provenance_from_parent(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run = root / "run"
            production = run / "production"
            production.mkdir(parents=True)
            (run / "mc.log").write_text("x", encoding="utf-8")
            (run / "trajectory.gro").write_text("x", encoding="utf-8")
            (production / "md.tpr").write_text("x", encoding="utf-8")
            (production / "md.xtc").write_text("x", encoding="utf-8")
            found = [item for item in discover_cases(root, max_depth=3) if item.directory == production.resolve()]
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].pocketmc_status, "confirmed")
            self.assertTrue(found[0].pocketmc_derived_md)
            self.assertIn("ancestor:mc.log", found[0].evidence)

    def test_slurm_pipeline_contains_both_arrays_and_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trajectory = root / "physical.pdb"
            _write_pose_trajectory(trajectory, 2)
            config_path = root / "analyses.toml"
            config_path.write_text(
                """
[input]
kind = "md"
run_id = "one"
topology = "physical.pdb"
trajectory = "physical.pdb"

[molecule]
preset = "water"

[cavity]
mode = "sphere"
anchor = "10LIG"

[substrate]
enabled = true
selection = "resid 10 and resname LIG"

[analysis]
tasks = ["pose-clusters"]

[output]
root = "results"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            config = load_analysis_config(config_path)
            outputs = render_analysis_sbatch(config, root / "jobs")
            self.assertTrue(any(path.name == "01_features.sbatch" for path in outputs))
            helper = (root / "jobs" / "submit_pipeline.sh").read_text(encoding="utf-8")
            self.assertIn("afterany:${feature_job}", helper)
            self.assertIn("afterany:${hydrate_job}", helper)

    def test_discovery_wizard_writes_frame_cap_and_multi_residue_substrate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            topology = root / "md.tpr"
            trajectory = root / "md.xtc"
            topology.write_text("x", encoding="utf-8")
            trajectory.write_text("x", encoding="utf-8")
            discovered = DiscoveredCase(
                case_id="protein-a", directory=root, md_status="confirmed", pocketmc_status="not_detected",
                pocketmc_derived_md=False, topology=topology, trajectory=trajectory, md_alternatives=(),
                mc_trajectory=None, mc_log=None, trajectory_meta=None, gcmc_configs=(), evidence=(), notes=(),
            )
            answers = [
                str(root), "4", "1", "water", "sphere", "10LIG", "C1,C2,C3", "0.5", "all", "yes",
                "10LIG,11OPP", "", "5000", "3", "", "", "", "results",
            ]
            destination = root / "wizard.toml"
            with (
                patch("gcmc_port.analysis.wizard.discover_cases", return_value=[discovered]),
                patch("gcmc_port.analysis.wizard._substrate_candidates", return_value=["10LIG", "11OPP"]),
                patch("gcmc_port.analysis.wizard._frame_count", return_value=10000),
                patch("builtins.input", side_effect=answers),
            ):
                path, config = run_wizard(destination=destination, discover_first=True)
            self.assertEqual(path, destination)
            self.assertEqual(config.pose.max_frames_per_trajectory, 5000)
            self.assertIn("resid 10", config.substrate.selection)
            self.assertIn("resid 11", config.substrate.selection)
            self.assertTrue((config.output.root).is_absolute())

    def test_discovery_wizard_prefers_md_without_capability_question(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            topology = root / "md.tpr"
            trajectory = root / "md.xtc"
            mc_trajectory = root / "trajectory.gro"
            for path in (topology, trajectory, mc_trajectory):
                path.write_text("x", encoding="utf-8")
            discovered = DiscoveredCase(
                case_id="both", directory=root, md_status="confirmed", pocketmc_status="confirmed",
                pocketmc_derived_md=True, topology=topology, trajectory=trajectory, md_alternatives=(),
                mc_trajectory=mc_trajectory, mc_log=None, trajectory_meta=None, gcmc_configs=(),
                evidence=("trajectory.gro",), notes=(), cavity_mode="sphere",
            )
            answers = [
                str(root), "4", "1", "water", "sphere", "10LIG", "C1", "0.5",
                "lifetime", "no", "results",
            ]
            destination = root / "analyses.toml"
            with (
                patch("gcmc_port.analysis.wizard.discover_cases", return_value=[discovered]),
                patch("builtins.input", side_effect=answers),
            ):
                _path, config = run_wizard(destination=destination, discover_first=True)
            self.assertEqual(config.kind, "md")
            self.assertEqual(len(config.datasets), 1)
            self.assertEqual(config.datasets[0].trajectory, trajectory.resolve())
            self.assertTrue((root / "run_analyses.sh").exists())
            self.assertTrue((root / "run_analyses_tahoma_only.sbatch").exists())

    def test_discovery_wizard_prompts_before_using_mc_only_results(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trajectory = root / "trajectory.gro"
            trajectory.write_text("x", encoding="utf-8")
            discovered = DiscoveredCase(
                case_id="mc-only", directory=root, md_status="not_detected", pocketmc_status="confirmed",
                pocketmc_derived_md=False, topology=None, trajectory=None, md_alternatives=(),
                mc_trajectory=trajectory, mc_log=None, trajectory_meta=None, gcmc_configs=(),
                evidence=("trajectory.gro",), notes=(), cavity_mode="sphere",
            )
            answers = [
                str(root), "4", "1", "yes", "water", "sphere", "800ATC", "C2,C4,C7", "0.6",
                "mc-states", "results",
            ]
            with (
                patch("gcmc_port.analysis.wizard.discover_cases", return_value=[discovered]),
                patch("builtins.input", side_effect=answers),
            ):
                _path, config = run_wizard(destination=root / "analyses.toml", discover_first=True)
            self.assertEqual(config.kind, "pocketmc")
            self.assertEqual(config.datasets[0].kind, "pocketmc")

    def test_discovery_wizard_keeps_distinct_mask_bundles_for_multiple_cases(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            discovered: list[DiscoveredCase] = []
            expected_masks: list[Path] = []
            for index in (1, 2):
                case_dir = root / f"case-{index}"
                cavity_dir = case_dir / "cavity-output"
                cavity_dir.mkdir(parents=True)
                topology = case_dir / "md.tpr"
                trajectory = case_dir / "md.xtc"
                topology.write_text("x", encoding="utf-8")
                trajectory.write_text("x", encoding="utf-8")
                mask = cavity_dir / "cavity_mask.dat"
                meta = cavity_dir / "cavity.meta.json"
                points = cavity_dir / "cavity_points.pdb"
                nearby = cavity_dir / "cavity_nearby_residues.tsv"
                for path in (mask, meta, points, nearby):
                    path.write_text("x", encoding="utf-8")
                expected_masks.append(mask.resolve())
                discovered.append(
                    DiscoveredCase(
                        case_id=f"case-{index}", directory=case_dir, md_status="confirmed",
                        pocketmc_status="confirmed", pocketmc_derived_md=True, topology=topology,
                        trajectory=trajectory, md_alternatives=(), mc_trajectory=None, mc_log=None,
                        trajectory_meta=None, gcmc_configs=(), evidence=("mc.log",), notes=(),
                        cavity_mode="mask", cavity_mask=mask, cavity_meta=meta,
                        cavity_points=points, cavity_nearby_residues=nearby,
                    )
                )
            answers = [
                str(root), "4", "0", "water", "mask", "800ATC", "C2,C4,C7", "",
                "lifetime", "no", "results",
            ]
            with (
                patch("gcmc_port.analysis.wizard.discover_cases", return_value=discovered),
                patch("builtins.input", side_effect=answers),
            ):
                _path, config = run_wizard(destination=root / "analyses.toml", discover_first=True)
            self.assertEqual([item.cavity.mask for item in config.datasets], expected_masks)

    def test_complete_launcher_set_reports_steps_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trajectory = root / "physical.pdb"
            _write_pose_trajectory(trajectory, 2)
            config_path = root / "analyses.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[input]", 'kind = "md"', 'run_id = "one"', 'topology = "physical.pdb"',
                        'trajectory = "physical.pdb"', "", "[molecule]", 'preset = "water"', "",
                        "[cavity]", 'mode = "sphere"', 'anchor = "10LIG"', "", "[analysis]",
                        'tasks = ["lifetime"]', "", "[output]", 'root = "results"',
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            paths = render_analysis_launchers(load_analysis_config(config_path), root / "jobs")
            self.assertEqual(len(paths), 3)
            self.assertEqual(
                {path.name for path in paths.values()},
                {"run_analyses.sh", "run_analyses.sbatch", "run_analyses_tahoma_only.sbatch"},
            )
            shell = paths["Direct shell"].read_text(encoding="utf-8")
            self.assertIn("STEP 1/3", shell)
            self.assertIn("minutes to hours", shell)
            self.assertIn("BASH_LINENO", shell)
            self.assertIn("analysis-run.log", shell)
            self.assertIn('POCKETMC_ANALYSES_BIN="${POCKETMC_ANALYSES_BIN:-pocketmc-analyses}"', shell)
            self.assertNotIn(root.resolve().as_posix(), shell)
            generic = paths["Generic Slurm"].read_text(encoding="utf-8")
            self.assertIn("#SBATCH --account=YOUR_ACCOUNT", generic)
            self.assertNotIn(root.resolve().as_posix(), generic)
            tahoma = paths["Tahoma-only Slurm"].read_text(encoding="utf-8")
            self.assertTrue(tahoma.startswith("#!/usr/bin/env bash\n"))
            self.assertIn("#SBATCH --account=YOUR_ACCOUNT", tahoma)
            self.assertIn("#SBATCH --time=48:00:00", tahoma)
            self.assertIn("#SBATCH --nodes=1", tahoma)
            self.assertIn("#SBATCH --ntasks-per-node=1", tahoma)
            self.assertIn("#SBATCH --cpus-per-task=32", tahoma)
            self.assertIn("#SBATCH --job-name=GCMC", tahoma)
            self.assertNotIn("#SBATCH --ntasks-per-node=32", tahoma)
            self.assertNotIn("#SBATCH --partition=compute", tahoma)
            self.assertNotIn("#SBATCH --mem=", tahoma)
            self.assertIn('--jobs "$ANALYSIS_JOBS"', tahoma)
            self.assertIn("one Python process", tahoma)
            self.assertIn("module load gromacs", tahoma)
            self.assertNotIn(root.resolve().as_posix(), tahoma)

    def test_deferred_mask_build_runs_from_saved_case_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "init.gro"
            trajectory = root / "md.xtc"
            topology = root / "md.tpr"
            for path in (source, trajectory, topology):
                path.write_text("x", encoding="utf-8")
            config_path = root / "analyses.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[[case]]", 'id = "one"', 'kind = "md"', 'topology = "md.tpr"',
                        'trajectory = "md.xtc"', 'cavity_mode = "mask"',
                        'cavity_mask = "cavity_mask.dat"', 'cavity_meta = "cavity.meta.json"',
                        'cavity_build_enabled = true', 'cavity_build_source = "init.gro"',
                        'cavity_build_output_prefix = "cavity"', 'cavity_build_mode = "seeded"', "",
                        "[molecule]", 'preset = "water"', "", "[cavity]", 'mode = "sphere"', "",
                        "[analysis]", 'tasks = ["lifetime"]', "", "[output]", 'root = "results"',
                    ]
                ) + "\n",
                encoding="utf-8",
            )

            def fake_build(_source: Path, *, outprefix: Path, **_kwargs: object) -> list[Path]:
                prefix = Path(outprefix)
                outputs = [
                    prefix.with_name(prefix.name + "_mask.dat"),
                    prefix.with_name(prefix.name + "_points.pdb"),
                    prefix.with_suffix(".meta.json"),
                    prefix.with_name(prefix.name + "_nearby_residues.tsv"),
                ]
                for path in outputs:
                    path.write_text("x", encoding="utf-8")
                return outputs

            config = load_analysis_config(config_path)
            with patch("gcmc_port.analysis.cavity_setup.build_cavity_from_structure", side_effect=fake_build):
                outputs = prepare_analysis_cavities(config)
            self.assertEqual(len(outputs), 4)
            self.assertTrue((root / "cavity_mask.dat").exists())
            self.assertTrue((root / "cavity_nearby_residues.tsv").exists())

    def test_anchor_falls_back_to_unique_same_resname_after_case_renumbering(self) -> None:
        class FakeAtoms:
            n_atoms = 2

            def select_atoms(self, _selection: str) -> "FakeAtoms":
                return self

        class FakeResidue:
            resid = 12413
            resnum = 12413
            resname = "ATC"
            segid = "A"
            atoms = FakeAtoms()

        class FakeUniverse:
            residues = [FakeResidue()]

        group, resolution = select_mda_anchor(FakeUniverse(), "800ATC", ("C1",), context="test")
        self.assertGreater(group.n_atoms, 0)
        self.assertEqual(resolution.resolved, "12413ATC")
        self.assertIn("resolved by residue name", resolution.warning)

    def test_partial_analysis_recovery_repairs_anchor_and_writes_resume_launchers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            trajectory = root / "physical.pdb"
            _write_pose_trajectory(trajectory, 2)
            config_path = root / "analyses.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[[case]]", 'id = "one"', 'kind = "md"', 'topology = "physical.pdb"',
                        'trajectory = "physical.pdb"', 'cavity_mode = "sphere"',
                        'cavity_anchor = "800LIG"', 'cavity_anchor_atoms = ["C1"]', "",
                        "[molecule]", 'preset = "water"', "", "[cavity]", 'mode = "sphere"',
                        'anchor = "800LIG"', "", "[analysis]", 'tasks = ["lifetime"]', "",
                        "[output]", 'root = "analysis-results"',
                    ]
                ) + "\n",
                encoding="utf-8",
            )
            output = root / "analysis-results"
            output.mkdir()
            manifest = output / "analysis_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            existing = ExistingAnalysis(
                manifest=manifest,
                config_path=config_path,
                output_root=output,
                failures=({"run_id": "one", "error": "cavity anchor matched no atoms: 800LIG"},),
                modified_ns=manifest.stat().st_mtime_ns,
            )
            with (
                patch("builtins.input", side_effect=[""]),
                patch("gcmc_port.analysis.wizard._repair_anchor_failures", return_value={"one": "10LIG"}),
            ):
                recovered = _resume_existing_analysis(existing)
            self.assertIsNotNone(recovered)
            recovered_path, recovered_config = recovered  # type: ignore[misc]
            self.assertEqual(recovered_config.datasets[0].cavity.anchor, "10LIG")
            self.assertIn("resume", recovered_path.stem)
            resume_shell = root / "resume_analyses.sh"
            self.assertTrue(resume_shell.exists())
            self.assertIn("Resuming from versioned caches", resume_shell.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
