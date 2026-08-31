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

from gcmc_port.cavity import (
    VoxelMask,
    align_voxel_mask_to_structure,
    build_cavity_from_structure,
    load_voxel_mask,
    molecule_residue_ids_in_mask,
    water_residue_ids_in_mask,
)
from gcmc_port.config import load_config
from gcmc_port.gro import Atom, GroStructure, parse_gro, write_gro
from gcmc_port.moves import write_mask_trajectory
from gcmc_port.workflow import GCMCWorkflow


class FixedMaskTests(unittest.TestCase):

    def test_molecule_residue_ids_in_mask_supports_co_residues(self) -> None:
        mask = VoxelMask(
            points=np.asarray([[0.20, 0.20, 0.20]], dtype=float),
            dx=0.10,
            reference_point=(0.20, 0.20, 0.20),
            effective_volume=0.001,
        )
        structure = GroStructure(
            title="co mask",
            atoms=[
                Atom(1, "ALA", "CA", 1, 0.20, 0.20, 0.20),
                Atom(50, "COM", "CJ", 2, 0.18, 0.20, 0.20),
                Atom(50, "COM", "J1", 3, 0.20, 0.20, 0.20),
                Atom(50, "COM", "OJ1", 4, 0.22, 0.20, 0.20),
                Atom(60, "COM", "CJ", 5, 0.80, 0.80, 0.80),
                Atom(60, "COM", "J1", 6, 0.82, 0.80, 0.80),
                Atom(60, "COM", "OJ1", 7, 0.84, 0.80, 0.80),
                Atom(70, "SOL", "OW", 8, 0.20, 0.20, 0.20),
                Atom(70, "SOL", "HW1", 9, 0.22, 0.20, 0.20),
                Atom(70, "SOL", "HW2", 10, 0.20, 0.22, 0.20),
            ],
            box_line="1.0 1.0 1.0",
        )

        self.assertEqual(molecule_residue_ids_in_mask(structure, mask, "COM"), [50])
        self.assertEqual(water_residue_ids_in_mask(structure, mask), [70])

    def test_mask_trajectory_writer_appends_gro_frames(self) -> None:
        gro_path = ROOT / "assets" / "example" / "init.gro"
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            build_cavity_from_structure(
                gro_path=gro_path,
                outprefix=outdir / "fixed",
                mode="seeded",
                dx=0.075,
                probe_radius=0.10,
                search_radius=0.9,
                seed_residue="800ATC",
                exclude_residues=["800ATC"],
            )
            mask = load_voxel_mask(outdir / "fixed_mask.dat", outdir / "fixed.meta.json")
            trajectory = outdir / "cavity_trajectory.gro"
            write_mask_trajectory(mask, gro_path, trajectory_path=trajectory, accepted=0, nmol=0, label="start")
            write_mask_trajectory(mask, gro_path, trajectory_path=trajectory, accepted=1, nmol=1, label="trial=1")

            lines = trajectory.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0].split()[0], "0")
            self.assertEqual(int(lines[1].strip()), mask.point_count)
            second_frame_line = mask.point_count + 3
            self.assertEqual(lines[second_frame_line].split()[0], "1")

    def test_mask_alignment_restores_membership_after_translation(self) -> None:
        gro_path = ROOT / "assets" / "example" / "init.gro"
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            build_cavity_from_structure(
                gro_path=gro_path,
                outprefix=outdir / "apo",
                mode="seeded",
                dx=0.075,
                probe_radius=0.10,
                search_radius=0.6,
                seed_residue="800ATC",
                exclude_residues=["800ATC"],
            )
            mask = load_voxel_mask(outdir / "apo_mask.dat", outdir / "apo.meta.json")
            original = parse_gro(gro_path)
            translated = GroStructure(
                title=original.title,
                atoms=[
                    Atom(
                        resid=atom.resid,
                        resname=atom.resname,
                        atomname=atom.atomname,
                        atomnr=atom.atomnr,
                        x=atom.x + 0.30,
                        y=atom.y - 0.20,
                        z=atom.z + 0.15,
                    )
                    for atom in original.atoms
                ],
                box_line=original.box_line,
            )
            translated_path = outdir / "translated.gro"
            write_gro(translated_path, translated)

            original_hits = water_residue_ids_in_mask(original, mask)
            shifted_hits = water_residue_ids_in_mask(translated, mask)
            aligned_mask, shift = align_voxel_mask_to_structure(mask, translated_path)
            realigned_hits = water_residue_ids_in_mask(translated, aligned_mask)

            self.assertEqual(len(original_hits), 5)
            self.assertLess(len(shifted_hits), len(original_hits))
            self.assertEqual(realigned_hits, original_hits)
            self.assertAlmostEqual(shift[0], 0.30, places=6)
            self.assertAlmostEqual(shift[1], -0.20, places=6)
            self.assertAlmostEqual(shift[2], 0.15, places=6)

    def test_alignment_ignores_excluded_ligand_motion(self) -> None:
        gro_path = ROOT / "assets" / "example" / "init.gro"
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            build_cavity_from_structure(
                gro_path=gro_path,
                outprefix=outdir / "apo",
                mode="seeded",
                dx=0.075,
                probe_radius=0.10,
                search_radius=0.6,
                seed_residue="800ATC",
                exclude_residues=["800ATC"],
            )
            mask = load_voxel_mask(outdir / "apo_mask.dat", outdir / "apo.meta.json")
            original = parse_gro(gro_path)
            ligand_moved = GroStructure(
                title=original.title,
                atoms=[
                    Atom(
                        resid=atom.resid,
                        resname=atom.resname,
                        atomname=atom.atomname,
                        atomnr=atom.atomnr,
                        x=atom.x + (0.40 if atom.resid == 800 and atom.resname == "ATC" else 0.0),
                        y=atom.y - (0.25 if atom.resid == 800 and atom.resname == "ATC" else 0.0),
                        z=atom.z + (0.18 if atom.resid == 800 and atom.resname == "ATC" else 0.0),
                    )
                    for atom in original.atoms
                ],
                box_line=original.box_line,
            )
            ligand_moved_path = outdir / "ligand-moved.gro"
            write_gro(ligand_moved_path, ligand_moved)

            aligned_mask, shift = align_voxel_mask_to_structure(mask, ligand_moved_path)

            self.assertAlmostEqual(shift[0], 0.0, places=6)
            self.assertAlmostEqual(shift[1], 0.0, places=6)
            self.assertAlmostEqual(shift[2], 0.0, places=6)
            self.assertEqual(mask.reference_point, aligned_mask.reference_point)

    def test_initial_delete_padding_expands_voxel_cleanup(self) -> None:
        gro_path = ROOT / "assets" / "example" / "init.gro"
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            build_cavity_from_structure(
                gro_path=gro_path,
                outprefix=outdir / "apo",
                mode="seeded",
                dx=0.075,
                probe_radius=0.10,
                search_radius=0.6,
                seed_residue="800ATC",
                exclude_residues=["800ATC"],
            )
            mask = load_voxel_mask(outdir / "apo_mask.dat", outdir / "apo.meta.json")
            structure = parse_gro(gro_path)

            default_hits = water_residue_ids_in_mask(structure, mask)
            buffered_hits = water_residue_ids_in_mask(structure, mask.with_membership_padding(mask.membership_padding + 0.02))

            self.assertEqual(len(default_hits), 5)
            self.assertGreater(len(buffered_hits), len(default_hits))

    def test_workflow_runtime_mask_realigns_each_structure(self) -> None:
        gro_path = ROOT / "assets" / "example" / "init.gro"
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            build_cavity_from_structure(
                gro_path=gro_path,
                outprefix=outdir / "apo",
                mode="seeded",
                dx=0.075,
                probe_radius=0.10,
                search_radius=0.6,
                seed_residue="800ATC",
                exclude_residues=["800ATC"],
            )

            original = parse_gro(gro_path)
            translated = GroStructure(
                title=original.title,
                atoms=[
                    Atom(
                        resid=atom.resid,
                        resname=atom.resname,
                        atomname=atom.atomname,
                        atomnr=atom.atomnr,
                        x=atom.x - 0.25,
                        y=atom.y + 0.18,
                        z=atom.z - 0.12,
                    )
                    for atom in original.atoms
                ],
                box_line=original.box_line,
            )
            translated_path = outdir / "translated.gro"
            write_gro(translated_path, translated)

            config_path = outdir / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'project_root = "."',
                        'work_root = "."',
                        f'forcefield_dir = "{(ROOT / "assets" / "defaults" / "amber14sb_parmbsc1.ff").as_posix()}"',
                        f'residue_types = "{(ROOT / "assets" / "defaults" / "residuetypes.dat").as_posix()}"',
                        f'topology = "{(ROOT / "assets" / "example" / "topol.top").as_posix()}"',
                        f'water_itp = "{(ROOT / "assets" / "defaults" / "WAT.itp").as_posix()}"',
                        f'chk_mdp = "{(ROOT / "assets" / "defaults" / "chk.mdp").as_posix()}"',
                        f'steep_mdp = "{(ROOT / "assets" / "defaults" / "steep.mdp").as_posix()}"',
                        f'em_mdp = "{(ROOT / "assets" / "defaults" / "em.mdp").as_posix()}"',
                        f'init_gro = "{gro_path.as_posix()}"',
                        f'gas_gro = "{(ROOT / "assets" / "defaults" / "COM.gro").as_posix()}"',
                        "",
                        "[cavity]",
                        'mode = "mask"',
                        'mask_file = "apo_mask.dat"',
                        'mask_meta = "apo.meta.json"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            workflow = GCMCWorkflow(load_config(config_path))
            original_mask = workflow._runtime_mask_for_structure(gro_path)
            translated_mask = workflow._runtime_mask_for_structure(translated_path)

            self.assertEqual(
                water_residue_ids_in_mask(original, original_mask),
                water_residue_ids_in_mask(translated, translated_mask),
            )

    def test_after_membership_should_use_realigned_mask(self) -> None:
        gro_path = ROOT / "assets" / "example" / "init.gro"
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            build_cavity_from_structure(
                gro_path=gro_path,
                outprefix=outdir / "apo",
                mode="seeded",
                dx=0.075,
                probe_radius=0.10,
                search_radius=0.6,
                seed_residue="800ATC",
                exclude_residues=["800ATC"],
            )

            config_path = outdir / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'project_root = "."',
                        'work_root = "."',
                        f'forcefield_dir = "{(ROOT / "assets" / "defaults" / "amber14sb_parmbsc1.ff").as_posix()}"',
                        f'residue_types = "{(ROOT / "assets" / "defaults" / "residuetypes.dat").as_posix()}"',
                        f'topology = "{(ROOT / "assets" / "example" / "topol.top").as_posix()}"',
                        f'water_itp = "{(ROOT / "assets" / "defaults" / "WAT.itp").as_posix()}"',
                        f'chk_mdp = "{(ROOT / "assets" / "defaults" / "chk.mdp").as_posix()}"',
                        f'steep_mdp = "{(ROOT / "assets" / "defaults" / "steep.mdp").as_posix()}"',
                        f'em_mdp = "{(ROOT / "assets" / "defaults" / "em.mdp").as_posix()}"',
                        f'init_gro = "{gro_path.as_posix()}"',
                        f'gas_gro = "{(ROOT / "assets" / "defaults" / "COM.gro").as_posix()}"',
                        "",
                        "[cavity]",
                        'mode = "mask"',
                        'mask_file = "apo_mask.dat"',
                        'mask_meta = "apo.meta.json"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            workflow = GCMCWorkflow(load_config(config_path))
            previous_mask = workflow._runtime_mask_for_structure(gro_path)
            original = parse_gro(gro_path)
            translated = GroStructure(
                title=original.title,
                atoms=[
                    Atom(
                        resid=atom.resid,
                        resname=atom.resname,
                        atomname=atom.atomname,
                        atomnr=atom.atomnr,
                        x=atom.x + 0.12,
                        y=atom.y - 0.09,
                        z=atom.z + 0.07,
                    )
                    for atom in original.atoms
                ],
                box_line=original.box_line,
            )
            translated_path = outdir / "confout-shifted.gro"
            write_gro(translated_path, translated)

            original_count = len(water_residue_ids_in_mask(original, previous_mask))
            old_count = len(water_residue_ids_in_mask(translated, previous_mask))
            new_count = len(water_residue_ids_in_mask(translated, workflow._runtime_mask_for_structure(translated_path)))

            self.assertNotEqual(old_count, new_count)
            self.assertEqual(new_count, original_count)

    def test_deletion_log_context_reports_provenance(self) -> None:
        gro_path = ROOT / "assets" / "example" / "init.gro"
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)
            build_cavity_from_structure(
                gro_path=gro_path,
                outprefix=outdir / "apo",
                mode="seeded",
                dx=0.075,
                probe_radius=0.10,
                search_radius=0.6,
                seed_residue="800ATC",
                exclude_residues=["800ATC"],
            )
            config_path = outdir / "config.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[paths]",
                        'project_root = "."',
                        'work_root = "."',
                        f'forcefield_dir = "{(ROOT / "assets" / "defaults" / "amber14sb_parmbsc1.ff").as_posix()}"',
                        f'residue_types = "{(ROOT / "assets" / "defaults" / "residuetypes.dat").as_posix()}"',
                        f'topology = "{(ROOT / "assets" / "example" / "topol.top").as_posix()}"',
                        f'water_itp = "{(ROOT / "assets" / "defaults" / "WAT.itp").as_posix()}"',
                        f'chk_mdp = "{(ROOT / "assets" / "defaults" / "chk.mdp").as_posix()}"',
                        f'steep_mdp = "{(ROOT / "assets" / "defaults" / "steep.mdp").as_posix()}"',
                        f'em_mdp = "{(ROOT / "assets" / "defaults" / "em.mdp").as_posix()}"',
                        f'init_gro = "{gro_path.as_posix()}"',
                        f'gas_gro = "{(ROOT / "assets" / "defaults" / "COM.gro").as_posix()}"',
                        "",
                        "[cavity]",
                        'mode = "mask"',
                        'mask_file = "apo_mask.dat"',
                        'mask_meta = "apo.meta.json"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            workflow = GCMCWorkflow(load_config(config_path))
            resname, resid, info = workflow._move_log_context(
                type_move="D",
                moved_resid=21086,
                confout_gro=gro_path,
                x_gro=gro_path,
                previous_gro=gro_path,
                gas_name="SOL",
                provenance={21086: "init"},
            )

            self.assertEqual(resname, "SOL")
            self.assertEqual(resid, "21086")
            self.assertEqual(info, "init")


if __name__ == "__main__":
    unittest.main()
