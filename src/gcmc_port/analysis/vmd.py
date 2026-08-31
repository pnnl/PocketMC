from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

from gcmc_port.cavity import load_voxel_mask
from gcmc_port.pathing import portable_path

from .density import hpd_thresholds
from .models import AnalysisConfig, FrameRecord, MoleculeFrame, RunResult


TRACE_SAMPLE_PS = 1000.0
TRACE_SURFACE_CUTOFF_NM = 1.0
TRACE_PAD_SAMPLES = 1
TRACE_MAX_SEGMENT_NM = 1.5
TRACE_POINT_RADIUS_A = 0.6


def _tcl(value: str | Path) -> str:
    return "{" + str(value).replace("\\", "/").replace("{", "\\{").replace("}", "\\}") + "}"


def _tcl_portable_path(path: str | Path | None, base_directory: Path) -> str:
    if path in (None, ""):
        return "{}"
    value = portable_path(Path(path), base_directory)
    if Path(value).is_absolute():
        return _tcl(value)
    return f"[file normalize [file join $script_dir {_tcl(value)}]]"


def _category(result: RunResult, uid: str) -> str:
    values = []
    for frame in result.frames:
        item = next((candidate for candidate in frame.molecules if candidate.uid == uid), None)
        values.append(bool(item and item.inside))
    if values and all(values):
        return "resident"
    if values and not values[0] and any(values):
        return "entry"
    if values and values[0] and not values[-1]:
        return "exit"
    return "entry"


def _write_fallback_structure(path: Path, result: RunResult) -> None:
    lines = ["REMARK PocketMC analysis representative points"]
    for serial, item in enumerate(result.frames[-1].molecules, start=1):
        x, y, z = (value * 10.0 for value in item.point_nm)
        lines.append(
            f"HETATM{serial:5d}  X   {item.resname[:3]:>3s} A{item.resid % 10000:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           O"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_VMD_COORDINATE_SUFFIXES = {".ent", ".gro", ".mae", ".mol2", ".pdb"}


def _write_vmd_md_structure(path: Path, result: RunResult) -> bool:
    """Write the MD reference frame in a format independent of VMD's TPR plugin."""
    topology = result.dataset.topology
    if topology is None or not topology.exists():
        return False
    source = result.dataset.reference
    if source is None or not source.exists():
        source = result.dataset.trajectory
    try:
        import MDAnalysis as mda

        try:
            universe = mda.Universe(str(topology), str(source))
        except Exception:
            universe = mda.Universe(str(source))
        universe.trajectory[0]
        with mda.Writer(str(path), n_atoms=universe.atoms.n_atoms) as writer:
            writer.write(universe.atoms)
        close = getattr(universe.trajectory, "close", None)
        if close is not None:
            close()
    except Exception:
        path.unlink(missing_ok=True)
        return False
    return path.exists() and path.stat().st_size > 0


def _vmd_structure(result: RunResult, vmd_dir: Path) -> tuple[Path, bool]:
    """Choose a portable coordinate-bearing structure and whether it fits the MD trajectory."""
    if result.dataset.kind != "md":
        structure = result.dataset.trajectory
        if structure.exists():
            return structure, True
        fallback = vmd_dir / "analysis_points.pdb"
        _write_fallback_structure(fallback, result)
        return fallback, False

    # A coordinate reference (normally previous.gro) is in the same frame as the
    # density.  Prefer it over the topology, and deliberately do not depend on
    # VMD's optional/fragile TPR molfile plugin.
    for candidate in (result.dataset.reference, result.dataset.topology):
        if (
            candidate is not None
            and candidate.exists()
            and candidate.suffix.lower() in _VMD_COORDINATE_SUFFIXES
        ):
            return candidate, True

    portable = vmd_dir / "analysis_structure.pdb"
    if _write_vmd_md_structure(portable, result):
        return portable, True

    fallback = vmd_dir / "analysis_points.pdb"
    _write_fallback_structure(fallback, result)
    return fallback, False


def _write_mask_structure(path: Path, config: AnalysisConfig, result: RunResult) -> None:
    saved_points = np.asarray(result.metadata.get("cavity_mask_points_nm", []), dtype=float)
    if saved_points.size:
        points = saved_points.reshape((-1, 3))
    else:
        assert config.cavity.mask is not None
        mask = load_voxel_mask(
            config.cavity.mask,
            config.cavity.meta,
            membership_padding=config.cavity.membership_padding_nm,
        )
        points = np.asarray(mask.points, dtype=float)
    lines = ["REMARK PocketMC cavity mask points"]
    for index, point in enumerate(points, start=1):
        serial = ((index - 1) % 99_999) + 1
        resid = ((index - 1) % 9_999) + 1
        x, y, z = (float(value) * 10.0 for value in point)
        lines.append(
            f"HETATM{serial:5d} HE   CAV M{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          He"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_substrate_structure(path: Path, result: RunResult) -> bool:
    """Write the plotted substrate image as a dedicated VMD molecule."""
    overlay = result.metadata.get("substrate_overlay")
    if not isinstance(overlay, dict):
        return False
    positions = np.asarray(overlay.get("positions_A", []), dtype=float)
    if positions.size == 0:
        return False
    positions = positions.reshape((-1, 3))
    names = [str(value) for value in overlay.get("atom_names", [])]
    elements = [str(value) for value in overlay.get("elements", [])]
    resnames = [str(value) for value in overlay.get("resnames", [])]
    resids = [int(value) for value in overlay.get("resids", [])]
    count = positions.shape[0]
    names = (names + ["X"] * count)[:count]
    elements = (elements + ["C"] * count)[:count]
    resnames = (resnames + ["SUB"] * count)[:count]
    resids = (resids + [1] * count)[:count]
    lines = ["REMARK substrate in the exact analysis/plot periodic image"]
    for index, (point, name, element, resname, resid) in enumerate(
        zip(positions, names, elements, resnames, resids), start=1
    ):
        x, y, z = (float(value) for value in point)
        lines.append(
            f"HETATM{index:5d} {name[:4]:^4s} {resname[:3]:>3s} S{resid % 10000:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element[:2].title():>2s}"
        )
    # Explicit bonds keep Licorice reliable even when the reference GRO did
    # not preserve topology.  Restrict inference to each selected residue.
    covalent = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "P": 1.07, "S": 1.05, "Cl": 1.02}
    neighbors: dict[int, list[int]] = {}
    for first in range(count):
        for second in range(first + 1, count):
            if (resnames[first], resids[first]) != (resnames[second], resids[second]):
                continue
            radius_first = covalent.get(elements[first].title(), 0.77)
            radius_second = covalent.get(elements[second].title(), 0.77)
            distance = float(np.linalg.norm(positions[first] - positions[second]))
            if 0.35 <= distance <= 1.25 * (radius_first + radius_second):
                neighbors.setdefault(first + 1, []).append(second + 1)
                neighbors.setdefault(second + 1, []).append(first + 1)
    for atom, bonded in sorted(neighbors.items()):
        lines.append(f"CONECT{atom:5d}" + "".join(f"{value:5d}" for value in bonded))
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def _sampled_trace_frames(result: RunResult, interval_ps: float = TRACE_SAMPLE_PS) -> list[FrameRecord]:
    if not result.frames:
        return []
    times = np.asarray([frame.time_ps for frame in result.frames], dtype=float)
    selected: list[FrameRecord] = []
    used: set[int] = set()
    target = float(times[0])
    while target <= float(times[-1]) + 1.0e-9:
        right = int(np.searchsorted(times, target, side="left"))
        candidates = [index for index in (right - 1, right) if 0 <= index < len(result.frames)]
        index = min(candidates, key=lambda value: (abs(float(times[value]) - target), value))
        if index not in used:
            used.add(index)
            selected.append(result.frames[index])
        target += interval_ps
    if len(result.frames) > 1 and len(result.frames) - 1 not in used:
        selected.append(result.frames[-1])
    return selected


def _trimmed_trace_points(
    result: RunResult,
    uid: str,
    interval_ps: float = TRACE_SAMPLE_PS,
) -> list[MoleculeFrame]:
    points = [
        item
        for frame in _sampled_trace_frames(result, interval_ps)
        for item in frame.molecules
        if item.uid == uid
    ]
    if not points:
        return []
    if _category(result, uid) == "resident":
        return points
    # Match the established trace workflow: prefer samples close to the
    # protein surface, and use inside-cavity samples only when no surface
    # sample exists at all.  Combining both sets retained long bulk tails for
    # large cavities.
    near = [
        index
        for index, point in enumerate(points)
        if np.isfinite(point.nearest_distance_nm)
        and point.nearest_distance_nm <= TRACE_SURFACE_CUTOFF_NM + 1.0e-9
    ]
    if not near:
        near = [index for index, point in enumerate(points) if point.inside]
    if not near:
        return []
    kept: set[int] = set()
    for index in near:
        kept.update(
            range(
                max(0, index - TRACE_PAD_SAMPLES),
                min(len(points), index + TRACE_PAD_SAMPLES + 1),
            )
        )
    return [point for index, point in enumerate(points) if index in kept]


def write_vmd_session(config: AnalysisConfig, result: RunResult, run_dir: Path) -> list[Path]:
    vmd_dir = run_dir / "vmd"
    vmd_dir.mkdir(parents=True, exist_ok=True)
    structure, structure_supports_trajectory = _vmd_structure(result, vmd_dir)
    mask_visual: Path | None = None
    if config.cavity.mode == "mask" and config.cavity.mask is not None:
        mask_visual = vmd_dir / "cavity_mask_points.pdb"
        _write_mask_structure(mask_visual, config, result)
    substrate_visual = vmd_dir / "substrate_overlay.pdb"
    if not _write_substrate_structure(substrate_visual, result):
        substrate_visual = None
    density_npz = run_dir / "density" / "density_maps.npz"
    density_cube = run_dir / "density" / "density.cube"
    thresholds: dict[float, float] = {}
    if density_npz.exists():
        with np.load(density_npz) as data:
            rho = np.asarray(data["rho_probability" if "rho_probability" in data.files else "rho"], dtype=float)
        if np.any(np.isfinite(rho) & (rho > 0.0)):
            thresholds = hpd_thresholds(rho)

    lines = [
        "# Unified PocketMC analysis VMD session",
        "set script_dir [file dirname [file normalize [info script]]]",
        f"set structure_path {_tcl_portable_path(structure, vmd_dir)}",
        f"set trajectory_path {_tcl_portable_path(result.dataset.trajectory, vmd_dir)}",
        f"set mask_path {_tcl_portable_path(mask_visual, vmd_dir)}",
        f"set substrate_path {_tcl_portable_path(substrate_visual, vmd_dir)}",
        f"set cube_path {_tcl_portable_path(density_cube, vmd_dir)}",
        "color Display Background white",
        "axes location Off",
        "proc pocketmc_load_session {} {",
        "global structure_path trajectory_path mask_path substrate_path cube_path",
        "if {[catch {mol new $structure_path waitfor all} base_mol]} { error \"Could not load structure: $structure_path ($base_mol)\" }",
        "set ::base_mol $base_mol",
    ]
    if (
        result.dataset.kind == "md"
        and structure_supports_trajectory
        and result.dataset.trajectory.resolve() != structure.resolve()
    ):
        alignment = result.metadata.get("alignment", {})
        indices = alignment.get("atom_indices_0based", []) if isinstance(alignment, dict) else []
        alignment_selection = (
            "index " + " ".join(str(int(value)) for value in indices)
            if indices
            else "protein and noh"
        )
        lines.extend(
            [
                "if {[catch {mol addfile $trajectory_path waitfor all molid $base_mol} trajectory_error]} { error \"Could not load MD trajectory: $trajectory_path ($trajectory_error)\" }",
                f"set alignment_selection {_tcl(alignment_selection)}",
                "set alignment_ref [atomselect $base_mol $alignment_selection frame 0]",
                "for {set iframe 1} {$iframe < [molinfo $base_mol get numframes]} {incr iframe} {",
                "  set alignment_mobile [atomselect $base_mol $alignment_selection frame $iframe]",
                "  set alignment_all [atomselect $base_mol \"all\" frame $iframe]",
                "  $alignment_all move [measure fit $alignment_mobile $alignment_ref]",
                "  $alignment_mobile delete",
                "  $alignment_all delete",
                "}",
                "$alignment_ref delete",
                "animate goto 0",
                "puts \"WARNING: Animated MD frames are rigid-fit from the input trajectory without PBC molecule reconstruction.\"",
                "puts \"WARNING: If protein/substrate atoms are wrapped or split across the unit cell, animated frames may not overlay the static density and traces.\"",
                "puts \"WARNING: Frame 0 and the static density, trace, cavity, and substrate overlays are the authoritative analysis-reference view.\"",
            ]
        )
    lines.extend(
        [
        "mol delrep 0 $base_mol",
        "mol representation NewCartoon",
        "mol color Structure",
        "mol selection protein",
        "mol material Opaque",
        "mol addrep $base_mol",
        "catch {array unset ::pocketmc_trace_commands}",
        "catch {array unset ::pocketmc_trace_colors}",
        "catch {array unset ::pocketmc_trace_ids}",
        "array set ::pocketmc_trace_commands {}",
        "array set ::pocketmc_trace_colors {}",
        "array set ::pocketmc_trace_ids {}",
        "proc pocketmc_trace_draw {label} {",
        "  if {![info exists ::pocketmc_trace_commands($label)]} { return }",
        "  if {[info exists ::pocketmc_trace_ids($label)] && [llength $::pocketmc_trace_ids($label)] > 0} { return }",
        "  graphics $::base_mol color $::pocketmc_trace_colors($label)",
        "  graphics $::base_mol material Opaque",
        "  set ids {}",
        "  foreach spec $::pocketmc_trace_commands($label) {",
        "    set command [linsert $spec 0 graphics $::base_mol]",
        "    lappend ids [uplevel #0 $command]",
        "  }",
        "  set ::pocketmc_trace_ids($label) $ids",
        "}",
        "proc trace_show {{pattern all}} {",
        "  if {$pattern eq \"all\"} { set pattern * }",
        "  foreach label [array names ::pocketmc_trace_commands $pattern] { pocketmc_trace_draw $label }",
        "}",
        "proc trace_hide {{pattern all}} {",
        "  if {$pattern eq \"all\"} { set pattern * }",
        "  foreach label [array names ::pocketmc_trace_commands $pattern] {",
        "    if {[info exists ::pocketmc_trace_ids($label)]} {",
        "      foreach id $::pocketmc_trace_ids($label) { graphics $::base_mol delete $id }",
        "      set ::pocketmc_trace_ids($label) {}",
        "    }",
        "  }",
        "}",
        ]
    )
    substrate = result.metadata.get("substrate_overlay", {})
    substrate_indices = substrate.get("atom_indices_0based", []) if isinstance(substrate, dict) else []
    substrate_selection = (
        "index " + " ".join(str(int(value)) for value in substrate_indices)
        if substrate_indices
        else (result.dataset.substrate_selection or config.substrate.selection)
    )
    if substrate_visual is not None:
        lines.extend(
            [
                "# Selected substrate in the same periodic image as density/traces",
                "mol new $substrate_path waitfor all",
                "set substrate_mol [molinfo top]",
                "mol delrep 0 $substrate_mol",
                "mol representation Licorice 0.24 16.0 16.0",
                "mol color Name",
                "mol selection all",
                "mol material Opaque",
                "mol addrep $substrate_mol",
            ]
        )
    elif substrate_selection:
        lines.extend(
            [
                "# Selected substrate (for example OPP/ATC)",
                "mol representation Licorice 0.24 16.0 16.0",
                "mol color Name",
                f"mol selection {_tcl(substrate_selection)}",
                "mol material Opaque",
                "mol addrep $base_mol",
            ]
        )
    if config.cavity.mode == "mask" and config.cavity.mask is not None:
        lines.extend(
            [
                "if {[file exists $mask_path]} {",
                "  mol new $mask_path waitfor all",
                "  set mask_mol [molinfo top]",
                "  mol delrep 0 $mask_mol",
                "  mol representation VDW 0.22 8.0",
                "  mol color ColorID 7",
                "  mol selection all",
                "  mol material Transparent",
                "  mol addrep $mask_mol",
                "}",
            ]
        )
    elif "cavity_center_nm" in result.metadata:
        center = [float(value) * 10.0 for value in result.metadata["cavity_center_nm"]]
        lines.extend(
            [
                "graphics $base_mol color yellow",
                "graphics $base_mol material Transparent",
                "graphics $base_mol sphere {"
                + " ".join(f"{value:.4f}" for value in center)
                + f"}} radius {config.cavity.radius_nm * 10.0:.4f} resolution 30",
            ]
        )
    if result.dataset.kind == "md":
        color_by_category = {"entry": "red", "exit": "blue", "resident": "gray"}
        visited = {visit.molecule_uid for visit in result.visits}
        visited.update(item.uid for frame in result.frames for item in frame.molecules if item.inside)
        for uid in sorted(visited):
            sample_ps = float(config.analysis.path_sample_ps)
            points = _trimmed_trace_points(result, uid, sample_ps)
            if not points:
                continue
            category = _category(result, uid)
            color = color_by_category[category]
            lines.append(
                f"# trace {uid} category={category}; sampled={sample_ps / 1000.0:g} ns; "
                f"surface_cutoff={TRACE_SURFACE_CUTOFF_NM * 10.0:g} A"
            )
            lines.append(f"set trace_label {_tcl(uid)}")
            lines.append(f"set ::pocketmc_trace_colors($trace_label) {color}")
            used: set[int] = set()
            for index, (first, second) in enumerate(zip(points, points[1:])):
                first_a = [value * 10.0 for value in first.point_nm]
                second_a = [value * 10.0 for value in second.point_nm]
                if float(np.linalg.norm(np.asarray(second.point_nm) - np.asarray(first.point_nm))) > TRACE_MAX_SEGMENT_NM:
                    continue
                used.update((index, index + 1))
                lines.append(
                    "lappend ::pocketmc_trace_commands($trace_label) [list line "
                    + "{" + " ".join(f"{value:.4f}" for value in first_a) + "} "
                    + "{" + " ".join(f"{value:.4f}" for value in second_a) + "} width 2 style solid]"
                )
            for index, point in enumerate(points):
                if index in used:
                    continue
                point_a = [value * 10.0 for value in point.point_nm]
                lines.append(
                    "lappend ::pocketmc_trace_commands($trace_label) [list sphere "
                    + "{" + " ".join(f"{value:.4f}" for value in point_a) + "} "
                    + f"radius {TRACE_POINT_RADIUS_A:.3f} resolution 10]"
                )
            lines.append("pocketmc_trace_draw $trace_label")
    else:
        lines.append("# PocketMC states are deliberately shown as unconnected points; MC moves are not physical paths.")
        lines.append("graphics $base_mol color green")
        lines.append("graphics $base_mol material Opaque")
        for frame in result.frames:
            for item in frame.molecules:
                if not item.inside:
                    continue
                point = [value * 10.0 for value in item.point_nm]
                lines.append("graphics $base_mol sphere {" + " ".join(f"{value:.4f}" for value in point) + "} radius 0.18 resolution 8")
    if density_cube.exists() and thresholds:
        lines.extend(
            [
                "if {[catch {mol new $cube_path type cube waitfor all} density_mol]} { error \"Could not load density cube: $cube_path\" }",
                "mol delrep 0 $density_mol",
            ]
        )
        colors = (8, 0, 1, 7)
        materials = ("Transparent", "Transparent", "Transparent", "Opaque")
        for (probability, level), color, material in zip(sorted(thresholds.items(), reverse=True), colors, materials):
            lines.extend(
                [
                    f"# HPD {probability:.0%}",
                    f"mol representation Isosurface {level:.10g} 0 0 0 1 1",
                    f"mol color ColorID {color}",
                    "mol selection all",
                    f"mol material {material}",
                    "mol addrep $density_mol",
                ]
            )
    lines.extend(
        [
            "mol top $base_mol",
            "display resetview",
            f"puts \"Loaded PocketMC {result.dataset.kind} analysis session.\"",
            (
                "puts \"WARNING: accepted-state density is not an equilibrium probability or physical trajectory.\""
                if result.dataset.kind == "pocketmc"
                else "puts \"MD traces are drawn in the analysis alignment frame.\""
            ),
            "}",
            "if {[catch {pocketmc_load_session} pocketmc_session_error]} {",
            "  puts stderr \"PocketMC VMD session could not be loaded:\"",
            "  puts stderr $pocketmc_session_error",
            "}",
        ]
    )
    tcl_path = vmd_dir / "session.vmd.tcl"
    tcl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    outputs = [tcl_path]
    if structure.parent == vmd_dir:
        outputs.append(structure)
    if mask_visual is not None:
        outputs.append(mask_visual)
    if substrate_visual is not None:
        outputs.append(substrate_visual)
    return outputs


def launch_vmd(tcl_path: str | Path, executable: str = "auto", *, wait: bool = False) -> int:
    path = Path(tcl_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    command = shutil.which("vmd") if executable == "auto" else executable
    if not command:
        raise RuntimeError("VMD executable was not found; pass --vmd with an explicit path")
    process = subprocess.Popen([command, "-e", str(path)])
    return process.wait() if wait else 0


def render_vmd(tcl_path: str | Path, executable: str = "auto") -> int:
    """Run an explicitly requested VMD render script without opening a display."""
    path = Path(tcl_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    command = shutil.which("vmd") if executable == "auto" else executable
    if not command:
        raise RuntimeError("VMD executable was not found; pass --vmd with an explicit path")
    result = subprocess.run(
        [command, "-dispdev", "text", "-e", str(path)],
        cwd=path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown VMD render failure"
        raise RuntimeError(detail)
    return 0
