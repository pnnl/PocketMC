from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterator

import numpy as np
from scipy.spatial import cKDTree

from gcmc_port.cavity import VoxelMask, load_voxel_mask
from gcmc_port.gro import GRO_INDEX_MODULUS, Atom, GroStructure, contiguous_residue_groups, coordinates_center, parse_atom_line

from .anchors import select_gro_anchor
from .geometry import apply_transform, kabsch_transform
from .models import AnalysisConfig, DatasetSpec, FrameRecord, MCMove, MoleculeFrame, RunResult


@dataclass(frozen=True, slots=True)
class ParsedState:
    index: int
    accepted: int
    energy: float | None
    structure: GroStructure


def iter_gro_states(path: Path) -> Iterator[ParsedState]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    cursor = 0
    state = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        if cursor + 2 >= len(lines):
            raise ValueError(f"Incomplete GRO state at line {cursor + 1}: {path}")
        title = lines[cursor]
        try:
            count = int(lines[cursor + 1].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid GRO state atom count at line {cursor + 2}: {path}") from exc
        end = cursor + 2 + count
        if end >= len(lines):
            raise ValueError(f"Incomplete GRO state {state}: {path}")
        atoms: list[Atom] = []
        residue_offset = 0
        previous_raw_resid: int | None = None
        for atom_index, line in enumerate(lines[cursor + 2 : end], start=1):
            atom = parse_atom_line(line)
            raw_resid = atom.resid
            if (
                previous_raw_resid is not None
                and raw_resid < previous_raw_resid
                and previous_raw_resid - raw_resid > GRO_INDEX_MODULUS // 2
            ):
                residue_offset += GRO_INDEX_MODULUS
            atom.resid = raw_resid + residue_offset
            atom.atomnr = atom_index
            atoms.append(atom)
            previous_raw_resid = raw_resid
        numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", title)
        accepted = int(float(numbers[0])) if numbers else state + 1
        energy = float(numbers[1]) if len(numbers) > 1 else None
        yield ParsedState(
            index=state,
            accepted=accepted,
            energy=energy,
            structure=GroStructure(title=title, atoms=atoms, box_line=lines[end]),
        )
        state += 1
        cursor = end + 1


def _mask_from_state(mask: VoxelMask, parsed: ParsedState, trajectory: Path) -> VoxelMask:
    points = np.asarray([(atom.x, atom.y, atom.z) for atom in parsed.structure.atoms], dtype=float)
    if points.shape[0] != mask.point_count:
        raise ValueError(
            f"Mask trajectory point count ({points.shape[0]}) does not match mask "
            f"({mask.point_count}) at accepted state {parsed.accepted}: {trajectory}"
        )
    return VoxelMask(
        points=points,
        dx=mask.dx,
        reference_point=tuple(float(value) for value in points.mean(axis=0)),
        effective_volume=mask.effective_volume,
        membership_padding=mask.membership_padding,
        probe_radius=mask.probe_radius,
        source_gro=mask.source_gro,
        exclude_residues=mask.exclude_residues,
    )


def _mask_frame_transform(
    state_mask: VoxelMask,
    reference_mask: VoxelMask,
    *,
    accepted: int,
    trajectory: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map one accepted-state mask image onto the first accepted-state mask."""
    mobile = np.asarray(state_mask.points, dtype=float)
    target = np.asarray(reference_mask.points, dtype=float)
    if mobile.shape != target.shape or mobile.shape[0] == 0:
        raise ValueError(
            f"PocketMC mask frames are not correspondence-compatible at accepted state {accepted}: {trajectory}"
        )
    mobile_centered = mobile - mobile.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    if mobile.shape[0] >= 3 and min(np.linalg.matrix_rank(mobile_centered), np.linalg.matrix_rank(target_centered)) >= 2:
        transform = kabsch_transform(mobile, target)
    else:
        transform = (np.eye(3), mobile.mean(axis=0), target.mean(axis=0))
    fitted = apply_transform(mobile, transform)
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    tolerance = max(0.002, 0.25 * float(state_mask.dx))
    if not np.isfinite(rmsd) or rmsd > tolerance:
        raise ValueError(
            f"PocketMC mask frame {accepted} cannot be rigidly aligned to the first accepted-state mask "
            f"(RMSD={rmsd:.6f} nm, tolerance={tolerance:.6f} nm): {trajectory}"
        )
    return transform


def read_sidecar(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    records: dict[int, dict[str, Any]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid trajectory sidecar JSON at {path}:{line_number}: {exc}") from exc
        records[int(record["accepted_state"])] = record
    return records


MC_ROW_RE = re.compile(
    r"^\s*(?P<trial>\d+)\s+(?P<accepted>\d+)\s+(?P<move>[IRTD])\s*\|\s*"
    r"(?P<e1>\S+)\s+(?P<e0>\S+)\s+(?P<de>\S+)\s*\|.*?\|\s*\d+\s+(?P<status>ACC\.|REJ\.)\s*\|\s*(?P<nins>\d+)"
)


def _number(value: str) -> float | None:
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def read_mc_moves(dataset: DatasetSpec) -> list[MCMove]:
    if dataset.mc_log is None or not dataset.mc_log.exists():
        return []
    moves: list[MCMove] = []
    for line in dataset.mc_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = MC_ROW_RE.match(line)
        if match is None:
            continue
        moves.append(
            MCMove(
                run_id=dataset.run_id,
                trial=int(match.group("trial")),
                accepted_before=int(match.group("accepted")),
                move=match.group("move"),
                accepted=match.group("status") == "ACC.",
                energy_kj_mol=_number(match.group("e1")),
                delta_energy_kj_mol=_number(match.group("de")),
                n_inside_before=int(match.group("nins")),
            )
        )
    return moves


def _is_legacy_dummy(group: list[Atom]) -> bool:
    if not group or group[0].resname.upper() != "WAT":
        return False
    if len(group) % 3 or {atom.atomname.upper() for atom in group} != {"OW", "HW1", "HW2"}:
        return False
    first = (group[0].x, group[0].y, group[0].z)
    names = [atom.atomname.upper() for atom in group]
    expected = ["OW", "HW1", "HW2"] * (len(group) // 3)
    return names == expected and all((atom.x, atom.y, atom.z) == first for atom in group[1:])


def _representative(group: list[Atom], config: AnalysisConfig) -> tuple[float, float, float]:
    molecule = config.molecule
    selected = [atom for atom in group if not molecule.atom_names or atom.atomname.upper() in molecule.atom_names]
    if molecule.point_mode == "atom":
        for name in molecule.atom_names:
            match = next((atom for atom in group if atom.atomname.upper() == name), None)
            if match is not None:
                return match.x, match.y, match.z
        raise ValueError(f"Residue {group[0].resid}{group[0].resname} has no representative atom")
    if molecule.point_mode == "com":
        raise ValueError("COM analysis of PocketMC GRO states requires mass metadata; use point_mode='cog'")
    return coordinates_center(selected or group)


def _anchor_point(structure: GroStructure, config: AnalysisConfig) -> np.ndarray:
    atoms, resolution = select_gro_anchor(
        structure,
        config.cavity.anchor,
        config.cavity.anchor_atoms,
        context="PocketMC cavity",
    )
    if resolution.warning:
        print(f"[anchor] {resolution.warning}", flush=True)
    return np.asarray(coordinates_center(atoms), dtype=float)


def read_mc_dataset(config: AnalysisConfig, dataset: DatasetSpec) -> RunResult:
    sidecar = read_sidecar(dataset.trajectory_meta)
    moves = read_mc_moves(dataset)
    accepted_moves = [move for move in moves if move.accepted]
    mask = None
    mask_states: Iterator[ParsedState] | None = None
    current_mask_state: ParsedState | None = None
    representative_mask: VoxelMask | None = None
    if config.cavity.mode == "mask":
        assert config.cavity.mask is not None
        mask = load_voxel_mask(config.cavity.mask, config.cavity.meta, membership_padding=config.cavity.membership_padding_nm)
        if config.cavity.mask_trajectory is not None:
            mask_states = iter_gro_states(config.cavity.mask_trajectory)
            current_mask_state = next(mask_states, None)
    frames: list[FrameRecord] = []
    generations: dict[tuple[int, str], int] = {}
    active_previous: set[tuple[int, str]] = set()
    cavity_center_nm: tuple[float, float, float] | None = None
    warnings: list[str] = []
    if not sidecar:
        warnings.append("legacy_identity_inferred: trajectory.meta.jsonl was not found; molecule identity was inferred")
    for parsed in iter_gro_states(dataset.trajectory):
        state_mask = mask
        state_to_reference = None
        if mask_states is not None:
            while current_mask_state is not None and current_mask_state.accepted < parsed.accepted:
                current_mask_state = next(mask_states, None)
            if current_mask_state is None or current_mask_state.accepted != parsed.accepted:
                raise ValueError(
                    f"No cavity mask trajectory frame matches accepted state {parsed.accepted}: "
                    f"{config.cavity.mask_trajectory}"
                )
            assert mask is not None and config.cavity.mask_trajectory is not None
            state_mask = _mask_from_state(mask, current_mask_state, config.cavity.mask_trajectory)
            if representative_mask is None:
                representative_mask = state_mask
            state_to_reference = _mask_frame_transform(
                state_mask,
                representative_mask,
                accepted=parsed.accepted,
                trajectory=config.cavity.mask_trajectory,
            )
        accepted_move = accepted_moves[parsed.index] if parsed.index < len(accepted_moves) else None
        record = sidecar.get(parsed.accepted, {})
        sidecar_by_key = {
            (int(item["resid"]), str(item["resname"]).upper()): str(item["uid"])
            for item in record.get("molecules", [])
        }
        configured_groups: list[list[Atom]] = []
        protein_atoms: list[Atom] = []
        for group in contiguous_residue_groups(parsed.structure):
            if group[0].resname.upper() in config.molecule.resnames:
                if _is_legacy_dummy(group):
                    continue
                configured_groups.append(group)
            elif group[0].resname.upper() not in {"SOL", "WAT", "HOH", "NA", "CL", "K", "COM"} and not group[0].atomname.upper().startswith("H"):
                protein_atoms.extend(group)
        if "molecules" in record:
            target_groups = [
                group
                for group in configured_groups
                if (group[0].resid, group[0].resname.upper()) in sidecar_by_key
            ]
        elif config.molecule.preset == "water":
            # Legacy PocketMC used WAT for inserted waters and SOL for bulk solvent.
            target_groups = [group for group in configured_groups if group[0].resname.upper() == "WAT"]
        else:
            target_groups = configured_groups
        if not record and accepted_move is not None and accepted_move.n_inside_before is not None:
            expected = accepted_move.n_inside_before + {"I": 1, "D": -1, "R": 0, "T": 0}.get(accepted_move.move, 0)
            if len(target_groups) != expected:
                warning = (
                    f"legacy_state_count_mismatch: accepted_state={parsed.accepted} "
                    f"mc.log_expected={expected} trajectory_inferred={len(target_groups)}"
                )
                if warning not in warnings:
                    warnings.append(warning)
        current_keys = {(group[0].resid, group[0].resname.upper()) for group in target_groups}
        for key in current_keys - active_previous:
            generations[key] = generations.get(key, -1) + 1
        active_previous = current_keys
        anchor = _anchor_point(parsed.structure, config) if state_mask is None else None
        if anchor is not None and cavity_center_nm is None:
            cavity_center_nm = tuple(float(value) for value in anchor)
        protein_points = np.asarray([(atom.x, atom.y, atom.z) for atom in protein_atoms], dtype=float)
        protein_labels = [f"{atom.resid}{atom.resname.upper()}" for atom in protein_atoms]
        tree = cKDTree(protein_points) if protein_points.size else None
        molecule_rows: list[MoleculeFrame] = []
        for group in target_groups:
            key = (group[0].resid, group[0].resname.upper())
            uid = sidecar_by_key.get(key, f"{key[0]}{key[1]}@{generations.get(key, 0)}")
            point = np.asarray(_representative(group, config), dtype=float)
            inside = state_mask.contains_point(tuple(point)) if state_mask is not None else float(np.linalg.norm(point - anchor)) <= (
                config.cavity.radius_nm + config.cavity.membership_padding_nm
            )
            saved_point = apply_transform(point.reshape(1, 3), state_to_reference)[0]
            nearest = ""
            distance = float("nan")
            if tree is not None:
                distance, hit = tree.query(point, k=1)
                nearest = protein_labels[int(hit)]
            molecule_rows.append(
                MoleculeFrame(
                    uid=uid,
                    resid=key[0],
                    resname=key[1],
                    point_nm=tuple(float(value) for value in saved_point),
                    inside=bool(inside),
                    nearest_residue=nearest,
                    nearest_distance_nm=float(distance),
                    nearest_residue_sim=nearest,
                )
            )
        frames.append(
            FrameRecord(
                frame=parsed.accepted,
                time_ps=float(parsed.accepted),
                molecules=tuple(molecule_rows),
                occupancy=sum(1 for item in molecule_rows if item.inside),
                energy_kj_mol=parsed.energy,
                trial=(int(record["trial"]) if "trial" in record else (accepted_move.trial if accepted_move else None)),
                move=(str(record.get("move", "")) or (accepted_move.move if accepted_move else "")),
            )
        )
    if not frames:
        raise ValueError(f"No PocketMC states found in {dataset.trajectory}")
    result = RunResult(dataset=dataset, frames=frames, mc_moves=moves, warnings=warnings)
    if cavity_center_nm is not None:
        result.metadata["cavity_center_nm"] = cavity_center_nm
    if mask is not None:
        saved_mask = representative_mask or mask
        result.metadata["cavity_mask_points_nm"] = np.asarray(saved_mask.points, dtype=float).tolist()
        result.metadata["cavity_mask_reference_nm"] = list(saved_mask.reference_point)
        if representative_mask is not None:
            result.metadata["coordinate_frame"] = "pocketmc-first-accepted-mask"
            result.metadata["mask_alignment"] = {
                "description": "Each accepted state's molecule coordinates were rigidly fitted from its ordered mask points to the first accepted-state mask.",
                "reference": str(config.cavity.mask_trajectory),
            }
    return result
